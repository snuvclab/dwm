import inspect
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any

from diffusers.configuration_utils import register_to_config
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.attention_processor import CogVideoXAttnProcessor2_0
import numpy as np
import PIL
import torch
import torch.nn as nn
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXTransformer3DModel,
)
from diffusers.image_processor import PipelineImageInput
from diffusers.models.embeddings import get_1d_rotary_pos_embed
from diffusers.models.normalization import CogVideoXLayerNormZero
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from transformers import AutoTokenizer, T5EncoderModel

from training.aether.utils.preprocess_utils import imcrop_center
from training.aether.aetherv1_pipeline_cogvideox import (
    get_3d_rotary_pos_embed,
    get_resize_crop_region_for_grid,
    retrieve_timesteps,
    retrieve_latents,
    AetherV1PipelineOutput,
    AetherV1PipelineCogVideoX,
)


class SMPLPoseEmbedding(nn.Module):
    """Embedding layer for SMPL pose parameters with frame sampling for AdaLN conditioning"""
    
    def __init__(self, pose_dim=63, embed_dim=512, hidden_dim=1024, sample_stride=4):
        super().__init__()
        self.pose_dim = pose_dim
        self.embed_dim = embed_dim
        self.sample_stride = sample_stride
        
        # Calculate flattened input dimension: pose_dim * num_sampled_frames
        # For 49 frames with stride 4: frames [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48] = 13 frames
        self.num_sampled_frames = (49 + sample_stride - 1) // sample_stride  # 13 frames
        self.flattened_dim = pose_dim * self.num_sampled_frames  # 63 * 13 = 819
        
        # Multi-layer perceptron to embed flattened pose parameters
        self.pose_projection = nn.Sequential(
            nn.Linear(self.flattened_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, pose_params: torch.Tensor) -> torch.Tensor:
        """
        Process SMPL pose parameters into embeddings.
        
        Args:
            pose_params: Tensor of shape (batch_size, num_frames, pose_dim)
            
        Returns:
            Tensor of shape (batch_size, embed_dim)
        """
        batch_size, num_frames, pose_dim = pose_params.shape
        
        # Sample frames with stride
        sampled_frames = pose_params[:, ::self.sample_stride, :]  # (B, 13, 63)
        
        # Ensure we have exactly the right number of frames
        if sampled_frames.shape[1] != self.num_sampled_frames:
            if sampled_frames.shape[1] < self.num_sampled_frames:
                # Pad with the last frame
                last_frame = sampled_frames[:, -1:, :]
                padding = last_frame.repeat(1, self.num_sampled_frames - sampled_frames.shape[1], 1)
                sampled_frames = torch.cat([sampled_frames, padding], dim=1)
            else:
                # Truncate to the correct number of frames
                sampled_frames = sampled_frames[:, :self.num_sampled_frames, :]
        
        # Flatten and process
        flattened = sampled_frames.reshape(batch_size, -1)  # (B, 13*63)
        embeddings = self.pose_projection(flattened)  # (B, embed_dim)
        embeddings = self.layer_norm(embeddings)
        
        return embeddings


class SMPLConditionedCogVideoXLayerNormZero(CogVideoXLayerNormZero):
    """Extended CogVideoXLayerNormZero with SMPL pose conditioning via AdaLN-zero"""
    
    def __init__(
        self,
        conditioning_dim: int,
        embedding_dim: int,
        smpl_embed_dim: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        bias: bool = True,
    ) -> None:
        super().__init__(conditioning_dim, embedding_dim, elementwise_affine, eps, bias)
        
        # Additional SMPL conditioning layers (zero-initialized for AdaLN-zero)
        self.smpl_linear = nn.Linear(smpl_embed_dim, 6 * embedding_dim, bias=bias)
        
        # Zero initialization (AdaLN-zero key feature)
        nn.init.zeros_(self.smpl_linear.weight)
        nn.init.zeros_(self.smpl_linear.bias)
        
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        encoder_hidden_states: torch.Tensor, 
        temb: torch.Tensor,
        smpl_emb: Optional[torch.Tensor] = None,
        smpl_emb_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: Input hidden states
            encoder_hidden_states: Input encoder hidden states  
            temb: Time embedding (original CogVideoX conditioning)
            smpl_emb: SMPL pose embedding (additional conditioning)
            smpl_emb_mask: Mask to zero out SMPL conditioning for certain samples
        Returns:
            Tuple of (norm_hidden_states, norm_encoder_hidden_states, gate, enc_gate)
        """
        # Original time conditioning
        time_shift, time_scale, time_gate, time_enc_shift, time_enc_scale, time_enc_gate = \
            self.linear(self.silu(temb)).chunk(6, dim=1)
        # SMPL conditioning (starts at zero due to initialization)
        if smpl_emb is not None:
            smpl_shift, smpl_scale, smpl_gate, smpl_enc_shift, smpl_enc_scale, smpl_enc_gate = \
                self.smpl_linear(self.silu(smpl_emb)).chunk(6, dim=1)
            
            # Apply mask to zero out SMPL conditioning for masked samples
            if smpl_emb_mask is not None:
                # Expand mask to match the shape of SMPL parameters
                # Use the same dtype as smpl_emb for consistency
                mask_expanded = smpl_emb_mask.to(dtype=smpl_emb.dtype)[:, None]  # (batch_size, 1)
                smpl_shift = smpl_shift * mask_expanded
                smpl_scale = smpl_scale * mask_expanded
                smpl_gate = smpl_gate * mask_expanded
                smpl_enc_shift = smpl_enc_shift * mask_expanded
                smpl_enc_scale = smpl_enc_scale * mask_expanded
                smpl_enc_gate = smpl_enc_gate * mask_expanded
            
            # Combine time + SMPL conditioning
            shift = time_shift + smpl_shift
            scale = time_scale + smpl_scale
            gate = time_gate + smpl_gate
            enc_shift = time_enc_shift + smpl_enc_shift
            enc_scale = time_enc_scale + smpl_enc_scale
            enc_gate = time_enc_gate + smpl_enc_gate
        else:
            # Fall back to original behavior (no SMPL conditioning)
            shift, scale, gate = time_shift, time_scale, time_gate
            enc_shift, enc_scale, enc_gate = time_enc_shift, time_enc_scale, time_enc_gate
        
        # Apply modulation (same as original CogVideoXLayerNormZero)
        hidden_states = self.norm(hidden_states) * (1 + scale)[:, None, :] + shift[:, None, :]
        encoder_hidden_states = self.norm(encoder_hidden_states) * (1 + enc_scale)[:, None, :] + enc_shift[:, None, :]
        
        return hidden_states, encoder_hidden_states, gate[:, None, :], enc_gate[:, None, :]


class SMPLConditionedCogVideoXBlock(nn.Module):
    """CogVideoXBlock with SMPL pose conditioning via extended LayerNormZero"""
    
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        time_embed_dim: int,
        smpl_embed_dim: int,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = False,
        qk_norm: bool = True,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = True,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        super().__init__()

        # 1. Self Attention with SMPL-conditioned LayerNormZero
        self.norm1 = SMPLConditionedCogVideoXLayerNormZero(
            time_embed_dim, dim, smpl_embed_dim, norm_elementwise_affine, norm_eps, bias=True
        )

        self.attn1 = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=attention_bias,
            out_bias=attention_out_bias,
            processor=CogVideoXAttnProcessor2_0(),
        )

        # 2. Feed Forward with SMPL-conditioned LayerNormZero
        self.norm2 = SMPLConditionedCogVideoXLayerNormZero(
            time_embed_dim, dim, smpl_embed_dim, norm_elementwise_affine, norm_eps, bias=True
        )

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        smpl_emb: Optional[torch.Tensor] = None,
        smpl_emb_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        text_seq_length = encoder_hidden_states.size(1)
        attention_kwargs = attention_kwargs or {}

        # 1. Self Attention with SMPL conditioning
        norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = self.norm1(
            hidden_states, encoder_hidden_states, temb, smpl_emb, smpl_emb_mask
        )

        # Attention
        attn_hidden_states, attn_encoder_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **attention_kwargs,
        )

        hidden_states = hidden_states + gate_msa * attn_hidden_states
        encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states

        # 2. Feed Forward with SMPL conditioning
        norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = self.norm2(
            hidden_states, encoder_hidden_states, temb, smpl_emb, smpl_emb_mask
        )

        # Feed-forward
        norm_combined = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
        ff_output = self.ff(norm_combined)

        hidden_states = hidden_states + gate_ff * ff_output[:, text_seq_length:]
        encoder_hidden_states = encoder_hidden_states + enc_gate_ff * ff_output[:, :text_seq_length]

        return hidden_states, encoder_hidden_states


class SMPLConditionedTransformer3DAdaLNZero(CogVideoXTransformer3DModel):
    """Variant of CogVideoXTransformer3DModel with SMPL pose conditioning via AdaLN-zero
    
    This model conditions on SMPL pose parameters by:
    1. Sampling frames with stride 4 (e.g., frames 0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48)
    2. Flattening the sampled pose parameters (63 * 13 = 819 dimensions)
    3. Encoding them into a global embedding for AdaLN-zero conditioning
    4. Applying the conditioning through extended LayerNormZero layers
    """
    
    _skip_layerwise_casting_patterns = ["patch_embed", "norm"]
    _supports_gradient_checkpointing = True
    _no_split_modules = ["SMPLConditionedCogVideoXBlock", "CogVideoXBlock", "CogVideoXPatchEmbed"]

    @register_to_config
    def __init__(self, *args, **kwargs):
        # Extract SMPL-specific parameters
        self.smpl_pose_dim = kwargs.pop('smpl_pose_dim', 63)
        self.smpl_embed_dim = kwargs.pop('smpl_embed_dim', 512)
        
        super().__init__(*args, **kwargs)
        
        # Initialize SMPL pose embedding
        self.smpl_pose_embedding = SMPLPoseEmbedding(
            pose_dim=self.smpl_pose_dim,
            embed_dim=self.smpl_embed_dim,
            sample_stride=4  # Sample every 4th frame for AdaLN conditioning
        )
        
        # Use the original block replacement approach
        self._replace_blocks_with_smpl_conditioned()
        
        # Ensure all SMPL components have the correct dtype AFTER creating all components
        self._ensure_smpl_dtype()
    
    def _replace_blocks_with_smpl_conditioned(self):
        """Original block replacement approach (kept for backward compatibility)"""
        # Replace transformer blocks with SMPL-conditioned versions
        smpl_conditioned_blocks = []
        for block in self.transformer_blocks:
            smpl_block = SMPLConditionedCogVideoXBlock(
                dim=block.norm1.norm.normalized_shape[0],  # Get dimension from existing norm
                num_attention_heads=self.config.num_attention_heads,
                attention_head_dim=self.config.attention_head_dim,
                time_embed_dim=self.config.time_embed_dim,
                smpl_embed_dim=self.smpl_embed_dim,
                dropout=self.config.dropout,
                activation_fn=self.config.activation_fn,
                attention_bias=self.config.attention_bias,
                norm_elementwise_affine=block.norm1.norm.elementwise_affine,
                norm_eps=block.norm1.norm.eps,
            )
            
            # Ensure SMPL block has the same dtype as the base model
            target_dtype = next(block.parameters()).dtype
            smpl_block = smpl_block.to(dtype=target_dtype)
            
            # Copy pretrained weights from original block to SMPL block for overlapping components
            # Copy attention weights
            smpl_block.attn1.load_state_dict(block.attn1.state_dict(), strict=False)
            
            # Copy feed-forward weights
            smpl_block.ff.load_state_dict(block.ff.state_dict(), strict=False)
            
            # Copy original LayerNorm weights to the base LayerNorm in SMPL-conditioned LayerNorm
            # Note: We only copy the base norm weights, not the SMPL-specific ones
            smpl_block.norm1.norm.load_state_dict(block.norm1.norm.state_dict(), strict=False)
            smpl_block.norm2.norm.load_state_dict(block.norm2.norm.state_dict(), strict=False)
            
            # Copy the original linear layers (time conditioning) to SMPL blocks
            smpl_block.norm1.linear.load_state_dict(block.norm1.linear.state_dict(), strict=False)
            smpl_block.norm2.linear.load_state_dict(block.norm2.linear.state_dict(), strict=False)
            
            smpl_conditioned_blocks.append(smpl_block)
        
        self.transformer_blocks = nn.ModuleList(smpl_conditioned_blocks)
    
    def _ensure_smpl_dtype(self):
        """Ensure all SMPL components have the correct dtype"""
        if hasattr(self, 'smpl_pose_embedding'):
            # Get the dtype from the base model
            base_dtype = next(self.parameters()).dtype
            self.smpl_pose_embedding = self.smpl_pose_embedding.to(dtype=base_dtype)
    
    @classmethod
    def from_aether_pretrained(cls, aether_model_path, *args, **kwargs):
        """Load original Aether weights and convert to SMPL-conditioned (initialization case)"""
        # Extract SMPL-specific parameters
        smpl_pose_dim = kwargs.pop('smpl_pose_dim', 63)
        smpl_embed_dim = kwargs.pop('smpl_embed_dim', 512)
        
        # Load the original Aether model using the correct class
        base_model = CogVideoXTransformer3DModel.from_pretrained(aether_model_path, *args, **kwargs)
        
        # Create our SMPL-conditioned model instance with proper initialization
        # Pass the config and SMPL parameters to __init__
        config_dict = dict(base_model.config)
        model = cls(
            **config_dict,
            smpl_pose_dim=smpl_pose_dim,
            smpl_embed_dim=smpl_embed_dim
        )
        
        # Copy all pretrained weights from the base model
        # This will copy weights for all overlapping layers (patch_embed, transformer_blocks, etc.)
        model.load_state_dict(base_model.state_dict(), strict=False)
        
        # Ensure the model has the same dtype as the base model
        base_dtype = next(base_model.parameters()).dtype
        model = model.to(dtype=base_dtype)
        
        return model

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        timestep=None,
        ofs=None,
        image_rotary_emb=None,
        attention_kwargs=None,
        pose_params=None,  # New parameter for SMPL pose
        pose_params_mask=None,  # New parameter to track None pose_params
        return_dict=True,
    ):
        # Process pose parameters if provided
        smpl_emb = None
        smpl_emb_mask = None
        if pose_params is not None:
            smpl_emb = self.smpl_pose_embedding(pose_params)
            if pose_params_mask is not None:
                smpl_emb_mask = pose_params_mask
        
        # Use the original approach with block replacement
        # We need to override the forward method to pass SMPL embeddings to blocks
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        batch_size, num_frames, channels, height, width = hidden_states.shape

        # 1. Time embedding (preserve original functionality)
        timesteps = timestep
        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb)

        if self.ofs_embedding is not None:
            ofs_emb = self.ofs_proj(ofs)
            ofs_emb = ofs_emb.to(dtype=hidden_states.dtype)
            ofs_emb = self.ofs_embedding(ofs_emb)
            emb = emb + ofs_emb

        # 2. Patch embedding
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)

        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]

        # 3. Transformer blocks with SMPL conditioning
        for i, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states, encoder_hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    emb,
                    smpl_emb,  # Pass SMPL embedding
                    smpl_emb_mask,  # Pass SMPL embedding mask
                    image_rotary_emb,
                    attention_kwargs,
                )
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=emb,
                    smpl_emb=smpl_emb,  # Pass SMPL embedding
                    smpl_emb_mask=smpl_emb_mask,  # Pass SMPL embedding mask
                    image_rotary_emb=image_rotary_emb,
                    attention_kwargs=attention_kwargs,
                )

        hidden_states = self.norm_final(hidden_states)

        # 4. Final block
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)

        # 5. Unpatchify
        p = self.config.patch_size
        p_t = self.config.patch_size_t

        if p_t is None:
            output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
            output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)
        else:
            output = hidden_states.reshape(
                batch_size, (num_frames + p_t - 1) // p_t, height // p, width // p, -1, p_t, p, p
            )
            output = output.permute(0, 1, 5, 4, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(1, 2)

        if not return_dict:
            return (output,)
        from diffusers.models.modeling_outputs import Transformer2DModelOutput
        return Transformer2DModelOutput(sample=output)


@dataclass
class AetherV1SMPLAdaLNZeroPipelineOutput(BaseOutput):
    rgb: np.ndarray
    disparity: np.ndarray
    raymap: np.ndarray


class AetherV1SMPLAdaLNZeroPipelineCogVideoX(AetherV1PipelineCogVideoX):
    """Variant of AetherV1PipelineCogVideoX with SMPL pose conditioning via extended LayerNormZero"""
    
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: T5EncoderModel,
        vae: AutoencoderKLCogVideoX,
        scheduler: CogVideoXDPMScheduler,
        transformer: SMPLConditionedTransformer3DAdaLNZero,  # Use our custom transformer
    ):
        super().__init__(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            scheduler=scheduler,
            transformer=transformer,
        )
        
        # Override the transformer call to handle pose_params
        self._original_transformer_call = self.transformer.forward
    
    def check_inputs(
        self,
        task,
        image,
        video,
        goal,
        raymap,
        pose_params,  # New parameter
        pose_params_mask,  # New parameter to track None pose_params
        height,
        width,
        num_frames,
        fps,
    ):
        # Call parent check_inputs
        super().check_inputs(
            task=task,
            image=image,
            video=video,
            goal=goal,
            raymap=raymap,
            height=height,
            width=width,
            num_frames=num_frames,
            fps=fps,
        )
        
        # Additional validation for SMPL pose parameters
        if pose_params is not None:
            if not isinstance(pose_params, (torch.Tensor, np.ndarray)):
                raise ValueError(f"pose_params must be a torch.Tensor or np.ndarray, got {type(pose_params)}")
            
            if isinstance(pose_params, np.ndarray):
                pose_params = torch.from_numpy(pose_params)
            
            # Validate pose_params shape: should be (batch_size, num_frames, 63)
            if pose_params.dim() != 3:
                raise ValueError(f"pose_params must have 3 dimensions (batch_size, num_frames, 63), got {pose_params.shape}")
            
            if pose_params.shape[-1] != 63:
                raise ValueError(f"pose_params last dimension must be 63 (SMPL pose parameters), got {pose_params.shape[-1]}")
        
        # Validate pose_params_mask if provided
        if pose_params_mask is not None:
            if not isinstance(pose_params_mask, (torch.Tensor, np.ndarray)):
                raise ValueError(f"pose_params_mask must be a torch.Tensor or np.ndarray, got {type(pose_params_mask)}")
            
            if isinstance(pose_params_mask, np.ndarray):
                pose_params_mask = torch.from_numpy(pose_params_mask)
            
            # Validate pose_params_mask shape: should be (batch_size,)
            if pose_params_mask.dim() != 1:
                raise ValueError(f"pose_params_mask must have 1 dimension (batch_size,), got {pose_params_mask.shape}")
            
            # Validate that pose_params_mask is boolean
            if pose_params_mask.dtype != torch.bool:
                raise ValueError(f"pose_params_mask must be boolean tensor, got {pose_params_mask.dtype}")
            
            # If pose_params is provided, validate that mask length matches batch size
            if pose_params is not None and pose_params_mask.shape[0] != pose_params.shape[0]:
                raise ValueError(f"pose_params_mask batch size ({pose_params_mask.shape[0]}) must match pose_params batch size ({pose_params.shape[0]})")
    
    def preprocess_inputs(
        self,
        image,
        goal,
        video,
        raymap,
        pose_params,  # New parameter
        pose_params_mask,  # New parameter to track None pose_params
        height,
        width,
        num_frames,
    ):
        # Call parent preprocess_inputs
        image, goal, video, raymap = super().preprocess_inputs(
            image=image,
            goal=goal,
            video=video,
            raymap=raymap,
            height=height,
            width=width,
            num_frames=num_frames,
        )
        
        # Preprocess pose_params
        if pose_params is not None:
            if isinstance(pose_params, np.ndarray):
                pose_params = torch.from_numpy(pose_params)
            
            # Ensure pose_params is on the correct device and dtype
            pose_params = pose_params.to(device=self._execution_device, dtype=self.transformer.dtype)
        
        if pose_params_mask is not None:
            if isinstance(pose_params_mask, np.ndarray):
                pose_params_mask = torch.from_numpy(pose_params_mask)
            
            # Ensure pose_params_mask is on the correct device and dtype
            pose_params_mask = pose_params_mask.to(device=self._execution_device, dtype=self.transformer.dtype)
        return image, goal, video, raymap, pose_params, pose_params_mask

    @torch.no_grad()
    def __call__(
        self,
        task: Optional[str] = None,
        image: Optional[PipelineImageInput] = None,
        video: Optional[PipelineImageInput] = None,
        goal: Optional[PipelineImageInput] = None,
        raymap: Optional[Union[torch.Tensor, np.ndarray]] = None,
        pose_params: Optional[Union[torch.Tensor, np.ndarray]] = None,  # New parameter
        pose_params_mask=None,  # New parameter to track None pose_params
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        timesteps: Optional[List[int]] = None,
        guidance_scale: Optional[float] = None,
        use_dynamic_cfg: bool = False,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict] = None,
        fps: Optional[int] = None,
    ) -> Union[AetherV1SMPLAdaLNZeroPipelineOutput, Tuple]:
        
        if task is None:
            if video is not None:
                task = "reconstruction"
            elif goal is not None:
                task = "planning"
            else:
                task = "prediction"

        height = (
            height
            or self.transformer.config.sample_height * self.vae_scale_factor_spatial
        )
        width = (
            width
            or self.transformer.config.sample_width * self.vae_scale_factor_spatial
        )
        num_frames = num_frames or self.transformer.config.sample_frames
        fps = fps or self._base_fps

        num_videos_per_prompt = 1

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            task=task,
            image=image,
            video=video,
            goal=goal,
            raymap=raymap,
            pose_params=pose_params,  # Add pose_params
            pose_params_mask=pose_params_mask,  # Add pose_params_mask
            height=height,
            width=width,
            num_frames=num_frames,
            fps=fps,
        )

        # 2. Preprocess inputs
        image, goal, video, raymap, pose_params, pose_params_mask = self.preprocess_inputs(
            image=image,
            goal=goal,
            video=video,
            raymap=raymap,
            pose_params=pose_params,  # Add pose_params
            pose_params_mask=pose_params_mask,  # Add pose_params_mask
            height=height,
            width=width,
            num_frames=num_frames,
        )
        self._guidance_scale = guidance_scale
        self._current_timestep = None
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        batch_size = 1

        device = self._execution_device

        # 3. Encode input prompt
        prompt_embeds = self.empty_prompt_embeds.to(device)

        num_inference_steps = (
            num_inference_steps or self._default_num_inference_steps[task]
        )
        guidance_scale = guidance_scale or self._default_guidance_scale[task]
        use_dynamic_cfg = use_dynamic_cfg or self._default_use_dynamic_cfg[task]

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps
        )
        self._num_timesteps = len(timesteps)

        # 5. Prepare latents
        latents, condition_latents = self.prepare_latents(
            image,
            goal,
            video,
            raymap,
            batch_size * num_videos_per_prompt,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Create rotary embeds if required
        image_rotary_emb = (
            self._prepare_rotary_positional_embeddings(
                height, width, latents.size(1), device, fps=fps
            )
            if self.transformer.config.use_rotary_positional_embeddings
            else None
        )

        # 8. Create ofs embeds if required
        ofs_emb = (
            None
            if self.transformer.config.ofs_embed_dim is None
            else latents.new_full((1,), fill_value=2.0)
        )

        # 9. Denoising loop
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            # for DPM-solver++
            old_pred_original_sample = None
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                if do_classifier_free_guidance:
                    if task == "planning":
                        assert goal is not None
                        uncond = condition_latents.clone()
                        uncond[:, :, : self.vae.config.latent_channels] = 0
                        latent_condition = torch.cat([uncond, condition_latents])
                    elif task == "prediction":
                        uncond = condition_latents.clone()
                        uncond[:, :1, : self.vae.config.latent_channels] = 0
                        latent_condition = torch.cat([uncond, condition_latents])
                    else:
                        raise ValueError(
                            f"Task {task} not supported for classifier-free guidance."
                        )

                else:
                    latent_condition = condition_latents

                latent_model_input = torch.cat(
                    [latent_model_input, latent_condition], dim=2
                )

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                # Handle pose_params for classifier-free guidance
                if do_classifier_free_guidance and pose_params is not None:
                    # For classifier-free guidance, we need to duplicate pose_params
                    # The first half will be for unconditional generation (no SMPL effect)
                    # The second half will be for conditional generation (with SMPL effect)
                    pose_params_input = torch.cat([pose_params] * 2, dim=0)
                    
                    # Also duplicate pose_params_mask if provided
                    if pose_params_mask is not None:
                        pose_params_mask_input = torch.cat([pose_params_mask] * 2, dim=0)
                    else:
                        pose_params_mask_input = None
                else:
                    pose_params_input = pose_params
                    pose_params_mask_input = pose_params_mask

                # Predict the noise residual (with or without SMPL pose conditioning)
                transformer_kwargs = {
                    "hidden_states": latent_model_input,
                    "encoder_hidden_states": prompt_embeds.repeat(
                        latent_model_input.shape[0], 1, 1
                    ),
                    "timestep": timestep,
                    "ofs": ofs_emb,
                    "image_rotary_emb": image_rotary_emb,
                    "attention_kwargs": attention_kwargs,
                    "return_dict": False,
                }
                
                # Add pose_params only if SMPL conditioning is enabled
                if pose_params_input is not None:
                    transformer_kwargs["pose_params"] = pose_params_input
                if pose_params_mask_input is not None:
                    transformer_kwargs["pose_params_mask"] = pose_params_mask_input
                
                noise_pred = self.transformer(**transformer_kwargs)[0]
                noise_pred = noise_pred.float()

                # perform guidance
                if use_dynamic_cfg:
                    self._guidance_scale = 1 + guidance_scale * (
                        (
                            1
                            - math.cos(
                                math.pi
                                * (
                                    (num_inference_steps - t.item())
                                    / num_inference_steps
                                )
                                ** 5.0
                            )
                        )
                        / 2
                    )

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                if not isinstance(self.scheduler, CogVideoXDPMScheduler):
                    latents = self.scheduler.step(
                        noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                    )[0]
                else:
                    latents, old_pred_original_sample = self.scheduler.step(
                        noise_pred,
                        old_pred_original_sample,
                        t,
                        timesteps[i - 1] if i > 0 else None,
                        latents,
                        **extra_step_kwargs,
                        return_dict=False,
                    )
                latents = latents.to(prompt_embeds.dtype)

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

        self._current_timestep = None

        rgb_latents = latents[:, :, : self.vae.config.latent_channels]
        disparity_latents = latents[
            :, :, self.vae.config.latent_channels : self.vae.config.latent_channels * 2
        ]
        camera_latents = latents[:, :, self.vae.config.latent_channels * 2 :]

        rgb_video = self.decode_latents(rgb_latents)
        rgb_video = self.video_processor.postprocess_video(
            video=rgb_video, output_type="np"
        )

        disparity_video = self.decode_latents(disparity_latents)
        disparity_video = disparity_video.mean(dim=1, keepdim=False)
        disparity_video = disparity_video * 0.5 + 0.5
        disparity_video = torch.square(disparity_video)
        disparity_video = disparity_video.float().cpu().numpy()

        raymap = (
            rearrange(camera_latents, "b t (n c) h w -> b (n t) c h w", n=4)[
                :, -rgb_video.shape[1] :, :, :
            ]
            .float()
            .cpu()
            .numpy()
        )

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (
                rgb_video,
                disparity_video,
                raymap,
            )

        return AetherV1SMPLAdaLNZeroPipelineOutput(
            rgb=rgb_video.squeeze(0),
            disparity=disparity_video.squeeze(0),
            raymap=raymap.squeeze(0),
        )


class SMPLPoseEmbeddingPerFrame(nn.Module):
    """Embedding layer for SMPL pose parameters with per-frame processing for AdaLN conditioning"""
    
    def __init__(self, pose_dim=63, embed_dim=512, hidden_dim=1024, sample_stride=4):
        super().__init__()
        self.pose_dim = pose_dim
        self.embed_dim = embed_dim
        self.sample_stride = sample_stride
        
        # Calculate number of sampled frames
        self.num_sampled_frames = (49 + sample_stride - 1) // sample_stride  # 13 frames
        
        # Shared MLP to process each frame's pose parameters independently
        self.pose_projection = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Zero initialization for AdaLN-zero (start with no SMPL conditioning effect)
        for layer in self.pose_projection:
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        # Initialize layer norm to identity (no effect initially)
        nn.init.ones_(self.layer_norm.weight)
        nn.init.zeros_(self.layer_norm.bias)
        
    def forward(self, pose_params: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pose_params: SMPL pose parameters of shape (batch_size, num_frames, pose_dim)
        Returns:
            pose_embedding: Embedded pose features of shape (batch_size, num_sampled_frames, embed_dim)
        """
        batch_size, num_frames, pose_dim = pose_params.shape
        
        # Sample frames with stride (e.g., frames 0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48)
        sampled_indices = torch.arange(0, min(num_frames, 49), self.sample_stride, device=pose_params.device)
        sampled_pose_params = pose_params[:, sampled_indices, :]  # (batch_size, 13, pose_dim)
        
        # Process each frame's pose parameters through shared MLP
        # Reshape to process all frames at once: (batch_size * 13, pose_dim)
        batch_size_sampled, num_sampled, pose_dim = sampled_pose_params.shape
        pose_reshaped = sampled_pose_params.reshape(-1, pose_dim)  # (batch_size * 13, pose_dim)
        
        # Project each frame's pose parameters to embedding space
        pose_embedding = self.pose_projection(pose_reshaped)  # (batch_size * 13, embed_dim)
        
        # Apply layer normalization
        pose_embedding = self.layer_norm(pose_embedding)
        
        # Reshape back to (batch_size, num_sampled_frames, embed_dim)
        pose_embedding = pose_embedding.reshape(batch_size_sampled, num_sampled, self.embed_dim)
        
        return pose_embedding


class SMPLConditionedCogVideoXLayerNormZeroPerFrame(CogVideoXLayerNormZero):
    """Extended CogVideoXLayerNormZero with per-frame SMPL pose conditioning via AdaLN-zero"""
    
    def __init__(
        self,
        conditioning_dim: int,
        embedding_dim: int,
        smpl_embed_dim: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        bias: bool = True,
    ) -> None:
        super().__init__(conditioning_dim, embedding_dim, elementwise_affine, eps, bias)
        
        # Additional SMPL conditioning layers (zero-initialized for AdaLN-zero)
        self.smpl_linear = nn.Linear(smpl_embed_dim, 6 * embedding_dim, bias=bias)
        
        # Zero initialization (AdaLN-zero key feature)
        nn.init.zeros_(self.smpl_linear.weight)
        nn.init.zeros_(self.smpl_linear.bias)
        
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        encoder_hidden_states: torch.Tensor, 
        temb: torch.Tensor,
        smpl_emb: Optional[torch.Tensor] = None,
        smpl_emb_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: Input hidden states of shape (batch_size, seq_len, embed_dim)
            encoder_hidden_states: Input encoder hidden states  
            temb: Time embedding (original CogVideoX conditioning) of shape (batch_size, time_embed_dim)
            smpl_emb: SMPL pose embedding of shape (batch_size, num_sampled_frames, smpl_embed_dim)
            smpl_emb_mask: Mask to zero out SMPL conditioning for certain samples
        Returns:
            Tuple of (norm_hidden_states, norm_encoder_hidden_states, gate, enc_gate)
        """
        # Original time conditioning
        time_shift, time_scale, time_gate, time_enc_shift, time_enc_scale, time_enc_gate = \
            self.linear(self.silu(temb)).chunk(6, dim=1)  # Each: (batch_size, embed_dim)
        
        # SMPL conditioning (starts at zero due to initialization)
        if smpl_emb is not None:
            batch_size, num_sampled_frames, smpl_embed_dim = smpl_emb.shape
            
            # Process each frame's SMPL embedding through shared linear layer
            # Reshape to process all frames at once: (batch_size * num_sampled_frames, smpl_embed_dim)
            smpl_emb_reshaped = smpl_emb.reshape(-1, smpl_embed_dim)
            
            # Apply linear transformation to each frame
            smpl_linear_output = self.smpl_linear(smpl_emb_reshaped)  # (batch_size * num_sampled_frames, 6 * embed_dim)
            
            # Reshape back and chunk: (batch_size, num_sampled_frames, embed_dim) for each component
            smpl_shift, smpl_scale, smpl_gate, smpl_enc_shift, smpl_enc_scale, smpl_enc_gate = \
                smpl_linear_output.reshape(batch_size, num_sampled_frames, -1).chunk(6, dim=2)
            
            # Unsqueeze time conditioning to match per-frame SMPL conditioning
            # time_*: (batch_size, embed_dim) -> (batch_size, 1, embed_dim)
            time_shift = time_shift[:, None, :]  # (batch_size, 1, embed_dim)
            time_scale = time_scale[:, None, :]  # (batch_size, 1, embed_dim)
            time_gate = time_gate[:, None, :]    # (batch_size, 1, embed_dim)
            time_enc_shift = time_enc_shift[:, None, :]  # (batch_size, 1, embed_dim)
            time_enc_scale = time_enc_scale[:, None, :]  # (batch_size, 1, embed_dim)
            time_enc_gate = time_enc_gate[:, None, :]    # (batch_size, 1, embed_dim)
            
            # Apply mask to zero out SMPL conditioning for masked samples
            if smpl_emb_mask is not None:
                # Expand mask to match the shape of SMPL parameters
                # Use the same dtype as smpl_emb for consistency
                mask_expanded = smpl_emb_mask.to(dtype=smpl_emb.dtype)[:, None, None]  # (batch_size, 1, 1)
                smpl_shift = smpl_shift * mask_expanded
                smpl_scale = smpl_scale * mask_expanded
                smpl_gate = smpl_gate * mask_expanded
                smpl_enc_shift = smpl_enc_shift * mask_expanded
                smpl_enc_scale = smpl_enc_scale * mask_expanded
                smpl_enc_gate = smpl_enc_gate * mask_expanded
            
            # Combine time + SMPL conditioning (per-frame)
            # smpl_*: (batch_size, num_sampled_frames, embed_dim)
            # time_*: (batch_size, 1, embed_dim) -> broadcast to (batch_size, num_sampled_frames, embed_dim)
            shift = time_shift + smpl_shift  # (batch_size, num_sampled_frames, embed_dim)
            scale = time_scale + smpl_scale  # (batch_size, num_sampled_frames, embed_dim)
            gate = time_gate + smpl_gate     # (batch_size, num_sampled_frames, embed_dim)
            enc_shift = time_enc_shift + smpl_enc_shift  # (batch_size, num_sampled_frames, embed_dim)
            enc_scale = time_enc_scale + smpl_enc_scale  # (batch_size, num_sampled_frames, embed_dim)
            enc_gate = time_enc_gate + smpl_enc_gate     # (batch_size, num_sampled_frames, embed_dim)
        else:
            # Fall back to original behavior (no SMPL conditioning)
            shift, scale, gate = time_shift, time_scale, time_gate
            enc_shift, enc_scale, enc_gate = time_enc_shift, time_enc_scale, time_enc_gate
        
        # Apply modulation (same as original CogVideoXLayerNormZero)
        # Note: shift, scale, gate now have shape (batch_size, num_sampled_frames, embed_dim) or (batch_size, embed_dim)
        # We need to handle both cases in the modulation
        if smpl_emb is not None:
            # Per-frame conditioning: average across frames for final modulation
            shift = shift.mean(dim=1)  # (batch_size, embed_dim)
            scale = scale.mean(dim=1)  # (batch_size, embed_dim)
            gate = gate.mean(dim=1)    # (batch_size, embed_dim)
            enc_shift = enc_shift.mean(dim=1)  # (batch_size, embed_dim)
            enc_scale = enc_scale.mean(dim=1)  # (batch_size, embed_dim)
            enc_gate = enc_gate.mean(dim=1)    # (batch_size, embed_dim)
        
        hidden_states = self.norm(hidden_states) * (1 + scale)[:, None, :] + shift[:, None, :]
        encoder_hidden_states = self.norm(encoder_hidden_states) * (1 + enc_scale)[:, None, :] + enc_shift[:, None, :]
        
        return hidden_states, encoder_hidden_states, gate[:, None, :], enc_gate[:, None, :]


class SMPLConditionedCogVideoXBlockPerFrame(nn.Module):
    """CogVideoXBlock with per-frame SMPL pose conditioning via extended LayerNormZero"""
    
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        time_embed_dim: int,
        smpl_embed_dim: int,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = False,
        qk_norm: bool = True,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = True,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        super().__init__()

        # 1. Self Attention with per-frame SMPL-conditioned LayerNormZero
        self.norm1 = SMPLConditionedCogVideoXLayerNormZeroPerFrame(
            time_embed_dim, dim, smpl_embed_dim, norm_elementwise_affine, norm_eps, bias=True
        )

        self.attn1 = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=attention_bias,
            out_bias=attention_out_bias,
            processor=CogVideoXAttnProcessor2_0(),
        )

        # 2. Feed Forward with per-frame SMPL-conditioned LayerNormZero
        self.norm2 = SMPLConditionedCogVideoXLayerNormZeroPerFrame(
            time_embed_dim, dim, smpl_embed_dim, norm_elementwise_affine, norm_eps, bias=True
        )

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        smpl_emb: Optional[torch.Tensor] = None,
        smpl_emb_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        text_seq_length = encoder_hidden_states.size(1)
        attention_kwargs = attention_kwargs or {}

        # 1. Self Attention with SMPL conditioning
        norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = self.norm1(
            hidden_states, encoder_hidden_states, temb, smpl_emb, smpl_emb_mask
        )

        # Attention
        attn_hidden_states, attn_encoder_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **attention_kwargs,
        )

        hidden_states = hidden_states + gate_msa * attn_hidden_states
        encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states

        # 2. Feed Forward with SMPL conditioning
        norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = self.norm2(
            hidden_states, encoder_hidden_states, temb, smpl_emb, smpl_emb_mask
        )

        # Feed-forward
        norm_combined = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
        ff_output = self.ff(norm_combined)

        hidden_states = hidden_states + gate_ff * ff_output[:, text_seq_length:]
        encoder_hidden_states = encoder_hidden_states + enc_gate_ff * ff_output[:, :text_seq_length]

        return hidden_states, encoder_hidden_states


class SMPLConditionedTransformer3DAdaLNZeroPerFrame(CogVideoXTransformer3DModel):
    """Variant of CogVideoXTransformer3DModel with per-frame SMPL pose conditioning via AdaLN-zero
    
    This model conditions on SMPL pose parameters by:
    1. Sampling frames with stride 4 (e.g., frames 0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48)
    2. Processing each sampled frame's pose parameters through a shared MLP
    3. Encoding them into per-frame embeddings for AdaLN-zero conditioning
    4. Applying the conditioning through extended LayerNormZero layers
    """
    
    _skip_layerwise_casting_patterns = ["patch_embed", "norm"]
    _supports_gradient_checkpointing = True
    _no_split_modules = ["CogVideoXBlock", "CogVideoXPatchEmbed"]

    @register_to_config
    def __init__(self, *args, **kwargs):
        # Extract SMPL-specific parameters
        self.smpl_pose_dim = kwargs.pop('smpl_pose_dim', 63)
        self.smpl_embed_dim = kwargs.pop('smpl_embed_dim', 512)
        
        super().__init__(*args, **kwargs)
        
        # Initialize per-frame SMPL pose embedding
        self.smpl_pose_embedding = SMPLPoseEmbeddingPerFrame(
            pose_dim=self.smpl_pose_dim,
            embed_dim=self.smpl_embed_dim,
            sample_stride=4  # Sample every 4th frame for AdaLN conditioning
        )
        
        # Use the original block replacement approach
        self._replace_blocks_with_smpl_conditioned()
        
        # Ensure all SMPL components have the correct dtype AFTER creating all components
        self._ensure_smpl_dtype()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        # Extract SMPL-specific parameters before calling parent
        smpl_pose_dim = kwargs.pop('smpl_pose_dim', 63)
        smpl_embed_dim = kwargs.pop('smpl_embed_dim', 512)
        sample_stride = kwargs.pop('sample_stride', 4)

        # Call parent from_pretrained to get the base model with proper Hugging Face loading
        base_model = CogVideoXTransformer3DModel.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        
        # Create a new instance of our class
        model = cls.__new__(cls)
        
        # Copy all attributes from the base model
        for attr_name, attr_value in base_model.__dict__.items():
            setattr(model, attr_name, attr_value)
        
        # Initialize per-frame SMPL pose embedding
        model.smpl_pose_embedding = SMPLPoseEmbeddingPerFrame(
            pose_dim=smpl_pose_dim,
            embed_dim=smpl_embed_dim,
            sample_stride=sample_stride  # Sample every 4th frame for AdaLN conditioning
        )
        
        # Replace transformer blocks with per-frame SMPL-conditioned versions
        smpl_conditioned_blocks = []
        for block in model.transformer_blocks:
            smpl_block = SMPLConditionedCogVideoXBlockPerFrame(
                dim=block.norm1.norm.normalized_shape[0],  # Get dimension from existing norm
                num_attention_heads=model.config.num_attention_heads,
                attention_head_dim=model.config.attention_head_dim,
                time_embed_dim=model.config.time_embed_dim,
                smpl_embed_dim=smpl_embed_dim,
                dropout=model.config.dropout,
                activation_fn=model.config.activation_fn,
                attention_bias=model.config.attention_bias,
                norm_elementwise_affine=block.norm1.norm.elementwise_affine,
                norm_eps=block.norm1.norm.eps,
            )
            
            # Copy pretrained weights from original block to SMPL block for overlapping components
            # Copy attention weights
            smpl_block.attn1.load_state_dict(block.attn1.state_dict(), strict=False)
            
            # Copy feed-forward weights
            smpl_block.ff.load_state_dict(block.ff.state_dict(), strict=False)
            
            # Copy original LayerNorm weights to the base LayerNorm in SMPL-conditioned LayerNorm
            # Note: We only copy the base norm weights, not the SMPL-specific ones
            smpl_block.norm1.norm.load_state_dict(block.norm1.norm.state_dict(), strict=False)
            smpl_block.norm2.norm.load_state_dict(block.norm2.norm.state_dict(), strict=False)
            
            # Copy the original linear layers (time conditioning) to SMPL blocks
            smpl_block.norm1.linear.load_state_dict(block.norm1.linear.state_dict(), strict=False)
            smpl_block.norm2.linear.load_state_dict(block.norm2.linear.state_dict(), strict=False)
            
            smpl_conditioned_blocks.append(smpl_block)
        
        model.transformer_blocks = nn.ModuleList(smpl_conditioned_blocks)
        
        return model
    
    @classmethod
    def from_aether_pretrained(cls, aether_model_path, *args, **kwargs):
        """Load original Aether weights and convert to SMPL-conditioned (initialization case)"""
        # Extract SMPL-specific parameters
        smpl_pose_dim = kwargs.pop('smpl_pose_dim', 63)
        smpl_embed_dim = kwargs.pop('smpl_embed_dim', 512)
        
        # Load the original Aether model using the correct class
        base_model = CogVideoXTransformer3DModel.from_pretrained(aether_model_path, *args, **kwargs)
        
        # Create our SMPL-conditioned model instance with proper initialization
        # Pass the config and SMPL parameters to __init__
        config_dict = dict(base_model.config)
        model = cls(
            **config_dict,
            smpl_pose_dim=smpl_pose_dim,
            smpl_embed_dim=smpl_embed_dim
        )
        
        # Copy all pretrained weights from the base model
        # This will copy weights for all overlapping layers (patch_embed, transformer_blocks, etc.)
        model.load_state_dict(base_model.state_dict(), strict=False)
        
        # Ensure the model has the same dtype as the base model
        base_dtype = next(base_model.parameters()).dtype
        model = model.to(dtype=base_dtype)
        
        return model
    
    def _replace_blocks_with_smpl_conditioned(self):
        """Original block replacement approach (kept for backward compatibility)"""
        # Replace transformer blocks with per-frame SMPL-conditioned versions
        smpl_conditioned_blocks = []
        for block in self.transformer_blocks:
            smpl_block = SMPLConditionedCogVideoXBlockPerFrame(
                dim=block.norm1.norm.normalized_shape[0],  # Get dimension from existing norm
                num_attention_heads=self.config.num_attention_heads,
                attention_head_dim=self.config.attention_head_dim,
                time_embed_dim=self.config.time_embed_dim,
                smpl_embed_dim=self.smpl_embed_dim,
                dropout=self.config.dropout,
                activation_fn=self.config.activation_fn,
                attention_bias=self.config.attention_bias,
                norm_elementwise_affine=block.norm1.norm.elementwise_affine,
                norm_eps=block.norm1.norm.eps,
            )
            
            # Ensure SMPL block has the same dtype as the base model
            target_dtype = next(block.parameters()).dtype
            smpl_block = smpl_block.to(dtype=target_dtype)
            
            # Copy pretrained weights from original block to SMPL block for overlapping components
            # Copy attention weights
            smpl_block.attn1.load_state_dict(block.attn1.state_dict(), strict=False)
            
            # Copy feed-forward weights
            smpl_block.ff.load_state_dict(block.ff.state_dict(), strict=False)
            
            # Copy original LayerNorm weights to the base LayerNorm in SMPL-conditioned LayerNorm
            # Note: We only copy the base norm weights, not the SMPL-specific ones
            smpl_block.norm1.norm.load_state_dict(block.norm1.norm.state_dict(), strict=False)
            smpl_block.norm2.norm.load_state_dict(block.norm2.norm.state_dict(), strict=False)
            
            # Copy the original linear layers (time conditioning) to SMPL blocks
            smpl_block.norm1.linear.load_state_dict(block.norm1.linear.state_dict(), strict=False)
            smpl_block.norm2.linear.load_state_dict(block.norm2.linear.state_dict(), strict=False)
            
            smpl_conditioned_blocks.append(smpl_block)
        
        self.transformer_blocks = nn.ModuleList(smpl_conditioned_blocks)
    
    def _ensure_smpl_dtype(self):
        """Ensure all SMPL components have the correct dtype"""
        if hasattr(self, 'smpl_pose_embedding'):
            # Get the dtype from the base model
            base_dtype = next(self.parameters()).dtype
            self.smpl_pose_embedding = self.smpl_pose_embedding.to(dtype=base_dtype)
    
    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        timestep=None,
        ofs=None,
        image_rotary_emb=None,
        attention_kwargs=None,
        pose_params=None,  # New parameter for SMPL pose
        pose_params_mask=None,  # New parameter to track None pose_params
        return_dict=True,
    ):
        # Process pose parameters if provided
        smpl_emb = None
        if pose_params is not None:
            # Per-frame SMPL pose embedding returns (batch_size, num_sampled_frames, embed_dim)
            smpl_emb = self.smpl_pose_embedding(pose_params)  # (batch_size, num_sampled_frames, embed_dim)
            if pose_params_mask is not None:
                smpl_emb_mask = pose_params_mask

        # Call parent forward method but pass smpl_emb to blocks
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        batch_size, num_frames, channels, height, width = hidden_states.shape

        # 1. Time embedding (preserve original functionality)
        timesteps = timestep
        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb)

        if self.ofs_embedding is not None:
            ofs_emb = self.ofs_proj(ofs)
            ofs_emb = ofs_emb.to(dtype=hidden_states.dtype)
            ofs_emb = self.ofs_embedding(ofs_emb)
            emb = emb + ofs_emb

        # 2. Patch embedding
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)

        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]

        # 3. Transformer blocks with per-frame SMPL conditioning
        for i, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states, encoder_hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    emb,
                    smpl_emb,  # Pass per-frame SMPL embedding
                    smpl_emb_mask,  # Pass SMPL embedding mask
                    image_rotary_emb,
                    attention_kwargs,
                )
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=emb,
                    smpl_emb=smpl_emb,  # Pass per-frame SMPL embedding
                    smpl_emb_mask=smpl_emb_mask,  # Pass SMPL embedding mask
                    image_rotary_emb=image_rotary_emb,
                    attention_kwargs=attention_kwargs,
                )

        hidden_states = self.norm_final(hidden_states)

        # 4. Final block
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)

        # 5. Unpatchify
        p = self.config.patch_size
        p_t = self.config.patch_size_t

        if p_t is None:
            output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
            output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)
        else:
            output = hidden_states.reshape(
                batch_size, (num_frames + p_t - 1) // p_t, height // p, width // p, -1, p_t, p, p
            )
            output = output.permute(0, 1, 5, 4, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(1, 2)

        if not return_dict:
            return (output,)
        from diffusers.models.modeling_outputs import Transformer2DModelOutput
        return Transformer2DModelOutput(sample=output)


class AetherV1SMPLAdaLNZeroPerFramePipelineCogVideoX(AetherV1PipelineCogVideoX):
    """Variant of AetherV1PipelineCogVideoX with per-frame SMPL pose conditioning via extended LayerNormZero"""
    
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: T5EncoderModel,
        vae: AutoencoderKLCogVideoX,
        scheduler: CogVideoXDPMScheduler,
        transformer: SMPLConditionedTransformer3DAdaLNZeroPerFrame,  # Use our per-frame custom transformer
    ):
        super().__init__(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            scheduler=scheduler,
            transformer=transformer,
        )
        
        # Override the transformer call to handle pose_params
        self._original_transformer_call = self.transformer.forward
    
    def check_inputs(
        self,
        task,
        image,
        video,
        goal,
        raymap,
        pose_params,  # New parameter
        pose_params_mask,  # New parameter to track None pose_params
        height,
        width,
        num_frames,
        fps,
    ):
        # Call parent check_inputs
        super().check_inputs(
            task=task,
            image=image,
            video=video,
            goal=goal,
            raymap=raymap,
            height=height,
            width=width,
            num_frames=num_frames,
            fps=fps,
        )
        
        # Additional validation for SMPL pose parameters
        if pose_params is not None:
            if not isinstance(pose_params, (torch.Tensor, np.ndarray)):
                raise ValueError(f"pose_params must be a torch.Tensor or np.ndarray, got {type(pose_params)}")
            
            if isinstance(pose_params, np.ndarray):
                pose_params = torch.from_numpy(pose_params)
            
            # Validate pose_params shape: should be (batch_size, num_frames, 63)
            if pose_params.dim() != 3:
                raise ValueError(f"pose_params must have 3 dimensions (batch_size, num_frames, 63), got {pose_params.shape}")
            
            if pose_params.shape[-1] != 63:
                raise ValueError(f"pose_params last dimension must be 63 (SMPL pose parameters), got {pose_params.shape[-1]}")
        
        # Validate pose_params_mask if provided
        if pose_params_mask is not None:
            if not isinstance(pose_params_mask, (torch.Tensor, np.ndarray)):
                raise ValueError(f"pose_params_mask must be a torch.Tensor or np.ndarray, got {type(pose_params_mask)}")
            
            if isinstance(pose_params_mask, np.ndarray):
                pose_params_mask = torch.from_numpy(pose_params_mask)
            
            # Validate pose_params_mask shape: should be (batch_size,)
            if pose_params_mask.dim() != 1:
                raise ValueError(f"pose_params_mask must have 1 dimension (batch_size,), got {pose_params_mask.shape}")
            
            # Validate that pose_params_mask is boolean
            if pose_params_mask.dtype != torch.bool:
                raise ValueError(f"pose_params_mask must be boolean tensor, got {pose_params_mask.dtype}")
            
            # If pose_params is provided, validate that mask length matches batch size
            if pose_params is not None and pose_params_mask.shape[0] != pose_params.shape[0]:
                raise ValueError(f"pose_params_mask batch size ({pose_params_mask.shape[0]}) must match pose_params batch size ({pose_params.shape[0]})")
    
    def preprocess_inputs(
        self,
        image,
        goal,
        video,
        raymap,
        pose_params,  # New parameter
        pose_params_mask,  # New parameter to track None pose_params
        height,
        width,
        num_frames,
    ):
        # Call parent preprocess_inputs
        image, goal, video, raymap = super().preprocess_inputs(
            image=image,
            goal=goal,
            video=video,
            raymap=raymap,
            height=height,
            width=width,
            num_frames=num_frames,
        )
        
        # Preprocess pose_params
        if pose_params is not None:
            if isinstance(pose_params, np.ndarray):
                pose_params = torch.from_numpy(pose_params)
            
            # Ensure pose_params is on the correct device and dtype
            pose_params = pose_params.to(device=self._execution_device, dtype=self.transformer.dtype)
        
        if pose_params_mask is not None:
            if isinstance(pose_params_mask, np.ndarray):
                pose_params_mask = torch.from_numpy(pose_params_mask)
            
            # Ensure pose_params_mask is on the correct device and dtype
            pose_params_mask = pose_params_mask.to(device=self._execution_device, dtype=self.transformer.dtype)
        return image, goal, video, raymap, pose_params, pose_params_mask

    @torch.no_grad()
    def __call__(
        self,
        task: Optional[str] = None,
        image: Optional[PipelineImageInput] = None,
        video: Optional[PipelineImageInput] = None,
        goal: Optional[PipelineImageInput] = None,
        raymap: Optional[Union[torch.Tensor, np.ndarray]] = None,
        pose_params: Optional[Union[torch.Tensor, np.ndarray]] = None,  # New parameter
        pose_params_mask=None,  # New parameter to track None pose_params
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        timesteps: Optional[List[int]] = None,
        guidance_scale: Optional[float] = None,
        use_dynamic_cfg: bool = False,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict] = None,
        fps: Optional[int] = None,
    ) -> Union[AetherV1SMPLAdaLNZeroPipelineOutput, Tuple]:
        
        if task is None:
            if video is not None:
                task = "reconstruction"
            elif goal is not None:
                task = "planning"
            else:
                task = "prediction"

        height = (
            height
            or self.transformer.config.sample_height * self.vae_scale_factor_spatial
        )
        width = (
            width
            or self.transformer.config.sample_width * self.vae_scale_factor_spatial
        )
        num_frames = num_frames or self.transformer.config.sample_frames
        fps = fps or self._base_fps

        num_videos_per_prompt = 1

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            task=task,
            image=image,
            video=video,
            goal=goal,
            raymap=raymap,
            pose_params=pose_params,  # Add pose_params
            pose_params_mask=pose_params_mask,  # Add pose_params_mask
            height=height,
            width=width,
            num_frames=num_frames,
            fps=fps,
        )

        # 2. Preprocess inputs
        image, goal, video, raymap, pose_params, pose_params_mask = self.preprocess_inputs(
            image=image,
            goal=goal,
            video=video,
            raymap=raymap,
            pose_params=pose_params,  # Add pose_params
            pose_params_mask=pose_params_mask,  # Add pose_params_mask
            height=height,
            width=width,
            num_frames=num_frames,
        )
        self._guidance_scale = guidance_scale
        self._current_timestep = None
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        batch_size = 1

        device = self._execution_device

        # 3. Encode input prompt
        prompt_embeds = self.empty_prompt_embeds.to(device)

        num_inference_steps = (
            num_inference_steps or self._default_num_inference_steps[task]
        )
        guidance_scale = guidance_scale or self._default_guidance_scale[task]
        use_dynamic_cfg = use_dynamic_cfg or self._default_use_dynamic_cfg[task]

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps
        )
        self._num_timesteps = len(timesteps)

        # 5. Prepare latents
        latents, condition_latents = self.prepare_latents(
            image,
            goal,
            video,
            raymap,
            batch_size * num_videos_per_prompt,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Create rotary embeds if required
        image_rotary_emb = (
            self._prepare_rotary_positional_embeddings(
                height, width, latents.size(1), device, fps=fps
            )
            if self.transformer.config.use_rotary_positional_embeddings
            else None
        )

        # 8. Create ofs embeds if required
        ofs_emb = (
            None
            if self.transformer.config.ofs_embed_dim is None
            else latents.new_full((1,), fill_value=2.0)
        )

        # 9. Denoising loop
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            # for DPM-solver++
            old_pred_original_sample = None
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                if do_classifier_free_guidance:
                    if task == "planning":
                        assert goal is not None
                        uncond = condition_latents.clone()
                        uncond[:, :, : self.vae.config.latent_channels] = 0
                        latent_condition = torch.cat([uncond, condition_latents])
                    elif task == "prediction":
                        uncond = condition_latents.clone()
                        uncond[:, :1, : self.vae.config.latent_channels] = 0
                        latent_condition = torch.cat([uncond, condition_latents])
                    else:
                        raise ValueError(
                            f"Task {task} not supported for classifier-free guidance."
                        )

                else:
                    latent_condition = condition_latents

                latent_model_input = torch.cat(
                    [latent_model_input, latent_condition], dim=2
                )

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                # Handle pose_params for classifier-free guidance
                if do_classifier_free_guidance and pose_params is not None:
                    # For classifier-free guidance, we need to duplicate pose_params
                    # The first half will be for unconditional generation (no SMPL effect)
                    # The second half will be for conditional generation (with SMPL effect)
                    pose_params_input = torch.cat([pose_params] * 2, dim=0)
                    
                    # Also duplicate pose_params_mask if provided
                    if pose_params_mask is not None:
                        pose_params_mask_input = torch.cat([pose_params_mask] * 2, dim=0)
                    else:
                        pose_params_mask_input = None
                else:
                    pose_params_input = pose_params
                    pose_params_mask_input = pose_params_mask

                # Predict the noise residual (with or without SMPL pose conditioning)
                transformer_kwargs = {
                    "hidden_states": latent_model_input,
                    "encoder_hidden_states": prompt_embeds.repeat(
                        latent_model_input.shape[0], 1, 1
                    ),
                    "timestep": timestep,
                    "ofs": ofs_emb,
                    "image_rotary_emb": image_rotary_emb,
                    "attention_kwargs": attention_kwargs,
                    "return_dict": False,
                }
                
                # Add pose_params only if SMPL conditioning is enabled
                if pose_params_input is not None:
                    transformer_kwargs["pose_params"] = pose_params_input
                if pose_params_mask_input is not None:
                    transformer_kwargs["pose_params_mask"] = pose_params_mask_input
                
                noise_pred = self.transformer(**transformer_kwargs)[0]
                noise_pred = noise_pred.float()

                # perform guidance
                if use_dynamic_cfg:
                    self._guidance_scale = 1 + guidance_scale * (
                        (
                            1
                            - math.cos(
                                math.pi
                                * (
                                    (num_inference_steps - t.item())
                                    / num_inference_steps
                                )
                                ** 5.0
                            )
                        )
                        / 2
                    )

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                if not isinstance(self.scheduler, CogVideoXDPMScheduler):
                    latents = self.scheduler.step(
                        noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                    )[0]
                else:
                    latents, old_pred_original_sample = self.scheduler.step(
                        noise_pred,
                        old_pred_original_sample,
                        t,
                        timesteps[i - 1] if i > 0 else None,
                        latents,
                        **extra_step_kwargs,
                        return_dict=False,
                    )
                latents = latents.to(prompt_embeds.dtype)

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

        self._current_timestep = None

        rgb_latents = latents[:, :, : self.vae.config.latent_channels]
        disparity_latents = latents[
            :, :, self.vae.config.latent_channels : self.vae.config.latent_channels * 2
        ]
        camera_latents = latents[:, :, self.vae.config.latent_channels * 2 :]

        rgb_video = self.decode_latents(rgb_latents)
        rgb_video = self.video_processor.postprocess_video(
            video=rgb_video, output_type="np"
        )

        disparity_video = self.decode_latents(disparity_latents)
        disparity_video = disparity_video.mean(dim=1, keepdim=False)
        disparity_video = disparity_video * 0.5 + 0.5
        disparity_video = torch.square(disparity_video)
        disparity_video = disparity_video.float().cpu().numpy()

        raymap = (
            rearrange(camera_latents, "b t (n c) h w -> b (n t) c h w", n=4)[
                :, -rgb_video.shape[1] :, :, :
            ]
            .float()
            .cpu()
            .numpy()
        )

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (
                rgb_video,
                disparity_video,
                raymap,
            )

        return AetherV1SMPLAdaLNZeroPipelineOutput(
            rgb=rgb_video.squeeze(0),
            disparity=disparity_video.squeeze(0),
            raymap=raymap.squeeze(0),
        ) 