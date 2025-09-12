# Copyright 2024 The CogVideoX team, Tsinghua University & ZhipuAI and The HuggingFace Team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
import torch.nn as nn
from typing import Optional, Union, Tuple, Dict, Any
from diffusers.models import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import is_torch_version, USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers, logging
from diffusers.models.embeddings import CogVideoXPatchEmbed, TimestepEmbedding, Timesteps
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.attention_processor import AttentionProcessor
from diffusers.models.normalization import AdaLayerNorm, CogVideoXLayerNormZero

from diffusers import CogVideoXTransformer3DModel

logger = logging.get_logger(__name__)

# copied from TrajectoryCrafter/models/crosstransformer3d.py
class PerceiverCrossAttention(nn.Module):
    """

    Args:
        dim (int): Dimension of the input latent and output. Default is 3072.
        dim_head (int): Dimension of each attention head. Default is 128.
        heads (int): Number of attention heads. Default is 16.
        kv_dim (int): Dimension of the key/value input, allowing flexible cross-attention. Default is 2048.

    Attributes:
        scale (float): Scaling factor used in dot-product attention for numerical stability.
        norm1 (nn.LayerNorm): Layer normalization applied to the input image features.
        norm2 (nn.LayerNorm): Layer normalization applied to the latent features.
        to_q (nn.Linear): Linear layer for projecting the latent features into queries.
        to_kv (nn.Linear): Linear layer for projecting the input features into keys and values.
        to_out (nn.Linear): Linear layer for outputting the final result after attention.

    """

    def __init__(self, *, dim=3072, dim_head=128, heads=16, kv_dim=2048):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        # Layer normalization to stabilize training
        self.norm1 = nn.LayerNorm(dim if kv_dim is None else kv_dim)
        self.norm2 = nn.LayerNorm(dim)

        # Linear transformations to produce queries, keys, and values
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(
            dim if kv_dim is None else kv_dim, inner_dim * 2, bias=False
        )
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        """

        Args:
            x (torch.Tensor): Input image features with shape (batch_size, n1, D), where:
                - batch_size (b): Number of samples in the batch.
                - n1: Sequence length (e.g., number of patches or tokens).
                - D: Feature dimension.

            latents (torch.Tensor): Latent feature representations with shape (batch_size, n2, D), where:
                - n2: Number of latent elements.

        Returns:
            torch.Tensor: Attention-modulated features with shape (batch_size, n2, D).

        """
        # Apply layer normalization to the input image and latent features
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, seq_len, _ = latents.shape

        # Compute queries, keys, and values
        q = self.to_q(latents)
        k, v = self.to_kv(x).chunk(2, dim=-1)

        # Reshape tensors to split into attention heads
        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # Compute attention weights
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(
            -2, -1
        )  # More stable scaling than post-division
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)

        # Compute the output via weighted combination of values
        out = weight @ v

        # Reshape and permute to prepare for final linear transformation
        out = out.permute(0, 2, 1, 3).reshape(b, seq_len, -1)

        return self.to_out(out)

class CogVideoXFunTransformer3DModel(CogVideoXTransformer3DModel):
    """
    VideoX-Fun Transformer model for video-like data.
    
    This class extends the standard CogVideoXTransformer3DModel to support VideoX-Fun specific features:
    1. Handles `inpaint_latents` and `control_latents` inputs
    2. Supports single frame processing with padding
    3. VideoX-Fun specific normalization logic
    """
    
    def __init__(self, *args, **kwargs):
        # Accept VideoX-Fun specific parameters but don't use them
        self.add_noise_in_inpaint_model = kwargs.pop("add_noise_in_inpaint_model", False)
        super().__init__(*args, **kwargs)
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path=None,
        base_model_name_or_path="alibaba-pai/CogVideoX-Fun-V1.1-5b-InP",
        subfolder="transformer",
        **kwargs
    ):
        """
        Load a CogVideoXFunTransformer3DModel from a pretrained model.

        Args:
            pretrained_model_name_or_path (str): Fine-tuned checkpoint path
            base_model_name_or_path (str): Base CogVideoX-Fun model path
        """
        if pretrained_model_name_or_path is not None:
            # === Load fine-tuned checkpoint ===
            print(f"📥 Loading fine-tuned VideoX-Fun transformer: {pretrained_model_name_or_path}")

            # 1) Create child class structure first
            config_path = os.path.join(pretrained_model_name_or_path, subfolder, "config.json")
            if os.path.exists(config_path):
                import json
                with open(config_path, "r") as f:
                    config = json.load(f)
                model = cls(**config)
            else:
                # Use base config
                model = cls()

            # 2) Load checkpoint state_dict directly
            state_dict_path = os.path.join(pretrained_model_name_or_path, subfolder, "diffusion_pytorch_model.safetensors")
            if not os.path.exists(state_dict_path):
                # Try alternative paths
                state_dict_path = os.path.join(pretrained_model_name_or_path, subfolder, "pytorch_model.bin")
                if not os.path.exists(state_dict_path):
                    raise FileNotFoundError(f"No model file found in {pretrained_model_name_or_path}")
            
            if state_dict_path.endswith(".safetensors"):
                from safetensors.torch import load_file
                state_dict = load_file(state_dict_path)
            else:
                state_dict = torch.load(state_dict_path, map_location="cpu")

            # 3) Load with strict=False for flexibility
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"⚠️ Missing keys (new layers, need initialization during training): {missing}")
            if unexpected:
                print(f"⚠️ Unexpected keys (replaced layers, etc.): {unexpected}")

            return model

        else:
            # === Initialize from base model ===
            print(f"🔧 Initializing from base model: {base_model_name_or_path}")
            # First load the base CogVideoX transformer
            base_model = CogVideoXTransformer3DModel.from_pretrained(
                base_model_name_or_path,
                subfolder=subfolder,
                torch_dtype=kwargs.get("torch_dtype", torch.bfloat16),
                revision=kwargs.get("revision", None),
                variant=kwargs.get("variant", None),
            )
            
            # Then create CogVideoXFun transformer and copy state_dict
            fun_model = CogVideoXFunTransformer3DModel(**base_model.config)
            fun_model.load_state_dict(base_model.state_dict())
            
            return fun_model
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
        inpaint_latents: Optional[torch.Tensor] = None,
        control_latents: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_dict: bool = True,
        **kwargs
    ):
        """
        Forward pass with VideoX-Fun specific inputs.
        
        Args:
            hidden_states: Input tensor with shape [batch, frames, channels, height, width]
            encoder_hidden_states: Text embeddings
            timestep: Timestep tensor
            timestep_cond: Timestep condition tensor
            inpaint_latents: Inpainting latents (VideoX-Fun specific)
            control_latents: Control latents (VideoX-Fun specific)
            image_rotary_emb: Rotary embeddings
            return_dict: Whether to return a dict
            **kwargs: Additional arguments
        """
        batch_size, num_frames, channels, height, width = hidden_states.shape
        
        # Handle single frame case (VideoX-Fun specific)
        if num_frames == 1 and self.patch_size_t is not None:
            hidden_states = torch.cat([hidden_states, torch.zeros_like(hidden_states)], dim=1)
            if inpaint_latents is not None:
                inpaint_latents = torch.concat([inpaint_latents, torch.zeros_like(inpaint_latents)], dim=1)
            if control_latents is not None:
                control_latents = torch.concat([control_latents, torch.zeros_like(control_latents)], dim=1)
            local_num_frames = num_frames + 1
        else:
            local_num_frames = num_frames
        
        # Time embedding
        timesteps = timestep
        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)
        
        # Patch embedding with VideoX-Fun specific concatenation
        if inpaint_latents is not None:
            hidden_states = torch.concat([hidden_states, inpaint_latents], 2)
        if control_latents is not None:
            hidden_states = torch.concat([hidden_states, control_latents], 2)
        
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)
        
        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]
        
        # Transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                
                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    emb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=emb,
                    image_rotary_emb=image_rotary_emb,
                )
        
        # Final normalization (VideoX-Fun specific)
        if not self.config.use_rotary_positional_embeddings:
            # CogVideoX-2B
            hidden_states = self.norm_final(hidden_states)
        else:
            # CogVideoX-5B
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            hidden_states = self.norm_final(hidden_states)
            hidden_states = hidden_states[:, text_seq_length:]
        
        # Final block
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)
        
        # Unpatchify
        p = self.config.patch_size
        p_t = self.config.patch_size_t
        
        if p_t is None:
            output = hidden_states.reshape(batch_size, local_num_frames, height // p, width // p, -1, p, p)
            output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)
        else:
            output = hidden_states.reshape(
                batch_size, (local_num_frames + p_t - 1) // p_t, height // p, width // p, -1, p_t, p, p
            )
            output = output.permute(0, 1, 5, 4, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(1, 2)
        
        # Handle single frame case (VideoX-Fun specific)
        if num_frames == 1:
            output = output[:, :num_frames, :]
        
        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)


class CogVideoXFunTransformer3DModelWithConcat(CogVideoXFunTransformer3DModel):
    """
    VideoX-Fun Transformer with conditional input support via channel concatenation.
    
    This class extends CogVideoXFunTransformer3DModel to support conditional inputs
    by concatenating condition channels directly to the input channels.
    
    Key differences from CogVideoXFunTransformer3DModel:
    1. Supports `add_noise_in_inpaint_model` parameter (VideoX-Fun specific)
    2. Handles `inpaint_latents` and `control_latents` in forward pass
    3. Extends input channels to accommodate condition channels
    """
    
    def __init__(self, *args, condition_channels: int = 0, **kwargs):
        # Store original in_channels before modification
        original_in_channels = kwargs.get("in_channels", 16)
        self.original_in_channels = original_in_channels
        
        self.add_noise_in_inpaint_model = kwargs.pop("add_noise_in_inpaint_model", False)
        
        super().__init__(*args, **kwargs)
        
        # Setup conditional channels if specified
        if condition_channels > 0:
            self._setup_condition_channels(condition_channels)
            self.condition_channels = condition_channels
        else:
            self.condition_channels = 0
    
    def _setup_condition_channels(self, condition_channels: int, original_proj: Optional[nn.Conv2d] = None):
        """Extend the transformer to handle conditional input channels.
        
        If `original_proj` is provided, use its weights as the source (for base→finetune case).
        Otherwise fall back to self.patch_embed.proj (for reload case).
        """
        if original_proj is None:
            original_proj = self.patch_embed.proj

        original_in_channels = original_proj.in_channels
        new_in_channels = original_in_channels + condition_channels

        print(f"Extending transformer input channels for concat approach:")
        print(f"  Original channels: {original_in_channels}")
        print(f"  Condition channels: {condition_channels}")
        print(f"  Total channels: {new_in_channels}")

        new_proj = nn.Conv2d(
            in_channels=new_in_channels,
            out_channels=original_proj.out_channels,
            kernel_size=original_proj.kernel_size,
            stride=original_proj.stride,
            padding=original_proj.padding,
            bias=original_proj.bias is not None,
        )

        with torch.no_grad():
            # copy pretrained weights into the original channels
            new_proj.weight[:, :original_in_channels] = original_proj.weight
            # init condition channels to zeros
            new_proj.weight[:, original_in_channels:] = 0.0
            if original_proj.bias is not None:
                new_proj.bias.data = original_proj.bias.data

        self.patch_embed.proj = new_proj
        self.register_to_config(
            in_channels=new_in_channels,
            original_in_channels=original_in_channels,
            condition_channels=condition_channels,
        )
        self._conditioning_channels_added = True
        print(f"✅ Successfully extended transformer for {condition_channels} condition channels")
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path=None,
        base_model_name_or_path="alibaba-pai/CogVideoX-Fun-V1.1-5b-InP",
        subfolder="transformer",
        condition_channels: Optional[int] = None,
        **kwargs
    ):
        """
        Load a CogVideoXFunTransformer3DModelWithConcat from a pretrained model.

        Args:
            pretrained_model_name_or_path (str): Fine-tuned checkpoint path
            base_model_name_or_path (str): Base CogVideoX-Fun model path
            condition_channels (int): Number of extra condition channels to add
        """
        if pretrained_model_name_or_path is not None:
            # === Load fine-tuned checkpoint ===
            print(f"📥 Loading fine-tuned concat pose-conditioned transformer: {pretrained_model_name_or_path}")

            # 1) Create child class structure first
            config_path = os.path.join(pretrained_model_name_or_path, subfolder, "config.json")
            if os.path.exists(config_path):
                import json
                with open(config_path, "r") as f:
                    config = json.load(f)
                model = cls(**config)
            else:
                # Use base config
                if condition_channels is None:
                    condition_channels = 32  # default
                model = cls(condition_channels=condition_channels)

            # 2) Load checkpoint state_dict directly
            state_dict_path = os.path.join(pretrained_model_name_or_path, subfolder, "diffusion_pytorch_model.safetensors")
            if not os.path.exists(state_dict_path):
                # Try alternative paths
                state_dict_path = os.path.join(pretrained_model_name_or_path, subfolder, "pytorch_model.bin")
                if not os.path.exists(state_dict_path):
                    raise FileNotFoundError(f"No model file found in {pretrained_model_name_or_path}")
            
            if state_dict_path.endswith(".safetensors"):
                from safetensors.torch import load_file
                state_dict = load_file(state_dict_path)
            else:
                state_dict = torch.load(state_dict_path, map_location="cpu")

            # 3) Load with strict=False for flexibility
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"⚠️ Missing keys (new layers, need initialization during training): {missing}")
            if unexpected:
                print(f"⚠️ Unexpected keys (replaced layers, etc.): {unexpected}")

            return model

        else:
            # === Initialize from base model ===
            print(f"🔧 Initializing from base model: {base_model_name_or_path}")
            # First load the base CogVideoX transformer
            base_model = CogVideoXTransformer3DModel.from_pretrained(
                base_model_name_or_path,
                subfolder=subfolder,
                torch_dtype=kwargs.get("torch_dtype", torch.bfloat16),
                revision=kwargs.get("revision", None),
                variant=kwargs.get("variant", None),
            )
            
            # Then create CogVideoXFun transformer and copy state_dict
            fun_model = CogVideoXFunTransformer3DModel(**base_model.config)
            fun_model.load_state_dict(base_model.state_dict())
            base_model = fun_model

            # Determine condition_channels
            if condition_channels is None:
                condition_channels = getattr(base_model.config, "condition_channels", 32)

            # Create extended model
            model = cls(**base_model.config, condition_channels=0)

            # Load base weights into extended model
            missing, unexpected = model.load_state_dict(base_model.state_dict(), strict=False)
            if missing:
                print(f"⚠️ Missing keys (new condition layers): {missing}")
            if unexpected:
                print(f"⚠️ Unexpected keys (replaced existing layers): {unexpected}")

            # Expand input projection for condition channels
            if condition_channels > 0:
                print(f"🔗 Expanding projection layer for {condition_channels} condition channels")
                model._setup_condition_channels(
                    condition_channels,
                    original_proj=base_model.patch_embed.proj
                )

            return model
    
    def get_condition_info(self) -> Dict[str, Any]:
        """Get information about the conditional setup."""
        info = {
            "approach": "concat",
            "has_conditions": hasattr(self, 'condition_channels') and self.condition_channels > 0,
            "condition_channels": getattr(self, 'condition_channels', 0),
            "total_input_channels": self.patch_embed.proj.in_channels,
            "base_channels": getattr(self, 'original_in_channels', self.patch_embed.proj.in_channels),
        }
        return info
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
        inpaint_latents: Optional[torch.Tensor] = None,
        control_latents: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_dict: bool = True,
        **kwargs
    ):
        """
        Forward pass with VideoX-Fun specific inputs and conditional concatenation.
        
        Args:
            hidden_states: Input tensor with shape [batch, frames, channels, height, width]
            encoder_hidden_states: Text embeddings
            timestep: Timestep tensor
            timestep_cond: Timestep condition tensor
            inpaint_latents: Inpainting latents (VideoX-Fun specific)
            control_latents: Control latents (VideoX-Fun specific)
            image_rotary_emb: Rotary embeddings
            return_dict: Whether to return a dict
            **kwargs: Additional arguments
        """
        # Validate input dimensions if condition channels are expected
        if hasattr(self, 'condition_channels') and self.condition_channels > 0:
            expected_channels = self.patch_embed.proj.in_channels
            actual_channels = hidden_states.shape[2]
            
            if actual_channels != expected_channels:
                raise ValueError(
                    f"Expected {expected_channels} channels (base + condition), "
                    f"but got {actual_channels} channels in hidden_states"
                )
        
        # Use parent class forward method which handles VideoX-Fun specific logic
        return super().forward(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            timestep_cond=timestep_cond,
            inpaint_latents=inpaint_latents,
            control_latents=control_latents,
            image_rotary_emb=image_rotary_emb,
            return_dict=return_dict,
            **kwargs
        )

class CogVideoXPatchEmbedWithAdapter(CogVideoXPatchEmbed):
    """
    VideoX-Fun Patch Embed with conditional input support via adapter.
    
    This class extends CogVideoXPatchEmbed to support conditional inputs
    by using adapter.
    """
    
    def __init__(self, *args, **kwargs):
        # Extract condition_channels from kwargs before passing to parent
        condition_channels = kwargs.pop("condition_channels", None)
        super().__init__(*args, **kwargs)
        in_channels = kwargs.get("in_channels", 16)
        patch_size = kwargs.get("patch_size", 16)
        patch_size_t = kwargs.get("patch_size_t", None)
        embed_dim = kwargs.get("embed_dim", 16)
        bias = kwargs.get("bias", True)

        if condition_channels is None:
            condition_channels = in_channels

        # Setup adapter
        if patch_size_t is None:
            # CogVideoX 1.0 checkpoints
            self.cond_proj = nn.Conv2d(
                condition_channels, embed_dim, kernel_size=(patch_size, patch_size), stride=patch_size, bias=bias
            )
        else:
            # CogVideoX 1.5 checkpoints
            self.cond_proj = nn.Linear(condition_channels * patch_size * patch_size * patch_size_t, embed_dim)

    def forward(self, text_embeds: torch.Tensor, image_embeds: torch.Tensor, 
                ref_image_embeds: torch.Tensor = None, cond_embeds: torch.Tensor = None):
        r"""
        Args:
            text_embeds (`torch.Tensor`):
                Input text embeddings. Expected shape: (batch_size, seq_length, embedding_dim).
            image_embeds (`torch.Tensor`):
                Input image embeddings. Expected shape: (batch_size, num_frames, channels, height, width).
            ref_image_embeds (`torch.Tensor`, optional):
                Input reference image embeddings. Expected shape: (batch_size, num_frames, channels, height, width).
            cond_embeds (`torch.Tensor`, optional):
                Input condition embeddings. Expected shape: (batch_size, num_frames, channels, height, width).
        """
        text_embeds = self.text_proj(text_embeds)

        batch_size, num_frames, channels, height, width = image_embeds.shape
        
        if self.patch_size_t is None:
            # Process image and ref image together
            if ref_image_embeds is not None:
                combined_embeds = torch.cat([image_embeds, ref_image_embeds], dim=0)
            else:
                combined_embeds = image_embeds
                
            combined_embeds = combined_embeds.reshape(-1, channels, height, width)
            combined_embeds = self.proj(combined_embeds)
            
            # Split back into image and ref embeds
            if ref_image_embeds is not None:
                image_embeds, ref_image_embeds = torch.chunk(combined_embeds, 2, dim=0)
                image_embeds = image_embeds.view(batch_size, num_frames, *image_embeds.shape[1:])
                ref_image_embeds = ref_image_embeds.view(batch_size, num_frames, *ref_image_embeds.shape[1:])
            else:
                image_embeds = combined_embeds.view(batch_size, num_frames, *combined_embeds.shape[1:])
                
            image_embeds = image_embeds.flatten(3).transpose(2, 3)  # [batch, num_frames, height x width, channels]
            image_embeds = image_embeds.flatten(1, 2)  # [batch, num_frames x height x width, channels]
            
            if ref_image_embeds is not None:
                ref_image_embeds = ref_image_embeds.flatten(3).transpose(2, 3)
                ref_image_embeds = ref_image_embeds.flatten(1, 2)
        
            if cond_embeds is not None:
                cond_channels = cond_embeds.shape[2]
                cond_embeds = cond_embeds.reshape(-1, cond_channels, height, width)
                cond_embeds = self.cond_proj(cond_embeds)
                cond_embeds = cond_embeds.view(batch_size, num_frames, *cond_embeds.shape[1:])
                cond_embeds = cond_embeds.flatten(3).transpose(2, 3)  # [batch, num_frames, height x width, channels]
                cond_embeds = cond_embeds.flatten(1, 2)  # [batch, num_frames x height x width, channels]
                image_embeds = image_embeds + cond_embeds
        
        else:
            p = self.patch_size
            p_t = self.patch_size_t

            # Process image and ref image together
            if ref_image_embeds is not None:
                combined_embeds = torch.cat([image_embeds, ref_image_embeds], dim=0)
            else:
                combined_embeds = image_embeds
                
            combined_embeds = combined_embeds.permute(0, 1, 3, 4, 2)
            combined_embeds = combined_embeds.reshape(
                -1, num_frames // p_t, p_t, height // p, p, width // p, p, channels
            )
            combined_embeds = combined_embeds.permute(0, 1, 3, 5, 7, 2, 4, 6).flatten(4, 7).flatten(1, 3)
            combined_embeds = self.proj(combined_embeds)
            
            # Split back into image and ref embeds
            if ref_image_embeds is not None:
                image_embeds, ref_image_embeds = torch.chunk(combined_embeds, 2, dim=0)
            else:
                image_embeds = combined_embeds

            if cond_embeds is not None:
                cond_channels = cond_embeds.shape[2]
                cond_embeds = cond_embeds.permute(0, 1, 3, 4, 2)
                cond_embeds = cond_embeds.reshape(
                    batch_size, num_frames // p_t, p_t, height // p, p, width // p, p, cond_channels
                )
                cond_embeds = cond_embeds.permute(0, 1, 3, 5, 7, 2, 4, 6).flatten(4, 7).flatten(1, 3)
                cond_embeds = self.cond_proj(cond_embeds)
                image_embeds = image_embeds + cond_embeds

        embeds = torch.cat(
            [text_embeds, image_embeds], dim=1
        ).contiguous()  # [batch, seq_length + num_frames x height x width, channels]

        if self.use_positional_embeddings or self.use_learned_positional_embeddings:
            if self.use_learned_positional_embeddings and (self.sample_width != width or self.sample_height != height):
                raise ValueError(
                    "It is currently not possible to generate videos at a different resolution that the defaults. This should only be the case with 'THUDM/CogVideoX-5b-I2V'."
                    "If you think this is incorrect, please open an issue at https://github.com/huggingface/diffusers/issues."
                )

            pre_time_compression_frames = (num_frames - 1) * self.temporal_compression_ratio + 1

            if (
                self.sample_height != height
                or self.sample_width != width
                or self.sample_frames != pre_time_compression_frames
            ):
                pos_embedding = self._get_positional_embeddings(
                    height, width, pre_time_compression_frames, device=embeds.device
                )
            else:
                pos_embedding = self.pos_embedding

            pos_embedding = pos_embedding.to(dtype=embeds.dtype)
            embeds = embeds + pos_embedding

        return embeds, ref_image_embeds
            
class CogVideoXFunTransformer3DModelWithAdapter(CogVideoXFunTransformer3DModel):
    """
    VideoX-Fun Transformer with conditional input support via adapter.
    
    This class extends CogVideoXFunTransformer3DModel to support conditional inputs
    by using CogVideoXPatchEmbedWithAdapter instead of extending input channels.
    
    Key differences from CogVideoXFunTransformer3DModel:
    1. Supports `add_noise_in_inpaint_model` parameter (VideoX-Fun specific)
    2. Handles `inpaint_latents` and `control_latents` in forward pass
    3. Uses CogVideoXPatchEmbedWithAdapter for conditional processing
    """
    
    def __init__(self, *args, condition_channels: int = 0, **kwargs):
        self.add_noise_in_inpaint_model = kwargs.pop("add_noise_in_inpaint_model", False)
        
        super().__init__(*args, **kwargs)
        
        # Replace patch_embed with adapter version if condition_channels > 0
        if condition_channels > 0:
            self._setup_adapter_patch_embed(condition_channels)
            self.condition_channels = condition_channels
        else:
            self.condition_channels = 0
    
    def _setup_adapter_patch_embed(self, condition_channels: int):
        """Replace patch_embed with CogVideoXPatchEmbedWithAdapter.
        
        This creates a new adapter-based patch embed that processes conditions
        separately from the main image embeddings.
        """
        print(f"Setting up adapter-based patch embed for {condition_channels} condition channels")
        
        # Get current patch_embed config
        current_patch_embed = self.patch_embed
        
        # Create new adapter patch embed with same config
        adapter_patch_embed = CogVideoXPatchEmbedWithAdapter(
            condition_channels=condition_channels,
            in_channels=current_patch_embed.proj.in_channels,
            patch_size=getattr(current_patch_embed, 'patch_size', 16),
            patch_size_t=getattr(current_patch_embed, 'patch_size_t', None),
            embed_dim=current_patch_embed.proj.out_channels,
            bias=current_patch_embed.proj.bias is not None,
            use_positional_embeddings=getattr(current_patch_embed, 'use_positional_embeddings', True),
            use_learned_positional_embeddings=getattr(current_patch_embed, 'use_learned_positional_embeddings', False),
            sample_height=getattr(current_patch_embed, 'sample_height', None),
            sample_width=getattr(current_patch_embed, 'sample_width', None),
            sample_frames=getattr(current_patch_embed, 'sample_frames', None),
            temporal_compression_ratio=getattr(current_patch_embed, 'temporal_compression_ratio', 1),
        )
        
        # Copy pretrained weights from original patch_embed
        with torch.no_grad():
            # Copy main projection weights
            adapter_patch_embed.proj.weight.data = current_patch_embed.proj.weight.data
            if current_patch_embed.proj.bias is not None:
                adapter_patch_embed.proj.bias.data = current_patch_embed.proj.bias.data
            
            # Copy text projection weights
            adapter_patch_embed.text_proj.weight.data = current_patch_embed.text_proj.weight.data
            if current_patch_embed.text_proj.bias is not None:
                adapter_patch_embed.text_proj.bias.data = current_patch_embed.text_proj.bias.data
            
            # Copy positional embeddings if they exist
            if hasattr(current_patch_embed, 'pos_embedding'):
                adapter_patch_embed.pos_embedding.data = current_patch_embed.pos_embedding.data
        
        # Replace patch_embed
        self.patch_embed = adapter_patch_embed
        
        # Register config
        self.register_to_config(
            condition_channels=condition_channels,
            use_adapter_patch_embed=True,
        )
        
        print(f"✅ Successfully replaced patch_embed with adapter version")
        print(f"  Main projection: {adapter_patch_embed.proj}")
        print(f"  Condition projection: {adapter_patch_embed.cond_proj}")
        print(f"  Text projection: {adapter_patch_embed.text_proj}")
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path=None,
        base_model_name_or_path="alibaba-pai/CogVideoX-Fun-V1.1-5b-InP",
        subfolder="transformer",
        condition_channels: Optional[int] = None,
        **kwargs
    ):
        """
        Load a CogVideoXFunTransformer3DModelWithConcat from a pretrained model.

        Args:
            pretrained_model_name_or_path (str): Fine-tuned checkpoint path
            base_model_name_or_path (str): Base CogVideoX-Fun model path
            condition_channels (int): Number of extra condition channels to add
        """
        if pretrained_model_name_or_path is not None:
            # === Load fine-tuned checkpoint ===
            print(f"📥 Loading fine-tuned concat pose-conditioned transformer: {pretrained_model_name_or_path}")

            # 1) Create child class structure first
            config_path = os.path.join(pretrained_model_name_or_path, subfolder, "config.json")
            if os.path.exists(config_path):
                import json
                with open(config_path, "r") as f:
                    config = json.load(f)
                model = cls(**config)
            else:
                # Use base config
                if condition_channels is None:
                    condition_channels = 16  # default
                model = cls(condition_channels=condition_channels)

            # 2) Load checkpoint state_dict directly
            state_dict_path = os.path.join(pretrained_model_name_or_path, subfolder, "diffusion_pytorch_model.safetensors")
            if not os.path.exists(state_dict_path):
                # Try alternative paths
                state_dict_path = os.path.join(pretrained_model_name_or_path, subfolder, "pytorch_model.bin")
                if not os.path.exists(state_dict_path):
                    raise FileNotFoundError(f"No model file found in {pretrained_model_name_or_path}")
            
            if state_dict_path.endswith(".safetensors"):
                from safetensors.torch import load_file
                state_dict = load_file(state_dict_path)
            else:
                state_dict = torch.load(state_dict_path, map_location="cpu")

            # 3) Load with strict=False for flexibility
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"⚠️ Missing keys (new layers, need initialization during training): {missing}")
            if unexpected:
                print(f"⚠️ Unexpected keys (replaced layers, etc.): {unexpected}")

            return model

        else:
            # === Initialize from base model ===
            print(f"🔧 Initializing from base model: {base_model_name_or_path}")
            # First load the base CogVideoX transformer
            base_model = CogVideoXTransformer3DModel.from_pretrained(
                base_model_name_or_path,
                subfolder=subfolder,
                torch_dtype=kwargs.get("torch_dtype", torch.bfloat16),
                revision=kwargs.get("revision", None),
                variant=kwargs.get("variant", None),
            )
            
            # Then create CogVideoXFun transformer and copy state_dict
            fun_model = CogVideoXFunTransformer3DModel(**base_model.config)
            fun_model.load_state_dict(base_model.state_dict())
            base_model = fun_model

            # Determine condition_channels
            if condition_channels is None:
                condition_channels = getattr(base_model.config, "condition_channels", 16)

            # Create extended model
            model = cls(**base_model.config, condition_channels=condition_channels)

            # Load base weights into extended model
            missing, unexpected = model.load_state_dict(base_model.state_dict(), strict=False)
            if missing:
                print(f"⚠️ Missing keys (new condition layers): {missing}")
            if unexpected:
                print(f"⚠️ Unexpected keys (replaced existing layers): {unexpected}")

            return model
    
    def get_condition_info(self) -> Dict[str, Any]:
        """Get information about the conditional setup."""
        info = {
            "approach": "concat",
            "has_conditions": hasattr(self, 'condition_channels') and self.condition_channels > 0,
            "condition_channels": getattr(self, 'condition_channels', 0),
            "total_input_channels": self.patch_embed.proj.in_channels,
            "base_channels": getattr(self, 'original_in_channels', self.patch_embed.proj.in_channels),
        }
        return info
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
        inpaint_latents: Optional[torch.Tensor] = None,
        control_latents: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_dict: bool = True,
        **kwargs
    ):
        """
        Forward pass with VideoX-Fun specific inputs and adapter-based conditional processing.
        
        Args:
            hidden_states: Input tensor with shape [batch, frames, channels, height, width]
            encoder_hidden_states: Text embeddings
            timestep: Timestep tensor
            timestep_cond: Timestep condition tensor
            inpaint_latents: Inpainting latents (VideoX-Fun specific)
            control_latents: Control latents (VideoX-Fun specific)
            image_rotary_emb: Rotary embeddings
            return_dict: Whether to return a dict
            **kwargs: Additional arguments
        """
        if "attention_kwargs" in kwargs:
            attention_kwargs = kwargs["attention_kwargs"].copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            attention_kwargs = {}
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_frames, channels, height, width = hidden_states.shape

        # 1. Time embedding
        timesteps = timestep
        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        if self.ofs_embedding is not None and "ofs" in kwargs:
            ofs = kwargs["ofs"]
            ofs_emb = self.ofs_proj(ofs)
            ofs_emb = ofs_emb.to(dtype=hidden_states.dtype)
            ofs_emb = self.ofs_embedding(ofs_emb)
            emb = emb + ofs_emb

        # Handle VideoX-Fun specific inputs
        if inpaint_latents is not None:
            hidden_states = torch.concat([hidden_states, inpaint_latents], 2)
        
        if not hasattr(self.patch_embed, 'cond_proj') and control_latents is not None:
            # concat control latents if not using adapter
            hidden_states = torch.concat([hidden_states, control_latents], 2)

        # 2. Patch embedding with adapter support
        patch_embed_input = {'text_embeds': encoder_hidden_states, 
                             'image_embeds': hidden_states}
        if hasattr(self.patch_embed, 'cond_proj') and control_latents is not None:
            patch_embed_input['cond_embeds'] = control_latents
    
        # Use adapter patch embed
        hidden_states, ref_hidden_states = self.patch_embed(**patch_embed_input)
        
        hidden_states = self.embedding_dropout(hidden_states)

        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]

        # 3. Transformer blocks
        ca_idx = 0
        for i, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states, encoder_hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    emb,
                    image_rotary_emb,
                    attention_kwargs,
                )
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=emb,
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

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)


class CrossTransformer3DModelWithAdapter(CogVideoXFunTransformer3DModelWithAdapter):
    def __init__(self, *args, 
                is_train_cross: bool = True, 
                cross_attn_interval: int = 2,
                cross_attn_dim_head: int = 128,
                cross_attn_num_heads: int = 16,
                cross_attn_kv_dim: int = None,
                **kwargs):
        super().__init__(*args, **kwargs)
        num_attention_heads = kwargs.get("num_attention_heads", 30)
        attention_head_dim = kwargs.get("attention_head_dim", 64)
        self.inner_dim = num_attention_heads * attention_head_dim
        self.is_train_cross = is_train_cross
        self.cross_attn_interval = cross_attn_interval
        self.num_cross_attn = self.num_layers // self.cross_attn_interval
        self.cross_attn_dim_head = cross_attn_dim_head
        self.cross_attn_num_heads = cross_attn_num_heads
        self.cross_attn_kv_dim = cross_attn_kv_dim
        if self.is_train_cross:
            self.perceiver_cross_attention = nn.ModuleList(
                [
                    PerceiverCrossAttention(
                        dim=self.inner_dim,
                        dim_head=self.cross_attn_dim_head,
                        heads=self.cross_attn_num_heads,
                        kv_dim=self.cross_attn_kv_dim,
                    ).to(self.device, dtype=self.dtype)
                    for _ in range(self.num_cross_attn)
                ]
            )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path=None,
        base_model_name_or_path="alibaba-pai/CogVideoX-Fun-V1.1-5b-InP",
        subfolder="transformer",
        condition_channels: Optional[int] = None,
        is_train_cross: bool = True,
        cross_attn_interval: int = 2,
        cross_attn_dim_head: int = 128,
        cross_attn_num_heads: int = 16,
        cross_attn_kv_dim: int = None,
        **kwargs
    ):
        """
        Load a CogVideoXFunTransformer3DModelWithConcat from a pretrained model.

        Args:
            pretrained_model_name_or_path (str): Fine-tuned checkpoint path
            base_model_name_or_path (str): Base CogVideoX-Fun model path
            condition_channels (int): Number of extra condition channels to add
            is_train_cross (bool): Whether to train cross attention
            cross_attn_interval (int): Interval for cross attention
            cross_attn_dim_head (int): Dimension of head for cross attention
            cross_attn_num_heads (int): Number of heads for cross attention
            cross_attn_kv_dim (int): Dimension of key/value for cross attention
        """
        if pretrained_model_name_or_path is not None:
            # === Load fine-tuned checkpoint ===
            print(f"📥 Loading fine-tuned concat pose-conditioned transformer: {pretrained_model_name_or_path}")

            # 1) Create child class structure first
            config_path = os.path.join(pretrained_model_name_or_path, subfolder, "config.json")
            if os.path.exists(config_path):
                import json
                with open(config_path, "r") as f:
                    config = json.load(f)
                model = cls(**config)
            else:
                # Use base config
                if condition_channels is None:
                    condition_channels = 16  # default
                model = cls(condition_channels=condition_channels,
                            is_train_cross=is_train_cross,
                            cross_attn_interval=cross_attn_interval,
                            cross_attn_dim_head=cross_attn_dim_head,
                            cross_attn_num_heads=cross_attn_num_heads,
                            cross_attn_kv_dim=cross_attn_kv_dim)

            # 2) Load checkpoint state_dict directly
            state_dict_path = os.path.join(pretrained_model_name_or_path, subfolder, "diffusion_pytorch_model.safetensors")
            if not os.path.exists(state_dict_path):
                # Try alternative paths
                state_dict_path = os.path.join(pretrained_model_name_or_path, subfolder, "pytorch_model.bin")
                if not os.path.exists(state_dict_path):
                    raise FileNotFoundError(f"No model file found in {pretrained_model_name_or_path}")
            
            if state_dict_path.endswith(".safetensors"):
                from safetensors.torch import load_file
                state_dict = load_file(state_dict_path)
            else:
                state_dict = torch.load(state_dict_path, map_location="cpu")

            # 3) Load with strict=False for flexibility
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"⚠️ Missing keys (new layers, need initialization during training): {missing}")
            if unexpected:
                print(f"⚠️ Unexpected keys (replaced layers, etc.): {unexpected}")

            return model

        else:
            # === Initialize from base model ===
            print(f"🔧 Initializing from base model: {base_model_name_or_path}")
            # First load the base CogVideoX transformer
            base_model = CogVideoXTransformer3DModel.from_pretrained(
                base_model_name_or_path,
                subfolder=subfolder,
                torch_dtype=kwargs.get("torch_dtype", torch.bfloat16),
                revision=kwargs.get("revision", None),
                variant=kwargs.get("variant", None),
            )
            
            # Then create CogVideoXFun transformer and copy state_dict
            fun_model = CogVideoXFunTransformer3DModel(**base_model.config)
            fun_model.load_state_dict(base_model.state_dict())
            base_model = fun_model

            # Determine condition_channels
            if condition_channels is None:
                condition_channels = getattr(base_model.config, "condition_channels", 16)

            # Create extended model
            model = cls(**base_model.config, 
                        condition_channels=condition_channels,
                        is_train_cross=is_train_cross,
                        cross_attn_interval=cross_attn_interval,
                        cross_attn_dim_head=cross_attn_dim_head,
                        cross_attn_num_heads=cross_attn_num_heads,
                        cross_attn_kv_dim=cross_attn_kv_dim)

            # Load base weights into extended model
            missing, unexpected = model.load_state_dict(base_model.state_dict(), strict=False)
            if missing:
                print(f"⚠️ Missing keys (new condition layers): {missing}")
            if unexpected:
                print(f"⚠️ Unexpected keys (replaced existing layers): {unexpected}")

            return model
    

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
        inpaint_latents: Optional[torch.Tensor] = None,
        control_latents: Optional[torch.Tensor] = None,
        ref_latents: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_dict: bool = True,
        **kwargs
    ):
        """
        Forward pass with VideoX-Fun specific inputs and adapter-based conditional processing.
        
        Args:
            hidden_states: Input tensor with shape [batch, frames, channels, height, width]
            encoder_hidden_states: Text embeddings
            timestep: Timestep tensor
            timestep_cond: Timestep condition tensor
            inpaint_latents: Inpainting latents (VideoX-Fun specific)
            control_latents: Control latents (VideoX-Fun specific)
            ref_latents: Reference video latents
            image_rotary_emb: Rotary embeddings
            return_dict: Whether to return a dict
            **kwargs: Additional arguments
        """
        if "attention_kwargs" in kwargs:
            attention_kwargs = kwargs["attention_kwargs"].copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            attention_kwargs = {}
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_frames, channels, height, width = hidden_states.shape

        # 1. Time embedding
        timesteps = timestep
        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        if self.ofs_embedding is not None and "ofs" in kwargs:
            ofs = kwargs["ofs"]
            ofs_emb = self.ofs_proj(ofs)
            ofs_emb = ofs_emb.to(dtype=hidden_states.dtype)
            ofs_emb = self.ofs_embedding(ofs_emb)
            emb = emb + ofs_emb

        # Handle VideoX-Fun specific inputs
        if inpaint_latents is not None:
            hidden_states = torch.concat([hidden_states, inpaint_latents], 2)
        
        if not hasattr(self.patch_embed, 'cond_proj') and control_latents is not None:
            # concat control latents if not using adapter
            hidden_states = torch.concat([hidden_states, control_latents], 2)

        # 2. Patch embedding with adapter support
        patch_embed_input = {'text_embeds': encoder_hidden_states, 
                             'image_embeds': hidden_states}
        if hasattr(self.patch_embed, 'cond_proj') and control_latents is not None:
            patch_embed_input['cond_embeds'] = control_latents

        if ref_latents is not None:
            patch_embed_input['ref_image_embeds'] = ref_latents
    
        # Use adapter patch embed
        hidden_states, ref_hidden_states = self.patch_embed(**patch_embed_input)
        
        hidden_states = self.embedding_dropout(hidden_states)

        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]

        # 3. Transformer blocks
        ca_idx = 0
        for i, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states, encoder_hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    emb,
                    image_rotary_emb,
                    attention_kwargs,
                )
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=emb,
                    image_rotary_emb=image_rotary_emb,
                    attention_kwargs=attention_kwargs,
                )
            if self.is_train_cross and ref_hidden_states is not None:
                if i % self.cross_attn_interval == 0:
                    hidden_states = hidden_states + self.perceiver_cross_attention[
                        ca_idx
                    ](
                        ref_hidden_states, hidden_states
                    )  # torch.Size([2, 32, 2048])  torch.Size([2, 17550, 3072])
                    ca_idx += 1

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

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
