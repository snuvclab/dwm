import torch
import torch.nn as nn
import os
import glob
import math
from diffusers import CogVideoXTransformer3DModel
from diffusers.models.embeddings import CogVideoXPatchEmbed
from diffusers.models.attention_processor import AttnProcessor2_0, CogVideoXAttnProcessor2_0
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.normalization import CogVideoXLayerNormZero
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers, logging
from safetensors.torch import load_file
from typing import Optional, Dict, Any, Tuple, Union
import numpy as np

logger = logging.get_logger(__name__)

def reshape_tensor(tensor, num_heads):
    """Reshape tensor for multi-head attention."""
    batch_size, seq_len, dim = tensor.shape
    head_dim = dim // num_heads
    return tensor.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

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


class LatentCondAdapter3D(nn.Module):
    """Residual conditioning adapter for DiT/CogVideoX latents."""
    def __init__(self, channels: int, norm: str = "none", groups: int = 32, 
                 return_residual: bool = False, use_dropout: bool = False):
        super().__init__()
        self.channels = channels
        self.return_residual = return_residual

        # Create conv layers as proper nn.Module submodules
        self.hand_conv = nn.Conv3d(channels, channels, 1, bias=False)
        self.static_conv = nn.Conv3d(channels, channels, 1, bias=False)
        
        # Zero initialization
        nn.init.zeros_(self.hand_conv.weight)
        nn.init.zeros_(self.static_conv.weight)
        self.scale_hand = nn.Parameter(torch.zeros(1))
        self.scale_static = nn.Parameter(torch.zeros(1))

        if norm == "group":
            g = min(groups, channels)
            self.norm = nn.GroupNorm(g, channels, affine=True)
        else:
            self.norm = nn.Identity()

        self.dropout = nn.Dropout3d(p=0.1) if use_dropout else nn.Identity()

        # freeze flags
        self.freeze_hand = False
        self.freeze_static = False

    def freeze_hand_branch(self):
        for p in list(self.hand_conv.parameters()) + [self.scale_hand]:
            p.requires_grad = False
        self.freeze_hand = True
        print("🔒 Hand branch frozen")

    def freeze_static_branch(self):
        for p in list(self.static_conv.parameters()) + [self.scale_static]:
            p.requires_grad = False
        self.freeze_static = True
        print("🔒 Static branch frozen")

    def unfreeze_hand_branch(self):
        for p in list(self.hand_conv.parameters()) + [self.scale_hand]:
            p.requires_grad = True
        self.freeze_hand = False
        print("🔓 Hand branch unfrozen")

    def unfreeze_static_branch(self):
        for p in list(self.static_conv.parameters()) + [self.scale_static]:
            p.requires_grad = True
        self.freeze_static = False
        print("🔓 Static branch unfrozen")

    def get_freeze_status(self):
        return {"hand": self.freeze_hand, "static": self.freeze_static}

    @staticmethod
    def _to_c3d(x):  # [B,T,C,H,W] → [B,C,T,H,W]
        return x.permute(0,2,1,3,4)

    @staticmethod
    def _to_tfirst(x):  # [B,C,T,H,W] → [B,T,C,H,W]
        return x.permute(0,2,1,3,4)

    def forward(self, latents, hand=None, static=None, hand_mask=None, static_mask=None, return_residual=None):
        residual = torch.zeros_like(latents)

        if hand is not None:
            h = self._to_c3d(hand)
            h = self.hand_conv(h)
            h = self.dropout(h)
            h = self._to_tfirst(h)
            if hand_mask is not None:
                if hand_mask.dim()==4: hand_mask = hand_mask.unsqueeze(2)
                h = h * hand_mask.to(h)
            residual += self.scale_hand * h

        if static is not None:
            s = self._to_c3d(static)
            s = self.static_conv(s)
            s = self.dropout(s)
            s = self._to_tfirst(s)
            if static_mask is not None:
                if static_mask.dim()==4: static_mask = static_mask.unsqueeze(2)
                s = s * static_mask.to(s)
            residual += self.scale_static * s

        residual = self._to_tfirst(self.norm(self._to_c3d(residual)))

        if return_residual if return_residual is not None else self.return_residual:
            return residual
        return latents + residual


class CogVideoXTransformer3DModelWithConcat(CogVideoXTransformer3DModel):
    """CogVideoXTransformer3DModel with concatenation-based conditioning.
    
    This approach concatenates condition channels directly to the input channels,
    requiring the transformer to learn to handle the extended input.
    """
    
    def __init__(self, *args, condition_channels: int = 0, **kwargs):
        original_in_channels = kwargs.pop("original_in_channels", None)
        super().__init__(*args, **kwargs)
        
        # Setup conditional channels if specified
        if condition_channels > 0:
            if original_in_channels is None:
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
        base_model_name_or_path="THUDM/CogVideoX-5b",
        subfolder="transformer",
        condition_channels: Optional[int] = None,
        **kwargs
    ):
        """
        Load a CogVideoX transformer with optional concatenated condition channels.

        Args:
            pretrained_model_name_or_path (str): Fine-tuned checkpoint path
            base_model_name_or_path (str): Base CogVideoX model path
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
            base_model = CogVideoXTransformer3DModel.from_pretrained(
                base_model_name_or_path,
                subfolder=subfolder,
                torch_dtype=kwargs.get("torch_dtype", torch.bfloat16),
                revision=kwargs.get("revision", None),
                variant=kwargs.get("variant", None),
            )

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
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        ofs: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.FloatTensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ):
        """Forward pass with concatenated conditioning.
        
        Note: This approach expects the input hidden_states to already contain
        the concatenated condition channels.
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
        
        # Process through the parent transformer
        return super().forward(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            ofs=ofs,
            image_rotary_emb=image_rotary_emb,
            attention_kwargs=attention_kwargs,
            return_dict=return_dict
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
        use_zero_proj = kwargs.pop("use_zero_proj", False)
        use_ref_proj = kwargs.pop("use_ref_proj", False)
        super().__init__(*args, **kwargs)
        in_channels = kwargs.get("in_channels", 16)
        condition_channels = kwargs.get("condition_channels", 16)
        ref_channels = kwargs.get("ref_channels", 16)
        patch_size = kwargs.get("patch_size", 16)
        patch_size_t = kwargs.get("patch_size_t", None)
        embed_dim = kwargs.get("embed_dim", 16)
        bias = kwargs.get("bias", True)

        # Setup adapter
        if patch_size_t is None:
            # CogVideoX 1.0 checkpoints
            self.cond_proj = nn.Conv2d(
                condition_channels, embed_dim, kernel_size=(patch_size, patch_size), stride=patch_size, bias=bias
            )
        else:
            # CogVideoX 1.5 checkpoints
            self.cond_proj = nn.Linear(condition_channels * patch_size * patch_size * patch_size_t, embed_dim)
        
        self.cond_zero_proj = None
        if use_zero_proj:
            from diffusers.models.controlnet import zero_module
            # with cond_zero_proj, the cond_proj is not zero initialized
            self.cond_zero_proj = zero_module(nn.Linear(embed_dim, embed_dim))
        else:
            # Zero initialize cond_proj weights for stable training
            with torch.no_grad():
                self.cond_proj.weight.zero_()
                if hasattr(self.cond_proj, 'bias') and self.cond_proj.bias is not None:
                    self.cond_proj.bias.zero_()

        if use_ref_proj:
            self.ref_proj = nn.Conv2d(
                ref_channels, embed_dim, kernel_size=(patch_size, patch_size), stride=patch_size, bias=bias
            )
        else:
            self.ref_proj = None

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
            # Process image embeds
            image_embeds = image_embeds.reshape(-1, channels, height, width)
            image_embeds = self.proj(image_embeds)
            image_embeds = image_embeds.view(batch_size, num_frames, *image_embeds.shape[1:])
            image_embeds = image_embeds.flatten(3).transpose(2, 3)  # [batch, num_frames, height x width, channels]
            image_embeds = image_embeds.flatten(1, 2)  # [batch, num_frames x height x width, channels]
            
            # Process ref image embeds separately if ref_proj exists
            if ref_image_embeds is not None:
                ref_channels = ref_image_embeds.shape[2]
                if self.ref_proj is not None:
                    # Use ref_proj for ref_image_embeds
                    ref_image_embeds = ref_image_embeds.reshape(-1, ref_channels, height, width)
                    ref_image_embeds = self.ref_proj(ref_image_embeds)
                    ref_image_embeds = ref_image_embeds.view(batch_size, num_frames, *ref_image_embeds.shape[1:])
                else:
                    # Use same proj as image_embeds (original behavior)
                    ref_image_embeds = ref_image_embeds.reshape(-1, ref_channels, height, width)
                    ref_image_embeds = self.proj(ref_image_embeds)
                    ref_image_embeds = ref_image_embeds.view(batch_size, num_frames, *ref_image_embeds.shape[1:])
                
                ref_image_embeds = ref_image_embeds.flatten(3).transpose(2, 3)
                ref_image_embeds = ref_image_embeds.flatten(1, 2)
        
            if cond_embeds is not None:
                cond_channels = cond_embeds.shape[2]
                cond_embeds = cond_embeds.reshape(-1, cond_channels, height, width)
                cond_embeds = self.cond_proj(cond_embeds)
                cond_embeds = cond_embeds.view(batch_size, num_frames, *cond_embeds.shape[1:])
                cond_embeds = cond_embeds.flatten(3).transpose(2, 3)  # [batch, num_frames, height x width, channels]
                cond_embeds = cond_embeds.flatten(1, 2)  # [batch, num_frames x height x width, channels]
                if self.cond_zero_proj is not None:
                    cond_embeds = self.cond_zero_proj(cond_embeds)
                image_embeds = image_embeds + cond_embeds
        
        else:
            p = self.patch_size
            p_t = self.patch_size_t

            # Process image embeds
            image_embeds = image_embeds.permute(0, 1, 3, 4, 2)
            image_embeds = image_embeds.reshape(
                -1, num_frames // p_t, p_t, height // p, p, width // p, p, channels
            )
            image_embeds = image_embeds.permute(0, 1, 3, 5, 7, 2, 4, 6).flatten(4, 7).flatten(1, 3)
            image_embeds = self.proj(image_embeds)
            
            # Process ref image embeds separately if ref_proj exists
            if ref_image_embeds is not None:
                if self.ref_proj is not None:
                    # Use ref_proj for ref_image_embeds
                    ref_image_embeds = ref_image_embeds.permute(0, 1, 3, 4, 2)
                    ref_image_embeds = ref_image_embeds.reshape(
                        -1, num_frames // p_t, p_t, height // p, p, width // p, p, channels
                    )
                    ref_image_embeds = ref_image_embeds.permute(0, 1, 3, 5, 7, 2, 4, 6).flatten(4, 7).flatten(1, 3)
                    ref_image_embeds = self.ref_proj(ref_image_embeds)
                else:
                    # Use same proj as image_embeds (original behavior)
                    ref_image_embeds = ref_image_embeds.permute(0, 1, 3, 4, 2)
                    ref_image_embeds = ref_image_embeds.reshape(
                        -1, num_frames // p_t, p_t, height // p, p, width // p, p, channels
                    )
                    ref_image_embeds = ref_image_embeds.permute(0, 1, 3, 5, 7, 2, 4, 6).flatten(4, 7).flatten(1, 3)
                    ref_image_embeds = self.proj(ref_image_embeds)

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


class CogVideoXTransformer3DModelWithAdapter(CogVideoXTransformer3DModel):
    """CogVideoXTransformer3DModel with adapter-based conditioning.
    
    This approach uses CogVideoXPatchEmbedWithAdapter to condition the latents
    by processing condition channels separately and adding them to the main embeddings.
    """
    
    def __init__(self, *args, condition_channels: int = 0, use_zero_proj: bool = False, use_ref_proj: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Replace patch_embed with adapter version if condition_channels > 0
        if condition_channels > 0:
            self._setup_adapter_patch_embed(condition_channels, use_zero_proj, use_ref_proj)
            self.condition_channels = condition_channels
        else:
            self.condition_channels = 0
    
    def _setup_adapter_patch_embed(self, condition_channels: int, 
                                   use_zero_proj: bool = False, use_ref_proj: bool = False):
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
            use_zero_proj=use_zero_proj,
            use_ref_proj=use_ref_proj,
        )
        
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
        base_model_name_or_path="THUDM/CogVideoX-5b",
        subfolder="transformer",
        condition_channels: Optional[int] = None,
        use_zero_proj: bool = False,
        use_ref_proj: bool = False,
        **kwargs
    ):
        """
        Load a CogVideoX transformer with adapter-based conditioning.

        Args:
            pretrained_model_name_or_path (str): Fine-tuned checkpoint path
            base_model_name_or_path (str): Base CogVideoX model path
            condition_channels (int): Number of extra condition channels to add
        """
        if pretrained_model_name_or_path is not None:
            # === Load fine-tuned checkpoint ===
            print(f"📥 Loading fine-tuned adapter pose-conditioned transformer: {pretrained_model_name_or_path}")

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
                            use_zero_proj=use_zero_proj,
                            use_ref_proj=use_ref_proj)

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
            base_model = CogVideoXTransformer3DModel.from_pretrained(
                base_model_name_or_path,
                subfolder=subfolder,
                torch_dtype=kwargs.get("torch_dtype", torch.bfloat16),
                revision=kwargs.get("revision", None),
                variant=kwargs.get("variant", None),
            )

            # Determine condition_channels
            if condition_channels is None:
                condition_channels = getattr(base_model.config, "condition_channels", 16)

            # Create extended model
            model = cls(**base_model.config, condition_channels=condition_channels,
                        use_zero_proj=use_zero_proj,
                        use_ref_proj=use_ref_proj)

            # Load base weights into extended model
            missing, unexpected = model.load_state_dict(base_model.state_dict(), strict=False)
            
            # Copy pretrained weights for newly added layers only
            # (existing layers like proj, text_proj, pos_embedding are already copied by load_state_dict)
            with torch.no_grad():
                # Get references to patch embeddings
                current_patch_embed = base_model.patch_embed
                adapter_patch_embed = model.patch_embed
                
                # Only copy ref projection weights (newly added layer)
                if use_ref_proj:
                    adapter_patch_embed.ref_proj.weight.data = current_patch_embed.proj.weight.data
                    if current_patch_embed.proj.bias is not None:
                        adapter_patch_embed.ref_proj.bias.data = current_patch_embed.proj.bias.data
                    print(f"✅ Copied ref_proj weights from base model (shape: {adapter_patch_embed.ref_proj.weight.shape})")
                else:
                    print("ℹ️ No ref_proj to copy (use_ref_proj=False)")
                
                print("✅ Base model weights successfully loaded and copied to adapter model")
            
            if missing:
                print(f"⚠️ Missing keys (new condition layers): {missing}")
            if unexpected:
                print(f"⚠️ Unexpected keys (replaced existing layers): {unexpected}")

            return model
    
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        image_rotary_emb: Optional[torch.FloatTensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        hand_conditions: Optional[torch.FloatTensor] = None,
        static_conditions: Optional[torch.FloatTensor] = None,
        hand_mask: Optional[torch.FloatTensor] = None,
        static_mask: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,
    ):
        """Forward pass with adapter-based conditioning."""
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            attention_kwargs = {}
            lora_scale = 1.0

        batch_size, num_frames, channels, height, width = hidden_states.shape

        # 1. Time embedding
        timesteps = timestep
        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb)

        if self.ofs_embedding is not None:
            # Create a default ofs tensor if not provided
            ofs = torch.ones((batch_size,), device=hidden_states.device, dtype=hidden_states.dtype) * 2.0
            ofs_emb = self.ofs_proj(ofs)
            ofs_emb = ofs_emb.to(dtype=ofs_emb.dtype)
            ofs_emb = self.ofs_embedding(ofs_emb)
            emb = emb + ofs_emb

        # Handle condition inputs
        if not hasattr(self.patch_embed, 'cond_proj') and hand_conditions is not None:
            # concat condition latents if not using adapter
            hidden_states = torch.concat([hidden_states, hand_conditions], 2)

        # 2. Patch embedding with adapter support
        patch_embed_input = {'text_embeds': encoder_hidden_states, 
                             'image_embeds': hidden_states}
        if hasattr(self.patch_embed, 'cond_proj') and hand_conditions is not None:
            patch_embed_input['cond_embeds'] = hand_conditions

        # Use adapter patch embed
        hidden_states, _ = self.patch_embed(**patch_embed_input)
        
        hidden_states = self.embedding_dropout(hidden_states)

        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]

        # 3. Transformer blocks
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

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
    
    def get_condition_info(self) -> Dict[str, Any]:
        """Get information about the conditional setup."""
        info = {
            "approach": "adapter",
            "has_conditions": hasattr(self, 'condition_channels') and self.condition_channels > 0,
            "condition_channels": getattr(self, 'condition_channels', 0),
            "total_input_channels": self.patch_embed.proj.in_channels,
            "base_channels": getattr(self, 'original_in_channels', self.patch_embed.proj.in_channels),
        }
        return info


# Legacy class for backward compatibility
class CogVideoXTransformer3DModelWithConditions(CogVideoXTransformer3DModelWithAdapter):
    """Legacy class that defaults to adapter-based conditioning.
    
    This class is kept for backward compatibility but defaults to the adapter approach.
    For new code, use CogVideoXTransformer3DModelWithAdapter or CogVideoXTransformer3DModelWithConcat
    explicitly based on your needs.
    """
    
    def __init__(self, *args, condition_channels: int = 0, **kwargs):
        super().__init__(
            *args, 
            condition_channels=condition_channels,
            **kwargs
        )

class CrossTransformer3DModelWithAdapter(CogVideoXTransformer3DModelWithAdapter):
    def __init__(self, *args, 
                is_train_cross: bool = True, 
                cross_attn_interval: int = 2,
                cross_attn_dim_head: int = 128,
                cross_attn_num_heads: int = 16,
                cross_attn_kv_dim: int = None,
                **kwargs):
        super().__init__(*args, use_ref_proj=True, **kwargs)
        num_attention_heads = kwargs.get("num_attention_heads", 30)
        attention_head_dim = kwargs.get("attention_head_dim", 64)
        self.inner_dim = num_attention_heads * attention_head_dim
        self.is_train_cross = is_train_cross
        self.cross_attn_interval = cross_attn_interval
        self.num_cross_attn = self.config.num_layers // self.cross_attn_interval
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
        base_model_name_or_path="THUDM/CogVideoX-5b-I2V",
        subfolder="transformer",
        condition_channels: Optional[int] = None,
        use_zero_proj: bool = False,
        is_train_cross: bool = True,
        cross_attn_interval: int = 2,
        cross_attn_dim_head: int = 128,
        cross_attn_num_heads: int = 16,
        cross_attn_kv_dim: int = None,
        **kwargs
    ):
        """
        Load a CrossTransformer3DModelWithAdapter from a pretrained model.

        Args:
            pretrained_model_name_or_path (str): Fine-tuned checkpoint path
            base_model_name_or_path (str): Base CogVideoX model path
            condition_channels (int): Number of extra condition channels to add
            use_zero_proj (bool): Whether to use zero projection
            is_train_cross (bool): Whether to train cross attention
            cross_attn_interval (int): Interval for cross attention
            cross_attn_dim_head (int): Dimension of head for cross attention
            cross_attn_num_heads (int): Number of heads for cross attention
            cross_attn_kv_dim (int): Dimension of key/value for cross attention
        """
        if pretrained_model_name_or_path is not None:
            # === Load fine-tuned checkpoint ===
            print(f"📥 Loading fine-tuned cross pose-conditioned transformer: {pretrained_model_name_or_path}")

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
                            use_zero_proj=use_zero_proj,
                            is_train_cross=is_train_cross,
                            cross_attn_interval=cross_attn_interval,
                            cross_attn_dim_head=cross_attn_dim_head,
                            cross_attn_num_heads=cross_attn_num_heads,
                            cross_attn_kv_dim=cross_attn_kv_dim)

            # 2) Load checkpoint state_dict directly
            # Try to find all safetensors files first
            safetensors_files = glob.glob(os.path.join(pretrained_model_name_or_path, subfolder, "*.safetensors"))
            if safetensors_files:
                print(f"🔧 Loading weights from {len(safetensors_files)} safetensors files")
                # Load and merge all safetensors files
                state_dict = {}
                for file_path in safetensors_files:
                    print(f"   Loading: {os.path.basename(file_path)}")
                    file_state_dict = load_file(file_path)
                    state_dict.update(file_state_dict)
            else:
                # Fallback to single file approach
                state_dict_path = os.path.join(pretrained_model_name_or_path, subfolder, "diffusion_pytorch_model.safetensors")
                if not os.path.exists(state_dict_path):
                    # Try alternative paths
                    state_dict_path = os.path.join(pretrained_model_name_or_path, subfolder, "pytorch_model.bin")
                    if not os.path.exists(state_dict_path):
                        raise FileNotFoundError(f"No model file found in {pretrained_model_name_or_path}")
                
                if state_dict_path.endswith(".safetensors"):
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

            # Determine condition_channels
            if condition_channels is None:
                condition_channels = getattr(base_model.config, "condition_channels", 16)

            # Create extended model
            model = cls(**base_model.config, 
                        condition_channels=condition_channels,
                        use_zero_proj=use_zero_proj,
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
        control_latents: Optional[torch.Tensor] = None,
        ref_latents: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        ref_image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
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




# ============================================================================
# AdaLN Pose Conditioning Classes
# ============================================================================

class SMPLPoseEmbedding(nn.Module):
    """Embedding layer for SMPL pose parameters with frame-wise projection and 1D conv for AdaLN conditioning"""
    
    def __init__(self, pose_dim=63, embed_dim=512, hidden_dim=1024, stride=4):
        super().__init__()
        self.pose_dim = pose_dim
        self.embed_dim = embed_dim
        self.stride = stride
        
        # Frame-wise projection: each frame's pose parameters are projected independently
        self.frame_projection = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        
        # 1D convolution to aggregate temporal features
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=1),  # Final projection
        )
        
        # Global average pooling to get single feature vector
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, pose_params: torch.Tensor) -> torch.Tensor:
        """
        Process SMPL pose parameters into embeddings using frame-wise projection and 1D conv.
        
        Args:
            pose_params: Tensor of shape (batch_size, num_frames, pose_dim)
            
        Returns:
            Tensor of shape (batch_size, embed_dim)
        """
        batch_size, num_frames, pose_dim = pose_params.shape
        
        # Frame-wise projection: (batch_size, num_frames, pose_dim) -> (batch_size, num_frames, embed_dim)
        frame_features = self.frame_projection(pose_params)  # (B, T, embed_dim)
        
        # Transpose for 1D conv: (batch_size, embed_dim, num_frames)
        frame_features = frame_features.transpose(1, 2)  # (B, embed_dim, T)
        
        # Apply 1D convolution for temporal aggregation
        temporal_features = self.temporal_conv(frame_features)  # (B, embed_dim, T)
        
        # Global average pooling to get single feature vector
        global_feature = self.global_pool(temporal_features)  # (B, embed_dim, 1)
        global_feature = global_feature.squeeze(-1)  # (B, embed_dim)
        
        # Apply layer normalization
        embeddings = self.layer_norm(global_feature)
        
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


class CogVideoXTransformer3DModelWithAdaLNPose(CogVideoXTransformer3DModel):
    """CogVideoXTransformer3DModel with SMPL pose conditioning via AdaLN-zero
    
    This model conditions on SMPL pose parameters by:
    1. Sampling frames with stride 4 (e.g., frames 0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48)
    2. Flattening the sampled pose parameters (63 * 13 = 819 dimensions)
    3. Encoding them into a global embedding for AdaLN-zero conditioning
    4. Applying the conditioning through extended LayerNormZero layers
    """
    
    _skip_layerwise_casting_patterns = ["patch_embed", "norm"]
    _supports_gradient_checkpointing = True
    _no_split_modules = ["SMPLConditionedCogVideoXBlock", "CogVideoXBlock", "CogVideoXPatchEmbed"]

    def __init__(self, *args, smpl_pose_dim=63, smpl_embed_dim=512, **kwargs):
        # Extract SMPL-specific parameters
        self.smpl_pose_dim = smpl_pose_dim
        self.smpl_embed_dim = smpl_embed_dim
        
        super().__init__(*args, **kwargs)
        
        # Initialize SMPL pose embedding
        self.smpl_pose_embedding = SMPLPoseEmbedding(
            pose_dim=self.smpl_pose_dim,
            embed_dim=self.smpl_embed_dim,
            stride=4  # Stride for temporal processing
        )
        
        # Replace transformer blocks with SMPL-conditioned versions
        self._replace_blocks_with_smpl_conditioned()
        
        # Ensure all SMPL components have the correct dtype AFTER creating all components
        self._ensure_smpl_dtype()
    
    def _replace_blocks_with_smpl_conditioned(self):
        """Replace transformer blocks with SMPL-conditioned versions"""
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
    def from_pretrained(
        cls,
        pretrained_model_name_or_path=None,
        base_model_name_or_path="THUDM/CogVideoX-5b",
        smpl_pose_dim=63,
        smpl_embed_dim=512,
        **kwargs
    ):
        """
        Load a CogVideoX transformer with AdaLN pose conditioning.

        Args:
            pretrained_model_name_or_path (str): Fine-tuned checkpoint path
            base_model_name_or_path (str): Base CogVideoX model path
            smpl_pose_dim (int): SMPL pose parameter dimension
            smpl_embed_dim (int): SMPL embedding dimension
        """
        if pretrained_model_name_or_path is not None:
            # === Load fine-tuned checkpoint ===
            print(f"📥 Loading fine-tuned AdaLN pose-conditioned transformer: {pretrained_model_name_or_path}")

            # 1) Create child class structure first
            config_path = os.path.join(pretrained_model_name_or_path, "config.json")
            if os.path.exists(config_path):
                import json
                with open(config_path, "r") as f:
                    config = json.load(f)
                model = cls(**config)
            else:
                # Use base config
                model = cls(
                    smpl_pose_dim=smpl_pose_dim,
                    smpl_embed_dim=smpl_embed_dim
                )

            # 2) Load checkpoint state_dict directly
            state_dict_path = os.path.join(pretrained_model_name_or_path, "diffusion_pytorch_model.safetensors")
            if not os.path.exists(state_dict_path):
                # Try alternative paths
                state_dict_path = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
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
            base_model = CogVideoXTransformer3DModel.from_pretrained(
                base_model_name_or_path,
                subfolder=kwargs.get("subfolder", "transformer"),
                torch_dtype=kwargs.get("torch_dtype", torch.bfloat16),
                revision=kwargs.get("revision", None),
                variant=kwargs.get("variant", None),
            )

            # Create new model with SMPL conditioning
            model = cls(
                **base_model.config,
                smpl_pose_dim=smpl_pose_dim,
                smpl_embed_dim=smpl_embed_dim
            )

            # Load base weights into extended model
            missing, unexpected = model.load_state_dict(base_model.state_dict(), strict=False)
            if missing:
                print(f"⚠️ Missing keys (new AdaLN layers): {missing}")
            if unexpected:
                print(f"⚠️ Unexpected keys (replaced existing layers): {unexpected}")

            return model

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        ofs: Optional[torch.FloatTensor] = None,  # Add ofs parameter for compatibility
        image_rotary_emb: Optional[torch.FloatTensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        pose_params: Optional[torch.Tensor] = None,  # New parameter for SMPL pose
        pose_params_mask: Optional[torch.Tensor] = None,  # New parameter to track None pose_params
        return_dict: bool = True,
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
            # Create a default ofs tensor if not provided
            ofs = torch.ones((batch_size,), device=hidden_states.device, dtype=hidden_states.dtype) * 2.0
            ofs_emb = self.ofs_proj(ofs)
            ofs_emb = ofs_emb.to(dtype=ofs_emb.dtype)
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

        return Transformer2DModelOutput(sample=output)


# ============================================================================
# Per-Frame AdaLN Pose Conditioning Classes
# ============================================================================

class SMPLPoseEmbeddingPerFrame(nn.Module):
    """Embedding layer for SMPL pose parameters with per-frame processing and 1D conv for AdaLN conditioning"""
    
    def __init__(self, pose_dim=63, embed_dim=512, hidden_dim=1024, stride=4):
        super().__init__()
        self.pose_dim = pose_dim
        self.embed_dim = embed_dim
        self.stride = stride
        
        # Frame-wise projection: each frame's pose parameters are projected independently
        self.frame_projection = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        
        # 1D convolution to reduce temporal dimension: 49 -> 52 -> 26 -> 13 frames
        # First conv with stride=2: 52 -> 26, then second conv with stride=2: 26 -> 13
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1),  # 52 -> 26
            nn.SiLU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1),  # 26 -> 13
            nn.SiLU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=1),  # Final projection
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Zero initialization for AdaLN-zero (start with no SMPL conditioning effect)
        for layer in self.frame_projection:
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        # Initialize conv layers to identity-like behavior
        for layer in self.temporal_conv:
            if isinstance(layer, nn.Conv1d):
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
            pose_embedding: Embedded pose features of shape (batch_size, 13, embed_dim)
        """
        batch_size, num_frames, pose_dim = pose_params.shape
        
        # Frame-wise projection: (batch_size, num_frames, pose_dim) -> (batch_size, num_frames, embed_dim)
        frame_features = self.frame_projection(pose_params)  # (B, T, embed_dim)
        
        # Pad with first frame to make 49 -> 52: copy first frame 3 times to the left
        if num_frames == 49:
            first_frame = frame_features[:, :1, :]  # (B, 1, embed_dim)
            padding = first_frame.repeat(1, 3, 1)  # (B, 3, embed_dim)
            frame_features = torch.cat([padding, frame_features], dim=1)  # (B, 52, embed_dim)
        
        # Transpose for 1D conv: (batch_size, embed_dim, num_frames)
        frame_features = frame_features.transpose(1, 2)  # (B, embed_dim, T)
        
        # Apply 1D convolution to reduce temporal dimension: 52 -> 26 -> 13
        temporal_features = self.temporal_conv(frame_features)  # (B, embed_dim, 13)
        
        # Transpose back: (batch_size, 13, embed_dim)
        temporal_features = temporal_features.transpose(1, 2)  # (B, 13, embed_dim)
        
        # Apply layer normalization
        pose_embedding = self.layer_norm(temporal_features)
        
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


class CogVideoXTransformer3DModelWithAdaLNPosePerFrame(CogVideoXTransformer3DModel):
    """CogVideoXTransformer3DModel with per-frame SMPL pose conditioning via AdaLN-zero
    
    This model conditions on SMPL pose parameters by:
    1. Sampling frames with stride 4 (e.g., frames 0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48)
    2. Processing each sampled frame's pose parameters through a shared MLP
    3. Encoding them into per-frame embeddings for AdaLN-zero conditioning
    4. Applying the conditioning through extended LayerNormZero layers
    """
    
    _skip_layerwise_casting_patterns = ["patch_embed", "norm"]
    _supports_gradient_checkpointing = True
    _no_split_modules = ["SMPLConditionedCogVideoXBlockPerFrame", "CogVideoXBlock", "CogVideoXPatchEmbed"]

    def __init__(self, *args, smpl_pose_dim=63, smpl_embed_dim=512, **kwargs):
        # Extract SMPL-specific parameters
        self.smpl_pose_dim = smpl_pose_dim
        self.smpl_embed_dim = smpl_embed_dim
        
        super().__init__(*args, **kwargs)
        
        # Initialize per-frame SMPL pose embedding
        self.smpl_pose_embedding = SMPLPoseEmbeddingPerFrame(
            pose_dim=self.smpl_pose_dim,
            embed_dim=self.smpl_embed_dim,
            stride=4  # Stride for temporal processing
        )
        
        # Replace transformer blocks with per-frame SMPL-conditioned versions
        self._replace_blocks_with_smpl_conditioned()
        
        # Ensure all SMPL components have the correct dtype AFTER creating all components
        self._ensure_smpl_dtype()
    
    def _replace_blocks_with_smpl_conditioned(self):
        """Replace transformer blocks with per-frame SMPL-conditioned versions"""
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
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path=None,
        base_model_name_or_path="THUDM/CogVideoX-5b",
        smpl_pose_dim=63,
        smpl_embed_dim=512,
        **kwargs
    ):
        """
        Load a CogVideoX transformer with per-frame AdaLN pose conditioning.

        Args:
            pretrained_model_name_or_path (str): Fine-tuned checkpoint path
            base_model_name_or_path (str): Base CogVideoX model path
            smpl_pose_dim (int): SMPL pose parameter dimension
            smpl_embed_dim (int): SMPL embedding dimension
        """
        if pretrained_model_name_or_path is not None:
            # === Load fine-tuned checkpoint ===
            print(f"📥 Loading fine-tuned per-frame AdaLN pose-conditioned transformer: {pretrained_model_name_or_path}")

            # 1) Create child class structure first
            config_path = os.path.join(pretrained_model_name_or_path, "config.json")
            if os.path.exists(config_path):
                import json
                with open(config_path, "r") as f:
                    config = json.load(f)
                model = cls(**config)
            else:
                # Use base config
                model = cls(
                    smpl_pose_dim=smpl_pose_dim,
                    smpl_embed_dim=smpl_embed_dim
                )

            # 2) Load checkpoint state_dict directly
            state_dict_path = os.path.join(pretrained_model_name_or_path, "diffusion_pytorch_model.safetensors")
            if not os.path.exists(state_dict_path):
                # Try alternative paths
                state_dict_path = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
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
            base_model = CogVideoXTransformer3DModel.from_pretrained(
                base_model_name_or_path,
                subfolder=kwargs.get("subfolder", "transformer"),
                torch_dtype=kwargs.get("torch_dtype", torch.bfloat16),
                revision=kwargs.get("revision", None),
                variant=kwargs.get("variant", None),
            )

            # Create new model with per-frame SMPL conditioning
            model = cls(
                **base_model.config,
                smpl_pose_dim=smpl_pose_dim,
                smpl_embed_dim=smpl_embed_dim
            )

            # Load base weights into extended model
            missing, unexpected = model.load_state_dict(base_model.state_dict(), strict=False)
            if missing:
                print(f"⚠️ Missing keys (new per-frame AdaLN layers): {missing}")
            if unexpected:
                print(f"⚠️ Unexpected keys (replaced existing layers): {unexpected}")

            return model

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        ofs: Optional[torch.FloatTensor] = None,  # Add ofs parameter for compatibility
        image_rotary_emb: Optional[torch.FloatTensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        pose_params: Optional[torch.Tensor] = None,  # New parameter for SMPL pose
        pose_params_mask: Optional[torch.Tensor] = None,  # New parameter to track None pose_params
        return_dict: bool = True,
    ):
        # Process pose parameters if provided
        smpl_emb = None
        smpl_emb_mask = None
        if pose_params is not None:
            # Per-frame SMPL pose embedding returns (batch_size, num_sampled_frames, embed_dim)
            smpl_emb = self.smpl_pose_embedding(pose_params)  # (batch_size, num_sampled_frames, embed_dim)
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
            # Create a default ofs tensor if not provided
            ofs = torch.ones((batch_size,), device=hidden_states.device, dtype=hidden_states.dtype) * 2.0
            ofs_emb = self.ofs_proj(ofs)
            ofs_emb = ofs_emb.to(dtype=ofs_emb.dtype)
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
        
        return Transformer2DModelOutput(sample=output)

def create_conditioned_transformer(
    base_model_path: str,
    approach: str = "adapter",  # "adapter", "concat", "adaln_pose", or "adaln_pose_perframe"
    condition_channels: int = 0,  # Used for both adapter and concat approaches
    smpl_pose_dim: int = 63,  # Only used for adaln_pose approaches
    smpl_embed_dim: int = 512,  # Only used for adaln_pose approaches
    torch_dtype: Optional[torch.dtype] = None,
    **kwargs
) -> CogVideoXTransformer3DModel:
    """Convenience function to create a conditioned transformer from a base model.
    
    Args:
        base_model_path: Path to the base model
        approach: Conditioning approach - "adapter", "concat", "adaln_pose", or "adaln_pose_perframe"
        condition_channels: Number of condition channels (for adapter and concat approaches)
        smpl_pose_dim: SMPL pose parameter dimension (only for adaln_pose approaches)
        smpl_embed_dim: SMPL embedding dimension (only for adaln_pose approaches)
        torch_dtype: Data type for the model
        **kwargs: Additional arguments passed to from_pretrained
    """
    print(f"Creating conditioned transformer from: {base_model_path}")
    print(f"Approach: {approach}")
    
    if approach == "adapter":
        print(f"Using adapter approach with {condition_channels} condition channels")
        transformer = CogVideoXTransformer3DModelWithAdapter.from_pretrained(
            base_model_path,
            subfolder="transformer",
            condition_channels=condition_channels,
            torch_dtype=torch_dtype,
            **kwargs
        )
    elif approach == "concat":
        print(f"Using concatenation approach with {condition_channels} condition channels")
        transformer = CogVideoXTransformer3DModelWithConcat.from_pretrained(
            base_model_path,
            subfolder="transformer",
            condition_channels=condition_channels,
            torch_dtype=torch_dtype,
            **kwargs
        )
    elif approach == "adaln_pose":
        print(f"Using AdaLN pose conditioning with pose_dim={smpl_pose_dim}, embed_dim={smpl_embed_dim}")
        transformer = CogVideoXTransformer3DModelWithAdaLNPose.from_pretrained(
            base_model_path,
            subfolder="transformer",
            smpl_pose_dim=smpl_pose_dim,
            smpl_embed_dim=smpl_embed_dim,
            torch_dtype=torch_dtype,
            **kwargs
        )
    elif approach == "adaln_pose_perframe":
        print(f"Using per-frame AdaLN pose conditioning with pose_dim={smpl_pose_dim}, embed_dim={smpl_embed_dim}")
        transformer = CogVideoXTransformer3DModelWithAdaLNPosePerFrame.from_pretrained(
            base_model_path,
            subfolder="transformer",
            smpl_pose_dim=smpl_pose_dim,
            smpl_embed_dim=smpl_embed_dim,
            torch_dtype=torch_dtype,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown approach: {approach}. Use 'adapter', 'concat', 'adaln_pose', or 'adaln_pose_perframe'")
    
    if hasattr(transformer, 'get_condition_info'):
        condition_info = transformer.get_condition_info()
        print(f"Transformer condition info: {condition_info}")
    
    return transformer 