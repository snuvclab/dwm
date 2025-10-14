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
import glob
import math
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple, Dict, Any
from safetensors.torch import load_file
from diffusers.models import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import is_torch_version, USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers, logging
from diffusers.models.embeddings import CogVideoXPatchEmbed, TimestepEmbedding, Timesteps, apply_rotary_emb
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.attention_processor import CogVideoXAttnProcessor2_0
from diffusers.models.attention_processor import AttentionProcessor
from diffusers.models.normalization import AdaLayerNorm, CogVideoXLayerNormZero

from diffusers import CogVideoXTransformer3DModel
from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXBlock

logger = logging.get_logger(__name__)

class SimpleAdapter(nn.Module):
    """
    Simple adapter for control signals based on Wan camera adapter.
    Uses PixelUnshuffle and Conv2d to reduce spatial dimensions.
    """
    def __init__(self, in_dim, out_dim, kernel_size, stride, num_residual_blocks=1):
        super(SimpleAdapter, self).__init__()
        
        # Pixel Unshuffle: reduce spatial dimensions by a factor of 8
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor=8)
        
        # Convolution: reduce spatial dimensions by a factor of 2 (without overlap)
        self.conv = nn.Conv2d(in_dim * 64, out_dim, kernel_size=kernel_size, stride=stride, padding=0)
        
        # Residual blocks for feature extraction
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(out_dim) for _ in range(num_residual_blocks)]
        )

    def forward(self, x):
        # Reshape to merge the frame dimension into batch
        bs, c, f, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(bs * f, c, h, w)
        
        # Pixel Unshuffle operation
        x_unshuffled = self.pixel_unshuffle(x)
        
        # Convolution operation
        x_conv = self.conv(x_unshuffled)
        
        # Feature extraction with residual blocks
        out = self.residual_blocks(x_conv)
        
        # Reshape to restore original bf dimension
        out = out.view(bs, f, out.size(1), out.size(2), out.size(3))
        
        # Permute dimensions to reorder (if needed), e.g., swap channels and feature frames
        out = out.permute(0, 2, 1, 3, 4)

        return out

class ResidualBlock(nn.Module):
    """Residual block for SimpleAdapter."""
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        return out

def reshape_tensor(tensor, num_heads):
    """Reshape tensor for multi-head attention."""
    batch_size, seq_len, dim = tensor.shape
    head_dim = dim // num_heads
    return tensor.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

class TripleAttention(Attention):
    """
    Attention module that processes three hidden states simultaneously:
    - hidden_states: main video hidden states
    - encoder_hidden_states: text hidden states  
    - cond_hidden_states: condition hidden states
    
    All three states participate in the attention mechanism with the same rotary embeddings.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    # def forward(
    #     self,
    #     hidden_states: torch.Tensor,
    #     encoder_hidden_states: torch.Tensor,
    #     cond_hidden_states: torch.Tensor,
    #     image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    #     attention_kwargs: Optional[Dict[str, Any]] = None,
    # ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #     """
    #     Forward pass for triple attention.
        
    #     Args:
    #         hidden_states: Main video hidden states [B, seq_len, dim]
    #         encoder_hidden_states: Text hidden states [B, text_seq_len, dim]
    #         cond_hidden_states: Condition hidden states [B, cond_seq_len, dim]
    #         image_rotary_emb: Rotary embeddings for video sequences (applied to hidden_states and cond_hidden_states)
    #         attention_kwargs: Additional attention arguments
            
    #     Returns:
    #         Tuple of (attended_hidden_states, attended_encoder_hidden_states, attended_cond_hidden_states)
    #     """
    #     # Concatenate video states (hidden_states + cond_hidden_states) for joint attention
    #     # This allows image_rotary_emb to be applied to both video sequences
    #     combined_video_states = torch.cat([hidden_states, cond_hidden_states], dim=1)
        
    #     # Concatenate image_rotary_emb to match the combined video states size
    #     if image_rotary_emb is not None:
    #         cos_emb, sin_emb = image_rotary_emb
    #         combined_cos_emb = torch.cat([cos_emb, cos_emb], dim=0)
    #         combined_sin_emb = torch.cat([sin_emb, sin_emb], dim=0)
    #         combined_image_rotary_emb = (combined_cos_emb, combined_sin_emb)
    #     else:
    #         combined_image_rotary_emb = None
        
    #     # Apply attention with combined video states as hidden_states and text as encoder_hidden_states
    #     attended_combined, attended_encoder_hidden_states = super().forward(
    #         hidden_states=combined_video_states,
    #         encoder_hidden_states=encoder_hidden_states,
    #         image_rotary_emb=combined_image_rotary_emb,
    #         **attention_kwargs or {}
    #     )
        
    #     # Split back into individual states
    #     seq_len = hidden_states.shape[1]
    #     cond_seq_len = cond_hidden_states.shape[1]
        
    #     attended_hidden_states = attended_combined[:, :seq_len]
    #     attended_cond_hidden_states = attended_combined[:, seq_len:]
        
    #     return attended_hidden_states, attended_encoder_hidden_states, attended_cond_hidden_states
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        cond_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **cross_attention_kwargs,
    ) -> torch.Tensor:
        r"""
        The forward method of the `Attention` class.

        Args:
            hidden_states (`torch.Tensor`):
                The hidden states of the query.
            encoder_hidden_states (`torch.Tensor`, *optional*):
                The hidden states of the encoder.
            cond_hidden_states (`torch.Tensor`, *optional*):
                The hidden states of the condition.
            attention_mask (`torch.Tensor`, *optional*):
                The attention mask to use. If `None`, no mask is applied.
            **cross_attention_kwargs:
                Additional keyword arguments to pass along to the cross attention.

        Returns:
            `torch.Tensor`: The output of the attention layer.
        """
        # The `Attention` class can call different attention processors / attention functions
        # here we simply pass along all tensors to the selected processor class
        # For standard processors that are defined here, `**cross_attention_kwargs` is empty

        attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
        quiet_attn_parameters = {"ip_adapter_masks", "ip_hidden_states"}
        unused_kwargs = [
            k for k, _ in cross_attention_kwargs.items() if k not in attn_parameters and k not in quiet_attn_parameters
        ]
        if len(unused_kwargs) > 0:
            logger.warning(
                f"cross_attention_kwargs {unused_kwargs} are not expected by {self.processor.__class__.__name__} and will be ignored."
            )
        cross_attention_kwargs = {k: w for k, w in cross_attention_kwargs.items() if k in attn_parameters}

        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            cond_hidden_states=cond_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )

class CogVideoXTripleAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention for the CogVideoX model. It applies a rotary embedding on
    query and key vectors, but does not include spatial normalization.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("CogVideoXTripleAttnProcessor requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        cond_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        cond_key_cache: Optional[torch.Tensor] = None,
        cond_value_cache: Optional[torch.Tensor] = None,
        return_cond_cache: bool = False,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)

        q_hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        kv_hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        batch_size, sequence_length, _ = q_hidden_states.shape

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(q_hidden_states)
        key = attn.to_k(kv_hidden_states)
        value = attn.to_v(kv_hidden_states)
        
        # Handle condition key/value caching
        if cond_key_cache is None or cond_value_cache is None:
            # First time: compute cond key/value and cache them
            cond_key = attn.to_k(cond_hidden_states)
            cond_value = attn.to_v(cond_hidden_states)
            # Concatenate with main key/value
            key = torch.cat([key, cond_key], dim=1)
            value = torch.cat([value, cond_value], dim=1)
        else:
            # Use cached cond key/value
            key = torch.cat([key, cond_key_cache], dim=1)
            value = torch.cat([value, cond_value_cache], dim=1)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
            if not attn.is_cross_attention:
                cos_emb, sin_emb = image_rotary_emb
                combined_cos_emb = torch.cat([cos_emb, cos_emb], dim=0)
                combined_sin_emb = torch.cat([sin_emb, sin_emb], dim=0)
                combined_image_rotary_emb = (combined_cos_emb, combined_sin_emb)
                key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], combined_image_rotary_emb)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
        
        if return_cond_cache and (cond_key_cache is None or cond_value_cache is None):
            # Return computed cond key/value for caching
            return hidden_states, encoder_hidden_states, cond_key, cond_value
        else:
            return hidden_states, encoder_hidden_states

class TripleCogVideoXLayerNormZero(CogVideoXLayerNormZero):
    """
    Extended CogVideoXLayerNormZero that processes three hidden states simultaneously.
    """
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        cond_hidden_states: torch.Tensor,
        temb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for triple layer norm zero.
        
        Args:
            hidden_states: Main video hidden states
            encoder_hidden_states: Text hidden states
            cond_hidden_states: Condition hidden states
            temb: Time embedding
            
        Returns:
            Tuple of (norm_hidden_states, norm_encoder_hidden_states, norm_cond_hidden_states, gate_msa, enc_gate_msa, cond_gate_msa)
        """
        # Apply the original layer norm to each state separately
        norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = super().forward(
            hidden_states, encoder_hidden_states, temb
        )
        
        # Apply the same normalization to cond_hidden_states
        # Use the same time conditioning for consistency
        shift, scale, gate, enc_shift, enc_scale, enc_gate = self.linear(self.silu(temb)).chunk(6, dim=1)
        
        # Apply modulation to cond_hidden_states
        norm_cond_hidden_states = self.norm(cond_hidden_states) * (1 + scale)[:, None, :] + shift[:, None, :]
        
        return norm_hidden_states, norm_encoder_hidden_states, norm_cond_hidden_states, gate_msa, enc_gate_msa

class TripleCogVideoXBlock(nn.Module):
    """
    CogVideoX Block that processes three hidden states simultaneously:
    - hidden_states: main video hidden states
    - encoder_hidden_states: text hidden states
    - cond_hidden_states: condition hidden states
    
    This block applies the same attention and feed-forward operations to all three states
    with shared time embeddings and rotary embeddings.
    """
    
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        time_embed_dim: int,
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

        # 1. Self Attention with triple attention
        self.norm1 = TripleCogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)
        self.attn1 = TripleAttention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=attention_bias,
            out_bias=attention_out_bias,
            processor=CogVideoXTripleAttnProcessor2_0(),
        )

        # 2. Feed Forward with triple processing
        self.norm2 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)
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
        cond_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cond_key_cache: Optional[torch.Tensor] = None,
        cond_value_cache: Optional[torch.Tensor] = None,
        return_cond_cache: bool = False,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for triple CogVideoX block.
        
        Args:
            hidden_states: Main video hidden states [B, seq_len, dim]
            encoder_hidden_states: Text hidden states [B, text_seq_len, dim]
            cond_hidden_states: Condition hidden states [B, cond_seq_len, dim]
            temb: Time embedding [B, time_embed_dim]
            image_rotary_emb: Rotary embeddings
            attention_kwargs: Additional attention arguments
            
        Returns:
            Tuple of (output_hidden_states, output_encoder_hidden_states, output_cond_hidden_states)
        """
        attention_kwargs = attention_kwargs or {}

        # 1. Self Attention with triple attention
        norm_hidden_states, norm_encoder_hidden_states, norm_cond_hidden_states, gate_msa, enc_gate_msa = self.norm1(
            hidden_states, encoder_hidden_states, cond_hidden_states, temb
        )

        # Apply triple attention with caching
        attn_result = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            cond_hidden_states=norm_cond_hidden_states,
            image_rotary_emb=image_rotary_emb,
            cond_key_cache=cond_key_cache,
            cond_value_cache=cond_value_cache,
            return_cond_cache=return_cond_cache,
            **attention_kwargs,
        )
        
        if return_cond_cache and len(attn_result) == 4:
            attn_hidden_states, attn_encoder_hidden_states, cond_key_cache, cond_value_cache = attn_result
        else:
            attn_hidden_states, attn_encoder_hidden_states = attn_result

        # Apply gating and residual connections
        hidden_states = hidden_states + gate_msa * attn_hidden_states
        encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states

        # 2. Feed Forward with triple processing
        norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = self.norm2(
            hidden_states, encoder_hidden_states, temb
        )

        # Apply feed-forward to each state separately
        ff_hidden_states = self.ff(norm_hidden_states)
        ff_encoder_hidden_states = self.ff(norm_encoder_hidden_states)

        # Apply gating and residual connections
        hidden_states = hidden_states + gate_ff * ff_hidden_states
        encoder_hidden_states = encoder_hidden_states + enc_gate_ff * ff_encoder_hidden_states

        if return_cond_cache:
            return hidden_states, encoder_hidden_states, cond_key_cache, cond_value_cache
        else:
            return hidden_states, encoder_hidden_states

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

class CogVideoXFunTransformer3DModelWithCondToken(CogVideoXFunTransformer3DModel):
    """
    VideoX-Fun Transformer model for video-like data with condition token support.
    
    This class extends the standard CogVideoXTransformer3DModel to support:
    1. Condition token processing with CogVideoXPatchEmbedWithCondToken
    2. Triple block support for processing three hidden states simultaneously
    3. VideoX-Fun specific normalization logic
    """
    
    def __init__(self, *args, condition_channels: int = 16, use_zero_proj: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Replace patch_embed with adapter version if condition_channels > 0
        if condition_channels > 0:
            self._setup_adapter_patch_embed(condition_channels, use_zero_proj)
            self.condition_channels = condition_channels
        else:
            self.condition_channels = 0
            
        # Always use TripleCogVideoXBlock for processing three hidden states
        self._setup_triple_blocks()
        self.use_triple_blocks = True

    def _setup_adapter_patch_embed(self, condition_channels: int, 
                                   use_zero_proj: bool = False):
        """Replace patch_embed with CogVideoXPatchEmbedWithAdapter.
        
        This creates a new adapter-based patch embed that processes conditions
        separately from the main image embeddings.
        """
        print(f"Setting up adapter-based patch embed for {condition_channels} condition channels")
        
        # Get current patch_embed config
        current_patch_embed = self.patch_embed
        
        # Create new adapter patch embed with same config
        adapter_patch_embed = CogVideoXPatchEmbedWithCondToken(
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
    
    def _setup_triple_blocks(self):
        """Replace CogVideoXBlock with TripleCogVideoXBlock.
        
        This creates new triple blocks that can process three hidden states simultaneously.
        """
        print("Setting up triple blocks for processing three hidden states")
        
        # Replace each CogVideoXBlock with TripleCogVideoXBlock
        for i, block in enumerate(self.transformer_blocks):
            # Use self.config which contains all the necessary parameters
            innder_dim = self.config.num_attention_heads * self.config.attention_head_dim
            triple_block = TripleCogVideoXBlock(
                dim=innder_dim,
                num_attention_heads=self.config.num_attention_heads,
                attention_head_dim=self.config.attention_head_dim,
                time_embed_dim=self.config.time_embed_dim,
                dropout=self.config.dropout,
                activation_fn=self.config.activation_fn,
                attention_bias=self.config.attention_bias,
                norm_elementwise_affine=self.config.norm_elementwise_affine,
                norm_eps=self.config.norm_eps,
            )
                
            # Copy weights from original block to triple block
            self._copy_block_weights(block, triple_block)
            
            # Replace the block
            self.transformer_blocks[i] = triple_block
            print(f"✅ Replaced block {i} with TripleCogVideoXBlock")
        
        print(f"✅ Successfully replaced {len(self.transformer_blocks)} blocks with triple blocks")
    
    def _copy_block_weights(self, original_block, triple_block):
        """Copy weights from original CogVideoXBlock to TripleCogVideoXBlock by iterating through parameters."""
        with torch.no_grad():
            # Get state dicts for both blocks
            original_state_dict = original_block.state_dict()
            triple_state_dict = triple_block.state_dict()
            
            # Copy matching parameters
            for name, param in original_state_dict.items():
                if name in triple_state_dict:
                    if param.shape == triple_state_dict[name].shape:
                        triple_state_dict[name].copy_(param)
                    else:
                        print(f"⚠️ Shape mismatch for {name}: {param.shape} vs {triple_state_dict[name].shape}")
                else:
                    print(f"⚠️ Parameter {name} not found in triple block")
            
            print(f"✅ Copied weights from original block to triple block")
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path=None,
        base_model_name_or_path="alibaba-pai/CogVideoX-Fun-V1.1-5b-InP",
        subfolder="transformer",
        condition_channels: Optional[int] = 16,
        use_zero_proj: bool = True,
        **kwargs
    ):
        """
        Load a CogVideoXFunTransformer3DModelWithCondToken from a pretrained model.

        Args:
            pretrained_model_name_or_path (str): Fine-tuned checkpoint path
            base_model_name_or_path (str): Base CogVideoX-Fun model path
            condition_channels (int): Number of condition channels
            use_zero_proj (bool): Whether to use zero projection
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
                # Override with provided parameters
                config.update({
                    'condition_channels': condition_channels or config.get('condition_channels', 16),
                    'use_zero_proj': use_zero_proj,
                })
                model = cls(**config)
            else:
                # Use base config with provided parameters
                model = cls(
                    condition_channels=condition_channels,
                    use_zero_proj=use_zero_proj
                )

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
            condition_channels = getattr(base_model.config, "condition_channels", 16)
            
            # Create extended model with provided parameters
            fun_model = cls(
                **base_model.config,
                condition_channels=condition_channels,
                use_zero_proj=use_zero_proj
            )
            
            # Load base weights into extended model
            missing, unexpected = fun_model.load_state_dict(base_model.state_dict(), strict=False)
            
            # Copy weights for newly added layers (triple blocks)
            print("🔄 Copying weights for triple blocks...")
            with torch.no_grad():
                for i, (original_block, triple_block) in enumerate(zip(base_model.transformer_blocks, fun_model.transformer_blocks)):
                    if hasattr(triple_block, '_copy_block_weights'):
                        fun_model._copy_block_weights(original_block, triple_block)
                        print(f"✅ Copied weights for triple block {i}")
            
            if missing:
                print(f"⚠️ Missing keys (new condition/triple layers): {missing}")
            if unexpected:
                print(f"⚠️ Unexpected keys (replaced existing layers): {unexpected}")
            
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
            # if control_latents is not None:
            #     control_latents = torch.concat([control_latents, torch.zeros_like(control_latents)], dim=1)
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
        # if control_latents is not None:
        #     hidden_states = torch.concat([hidden_states, control_latents], 2)
        
        # Store original video dimensions before patch_embed
        batch_size, num_frames, channels, height, width = hidden_states.shape
        
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states, control_latents)
        hidden_states = self.embedding_dropout(hidden_states)
        
        text_seq_length = encoder_hidden_states.shape[1]
        # Calculate video sequence length (height * width * num_frames)
        video_seq_length = (height // self.config.patch_size) * (width // self.config.patch_size) * num_frames
        
        # Split concatenated embeddings: [text, video, cond]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        video_hidden_states = hidden_states[:, text_seq_length:text_seq_length + video_seq_length]
        cond_hidden_states = hidden_states[:, text_seq_length + video_seq_length:]
        
        # Use video_hidden_states as the main hidden_states for transformer blocks
        hidden_states = video_hidden_states
        
        # Transformer blocks with caching
        cond_key_cache = None
        cond_value_cache = None
        
        for i, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                
                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                if i == 0:
                    # First block: compute and cache cond key/value
                    hidden_states, encoder_hidden_states, cond_key_cache, cond_value_cache = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        cond_hidden_states,
                        emb,
                        image_rotary_emb,
                        cond_key_cache,
                        cond_value_cache,
                        True,  # return_cond_cache=True
                        **ckpt_kwargs,
                    )
                else:
                    # Subsequent blocks: use cached cond key/value
                    hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        cond_hidden_states,
                        emb,
                        image_rotary_emb,
                        cond_key_cache,
                        cond_value_cache,
                        False,  # return_cond_cache=False
                        **ckpt_kwargs,
                    )
            else:
                if i == 0:
                    # First block: compute and cache cond key/value
                    hidden_states, encoder_hidden_states, cond_key_cache, cond_value_cache = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        cond_hidden_states=cond_hidden_states,
                        temb=emb,
                        image_rotary_emb=image_rotary_emb,
                        cond_key_cache=cond_key_cache,
                        cond_value_cache=cond_value_cache,
                        return_cond_cache=True,
                    )
                else:
                    # Subsequent blocks: use cached cond key/value
                    hidden_states, encoder_hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        cond_hidden_states=cond_hidden_states,
                        temb=emb,
                        image_rotary_emb=image_rotary_emb,
                        cond_key_cache=cond_key_cache,
                        cond_value_cache=cond_value_cache,
                        return_cond_cache=False,
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
    4. Optional control adapter for additional control signals
    """
    
    def __init__(self, *args, condition_channels: int = 0, add_control_adapter: bool = False, 
                 in_dim_control_adapter: int = 12, **kwargs):
        # Store original in_channels before modification
        original_in_channels = kwargs.get("in_channels", 16)
        self.original_in_channels = original_in_channels
        
        # self.add_noise_in_inpaint_model = kwargs.pop("add_noise_in_inpaint_model", False)
        # self.add_control_adapter = add_control_adapter
        self.register_to_config(
            add_noise_in_inpaint_model=kwargs.pop("add_noise_in_inpaint_model", False),
            add_control_adapter=add_control_adapter,
        )
        super().__init__(*args, **kwargs)
        
        # Setup conditional channels if specified
        if condition_channels > 0:
            self._setup_condition_channels(condition_channels)
            self.condition_channels = condition_channels
        else:
            self.condition_channels = 0
        
        # Setup control adapter if specified
        if add_control_adapter:
            # Get patch size from config
            patch_size = self.config.patch_size if hasattr(self.config, 'patch_size') else 2
            # Get hidden dimension
            num_attention_heads = self.config.num_attention_heads if hasattr(self.config, 'num_attention_heads') else 30
            attention_head_dim = self.config.attention_head_dim if hasattr(self.config, 'attention_head_dim') else 64
            out_dim = num_attention_heads * attention_head_dim
            
            self.control_adapter = SimpleAdapter(
                in_dim=in_dim_control_adapter,
                out_dim=out_dim,
                kernel_size=(patch_size, patch_size),
                stride=(patch_size, patch_size),
                num_residual_blocks=1
            )
            print(f"✅ Added control adapter: {in_dim_control_adapter} -> {out_dim} channels")
        else:
            self.control_adapter = None
    
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
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 12,
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
                model = cls(
                    condition_channels=condition_channels,
                    add_control_adapter=add_control_adapter,
                    in_dim_control_adapter=in_dim_control_adapter
                )

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
            
            # Then create CogVideoXFun transformer and copy state_dict
            fun_model = CogVideoXFunTransformer3DModel(**base_model.config)
            fun_model.load_state_dict(base_model.state_dict())
            base_model = fun_model

            # Determine condition_channels
            if condition_channels is None:
                condition_channels = getattr(base_model.config, "condition_channels", 32)

            # Create extended model
            model = cls(
                **base_model.config, 
                condition_channels=0,
                add_control_adapter=add_control_adapter,
                in_dim_control_adapter=in_dim_control_adapter
            )

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
        adapter_control: Optional[torch.Tensor] = None,
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
            adapter_control: Additional control signal for adapter (e.g., camera params) [B, C, F, H, W]
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
        
        # If control adapter is enabled and adapter_control is provided, process it
        if self.control_adapter is not None and adapter_control is not None:
            # Store for later use after patch embedding
            batch_size, num_frames, channels, height, width = hidden_states.shape
            
            # Time embedding
            timesteps = timestep
            t_emb = self.time_proj(timesteps)
            t_emb = t_emb.to(dtype=hidden_states.dtype)
            emb = self.time_embedding(t_emb, timestep_cond)
            
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
            
            # Patch embedding with VideoX-Fun specific concatenation
            if inpaint_latents is not None:
                hidden_states = torch.concat([hidden_states, inpaint_latents], 2)
            if control_latents is not None:
                hidden_states = torch.concat([hidden_states, control_latents], 2)
            
            hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
            hidden_states = self.embedding_dropout(hidden_states)
            
            # Apply control adapter
            adapter_features = self.control_adapter(adapter_control)  # [B, out_dim, F, H', W']
            # Flatten adapter features: [B, out_dim, F, H', W'] -> [B, F*H'*W', out_dim]
            adapter_features = adapter_features.flatten(2).transpose(1, 2)
            
            text_seq_length = encoder_hidden_states.shape[1]
            encoder_hidden_states = hidden_states[:, :text_seq_length]
            hidden_states = hidden_states[:, text_seq_length:]
            
            # Add adapter features to hidden states
            if adapter_features.shape[1] == hidden_states.shape[1]:
                hidden_states = hidden_states + adapter_features
            else:
                print(f"⚠️ Adapter features shape mismatch: {adapter_features.shape} vs {hidden_states.shape}")
            
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
        else:
            # No control adapter, use parent class forward method
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
        use_zero_proj = kwargs.pop("use_zero_proj", False)
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
                if self.cond_zero_proj is not None:
                    cond_embeds = self.cond_zero_proj(cond_embeds)
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

        return embeds

class CogVideoXPatchEmbedWithCondToken(CogVideoXPatchEmbed):
    """
    VideoX-Fun Patch Embed with conditional input support via adapter.
    
    This class extends CogVideoXPatchEmbed to support conditional inputs
    by using cond token.
    """
    
    def __init__(self, *args, **kwargs):
        # Extract condition_channels from kwargs before passing to parent
        condition_channels = kwargs.pop("condition_channels", 16)
        use_zero_proj = kwargs.pop("use_zero_proj", False)
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

    def forward(self, text_embeds: torch.Tensor, image_embeds: torch.Tensor, 
                cond_embeds: torch.Tensor = None):
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
        
            if cond_embeds is not None:
                cond_channels = cond_embeds.shape[2]
                cond_embeds = cond_embeds.reshape(-1, cond_channels, height, width)
                cond_embeds = self.cond_proj(cond_embeds)
                cond_embeds = cond_embeds.view(batch_size, num_frames, *cond_embeds.shape[1:])
                cond_embeds = cond_embeds.flatten(3).transpose(2, 3)  # [batch, num_frames, height x width, channels]
                cond_embeds = cond_embeds.flatten(1, 2)  # [batch, num_frames x height x width, channels]
                if self.cond_zero_proj is not None:
                    cond_embeds = self.cond_zero_proj(cond_embeds)
        
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

            if cond_embeds is not None:
                cond_channels = cond_embeds.shape[2]
                cond_embeds = cond_embeds.permute(0, 1, 3, 4, 2)
                cond_embeds = cond_embeds.reshape(
                    batch_size, num_frames // p_t, p_t, height // p, p, width // p, p, cond_channels
                )
                cond_embeds = cond_embeds.permute(0, 1, 3, 5, 7, 2, 4, 6).flatten(4, 7).flatten(1, 3)
                cond_embeds = self.cond_proj(cond_embeds)

        embeds = torch.cat(
            [text_embeds, image_embeds, cond_embeds], dim=1
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

        return embeds


class CogVideoXPatchEmbedWithAdapterV2(CogVideoXPatchEmbed):
    """
    VideoX-Fun Patch Embed with conditional input support via adapter.
    Adds GroupNorm and gating to stabilize conditioning signal.
    """

    def __init__(self, *args, **kwargs):
        condition_channels = kwargs.pop("condition_channels", None)
        super().__init__(*args, **kwargs)

        in_channels = kwargs.get("in_channels", 16)
        patch_size = kwargs.get("patch_size", 16)
        patch_size_t = kwargs.get("patch_size_t", None)
        embed_dim = kwargs.get("embed_dim", 16)
        bias = kwargs.get("bias", True)

        if condition_channels is None:
            condition_channels = in_channels

        # Adapter projection layer
        if patch_size_t is None:
            # CogVideoX 1.0
            self.cond_proj = nn.Conv2d(
                condition_channels,
                embed_dim,
                kernel_size=(patch_size, patch_size),
                stride=patch_size,
                bias=bias,
            )
        else:
            # CogVideoX 1.5
            self.cond_proj = nn.Linear(
                condition_channels * patch_size * patch_size * patch_size_t, embed_dim
            )

        # Zero-init for stable start
        with torch.no_grad():
            self.cond_proj.weight.zero_()
            if hasattr(self.cond_proj, "bias") and self.cond_proj.bias is not None:
                self.cond_proj.bias.zero_()

        # GroupNorm for cond stabilization
        self.cond_norm = nn.GroupNorm(1, embed_dim, affine=True)
        with torch.no_grad():
            self.cond_norm.weight.fill_(1.0)
            self.cond_norm.bias.zero_()

        # Learnable gate for gradual conditioning
        self.cond_gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        text_embeds: torch.Tensor,
        image_embeds: torch.Tensor,
        ref_image_embeds: torch.Tensor = None,
        cond_embeds: torch.Tensor = None,
    ):
        """
        Args:
            text_embeds (`torch.Tensor`): [B, seq_length, embed_dim]
            image_embeds (`torch.Tensor`): [B, F, C, H, W]
            ref_image_embeds (`torch.Tensor`, optional): [B, F, C, H, W]
            cond_embeds (`torch.Tensor`, optional): [B, F, C, H, W]
        """
        text_embeds = self.text_proj(text_embeds)
        batch_size, num_frames, channels, height, width = image_embeds.shape

        # ---- Patchify image + ref ----
        if self.patch_size_t is None:
            # Combine image & ref
            if ref_image_embeds is not None:
                combined_embeds = torch.cat([image_embeds, ref_image_embeds], dim=0)
            else:
                combined_embeds = image_embeds

            combined_embeds = combined_embeds.reshape(-1, channels, height, width)
            combined_embeds = self.proj(combined_embeds)

            # Split back
            if ref_image_embeds is not None:
                image_embeds, ref_image_embeds = torch.chunk(combined_embeds, 2, dim=0)
                image_embeds = image_embeds.view(batch_size, num_frames, *image_embeds.shape[1:])
                ref_image_embeds = ref_image_embeds.view(batch_size, num_frames, *ref_image_embeds.shape[1:])
            else:
                image_embeds = combined_embeds.view(batch_size, num_frames, *combined_embeds.shape[1:])

            # Flatten for transformer
            image_embeds = image_embeds.flatten(3).transpose(2, 3).flatten(1, 2)
            if ref_image_embeds is not None:
                ref_image_embeds = ref_image_embeds.flatten(3).transpose(2, 3).flatten(1, 2)

            # ---- Process conditioning ----
            if cond_embeds is not None:
                cond_channels = cond_embeds.shape[2]
                cond_embeds = cond_embeds.reshape(-1, cond_channels, height, width)
                cond_embeds = self.cond_proj(cond_embeds)  # [B*F, embed_dim, H/P, W/P]
                cond_embeds = cond_embeds.view(batch_size, num_frames, *cond_embeds.shape[1:])
                cond_embeds = cond_embeds.flatten(3).transpose(2, 3).flatten(1, 2)

                # Normalize + Gate
                # GroupNorm expects [batch, channels, ...] but we have [batch, seq_len, channels]
                # Reshape to [batch, channels, seq_len] for GroupNorm
                batch_size, seq_len, channels = cond_embeds.shape
                cond_embeds = cond_embeds.transpose(1, 2)  # [batch, channels, seq_len]
                cond_embeds = self.cond_norm(cond_embeds)
                cond_embeds = cond_embeds.transpose(1, 2)  # [batch, seq_len, channels]
                image_embeds = image_embeds + self.cond_gate * cond_embeds

        else:
            # 3D patchify for CogVideoX 1.5
            p = self.patch_size
            p_t = self.patch_size_t

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

                # Normalize + Gate
                # GroupNorm expects [batch, channels, ...] but we have [batch, seq_len, channels]
                # Reshape to [batch, channels, seq_len] for GroupNorm
                batch_size, seq_len, channels = cond_embeds.shape
                cond_embeds = cond_embeds.transpose(1, 2)  # [batch, channels, seq_len]
                cond_embeds = self.cond_norm(cond_embeds)
                cond_embeds = cond_embeds.transpose(1, 2)  # [batch, seq_len, channels]
                image_embeds = image_embeds + self.cond_gate * cond_embeds

        # ---- Merge text & image tokens ----
        embeds = torch.cat([text_embeds, image_embeds], dim=1).contiguous()

        # ---- Positional embeddings ----
        if self.use_positional_embeddings or self.use_learned_positional_embeddings:
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

        return embeds

class CogVideoXPatchEmbedWithAdapterV3(CogVideoXPatchEmbed):
    """
    VideoX-Fun Patch Embed with conditional input support via adapter.
    Uses Conv3D for temporal processing and zero-initialized projection.
    """

    def __init__(self, *args, **kwargs):
        condition_channels = kwargs.pop("condition_channels", None)
        use_enhanced_processing = kwargs.pop("use_enhanced_processing", False)
        super().__init__(*args, **kwargs)

        in_channels = kwargs.get("in_channels", 16)
        patch_size = kwargs.get("patch_size", 16)
        patch_size_t = kwargs.get("patch_size_t", None)
        embed_dim = kwargs.get("embed_dim", 16)
        bias = kwargs.get("bias", True)

        if condition_channels is None:
            condition_channels = in_channels

        self.condition_channels = condition_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.use_enhanced_processing = use_enhanced_processing

        # 3D convolution for condition processing
        if patch_size_t is None:
            # CogVideoX 1.0 - Use 2D conv
            self.add_conv_in = nn.Conv2d(
                condition_channels, embed_dim,
                kernel_size=patch_size, stride=patch_size, bias=bias
            )
        else:
            # CogVideoX 1.5 - Use 3D conv
            self.add_conv_in = nn.Conv3d(
                condition_channels, embed_dim,
                kernel_size=(patch_size, patch_size, patch_size_t), 
                stride=(patch_size, patch_size, patch_size_t), bias=bias
            )

        # Enhanced processing layers (optional)
        if use_enhanced_processing:
            self.add_norm = nn.GroupNorm(32, embed_dim)
            self.add_act = nn.SiLU()
            print(f"✅ Enhanced processing enabled: GroupNorm + SiLU")
        else:
            self.add_norm = None
            self.add_act = None
            print(f"✅ Standard processing: Conv3D + Zero projection only")

        # Zero-initialized projection
        from diffusers.models.controlnet import zero_module
        self.add_proj = zero_module(nn.Linear(embed_dim, embed_dim))

    def forward(
        self,
        text_embeds: torch.Tensor,
        image_embeds: torch.Tensor,
        cond_embeds: torch.Tensor = None,
    ):
        """
        Args:
            text_embeds (`torch.Tensor`): [B, seq_length, embed_dim]
            image_embeds (`torch.Tensor`): [B, F, C, H, W]
            cond_embeds (`torch.Tensor`, optional): [B, F, C, H, W]
        """
        text_embeds = self.text_proj(text_embeds)
        batch_size, num_frames, channels, height, width = image_embeds.shape

        # ---- Process main image ----
        if self.patch_size_t is None:
            # CogVideoX 1.0 - 2D processing
            image_embeds = image_embeds.reshape(-1, channels, height, width)
            (-1, channels, height, width)
            image_embeds = self.proj(image_embeds)
            image_embeds = image_embeds.view(batch_size, num_frames, *image_embeds.shape[1:])
            image_embeds = image_embeds.flatten(3).transpose(2, 3).flatten(1, 2)  # [B, F*H*W, embed_dim]
        else:
            # CogVideoX 1.5 - 3D processing
            p = self.patch_size
            p_t = self.patch_size_t
            image_embeds = image_embeds.permute(0, 1, 3, 4, 2)
            image_embeds = image_embeds.reshape(
                -1, num_frames // p_t, p_t, height // p, p, width // p, p, channels
            )
            image_embeds = image_embeds.permute(0, 1, 3, 5, 7, 2, 4, 6).flatten(4, 7).flatten(1, 3)
            image_embeds = self.proj(image_embeds)

        # ---- Process condition ----
        if cond_embeds is not None:
            if self.patch_size_t is None:
                # CogVideoX 1.0 - 2D conv
                cond_embeds = cond_embeds.reshape(-1, self.condition_channels, height, width)
                add_cond = self.add_conv_in(cond_embeds)  # [B*F, embed_dim, H/P, W/P]
                add_cond = add_cond.view(batch_size, num_frames, *add_cond.shape[1:])
                add_cond = add_cond.flatten(3).transpose(2, 3).flatten(1, 2)  # [B, F*H*W, embed_dim]
            else:
                # CogVideoX 1.5 - 3D conv
                p = self.patch_size
                p_t = self.patch_size_t
                cond_embeds = cond_embeds.permute(0, 1, 3, 4, 2)
                cond_embeds = cond_embeds.reshape(
                    batch_size, num_frames // p_t, p_t, height // p, p, width // p, p, self.condition_channels
                )
                cond_embeds = cond_embeds.permute(0, 1, 3, 5, 7, 2, 4, 6).flatten(4, 7).flatten(1, 3)
                add_cond = self.add_conv_in(cond_embeds)  # [B, embed_dim, ...]
                add_cond = add_cond.flatten(2).transpose(1, 2)  # [B, seq_len, embed_dim]

            # Enhanced processing (optional)
            if self.use_enhanced_processing:
                # Apply GroupNorm + SiLU
                add_cond = self.add_norm(add_cond)
                add_cond = self.add_act(add_cond)

            # Apply zero-initialized projection and add to hidden states
            image_embeds = image_embeds + self.add_proj(add_cond)

        # ---- Merge text & image tokens ----
        embeds = torch.cat([text_embeds, image_embeds], dim=1).contiguous()

        # ---- Positional embeddings ----
        if self.use_positional_embeddings or self.use_learned_positional_embeddings:
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

        return embeds
            
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
    
    def __init__(self, *args, condition_channels: int = 0, adapter_version: str = "v1", use_enhanced_processing: bool = False, **kwargs):
        self.add_noise_in_inpaint_model = kwargs.pop("add_noise_in_inpaint_model", False)
        self.adapter_version = adapter_version
        self.use_enhanced_processing = use_enhanced_processing
        super().__init__(*args, **kwargs)
        
        # Replace patch_embed with adapter version if condition_channels > 0
        if condition_channels > 0:
            self._setup_adapter_patch_embed(condition_channels, adapter_version)
            self.condition_channels = condition_channels
        else:
            self.condition_channels = 0
    
    def _setup_adapter_patch_embed(self, condition_channels: int, adapter_version: str = "v1"):
        """Replace patch_embed with CogVideoXPatchEmbedWithAdapter, V2, or V3.
        
        This creates a new adapter-based patch embed that processes conditions
        separately from the main image embeddings.
        
        Args:
            condition_channels: Number of condition channels
            adapter_version: Version of adapter to use ("v1", "v2", or "v3")
        """
        print(f"Setting up adapter-based patch embed (version {adapter_version}) for {condition_channels} condition channels")
        
        # Get current patch_embed config
        current_patch_embed = self.patch_embed
        
        # Choose adapter class based on version
        if adapter_version == "v2":
            adapter_class = CogVideoXPatchEmbedWithAdapterV2
            print("  Using CogVideoXPatchEmbedWithAdapterV2 (with GroupNorm and gating)")
        elif adapter_version in ["v3", "v4"]:
            adapter_class = CogVideoXPatchEmbedWithAdapterV3
            if adapter_version == "v3":
                print("  Using CogVideoXPatchEmbedWithAdapterV3 (with Conv3D and zero projection)")
            else:  # v4
                print("  Using CogVideoXPatchEmbedWithAdapterV3 (with Conv3D, GroupNorm, SiLU by default)")
        else:
            adapter_class = CogVideoXPatchEmbedWithAdapter
            print("  Using CogVideoXPatchEmbedWithAdapter (standard version)")
        
        # Create new adapter patch embed with same config
        adapter_kwargs = {
            'condition_channels': condition_channels,
            'in_channels': current_patch_embed.proj.in_channels,
            'patch_size': getattr(current_patch_embed, 'patch_size', 16),
            'patch_size_t': getattr(current_patch_embed, 'patch_size_t', None),
            'embed_dim': current_patch_embed.proj.out_channels,
            'bias': current_patch_embed.proj.bias is not None,
            'use_positional_embeddings': getattr(current_patch_embed, 'use_positional_embeddings', True),
            'use_learned_positional_embeddings': getattr(current_patch_embed, 'use_learned_positional_embeddings', False),
            'sample_height': getattr(current_patch_embed, 'sample_height', None),
            'sample_width': getattr(current_patch_embed, 'sample_width', None),
            'sample_frames': getattr(current_patch_embed, 'sample_frames', None),
            'temporal_compression_ratio': getattr(current_patch_embed, 'temporal_compression_ratio', 1),
        }
        
        # Add enhanced processing flag for V3 and V4 adapters
        if adapter_version in ["v3", "v4"]:
            # V3: use_enhanced_processing=False, V4: use_enhanced_processing=True
            if adapter_version == "v3":
                adapter_kwargs['use_enhanced_processing'] = False
            else:  # v4
                adapter_kwargs['use_enhanced_processing'] = True
        
        adapter_patch_embed = adapter_class(**adapter_kwargs)
        
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
            adapter_version=adapter_version,
        )
        
        print(f"✅ Successfully replaced patch_embed with adapter version {adapter_version}")
        print(f"  Main projection: {adapter_patch_embed.proj}")
        if hasattr(adapter_patch_embed, 'cond_proj'):
            print(f"  Condition projection: {adapter_patch_embed.cond_proj}")
        if hasattr(adapter_patch_embed, 'add_conv_in'):
            print(f"  Condition conv: {adapter_patch_embed.add_conv_in}")
        print(f"  Text projection: {adapter_patch_embed.text_proj}")
        if adapter_version == "v2":
            print(f"  Condition norm: {adapter_patch_embed.cond_norm}")
            print(f"  Condition gate: {adapter_patch_embed.cond_gate}")
        elif adapter_version == "v3":
            print(f"  Zero projection: {adapter_patch_embed.add_proj}")
        elif adapter_version == "v4":
            print(f"  Zero projection: {adapter_patch_embed.add_proj}")
            if hasattr(adapter_patch_embed, 'add_norm'):
                print(f"  GroupNorm: {adapter_patch_embed.add_norm}")
            if hasattr(adapter_patch_embed, 'add_act'):
                print(f"  SiLU: {adapter_patch_embed.add_act}")
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path=None,
        base_model_name_or_path="alibaba-pai/CogVideoX-Fun-V1.1-5b-InP",
        subfolder="transformer",
        condition_channels: Optional[int] = None,
        adapter_version: str = "v1",
        use_enhanced_processing: bool = False,
        **kwargs
    ):
        """
        Load a CogVideoXFunTransformer3DModelWithAdapter from a pretrained model.

        Args:
            pretrained_model_name_or_path (str): Fine-tuned checkpoint path
            base_model_name_or_path (str): Base CogVideoX-Fun model path
            condition_channels (int): Number of extra condition channels to add
            adapter_version (str): Version of adapter to use ("v1" or "v2")
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
                model = cls(condition_channels=condition_channels, adapter_version=adapter_version)

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
            
            # Then create CogVideoXFun transformer and copy state_dict
            fun_model = CogVideoXFunTransformer3DModel(**base_model.config)
            fun_model.load_state_dict(base_model.state_dict())
            base_model = fun_model

            # Determine condition_channels
            if condition_channels is None:
                condition_channels = getattr(base_model.config, "condition_channels", 16)

            # Create extended model
            model = cls(**base_model.config, condition_channels=condition_channels, adapter_version=adapter_version, use_enhanced_processing=use_enhanced_processing)

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
            "approach": "adapter",
            "has_conditions": hasattr(self, 'condition_channels') and self.condition_channels > 0,
            "condition_channels": getattr(self, 'condition_channels', 0),
            "adapter_version": getattr(self, 'adapter_version', 'v1'),
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
        
        if self.adapter_version == "v1" and control_latents is not None:
            # concat control latents if not using adapter
            hidden_states = torch.concat([hidden_states, control_latents], 2)

        # 2. Patch embedding with adapter support
        patch_embed_input = {'text_embeds': encoder_hidden_states, 
                             'image_embeds': hidden_states}
        if self.adapter_version == "v1" and control_latents is not None:
            patch_embed_input['cond_embeds'] = control_latents
    
        # Use adapter patch embed
        hidden_states = self.patch_embed(**patch_embed_input)
        
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


class CrossTransformer3DModel(CogVideoXFunTransformer3DModel):
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
        is_train_cross: bool = True,
        cross_attn_interval: int = 2,
        cross_attn_dim_head: int = 128,
        cross_attn_num_heads: int = 16,
        cross_attn_kv_dim: int = None,
        **kwargs
    ):
        """
        Load a CrossTransformer3DModel from a pretrained model.

        Args:
            pretrained_model_name_or_path (str): Fine-tuned checkpoint path
            base_model_name_or_path (str): Base CogVideoX-Fun model path
            is_train_cross (bool): Whether to train cross attention
            cross_attn_interval (int): Interval for cross attention
            cross_attn_dim_head (int): Dimension of head for cross attention
            cross_attn_num_heads (int): Number of heads for cross attention
            cross_attn_kv_dim (int): Dimension of key/value for cross attention
        """
        if pretrained_model_name_or_path is not None:
            # === Load fine-tuned checkpoint ===
            print(f"📥 Loading fine-tuned cross-attention transformer: {pretrained_model_name_or_path}")

            # 1) Create child class structure first
            config_path = os.path.join(pretrained_model_name_or_path, subfolder, "config.json")
            if os.path.exists(config_path):
                import json
                with open(config_path, "r") as f:
                    config = json.load(f)
                model = cls(**config)
            else:
                # Use base config
                model = cls(is_train_cross=is_train_cross,
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
            
            # Then create CogVideoXFun transformer and copy state_dict
            fun_model = CogVideoXFunTransformer3DModel(**base_model.config)
            fun_model.load_state_dict(base_model.state_dict())
            base_model = fun_model

            # Create extended model
            model = cls(**base_model.config, 
                        is_train_cross=is_train_cross,
                        cross_attn_interval=cross_attn_interval,
                        cross_attn_dim_head=cross_attn_dim_head,
                        cross_attn_num_heads=cross_attn_num_heads,
                        cross_attn_kv_dim=cross_attn_kv_dim)

            # Load base weights into extended model
            missing, unexpected = model.load_state_dict(base_model.state_dict(), strict=False)
            if missing:
                print(f"⚠️ Missing keys (new cross-attention layers): {missing}")
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
        Forward pass with VideoX-Fun specific inputs and cross-attention processing.
        
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
        
        if control_latents is not None:
            # concat control latents (no adapter, so always concat)
            hidden_states = torch.concat([hidden_states, control_latents], 2)

        # 2. Patch embedding (no adapter)
        patch_embed_input = {'text_embeds': encoder_hidden_states, 
                             'image_embeds': hidden_states}

        if ref_latents is not None:
            patch_embed_input['ref_image_embeds'] = ref_latents
    
        # Use standard patch embed
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
        adapter_version: str = "v1",
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
            base_model_name_or_path (str): Base CogVideoX-Fun model path
            condition_channels (int): Number of extra condition channels to add
            adapter_version (str): Version of adapter to use ("v1" or "v2")
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
                            adapter_version=adapter_version,
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
                        adapter_version=adapter_version,
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
