# Copyright 2024 The Wan team.
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

"""
WanTransformer3DModelWithConcat: Wan Transformer with conditional input support.

Extends WanTransformer3DModel to support additional condition channels
that are concatenated to the input latents (e.g., hand pose latents).

Usage in pipeline:
    transformer = WanTransformer3DModelWithConcat.from_pretrained(
        base_model_path,
        subfolder="transformer",
        transformer_additional_kwargs={"condition_channels": 16},
    )
"""

import torch
import torch.nn as nn
from typing import Dict, Any
from .wan_transformer3d import WanTransformer3DModel


class WanTransformer3DModelWithConcat(WanTransformer3DModel):
    """
    Wan Transformer with conditional input support via channel concatenation.
    
    Extends WanTransformer3DModel to support additional condition channels
    that are concatenated to the input latents before patch embedding.
    
    Channel layout (for inpainting with hand condition):
        - x: [C=16] noisy latents
        - y: [C=17] mask (1) + static latents (16) - handled by parent
        - condition_latents: [C=condition_channels] e.g., hand latents
        
    The extended patch_embedding handles the concatenated input:
        Total input channels = in_dim + 17 (y channels) + condition_channels
    """
    
    _supports_gradient_checkpointing = True
    
    def __init__(
        self,
        # WanTransformer3DModel parameters (must be explicit for ConfigMixin)
        model_type='t2v',
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        in_channels=16,
        hidden_size=2048,
        add_control_adapter=False,
        in_dim_control_adapter=24,
        add_ref_conv=False,
        in_dim_ref_conv=16,
        cross_attn_type=None,
        fps: int = 16,
        # WanTransformer3DModelWithConcat specific parameters
        condition_channels: int = 0,
        is_wan2_2: bool = False,
    ):
        """
        Initialize WanTransformer3DModelWithConcat.
        
        Args:
            condition_channels: Number of additional condition channels to concatenate.
                              Default 0 means no extra condition (behaves like parent).
            is_wan2_2: If True, applies WAN 2.2 specific settings:
                      - Sets cross_attn_type="cross_attn"
                      - Removes img_emb (CLIP image encoder)
            Other args: See WanTransformer3DModel.
        """
        # Apply WAN 2.2 specific settings
        if is_wan2_2:
            cross_attn_type = "cross_attn"
        
        super().__init__(
            model_type=model_type,
            patch_size=patch_size,
            text_len=text_len,
            in_dim=in_dim,
            dim=dim,
            ffn_dim=ffn_dim,
            freq_dim=freq_dim,
            text_dim=text_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            window_size=window_size,
            qk_norm=qk_norm,
            cross_attn_norm=cross_attn_norm,
            eps=eps,
            in_channels=in_channels,
            hidden_size=hidden_size,
            add_control_adapter=add_control_adapter,
            in_dim_control_adapter=in_dim_control_adapter,
            add_ref_conv=add_ref_conv,
            in_dim_ref_conv=in_dim_ref_conv,
            cross_attn_type=cross_attn_type,
            fps=fps,
        )
        
        # Remove img_emb for WAN 2.2 (no CLIP image encoder)
        if is_wan2_2 and hasattr(self, "img_emb"):
            del self.img_emb
        
        self.condition_channels = condition_channels
        self.is_wan2_2 = is_wan2_2
        
        if condition_channels > 0:
            self._extend_patch_embedding(condition_channels)
    
    def _extend_patch_embedding(self, condition_channels: int):
        """
        Extend patch_embedding to handle additional condition channels.
        
        Args:
            condition_channels: Number of condition channels to add.
        """
        original_proj = self.patch_embedding
        original_in_channels = original_proj.in_channels
        new_in_channels = original_in_channels + condition_channels
        
        print(f"🔗 Extending Wan transformer patch_embedding:")
        print(f"   Original in_channels: {original_in_channels}")
        print(f"   Condition channels: {condition_channels}")
        print(f"   New in_channels: {new_in_channels}")
        
        # Create new conv with extended input channels
        new_proj = nn.Conv3d(
            in_channels=new_in_channels,
            out_channels=original_proj.out_channels,
            kernel_size=original_proj.kernel_size,
            stride=original_proj.stride,
            padding=original_proj.padding,
            bias=original_proj.bias is not None,
        )
        
        with torch.no_grad():
            # Copy pretrained weights for original channels
            new_proj.weight[:, :original_in_channels] = original_proj.weight
            # Initialize new condition channels to zeros
            new_proj.weight[:, original_in_channels:] = 0.0
            if original_proj.bias is not None:
                new_proj.bias.data = original_proj.bias.data
        
        self.patch_embedding = new_proj
        
        # Register to config for serialization
        self.register_to_config(
            in_channels=new_in_channels,
            original_in_channels=original_in_channels,
            condition_channels=condition_channels,
        )
        
        print(f"✅ Extended patch_embedding for {condition_channels} condition channels")
    
    def get_condition_info(self) -> Dict[str, Any]:
        """Get information about the conditional setup."""
        return {
            "approach": "concat",
            "has_conditions": self.condition_channels > 0,
            "condition_channels": self.condition_channels,
            "total_input_channels": self.patch_embedding.in_channels,
            "is_wan2_2": self.is_wan2_2,
        }
    
    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        y_camera=None,
        full_ref=None,
        subject_ref=None,
        cond_flag=True,
        condition_latents=None,
    ):
        """
        Forward pass with conditional latent concatenation.
        
        Args:
            x: Noisy latent input.
               - List of [C, F, H, W] tensors (WAN native format), or
               - Batched tensor [B, C, F, H, W]
            t: Timestep tensor [B] or scalar
            context: Text embeddings - list of [L, D] tensors
            seq_len: Sequence length for positional encoding
            clip_fea: CLIP image features (optional)
            y: Inpaint latents [mask + static latents] (optional)
               - List of [C_inpaint, F, H, W] or [B, C_inpaint, F, H, W]
            y_camera: Camera parameters for control adapter (optional)
            full_ref: Full reference image (optional)
            subject_ref: Subject reference video (optional)
            cond_flag: Condition flag for TeaCache
            condition_latents: Additional condition latents to concat (optional)
               - [B, C_cond, F, H, W] tensor for extra condition channels
        
        Returns:
            Denoised output tensors with shape [B, C_out, F, H, W]
        """
        # Convert x to list format if needed
        if isinstance(x, torch.Tensor) and x.dim() == 5:
            x = [x[i] for i in range(x.shape[0])]
        
        # Concatenate condition_latents to y if provided
        if condition_latents is not None:
            if y is None:
                # If y is None, create y from condition_latents
                if isinstance(condition_latents, torch.Tensor) and condition_latents.dim() == 5:
                    y = [condition_latents[i] for i in range(condition_latents.shape[0])]
                else:
                    y = condition_latents
            else:
                # Concatenate condition_latents to y
                if isinstance(y, list):
                    # y is list of [C, F, H, W], condition_latents is [B, C_cond, F, H, W]
                    y = [
                        torch.cat([y_i, condition_latents[i]], dim=0) 
                        for i, y_i in enumerate(y)
                    ]
                elif isinstance(y, torch.Tensor) and y.dim() == 5:
                    # y is [B, C, F, H, W], condition_latents is [B, C_cond, F, H, W]
                    y = torch.cat([y, condition_latents], dim=1)
                    # Convert to list format expected by WAN
                    y = [y[i] for i in range(y.shape[0])]
        
        # Convert y to list format if needed (when condition_latents is None)
        elif y is not None and isinstance(y, torch.Tensor) and y.dim() == 5:
            y = [y[i] for i in range(y.shape[0])]
        
        # Call parent forward
        return super().forward(
            x=x,
            t=t,
            context=context,
            seq_len=seq_len,
            clip_fea=clip_fea,
            y=y,
            y_camera=y_camera,
            full_ref=full_ref,
            subject_ref=subject_ref,
            cond_flag=cond_flag,
        )
