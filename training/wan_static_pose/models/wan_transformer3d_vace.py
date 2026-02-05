# Copyright 2024 The Wan team and VACE team.
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
WanTransformer3DVace: Wan Transformer with VACE-style conditioning.

Implements VACE (Video Anything Controllable Editor) conditioning mechanism
where hand conditions are processed through parallel VACE blocks and injected
as hints into the main transformer blocks via skip connections.

Architecture:
    - VaceWanAttentionBlock: Processes condition input, generates hints
    - BaseWanAttentionBlock: Main blocks that receive hints via skip connection
    - vace_patch_embedding: Separate embedding for condition input

Reference: VACE (Alibaba) - https://github.com/ali-vilab/VACE
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import register_to_config

from .wan_transformer3d import (
    WanAttentionBlock,
    WanTransformer3DModel,
    sinusoidal_embedding_1d,
)


class VaceWanAttentionBlock(WanAttentionBlock):
    """
    VACE attention block for processing condition input.

    Processes the condition (e.g., hand video) in parallel with the main blocks
    and generates hints that are injected into the main transformer.

    Key components:
        - before_proj: Projects main input x to condition space (only block_id=0)
        - after_proj: Projects block output to hint space (zero-initialized)

    The output is a stack of hints from all previous blocks plus current block output.
    """

    def __init__(
        self,
        cross_attn_type: str,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        window_size: Tuple[int, int] = (-1, -1),
        qk_norm: bool = True,
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        block_id: int = 0,
    ):
        super().__init__(
            cross_attn_type, dim, ffn_dim, num_heads,
            window_size, qk_norm, cross_attn_norm, eps
        )
        self.block_id = block_id

        # First block needs to project x into condition space
        if block_id == 0:
            self.before_proj = nn.Linear(dim, dim)
            nn.init.zeros_(self.before_proj.weight)
            nn.init.zeros_(self.before_proj.bias)

        # All blocks project output to hint space (zero-init for stable training)
        self.after_proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.after_proj.weight)
        nn.init.zeros_(self.after_proj.bias)

    def forward(
        self,
        c: torch.Tensor,
        x: torch.Tensor,
        e: torch.Tensor,
        seq_lens: torch.Tensor,
        grid_sizes: torch.Tensor,
        freqs: torch.Tensor,
        context: torch.Tensor,
        context_lens: Optional[torch.Tensor] = None,
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass for VACE block.

        Args:
            c: Condition features.
               - For block_id=0: [B, L, C] condition embeddings
               - For block_id>0: [num_hints+1, B, L, C] stacked hints + current
            x: Main input features [B, L, C] (used only in first block)
            e: Time embeddings
            seq_lens: Sequence lengths
            grid_sizes: Grid sizes for RoPE
            freqs: RoPE frequencies
            context: Text context embeddings
            context_lens: Context lengths
            dtype: Computation dtype

        Returns:
            Stacked tensor [num_hints+2, B, L, C] containing:
            - Previous hints (if any)
            - Current hint (after_proj output)
            - Current block output (for next block)
        """
        if self.block_id == 0:
            # First block: project x and add to condition c
            c = self.before_proj(c) + x
            all_c = []
        else:
            # Subsequent blocks: unstack previous hints
            all_c = list(torch.unbind(c))
            c = all_c.pop(-1)  # Get the last one as current input

        # Process through parent attention block
        c = super().forward(
            c, e, seq_lens, grid_sizes, freqs, context, context_lens, dtype, **kwargs
        )

        # Generate hint and stack
        c_skip = self.after_proj(c)
        all_c += [c_skip, c]  # Add hint and output
        c = torch.stack(all_c)

        return c


class BaseWanAttentionBlock(WanAttentionBlock):
    """
    Base attention block that receives hints from VACE blocks.

    Extends WanAttentionBlock to add hint injection after the forward pass:
        output = super().forward(x) + hints[block_id] * context_scale

    Only blocks with assigned block_id receive hints.
    """

    def __init__(
        self,
        cross_attn_type: str,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        window_size: Tuple[int, int] = (-1, -1),
        qk_norm: bool = True,
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        block_id: Optional[int] = None,
    ):
        super().__init__(
            cross_attn_type, dim, ffn_dim, num_heads,
            window_size, qk_norm, cross_attn_norm, eps
        )
        self.block_id = block_id

    def forward(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        seq_lens: torch.Tensor,
        grid_sizes: torch.Tensor,
        freqs: torch.Tensor,
        context: torch.Tensor,
        context_lens: Optional[torch.Tensor] = None,
        dtype: torch.dtype = torch.float32,
        hints: Optional[List[torch.Tensor]] = None,
        context_scale: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass with optional hint injection.

        Args:
            x: Input features [B, L, C]
            hints: List of hint tensors from VACE blocks
            context_scale: Scaling factor for hint injection
            (other args same as WanAttentionBlock)

        Returns:
            Output features with hint added (if applicable)
        """
        x = super().forward(
            x, e, seq_lens, grid_sizes, freqs, context, context_lens, dtype, **kwargs
        )

        # Inject hint if this block has one assigned
        if self.block_id is not None and hints is not None:
            x = x + hints[self.block_id] * context_scale

        return x


class WanTransformer3DVace(WanTransformer3DModel):
    """
    Wan Transformer with VACE-style conditioning for hand video.

    Extends WanTransformer3DModel with:
        - vace_blocks: Parallel transformer blocks for processing hand conditions
        - vace_patch_embedding: Separate patch embedding for hand latents
        - Hint injection mechanism into main blocks

    The conditioning flow:
        1. Hand latents → vace_patch_embedding → vace features
        2. vace_blocks process features in parallel with main blocks
        3. Each vace_block generates a hint
        4. Hints are injected into corresponding main blocks via skip connections

    Args:
        vace_layers: Which layers have corresponding VACE blocks (default: every 2nd layer)
        vace_in_dim: Input channels for vace_patch_embedding (default: same as in_dim)
        (other args same as WanTransformer3DModel)
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        # WanTransformer3DModel parameters
        model_type: str = 't2v',
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        text_len: int = 512,
        in_dim: int = 16,
        dim: int = 2048,
        ffn_dim: int = 8192,
        freq_dim: int = 256,
        text_dim: int = 4096,
        out_dim: int = 16,
        num_heads: int = 16,
        num_layers: int = 32,
        window_size: Tuple[int, int] = (-1, -1),
        qk_norm: bool = True,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
        in_channels: int = 16,
        hidden_size: int = 2048,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        add_ref_conv: bool = False,
        in_dim_ref_conv: int = 16,
        cross_attn_type: Optional[str] = None,
        fps: int = 16,
        # VACE-specific parameters
        vace_layers: Optional[List[int]] = None,
        vace_in_dim: Optional[int] = None,
        is_wan2_2: bool = False,
    ):
        """
        Initialize WanTransformer3DVace.

        Args:
            vace_layers: List of layer indices that have VACE blocks.
                        Default: every 2nd layer [0, 2, 4, ..., num_layers-2]
            vace_in_dim: Input channels for VACE patch embedding.
                        Default: same as in_dim (16 for hand latents)
            is_wan2_2: If True, use WAN 2.2 settings (cross_attn, no CLIP)
        """
        # Apply WAN 2.2 specific settings
        if is_wan2_2:
            cross_attn_type = "cross_attn"

        # Initialize parent (creates standard blocks)
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

        self.is_wan2_2 = is_wan2_2

        # VACE configuration
        self.vace_layers = vace_layers if vace_layers is not None else list(range(0, num_layers, 2))
        self.vace_in_dim = vace_in_dim if vace_in_dim is not None else in_dim

        assert 0 in self.vace_layers, "Layer 0 must be in vace_layers"
        self.vace_layers_mapping = {layer_idx: vace_idx for vace_idx, layer_idx in enumerate(self.vace_layers)}

        # Determine cross attention type for blocks
        block_cross_attn_type = cross_attn_type if cross_attn_type is not None else (
            't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        )

        # Replace standard blocks with BaseWanAttentionBlock (same weights, adds hint support)
        self.blocks = nn.ModuleList([
            BaseWanAttentionBlock(
                block_cross_attn_type, dim, ffn_dim, num_heads, window_size,
                qk_norm, cross_attn_norm, eps,
                block_id=self.vace_layers_mapping.get(i)  # None if not in vace_layers
            )
            for i in range(num_layers)
        ])

        # Set layer indices for attention
        for layer_idx, block in enumerate(self.blocks):
            block.self_attn.layer_idx = layer_idx
            block.self_attn.num_layers = num_layers

        # Create VACE blocks (parallel processing for condition)
        self.vace_blocks = nn.ModuleList([
            VaceWanAttentionBlock(
                block_cross_attn_type, dim, ffn_dim, num_heads, window_size,
                qk_norm, cross_attn_norm, eps,
                block_id=i
            )
            for i in range(len(self.vace_layers))
        ])

        # VACE patch embedding for condition input
        self.vace_patch_embedding = nn.Conv3d(
            self.vace_in_dim, dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        print(f"🎛️ WanTransformer3DVace initialized:")
        print(f"   VACE layers: {self.vace_layers}")
        print(f"   VACE input dim: {self.vace_in_dim}")
        print(f"   Number of VACE blocks: {len(self.vace_blocks)}")

    def forward_vace(
        self,
        x: torch.Tensor,
        vace_context: List[torch.Tensor],
        seq_len: int,
        e: torch.Tensor,
        seq_lens: torch.Tensor,
        grid_sizes: torch.Tensor,
        freqs: torch.Tensor,
        context: torch.Tensor,
        context_lens: Optional[torch.Tensor] = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> List[torch.Tensor]:
        """
        Process condition input through VACE blocks to generate hints.

        Args:
            x: Main input embeddings [B, L, C] (used for residual in first block)
            vace_context: List of condition tensors [C_vace, F, H, W] per batch
            seq_len: Maximum sequence length
            e: Time embeddings
            seq_lens: Sequence lengths
            grid_sizes: Grid sizes
            freqs: RoPE frequencies
            context: Text context
            context_lens: Context lengths
            dtype: Computation dtype

        Returns:
            List of hint tensors, one per VACE layer
        """
        # Embed condition inputs (convert to patch embedding's dtype for compatibility)
        vace_dtype = self.vace_patch_embedding.weight.dtype
        c = [self.vace_patch_embedding(u.unsqueeze(0).to(vace_dtype)) for u in vace_context]
        c = [u.flatten(2).transpose(1, 2).to(dtype) for u in c]  # Convert back to main dtype

        # Pad to seq_len and concatenate batch
        c = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
            for u in c
        ])

        # Process through VACE blocks
        for block in self.vace_blocks:
            c = block(
                c, x,
                e=e,
                seq_lens=seq_lens,
                grid_sizes=grid_sizes,
                freqs=freqs,
                context=context,
                context_lens=context_lens,
                dtype=dtype,
            )

        # Extract hints (all but the last output)
        hints = list(torch.unbind(c))[:-1]
        return hints

    def forward(
        self,
        x: Union[torch.Tensor, List[torch.Tensor]],
        t: torch.Tensor,
        context: Union[torch.Tensor, List[torch.Tensor]],
        seq_len: int,
        clip_fea: Optional[torch.Tensor] = None,
        y: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        y_camera: Optional[torch.Tensor] = None,
        full_ref: Optional[torch.Tensor] = None,
        subject_ref: Optional[torch.Tensor] = None,
        cond_flag: bool = True,
        # VACE-specific parameters
        vace_context: Optional[List[torch.Tensor]] = None,
        vace_context_scale: float = 1.0,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass with VACE conditioning.

        Args:
            x: Noisy latent input
            t: Timestep tensor
            context: Text embeddings
            seq_len: Sequence length
            clip_fea: CLIP features (optional)
            y: Inpaint latents [mask + static] (optional)
            vace_context: List of condition tensors (e.g., hand latents)
                         Each tensor: [C_vace, F, H, W]
            vace_context_scale: Scale factor for hint injection (default 1.0)
            (other args for compatibility)

        Returns:
            Denoised output tensor
        """
        # Get device and ensure freqs are on correct device
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # Handle input format (list vs tensor)
        if isinstance(x, torch.Tensor) and x.dim() == 5:
            dtype = x.dtype
            x = [x[i] for i in range(x.shape[0])]
        else:
            dtype = x[0].dtype

        # Concatenate y (inpaint condition) if provided
        if y is not None:
            if isinstance(y, torch.Tensor) and y.dim() == 5:
                y = [y[i] for i in range(y.shape[0])]
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # Patch embedding for main input
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len

        # Pad and concatenate batch
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
            for u in x
        ])

        # Time embeddings
        with amp.autocast(device_type='cuda', dtype=torch.float32):
            e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # Context (text) embedding
        context_lens = None
        if isinstance(context, list):
            context = self.text_embedding(
                torch.stack([
                    torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                    for u in context
                ])
            )
        else:
            context = self.text_embedding(context)

        # CLIP image embedding (for WAN 2.1 i2v mode)
        if clip_fea is not None and hasattr(self, 'img_emb'):
            context_clip = self.img_emb(clip_fea)
            context = torch.cat([context_clip, context], dim=1)

        # Prepare common kwargs for blocks
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            dtype=dtype,
        )

        # Process VACE context if provided
        hints = None
        if vace_context is not None:
            hints = self.forward_vace(
                x, vace_context, seq_len,
                e=e0,
                seq_lens=seq_lens,
                grid_sizes=grid_sizes,
                freqs=self.freqs,
                context=context,
                context_lens=context_lens,
                dtype=dtype,
            )

        # Add hints to kwargs
        kwargs['hints'] = hints
        kwargs['context_scale'] = vace_context_scale

        # Process through main blocks with hint injection
        for block in self.blocks:
            x = block(x, **kwargs)

        # Head (final projection)
        x = self.head(x, e)

        # Unpatchify
        x = self.unpatchify(x, grid_sizes)

        return [u.float() for u in x]

    def get_condition_info(self) -> Dict[str, Any]:
        """Get information about the VACE conditioning setup."""
        return {
            "approach": "vace",
            "vace_layers": self.vace_layers,
            "vace_in_dim": self.vace_in_dim,
            "num_vace_blocks": len(self.vace_blocks),
            "is_wan2_2": self.is_wan2_2,
        }
