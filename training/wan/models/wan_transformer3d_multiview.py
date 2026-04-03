import math
from typing import Dict, Optional

import torch
import torch.amp as amp
import torch.nn as nn
from diffusers.utils import is_torch_version

from ..utils import cfg_skip
from .wan_transformer3d import (
    WanAttentionBlock,
    WanCrossAttention,
    WanI2VCrossAttention,
    WanRMSNorm,
    WanT2VCrossAttention,
    rope_apply,
    sinusoidal_embedding_1d,
)
from .wan_transformer3d_with_conditions import WanTransformer3DModelWithConcat

try:
    import flash_attn

    FLASH_FUSION_AVAILABLE = True
except ModuleNotFoundError:
    flash_attn = None
    FLASH_FUSION_AVAILABLE = False


def _resolve_cross_attn_type(module: nn.Module) -> str:
    if isinstance(module, WanT2VCrossAttention):
        return "t2v_cross_attn"
    if isinstance(module, WanI2VCrossAttention):
        return "i2v_cross_attn"
    if isinstance(module, WanCrossAttention):
        return "cross_attn"
    raise ValueError(f"Unsupported cross attention module type: {type(module)}")


def _require_flash_fusion() -> None:
    if not FLASH_FUSION_AVAILABLE:
        raise RuntimeError(
            "same_time_flash_fusion requires the flash_attn package. "
            "No fallback path is implemented."
        )


def _flash_attn_with_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    _require_flash_fusion()

    out, softmax_lse, _ = flash_attn.flash_attn_varlen_func(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        deterministic=False,
        return_attn_probs=True,
    )
    return out, softmax_lse


def _swap_two_views(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] != 2:
        raise ValueError(f"Expected exactly 2 views, got shape {tuple(x.shape)}")
    return torch.cat((x[:, 1:2], x[:, 0:1]), dim=1)


class SameTimeFlashFusionMultiViewSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size=(-1, -1),
        qk_norm: bool = True,
        eps: float = 1e-6,
    ):
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    @staticmethod
    def _validate_layout(
        *,
        x: torch.Tensor,
        view_grid_sizes: torch.Tensor,
        view_seq_lens: torch.Tensor,
        total_seq_lens: torch.Tensor,
    ) -> tuple[int, int, int, int, int, int]:
        if view_grid_sizes.dim() != 3:
            raise ValueError(f"Expected view_grid_sizes as [B, V, 3], got {tuple(view_grid_sizes.shape)}")

        batch_size, num_views = int(view_grid_sizes.shape[0]), int(view_grid_sizes.shape[1])
        if num_views != 2:
            raise NotImplementedError(
                f"same_time_flash_fusion currently supports exactly 2 views, got {num_views}."
            )

        reference_grid = view_grid_sizes[0, 0]
        if not torch.equal(reference_grid.view(1, 1, 3).expand_as(view_grid_sizes), view_grid_sizes):
            raise ValueError("same_time_flash_fusion requires identical [F, H, W] patch grids across batch and views.")

        if not torch.equal(view_seq_lens[:, :1].expand_as(view_seq_lens), view_seq_lens):
            raise ValueError("same_time_flash_fusion requires identical per-view sequence lengths across views.")

        if not torch.equal(total_seq_lens[:1].expand_as(total_seq_lens), total_seq_lens):
            raise ValueError("same_time_flash_fusion requires identical total sequence lengths across batch.")

        frames, height_patches, width_patches = [int(value) for value in reference_grid.tolist()]
        tokens_per_frame = height_patches * width_patches
        view_seq_len = int(view_seq_lens[0, 0].item())
        expected_view_seq_len = frames * tokens_per_frame
        if view_seq_len != expected_view_seq_len:
            raise ValueError(
                f"view_seq_len mismatch: expected {expected_view_seq_len}, got {view_seq_len}."
            )

        total_seq_len = int(total_seq_lens[0].item())
        expected_total_seq_len = num_views * view_seq_len
        if total_seq_len != expected_total_seq_len:
            raise ValueError(
                f"total_seq_len mismatch: expected {expected_total_seq_len}, got {total_seq_len}."
            )
        if x.shape[1] < total_seq_len:
            raise ValueError(f"Input seq_len {x.shape[1]} is smaller than required total_seq_len {total_seq_len}.")

        return batch_size, num_views, frames, tokens_per_frame, view_seq_len, total_seq_len

    def forward(
        self,
        x: torch.Tensor,
        seq_lens: torch.Tensor,
        view_grid_sizes: torch.Tensor,
        view_seq_lens: torch.Tensor,
        total_seq_lens: torch.Tensor,
        freqs: torch.Tensor,
        dtype: torch.dtype,
        t,
    ) -> torch.Tensor:
        del seq_lens, t

        batch_size, num_views, frames, tokens_per_frame, view_seq_len, total_seq_len = self._validate_layout(
            x=x,
            view_grid_sizes=view_grid_sizes,
            view_seq_lens=view_seq_lens,
            total_seq_lens=total_seq_lens,
        )

        x_valid = x[:, :total_seq_len].unflatten(1, (num_views, view_seq_len))
        x_valid = x_valid.unflatten(2, (frames, tokens_per_frame)).contiguous()

        q = self.norm_q(self.q(x_valid.to(dtype))).view(
            batch_size, num_views, frames, tokens_per_frame, self.num_heads, self.head_dim
        )
        k = self.norm_k(self.k(x_valid.to(dtype))).view(
            batch_size, num_views, frames, tokens_per_frame, self.num_heads, self.head_dim
        )
        v = self.v(x_valid.to(dtype)).view(
            batch_size, num_views, frames, tokens_per_frame, self.num_heads, self.head_dim
        )

        q_view = q.reshape(batch_size * num_views, view_seq_len, self.num_heads, self.head_dim)
        k_view = k.reshape(batch_size * num_views, view_seq_len, self.num_heads, self.head_dim)
        v_view = v.reshape(batch_size * num_views, view_seq_len, self.num_heads, self.head_dim)
        view_grids = view_grid_sizes.reshape(batch_size * num_views, 3)

        q_view = rope_apply(q_view, view_grids, freqs).to(dtype=v_view.dtype)
        k_view = rope_apply(k_view, view_grids, freqs).to(dtype=v_view.dtype)

        cu_seqlens_intra = (
            torch.arange(batch_size * num_views + 1, device=q_view.device, dtype=torch.int32) * view_seq_len
        )
        q_intra = q_view.reshape(batch_size * num_views * view_seq_len, self.num_heads, self.head_dim)
        k_intra = k_view.reshape(batch_size * num_views * view_seq_len, self.num_heads, self.head_dim)
        v_intra = v_view.reshape(batch_size * num_views * view_seq_len, self.num_heads, self.head_dim)
        out_intra, lse_intra = _flash_attn_with_lse(
            q_intra,
            k_intra,
            v_intra,
            cu_seqlens_q=cu_seqlens_intra,
            cu_seqlens_k=cu_seqlens_intra,
            max_seqlen_q=view_seq_len,
            max_seqlen_k=view_seq_len,
        )
        out_intra = out_intra.reshape(
            batch_size, num_views, frames, tokens_per_frame, self.num_heads, self.head_dim
        )
        lse_intra = lse_intra.transpose(0, 1).contiguous().reshape(
            batch_size, num_views, frames, tokens_per_frame, self.num_heads
        )

        q_cross = q_view.reshape(
            batch_size, num_views, frames, tokens_per_frame, self.num_heads, self.head_dim
        ).reshape(batch_size * num_views * frames, tokens_per_frame, self.num_heads, self.head_dim)
        k_view_grouped = k_view.reshape(
            batch_size, num_views, frames, tokens_per_frame, self.num_heads, self.head_dim
        )
        v_view_grouped = v_view.reshape(
            batch_size, num_views, frames, tokens_per_frame, self.num_heads, self.head_dim
        )
        k_other = _swap_two_views(k_view_grouped).reshape(
            batch_size * num_views * frames, tokens_per_frame, self.num_heads, self.head_dim
        )
        v_other = _swap_two_views(v_view_grouped).reshape(
            batch_size * num_views * frames, tokens_per_frame, self.num_heads, self.head_dim
        )

        cu_seqlens_cross = (
            torch.arange(batch_size * num_views * frames + 1, device=q_cross.device, dtype=torch.int32)
            * tokens_per_frame
        )
        q_cross_flat = q_cross.reshape(batch_size * num_views * frames * tokens_per_frame, self.num_heads, self.head_dim)
        k_cross_flat = k_other.reshape(
            batch_size * num_views * frames * tokens_per_frame, self.num_heads, self.head_dim
        )
        v_cross_flat = v_other.reshape(
            batch_size * num_views * frames * tokens_per_frame, self.num_heads, self.head_dim
        )
        out_cross, lse_cross = _flash_attn_with_lse(
            q_cross_flat,
            k_cross_flat,
            v_cross_flat,
            cu_seqlens_q=cu_seqlens_cross,
            cu_seqlens_k=cu_seqlens_cross,
            max_seqlen_q=tokens_per_frame,
            max_seqlen_k=tokens_per_frame,
        )
        out_cross = out_cross.reshape(
            batch_size, num_views, frames, tokens_per_frame, self.num_heads, self.head_dim
        )
        lse_cross = lse_cross.transpose(0, 1).contiguous().reshape(
            batch_size, num_views, frames, tokens_per_frame, self.num_heads
        )

        lse_union = torch.logaddexp(lse_intra, lse_cross)
        weight_intra = torch.exp(lse_intra - lse_union).to(dtype=out_intra.dtype).unsqueeze(-1)
        weight_cross = torch.exp(lse_cross - lse_union).to(dtype=out_cross.dtype).unsqueeze(-1)

        fused = weight_intra * out_intra + weight_cross * out_cross
        fused = fused.reshape(batch_size, total_seq_len, self.dim)
        fused = self.o(fused)

        if total_seq_len == x.shape[1]:
            return fused

        padded = fused.new_zeros(batch_size, x.shape[1], self.dim)
        padded[:, :total_seq_len] = fused
        return padded


class MultiViewWanAttentionBlock(WanAttentionBlock):
    def __init__(
        self,
        cross_attn_type,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        cross_view_mode: str = "same_time_flash_fusion",
    ):
        super().__init__(
            cross_attn_type=cross_attn_type,
            dim=dim,
            ffn_dim=ffn_dim,
            num_heads=num_heads,
            window_size=window_size,
            qk_norm=qk_norm,
            cross_attn_norm=cross_attn_norm,
            eps=eps,
        )
        self.cross_view_mode = str(cross_view_mode).strip().lower()
        if self.cross_view_mode != "same_time_flash_fusion":
            raise ValueError(f"Unsupported cross_view_mode: {cross_view_mode}")

        self.self_attn = SameTimeFlashFusionMultiViewSelfAttention(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            qk_norm=qk_norm,
            eps=eps,
        )

    @classmethod
    def from_base_block(
        cls,
        block: WanAttentionBlock,
        *,
        cross_view_mode: str,
    ) -> "MultiViewWanAttentionBlock":
        new_block = cls(
            cross_attn_type=_resolve_cross_attn_type(block.cross_attn),
            dim=block.dim,
            ffn_dim=block.ffn_dim,
            num_heads=block.num_heads,
            window_size=block.window_size,
            qk_norm=block.qk_norm,
            cross_attn_norm=block.cross_attn_norm,
            eps=block.eps,
            cross_view_mode=cross_view_mode,
        )
        missing, unexpected = new_block.load_state_dict(block.state_dict(), strict=False)
        if unexpected:
            raise ValueError(f"Unexpected keys while converting base block to multiview block: {unexpected}")
        if missing:
            raise ValueError(f"Unexpected missing keys while converting base block: {missing}")
        return new_block

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        dtype=torch.float32,
        t=0,
        multiview_batch_size: Optional[int] = None,
        num_views: Optional[int] = None,
        view_grid_sizes: Optional[torch.Tensor] = None,
        view_seq_lens: Optional[torch.Tensor] = None,
        total_seq_lens: Optional[torch.Tensor] = None,
    ):
        del grid_sizes, multiview_batch_size, num_views

        if view_grid_sizes is None or view_seq_lens is None or total_seq_lens is None:
            raise ValueError("same_time_flash_fusion requires multiview layout metadata for every block.")

        if e.dim() > 3:
            e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
            e = [e.squeeze(2) for e in e]
        else:
            e = (self.modulation + e).chunk(6, dim=1)

        temp_x = self.norm1(x) * (1 + e[1]) + e[0]
        temp_x = temp_x.to(dtype)

        y = self.self_attn(
            temp_x,
            seq_lens=seq_lens,
            view_grid_sizes=view_grid_sizes,
            view_seq_lens=view_seq_lens,
            total_seq_lens=total_seq_lens,
            freqs=freqs,
            dtype=dtype,
            t=t,
        )
        x = x + y * e[2]

        x = x + self.cross_attn(self.norm3(x), context, context_lens, dtype, t=t)

        temp_x = self.norm2(x) * (1 + e[4]) + e[3]
        temp_x = temp_x.to(dtype)
        y = self.ffn(temp_x)
        x = x + y * e[5]
        return x


class MultiViewWanTransformer3DModelWithConcat(WanTransformer3DModelWithConcat):
    _supports_gradient_checkpointing = True

    def __init__(
        self,
        model_type="t2v",
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
        condition_channels: int = 0,
        is_wan2_2: bool = False,
        multiview_mode: str = "same_time_flash_fusion",
        cross_view_num_heads: Optional[int] = None,
    ):
        del cross_view_num_heads

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
            condition_channels=condition_channels,
            is_wan2_2=is_wan2_2,
        )

        self.multiview_mode = str(multiview_mode).strip().lower()
        if self.multiview_mode != "same_time_flash_fusion":
            raise ValueError(f"Unsupported multiview_mode: {multiview_mode}")

        self._replace_multiview_blocks()
        self.register_to_config(multiview_mode=self.multiview_mode)

    def _replace_multiview_blocks(self) -> None:
        converted_blocks = []
        for layer_idx, block in enumerate(self.blocks):
            mv_block = MultiViewWanAttentionBlock.from_base_block(
                block,
                cross_view_mode="same_time_flash_fusion",
            )
            mv_block.self_attn.layer_idx = layer_idx
            mv_block.self_attn.num_layers = self.num_layers
            converted_blocks.append(mv_block)
        self.blocks = nn.ModuleList(converted_blocks)

    @staticmethod
    def _concat_condition_inputs_6d(
        y: Optional[torch.Tensor],
        condition_latents: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if condition_latents is None:
            return y
        if y is None:
            return condition_latents
        if not isinstance(y, torch.Tensor) or y.dim() != 6:
            raise TypeError("Expected `y` as [B, V, C, F, H, W] in same_time_flash_fusion mode.")
        if not isinstance(condition_latents, torch.Tensor) or condition_latents.dim() != 6:
            raise TypeError("Expected `condition_latents` as [B, V, C, F, H, W] in same_time_flash_fusion mode.")
        return torch.cat([y, condition_latents], dim=2)

    def _unpatchify_multiview(
        self,
        x: torch.Tensor,
        view_grid_sizes: torch.Tensor,
        view_seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        batch_outputs = []
        batch_size, num_views = int(view_grid_sizes.shape[0]), int(view_grid_sizes.shape[1])
        for batch_idx in range(batch_size):
            offset = 0
            view_outputs = []
            for view_idx in range(num_views):
                view_seq_len = int(view_seq_lens[batch_idx, view_idx].item())
                view_tokens = x[batch_idx : batch_idx + 1, offset : offset + view_seq_len]
                view_grid = view_grid_sizes[batch_idx, view_idx : view_idx + 1]
                view_outputs.append(self.unpatchify(view_tokens, view_grid)[0])
                offset += view_seq_len
            batch_outputs.append(torch.stack(view_outputs, dim=0))
        return torch.stack(batch_outputs, dim=0)

    def _forward_same_time_flash_fusion(
        self,
        x: torch.Tensor,
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
        del cond_flag

        _require_flash_fusion()

        if self.sp_world_size > 1:
            raise NotImplementedError("same_time_flash_fusion does not support sequence-parallel inference/training.")
        if y_camera is not None or full_ref is not None or subject_ref is not None:
            raise NotImplementedError(
                "same_time_flash_fusion currently supports only x/y/condition_latents inputs without "
                "camera/full_ref/subject_ref auxiliaries."
            )
        if not isinstance(x, torch.Tensor) or x.dim() != 6:
            raise TypeError("same_time_flash_fusion expects `x` as [B, V, C, F, H, W].")
        if y is not None and (not isinstance(y, torch.Tensor) or y.dim() != 6):
            raise TypeError("same_time_flash_fusion expects `y` as [B, V, C, F, H, W].")
        if condition_latents is not None and (
            not isinstance(condition_latents, torch.Tensor) or condition_latents.dim() != 6
        ):
            raise TypeError("same_time_flash_fusion expects `condition_latents` as [B, V, C, F, H, W].")

        y = self._concat_condition_inputs_6d(y, condition_latents)

        device = self.patch_embedding.weight.device
        dtype = x.dtype
        if self.freqs.device != device and torch.device(type="meta") != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = torch.cat([x, y], dim=2)

        batch_size, num_views = x.shape[:2]
        if num_views != 2:
            raise NotImplementedError(f"same_time_flash_fusion currently supports exactly 2 views, got {num_views}.")

        x = x.reshape(batch_size * num_views, *x.shape[2:])
        x = self.patch_embedding(x)

        view_grid_sizes = (
            torch.tensor(x.shape[2:], device=x.device, dtype=torch.long)
            .view(1, 1, 3)
            .expand(batch_size, num_views, 3)
            .contiguous()
        )
        view_seq_lens = view_grid_sizes.prod(dim=-1, dtype=torch.long)
        total_seq_lens = view_seq_lens.sum(dim=1)

        x = x.flatten(2).transpose(1, 2).unflatten(0, (batch_size, num_views))
        x = x.reshape(batch_size, int(total_seq_lens.max().item()), self.dim)

        seq_len = max(int(seq_len), int(total_seq_lens.max().item()))
        if x.size(1) < seq_len:
            x = torch.cat([x, x.new_zeros(batch_size, seq_len - x.size(1), x.size(2))], dim=1)
        seq_lens = total_seq_lens

        with amp.autocast(device_type="cuda", dtype=torch.float32):
            if t.dim() != 1:
                bt = t.size(0)
                ft = t.flatten()
                e = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim, ft).unflatten(0, (bt, seq_len)).float()
                )
                e0 = self.time_projection(e).unflatten(2, (6, self.dim))
            else:
                e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float())
                e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            e0 = e0.to(dtype)
            e = e.to(dtype)

        context_lens = None
        context = self.text_embedding(
            torch.stack(
                [torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context]
            )
        )
        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)
            context = torch.concat([context_clip, context], dim=1)

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def create_custom_forward(module, local_view_grid_sizes, local_view_seq_lens, local_total_seq_lens):
                    def custom_forward(x_, e_, seq_lens_, context_, context_lens_, dtype_, t_):
                        return module(
                            x_,
                            e=e_,
                            seq_lens=seq_lens_,
                            grid_sizes=local_view_grid_sizes[:, 0],
                            view_grid_sizes=local_view_grid_sizes,
                            view_seq_lens=local_view_seq_lens,
                            total_seq_lens=local_total_seq_lens,
                            freqs=self.freqs,
                            context=context_,
                            context_lens=context_lens_,
                            dtype=dtype_,
                            t=t_,
                        )

                    return custom_forward

                ckpt_kwargs: Dict[str, object] = (
                    {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                )
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block, view_grid_sizes, view_seq_lens, total_seq_lens),
                    x,
                    e0,
                    seq_lens,
                    context,
                    context_lens,
                    dtype,
                    t,
                    **ckpt_kwargs,
                )
            else:
                x = block(
                    x,
                    e=e0,
                    seq_lens=seq_lens,
                    grid_sizes=view_grid_sizes[:, 0],
                    view_grid_sizes=view_grid_sizes,
                    view_seq_lens=view_seq_lens,
                    total_seq_lens=total_seq_lens,
                    freqs=self.freqs,
                    context=context,
                    context_lens=context_lens,
                    dtype=dtype,
                    t=t,
                )

        if torch.is_grad_enabled() and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            ckpt_kwargs: Dict[str, object] = (
                {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            )
            x = torch.utils.checkpoint.checkpoint(create_custom_forward(self.head), x, e, **ckpt_kwargs)
        else:
            x = self.head(x, e)

        return self._unpatchify_multiview(x, view_grid_sizes, view_seq_lens)

    @cfg_skip()
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
        multiview_batch_size: Optional[int] = None,
        num_views: Optional[int] = None,
    ):
        del multiview_batch_size, num_views
        return self._forward_same_time_flash_fusion(
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
            condition_latents=condition_latents,
        )
