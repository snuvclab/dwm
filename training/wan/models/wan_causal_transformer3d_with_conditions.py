import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from diffusers.configuration_utils import register_to_config
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

from .wan_transformer3d import (
    MLPProj,
    WAN_CROSSATTENTION_CLASSES,
    WanLayerNorm,
    WanRMSNorm,
    rope_apply,
    sinusoidal_embedding_1d,
)
from .wan_transformer3d_with_conditions import WanTransformer3DModelWithConcat


_flex_attention_impl = flex_attention
if torch.cuda.is_available():
    try:
        _flex_attention_impl = torch.compile(flex_attention, dynamic=False, mode="default")
    except Exception:
        _flex_attention_impl = flex_attention


class CausalWanSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, qk_norm=True, eps=1e-6):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs, block_mask):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)

        is_teacher_forcing = bool(seq_lens.numel() > 0 and s == int(seq_lens[0].item()) * 2)
        if is_teacher_forcing:
            q_clean, q_noisy = torch.chunk(q, 2, dim=1)
            k_clean, k_noisy = torch.chunk(k, 2, dim=1)
            roped_query = torch.cat(
                [rope_apply(q_clean, grid_sizes, freqs), rope_apply(q_noisy, grid_sizes, freqs)],
                dim=1,
            ).to(v.dtype)
            roped_key = torch.cat(
                [rope_apply(k_clean, grid_sizes, freqs), rope_apply(k_noisy, grid_sizes, freqs)],
                dim=1,
            ).to(v.dtype)
        else:
            roped_query = rope_apply(q, grid_sizes, freqs).to(v.dtype)
            roped_key = rope_apply(k, grid_sizes, freqs).to(v.dtype)

        padded_length = math.ceil(s / 128) * 128 - s
        if padded_length > 0:
            pad_shape = (b, padded_length, n, d)
            roped_query = torch.cat(
                [roped_query, torch.zeros(pad_shape, device=x.device, dtype=v.dtype)],
                dim=1,
            )
            roped_key = torch.cat(
                [roped_key, torch.zeros(pad_shape, device=x.device, dtype=v.dtype)],
                dim=1,
            )
            v = torch.cat(
                [v, torch.zeros(pad_shape, device=x.device, dtype=v.dtype)],
                dim=1,
            )

        x = _flex_attention_impl(
            query=roped_query.transpose(2, 1),
            key=roped_key.transpose(2, 1),
            value=v.transpose(2, 1),
            block_mask=block_mask,
        )
        if padded_length > 0:
            x = x[:, :, :-padded_length]
        x = x.transpose(2, 1).flatten(2)
        return self.o(x)


class CausalWanAttentionBlock(nn.Module):
    def __init__(
        self,
        cross_attn_type,
        dim,
        ffn_dim,
        num_heads,
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
    ):
        super().__init__()
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(dim, num_heads, qk_norm=qk_norm, eps=eps)
        self.norm3 = (
            WanLayerNorm(dim, eps, elementwise_affine=True)
            if cross_attn_norm
            else nn.Identity()
        )
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](
            dim,
            num_heads,
            (-1, -1),
            qk_norm,
            eps,
        )
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        block_mask,
        dtype=torch.float32,
        t=None,
    ):
        num_frames = e.shape[1]
        frame_seqlen = x.shape[1] // num_frames
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)

        temp_x = (
            self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]
        ).flatten(1, 2)
        y = self.self_attn(temp_x.to(dtype), seq_lens, grid_sizes, freqs, block_mask)
        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)

        x = x + self.cross_attn(self.norm3(x), context, context_lens, dtype, t=t)
        temp_x = (
            self.norm2(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[4]) + e[3]
        ).flatten(1, 2)
        y = self.ffn(temp_x.to(dtype))
        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[5]).flatten(1, 2)
        return x


class CausalHead(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, math.prod(patch_size) * out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        num_frames = e.shape[1]
        frame_seqlen = x.shape[1] // num_frames
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = self.head(
            self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]
        )
        return x


class CausalWanTransformer3DModelWithConcat(WanTransformer3DModelWithConcat):
    @register_to_config
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
        local_attn_size: int = -1,
    ):
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
        if cross_attn_type is None:
            cross_attn_type = "t2v_cross_attn" if model_type == "t2v" else "i2v_cross_attn"

        self.blocks = nn.ModuleList(
            [
                CausalWanAttentionBlock(
                    cross_attn_type,
                    dim,
                    ffn_dim,
                    num_heads,
                    qk_norm=qk_norm,
                    cross_attn_norm=cross_attn_norm,
                    eps=eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.head = CausalHead(dim, out_dim, patch_size, eps)
        self.local_attn_size = int(local_attn_size)
        self.block_mask = None
        self._cached_mask_signature = None
        self.num_frame_per_block = 1
        self.independent_first_frame = False
        self.register_to_config(local_attn_size=self.local_attn_size)

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str,
        num_frames: int,
        frame_seqlen: int,
        num_frame_per_block: int = 1,
        local_attn_size: int = -1,
    ) -> BlockMask:
        total_length = num_frames * frame_seqlen
        padded_length = math.ceil(total_length / 128) * 128 - total_length
        ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        frame_indices = torch.arange(
            start=0,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device,
        )

        for start in frame_indices:
            ends[start : start + frame_seqlen * num_frame_per_block] = (
                start + frame_seqlen * num_frame_per_block
            )

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            return (
                (kv_idx < ends[q_idx])
                & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))
            ) | (q_idx == kv_idx)

        return create_block_mask(
            attention_mask,
            B=None,
            H=None,
            Q_LEN=total_length + padded_length,
            KV_LEN=total_length + padded_length,
            _compile=False,
            device=device,
        )

    @staticmethod
    def _prepare_teacher_forcing_mask(
        device: torch.device | str,
        num_frames: int,
        frame_seqlen: int,
        num_frame_per_block: int = 1,
    ) -> BlockMask:
        total_length = num_frames * frame_seqlen * 2
        padded_length = math.ceil(total_length / 128) * 128 - total_length
        clean_ends = num_frames * frame_seqlen

        context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_context_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

        attention_block_size = frame_seqlen * num_frame_per_block
        frame_indices = torch.arange(
            start=0,
            end=num_frames * frame_seqlen,
            step=attention_block_size,
            device=device,
            dtype=torch.long,
        )
        for start in frame_indices:
            context_ends[start : start + attention_block_size] = start + attention_block_size

        noisy_image_start_list = torch.arange(
            num_frames * frame_seqlen,
            total_length,
            step=attention_block_size,
            device=device,
            dtype=torch.long,
        )
        noisy_image_end_list = noisy_image_start_list + attention_block_size
        for block_index, (start, end) in enumerate(zip(noisy_image_start_list, noisy_image_end_list)):
            noise_noise_starts[start:end] = start
            noise_noise_ends[start:end] = end
            noise_context_ends[start:end] = block_index * attention_block_size

        def attention_mask(b, h, q_idx, kv_idx):
            clean_mask = (q_idx < clean_ends) & (kv_idx < context_ends[q_idx])
            same_noisy_block = (kv_idx < noise_noise_ends[q_idx]) & (kv_idx >= noise_noise_starts[q_idx])
            prior_context = (kv_idx < noise_context_ends[q_idx]) & (kv_idx >= noise_context_starts[q_idx])
            noise_mask = (q_idx >= clean_ends) & (same_noisy_block | prior_context)
            return (q_idx == kv_idx) | clean_mask | noise_mask

        return create_block_mask(
            attention_mask,
            B=None,
            H=None,
            Q_LEN=total_length + padded_length,
            KV_LEN=total_length + padded_length,
            _compile=False,
            device=device,
        )

    def _get_block_mask(self, device, num_frames: int, frame_seqlen: int, teacher_forcing: bool):
        signature = (teacher_forcing, num_frames, frame_seqlen, self.num_frame_per_block, self.local_attn_size)
        if self.block_mask is not None and self._cached_mask_signature == signature:
            return self.block_mask

        if teacher_forcing:
            block_mask = self._prepare_teacher_forcing_mask(
                device=device,
                num_frames=num_frames,
                frame_seqlen=frame_seqlen,
                num_frame_per_block=self.num_frame_per_block,
            )
        else:
            block_mask = self._prepare_blockwise_causal_attn_mask(
                device=device,
                num_frames=num_frames,
                frame_seqlen=frame_seqlen,
                num_frame_per_block=self.num_frame_per_block,
                local_attn_size=self.local_attn_size,
            )

        self.block_mask = block_mask
        self._cached_mask_signature = signature
        return block_mask

    @staticmethod
    def _maybe_tensor_to_list(value: Optional[torch.Tensor]):
        if value is None:
            return None
        if isinstance(value, torch.Tensor) and value.dim() == 5:
            return [value[i] for i in range(value.shape[0])]
        return value

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
        clean_x=None,
        aug_t=None,
    ):
        del y_camera, full_ref, subject_ref, cond_flag

        x = self._maybe_tensor_to_list(x)
        if x is None:
            raise ValueError("x must be provided")

        if condition_latents is not None:
            condition_latents = self._maybe_tensor_to_list(condition_latents)
            if y is None:
                y = condition_latents
            else:
                y = self._maybe_tensor_to_list(y)
                y = [torch.cat([y_i, condition_latents[i]], dim=0) for i, y_i in enumerate(y)]
        else:
            y = self._maybe_tensor_to_list(y)

        clean_x = self._maybe_tensor_to_list(clean_x)

        device = self.patch_embedding.weight.device
        dtype = x[0].dtype
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long, device=device) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long, device=device)
        if seq_lens.max() > seq_len:
            raise ValueError(f"seq_lens.max()={int(seq_lens.max())} exceeds seq_len={seq_len}")
        x = torch.cat(
            [torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x],
            dim=0,
        )

        num_frames = int(grid_sizes[0][0].item())
        frame_seqlen = int(grid_sizes[0][1].item() * grid_sizes[0][2].item())
        teacher_forcing = clean_x is not None
        block_mask = self._get_block_mask(device, num_frames, frame_seqlen, teacher_forcing)

        if t.dim() == 1:
            t = t.unsqueeze(1).repeat(1, num_frames)
        flat_t = t.flatten()
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, flat_t).type_as(x))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)

        context_lens = None
        if isinstance(context, torch.Tensor):
            context_tensor = context.to(device=device, dtype=dtype)
        else:
            context_tensor = torch.stack(
                [torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context]
            ).to(device=device, dtype=dtype)
        context = self.text_embedding(context_tensor)
        if clip_fea is not None:
            if not hasattr(self, "img_emb"):
                raise ValueError("clip_fea was provided but this backbone does not have img_emb")
            context = torch.cat([self.img_emb(clip_fea), context], dim=1)

        clean_token_count = 0
        if clean_x is not None:
            clean_x = [self.patch_embedding(u.unsqueeze(0)) for u in clean_x]
            clean_x = [u.flatten(2).transpose(1, 2) for u in clean_x]
            clean_token_count = int(clean_x[0].shape[1])
            clean_x = torch.cat(
                [torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in clean_x],
                dim=0,
            )
            x = torch.cat([clean_x, x], dim=1)
            if aug_t is None:
                aug_t = torch.zeros_like(t)
            flat_aug_t = aug_t.flatten()
            e_clean = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, flat_aug_t).type_as(x))
            e0_clean = self.time_projection(e_clean).unflatten(1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
            e0 = torch.cat([e0_clean, e0], dim=1)

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                def custom_forward(x_in, e0_in, seq_lens_in, grid_sizes_in, freqs_in, context_in, context_lens_in):
                    return block(
                        x_in,
                        e0_in,
                        seq_lens_in,
                        grid_sizes_in,
                        freqs_in,
                        context_in,
                        context_lens_in,
                        block_mask,
                        dtype=dtype,
                        t=t,
                    )

                x = torch.utils.checkpoint.checkpoint(
                    custom_forward,
                    x,
                    e0,
                    seq_lens,
                    grid_sizes,
                    self.freqs,
                    context,
                    context_lens,
                    use_reentrant=False,
                )
            else:
                x = block(
                    x,
                    e0,
                    seq_lens,
                    grid_sizes,
                    self.freqs,
                    context,
                    context_lens,
                    block_mask,
                    dtype=dtype,
                    t=t,
                )

        if clean_token_count > 0:
            x = x[:, clean_token_count:]

        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2).to(dtype))
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)
