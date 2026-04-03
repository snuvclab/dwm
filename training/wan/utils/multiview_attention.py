from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass(frozen=True)
class MultiViewTokenLayout:
    num_views: int
    frames_per_view: int
    height_patches: int
    width_patches: int
    tokens_per_frame: int
    total_tokens: int
    view_index: torch.Tensor
    frame_index: torch.Tensor
    spatial_index: torch.Tensor

    @property
    def total_frames(self) -> int:
        return self.num_views * self.frames_per_view


def build_multiview_token_layout(
    num_views: int,
    frames_per_view: int,
    height_patches: int,
    width_patches: int,
    *,
    device: torch.device | None = None,
) -> MultiViewTokenLayout:
    if num_views <= 0:
        raise ValueError(f"num_views must be positive, got {num_views}")
    if frames_per_view <= 0:
        raise ValueError(f"frames_per_view must be positive, got {frames_per_view}")
    if height_patches <= 0 or width_patches <= 0:
        raise ValueError(
            "height_patches and width_patches must be positive, "
            f"got {(height_patches, width_patches)}"
        )

    tokens_per_frame = height_patches * width_patches
    total_tokens = num_views * frames_per_view * tokens_per_frame

    token_ids = torch.arange(total_tokens, device=device, dtype=torch.long)
    view_index = token_ids // (frames_per_view * tokens_per_frame)
    frame_index = (token_ids // tokens_per_frame) % frames_per_view
    spatial_index = token_ids % tokens_per_frame

    return MultiViewTokenLayout(
        num_views=num_views,
        frames_per_view=frames_per_view,
        height_patches=height_patches,
        width_patches=width_patches,
        tokens_per_frame=tokens_per_frame,
        total_tokens=total_tokens,
        view_index=view_index,
        frame_index=frame_index,
        spatial_index=spatial_index,
    )


def build_same_time_cross_view_mask(
    layout: MultiViewTokenLayout,
    *,
    include_same_view_full_attention: bool = True,
    cross_view_temporal_radius: int = 0,
    include_self: bool = True,
    device: torch.device | None = None,
) -> torch.Tensor:
    if cross_view_temporal_radius < 0:
        raise ValueError(
            f"cross_view_temporal_radius must be non-negative, got {cross_view_temporal_radius}"
        )

    view = layout.view_index.to(device=device)
    frame = layout.frame_index.to(device=device)

    same_view = view[:, None] == view[None, :]
    same_frame = (frame[:, None] - frame[None, :]).abs() <= cross_view_temporal_radius

    if include_same_view_full_attention:
        allowed = same_view | ((~same_view) & same_frame)
    else:
        allowed = same_frame

    if include_self:
        idx = torch.arange(layout.total_tokens, device=allowed.device)
        allowed[idx, idx] = True

    return allowed


def summarize_mask_by_frame(mask: torch.Tensor, layout: MultiViewTokenLayout) -> torch.Tensor:
    if mask.ndim != 2 or mask.shape[0] != mask.shape[1]:
        raise ValueError(f"mask must be square 2D, got shape {tuple(mask.shape)}")
    if mask.shape[0] != layout.total_tokens:
        raise ValueError(
            "mask size does not match layout: "
            f"{mask.shape[0]} vs {layout.total_tokens}"
        )

    total_frames = layout.total_frames
    tpf = layout.tokens_per_frame
    frame_mask = mask.view(total_frames, tpf, total_frames, tpf)
    frame_mask = frame_mask.any(dim=3).any(dim=1)
    return frame_mask


def query_token_index(
    layout: MultiViewTokenLayout,
    *,
    query_view: int,
    query_frame: int,
    query_row: int,
    query_col: int,
) -> int:
    if not (0 <= query_view < layout.num_views):
        raise ValueError(f"query_view must be in [0, {layout.num_views}), got {query_view}")
    if not (0 <= query_frame < layout.frames_per_view):
        raise ValueError(
            f"query_frame must be in [0, {layout.frames_per_view}), got {query_frame}"
        )
    if not (0 <= query_row < layout.height_patches):
        raise ValueError(
            f"query_row must be in [0, {layout.height_patches}), got {query_row}"
        )
    if not (0 <= query_col < layout.width_patches):
        raise ValueError(
            f"query_col must be in [0, {layout.width_patches}), got {query_col}"
        )

    spatial_index = query_row * layout.width_patches + query_col
    return (
        query_view * layout.frames_per_view * layout.tokens_per_frame
        + query_frame * layout.tokens_per_frame
        + spatial_index
    )


def extract_query_attention_volume(
    mask: torch.Tensor,
    layout: MultiViewTokenLayout,
    *,
    query_view: int,
    query_frame: int,
    query_row: int,
    query_col: int,
) -> torch.Tensor:
    if mask.ndim != 2 or mask.shape[0] != mask.shape[1]:
        raise ValueError(f"mask must be square 2D, got shape {tuple(mask.shape)}")
    if mask.shape[0] != layout.total_tokens:
        raise ValueError(
            "mask size does not match layout: "
            f"{mask.shape[0]} vs {layout.total_tokens}"
        )

    query_idx = query_token_index(
        layout,
        query_view=query_view,
        query_frame=query_frame,
        query_row=query_row,
        query_col=query_col,
    )
    query_mask = mask[query_idx]
    return query_mask.view(
        layout.num_views,
        layout.frames_per_view,
        layout.height_patches,
        layout.width_patches,
    )


def summarize_query_attention_by_frame(query_attention_volume: torch.Tensor) -> torch.Tensor:
    if query_attention_volume.ndim != 4:
        raise ValueError(
            "query_attention_volume must have shape [num_views, frames_per_view, height_patches, width_patches], "
            f"got {tuple(query_attention_volume.shape)}"
        )
    return query_attention_volume.flatten(2).float().mean(dim=-1)


def mask_density(mask: torch.Tensor) -> float:
    if mask.numel() == 0:
        return 0.0
    return float(mask.float().mean().item())


def estimate_dense_mask_bytes(
    num_views: int,
    frames_per_view: int,
    height_patches: int,
    width_patches: int,
    *,
    bytes_per_entry: int,
) -> Tuple[int, int]:
    total_tokens = num_views * frames_per_view * height_patches * width_patches
    total_entries = total_tokens * total_tokens
    return total_tokens, total_entries * bytes_per_entry
