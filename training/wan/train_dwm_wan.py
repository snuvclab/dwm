# Copyright 2024 The HuggingFace Team.
# Modified for WAN-based training with VideoX-Fun style.
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""
WAN-based training script for DWM (Dexterous World Model).

Supports:
- LoRA training (VideoX-Fun lora_utils style)
- SFT training (full or partial fine-tuning)
- Flow Matching loss
- WanTransformer3DModelWithConcat for conditional generation
"""

import argparse
import gc
import logging
import math
import os
import time
import pickle
import random
import re
import shutil
import warnings
from datetime import datetime as dt, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from training.wan.diffusers_compat import disable_broken_torchao

disable_broken_torchao()

from safetensors.torch import load_file, save_file

import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import (
    DistributedDataParallelKwargs,
    DummyScheduler,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    set_seed,
)
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils import export_to_video
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, RandomSampler
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, T5Tokenizer, UMT5EncoderModel
from transformers.utils import ContextManagers
import imageio.v3 as iio
import cv2

# WAN models and utilities from training.wan
from training.wan.models import (
    AutoencoderKLWan,
    AutoencoderKLWan3_8,
    CLIPModel,
    WanT5EncoderModel,
    WanTransformer3DModel,
)
from training.wan.models.wan_transformer3d_with_conditions import (
    WanTransformer3DModelWithConcat,
)
from training.wan.models.wan_transformer3d_i2v_with_conditions import (
    WanI2VTransformer3DModelWithConcat,
)
from training.wan.models.wan_transformer3d_vace import (
    WanTransformer3DVace,
)
from training.wan.pipeline.pipeline_wan_fun_inpaint import (
    WanFunInpaintPipeline,
)
from training.wan.pipeline.pipeline_wan_fun_inpaint_hand_concat import (
    WanFunInpaintHandConcatPipeline,
)
from training.wan.pipeline.pipeline_wan2_2_fun_inpaint_hand_concat import (
    Wan2_2FunInpaintHandConcatPipeline,
)
from training.wan.pipeline.pipeline_wan_i2v_diffusers_hand_concat import (
    WanI2VDiffusersHandConcatPipeline,
)
from training.wan.pipeline.pipeline_wan_fun_inpaint_hand_vace import (
    WanFunInpaintHandVacePipeline,
)
from training.wan.utils.lora_utils import (
    create_network,
    merge_lora,
    unmerge_lora,
)
from training.wan.utils.lightx2v_compat import (
    load_lightx2v_with_fallback,
)
from training.wan.utils.checkpoint_utils import (
    extract_checkpoint_step,
    list_checkpoint_dirs,
    save_training_checkpoint,
)
from training.wan.utils.utils import (
    filter_kwargs,
    get_image_to_video_latent,
    save_videos_grid,
)
from training.wan.utils.discrete_sampler import DiscreteSampling
from training.wan.config_loader import load_experiment_config

# Dataset
from training.wan.dataset import (
    BucketSampler,
    VideoDatasetWithConditionsAndResizing,
)

# Wandb for logging
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

logger = get_logger(__name__)

PROMPT_MODE_NORMAL = "normal"
PROMPT_MODE_EMPTY = "empty"
VALIDATION_PROMPT_MODE_ORIGINAL = "original"
VALIDATION_PROMPT_MODE_BOTH = "both"


def resolve_prompt_mode(training_config: Dict[str, Any]) -> str:
    """Resolve training prompt mode with limited backward compatibility."""
    prompt_mode = str(training_config.get("prompt_mode", "")).strip().lower()
    if prompt_mode:
        if prompt_mode not in {PROMPT_MODE_NORMAL, PROMPT_MODE_EMPTY}:
            raise ValueError(f"Unsupported prompt_mode: {prompt_mode}")
        return prompt_mode

    if "prompt_dropout_prob" in training_config:
        legacy_prob = float(training_config["prompt_dropout_prob"])
        if legacy_prob == 0.0:
            warnings.warn(
                "training.prompt_dropout_prob is deprecated. Use training.prompt_mode=normal instead.",
                stacklevel=2,
            )
            return PROMPT_MODE_NORMAL
        if legacy_prob == 1.0:
            warnings.warn(
                "training.prompt_dropout_prob is deprecated. Use training.prompt_mode=empty instead.",
                stacklevel=2,
            )
            return PROMPT_MODE_EMPTY
        raise ValueError(
            "training.prompt_dropout_prob no longer supports fractional values. "
            "Use training.prompt_mode with one of: normal, empty."
        )

    return PROMPT_MODE_NORMAL


def resolve_validation_prompt_mode(training_config: Dict[str, Any]) -> str:
    """Resolve validation prompt mode."""
    validation_prompt_mode = str(
        training_config.get("validation_prompt_mode", VALIDATION_PROMPT_MODE_ORIGINAL)
    ).strip().lower()
    valid_modes = {
        VALIDATION_PROMPT_MODE_ORIGINAL,
        PROMPT_MODE_EMPTY,
        VALIDATION_PROMPT_MODE_BOTH,
    }
    if validation_prompt_mode not in valid_modes:
        raise ValueError(f"Unsupported validation_prompt_mode: {validation_prompt_mode}")
    return validation_prompt_mode


def validate_i2v_condition_latent_type(
    pipeline_type: str,
    conditioning_mode: str,
    i2v_condition_latent_type: str,
    use_diffusers_i2v_backend: bool = False,
) -> None:
    """Reject incompatible I2V conditioning setups early."""
    if conditioning_mode != "i2v":
        return

    normalized_pipeline_type = str(pipeline_type).strip().lower()
    if (
        "fun" in normalized_pipeline_type
        and not use_diffusers_i2v_backend
        and i2v_condition_latent_type == "image"
    ):
        raise ValueError(
            "training.i2v_condition_latent_type='image' is unsupported for Fun/Fun-InP pipelines "
            f"(pipeline.type='{pipeline_type}'). Use training.i2v_condition_latent_type='fun_inp' instead, "
            "because Fun-style I2V conditioning expects masked-video latents rather than a first-frame latent "
            "with zero-filled tail blocks."
        )


def get_validation_prompt_variants(prompt_text: str, validation_prompt_mode: str) -> List[tuple[str, str]]:
    """Return validation prompt variants as (label, prompt_text)."""
    if validation_prompt_mode == VALIDATION_PROMPT_MODE_ORIGINAL:
        return [(VALIDATION_PROMPT_MODE_ORIGINAL, prompt_text)]
    if validation_prompt_mode == PROMPT_MODE_EMPTY:
        return [(PROMPT_MODE_EMPTY, "")]
    if validation_prompt_mode == VALIDATION_PROMPT_MODE_BOTH:
        return [
            (VALIDATION_PROMPT_MODE_ORIGINAL, prompt_text),
            (PROMPT_MODE_EMPTY, ""),
        ]
    raise ValueError(f"Unsupported validation_prompt_mode: {validation_prompt_mode}")


def _load_path_entries_file(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip() and not line.lstrip().startswith("#")]


def _load_named_validation_sets(validation_entries: Any, data_root: str) -> List[tuple[str, List[str]]]:
    if isinstance(validation_entries, (list, tuple)):
        normalized_entries = [str(entry) for entry in validation_entries if entry]
    else:
        normalized_entries = [str(validation_entries)]

    validation_groups: List[tuple[str, List[str]]] = []
    for entry in normalized_entries:
        validation_set_path = os.path.join(data_root, entry)
        if os.path.exists(validation_set_path):
            validation_groups.append((entry, _load_path_entries_file(Path(validation_set_path))))
        else:
            logger.warning("Validation set file not found: %s", validation_set_path)
    return validation_groups


def _interleave_validation_groups(validation_groups: List[tuple[str, List[str]]]) -> List[str]:
    interleaved: List[str] = []
    max_group_len = max((len(paths) for _, paths in validation_groups), default=0)
    for index in range(max_group_len):
        for _, paths in validation_groups:
            if index < len(paths):
                interleaved.append(paths[index])
    return interleaved


def _normalize_exclude_videos_files(exclude_videos_file: Optional[Any]) -> List[str]:
    if not exclude_videos_file:
        return []
    if isinstance(exclude_videos_file, (str, Path)):
        return [str(exclude_videos_file)]
    try:
        return [str(path) for path in exclude_videos_file if path]
    except TypeError:
        return [str(exclude_videos_file)]


def load_excluded_video_entries(data_root: str, exclude_videos_file: Optional[Any]) -> set[str]:
    exclude_paths_raw = _normalize_exclude_videos_files(exclude_videos_file)
    if not exclude_paths_raw:
        return set()

    entries: set[str] = set()
    for exclude_path_raw in exclude_paths_raw:
        exclude_path = Path(exclude_path_raw)
        if not exclude_path.is_absolute():
            exclude_path = Path(data_root) / exclude_path
        if not exclude_path.exists():
            raise FileNotFoundError(f"Exclude videos file not found: {exclude_path}")

        path_entries = set(_load_path_entries_file(exclude_path))
        entries.update(path_entries)
        logger.info("Loaded %d excluded videos from %s", len(path_entries), exclude_path)

    logger.info("Loaded %d unique excluded videos from %d file(s)", len(entries), len(exclude_paths_raw))
    return entries


def _safe_read_validation_video(path: Path, role: str) -> Optional[np.ndarray]:
    try:
        return iio.imread(str(path)).astype(np.float32) / 255.0
    except Exception as e:
        logger.warning("Failed to decode %s video %s: %s. Skipping.", role, path, e)
        return None


def _build_validation_video_save_id(video_path: str | Path) -> str:
    """Build a readable, collision-resistant validation video id from a relative dataset path."""
    path_obj = Path(video_path)
    path_no_suffix = path_obj.with_suffix("")
    parts = list(path_no_suffix.parts)
    if len(parts) >= 2 and parts[-2] in {
        "videos",
        "video",
        "robot_mesh_videos",
        "comparison",
        "rgb",
    }:
        parts = parts[:-2] + [parts[-1]]

    sanitized_parts: List[str] = []
    for part in parts:
        cleaned = re.sub(r"[^A-Za-z0-9._@+-]+", "-", str(part)).strip("-")
        if cleaned:
            sanitized_parts.append(cleaned)

    if not sanitized_parts:
        fallback = re.sub(r"[^A-Za-z0-9._@+-]+", "-", path_no_suffix.name).strip("-")
        return fallback or "validation_video"
    return "__".join(sanitized_parts)


def resize_mask(mask, latent, process_first_frame_only=True):
    """Resize mask to match latent dimensions."""
    latent_size = latent.size()
    batch_size, channels, num_frames, height, width = mask.shape

    if process_first_frame_only:
        target_size = list(latent_size[2:])
        target_size[0] = 1
        first_frame_resized = F.interpolate(
            mask[:, :, 0:1, :, :],
            size=target_size,
            mode='trilinear',
            align_corners=False
        )
        
        target_size = list(latent_size[2:])
        target_size[0] = target_size[0] - 1
        if target_size[0] != 0:
            remaining_frames_resized = F.interpolate(
                mask[:, :, 1:, :, :],
                size=target_size,
                mode='trilinear',
                align_corners=False
            )
            resized_mask = torch.cat([first_frame_resized, remaining_frames_resized], dim=2)
        else:
            resized_mask = first_frame_resized
    else:
        target_size = list(latent_size[2:])
        resized_mask = F.interpolate(
            mask,
            size=target_size,
            mode='trilinear',
            align_corners=False
        )
    return resized_mask


def get_sigmas(noise_scheduler, timesteps, n_dim=4, dtype=torch.float32, device="cuda"):
    """Get sigmas for flow matching."""
    sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def custom_mse_loss(noise_pred, target, weighting=None, threshold=50):
    """Custom MSE loss with optional weighting and threshold."""
    noise_pred = noise_pred.float()
    target = target.float()
    diff = noise_pred - target
    mse_loss = F.mse_loss(noise_pred, target, reduction='none')
    mask = (diff.abs() <= threshold).float()
    masked_loss = mse_loss * mask
    if weighting is not None:
        masked_loss = masked_loss * weighting
    final_loss = masked_loss.mean()
    return final_loss


def encode_clip_context_batch(
    clip_image_encoder: Optional[CLIPVisionModel],
    clip_images: Optional[torch.Tensor],
    weight_dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """Encode first-frame images into CLIP hidden states for WAN i2v diffusers training."""
    if clip_image_encoder is None or clip_images is None:
        return None

    if clip_images.ndim == 5:
        # Raw i2v samples carry first-frame images as [B, 1, C, H, W].
        clip_images = clip_images[:, 0]
    elif clip_images.ndim == 3:
        clip_images = clip_images.unsqueeze(0)
    elif clip_images.ndim != 4:
        raise ValueError(
            f"Unexpected clip_images shape for CLIP encoding: {tuple(clip_images.shape)}. "
            "Expected [B, C, H, W] or [B, 1, C, H, W]."
        )

    image_size = int(getattr(clip_image_encoder.config, "image_size", 224))
    pixel_values = F.interpolate(
        clip_images,
        size=(image_size, image_size),
        mode="bicubic",
        align_corners=False,
    )
    pixel_values = pixel_values.mul(0.5).add(0.5).clamp(0, 1)
    mean = torch.tensor(
        [0.48145466, 0.4578275, 0.40821073],
        device=pixel_values.device,
        dtype=pixel_values.dtype,
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        [0.26862954, 0.26130258, 0.27577711],
        device=pixel_values.device,
        dtype=pixel_values.dtype,
    ).view(1, 3, 1, 1)
    pixel_values = (pixel_values - mean) / std

    outputs = clip_image_encoder(pixel_values=pixel_values.to(dtype=weight_dtype))
    return outputs.last_hidden_state


def _resolve_component_path(base_path: str, subpath: Optional[str], default_subpath: str) -> str:
    """
    Resolve model/tokenizer component path.
    - If local `<base_path>/<subpath>` exists, use it.
    - Otherwise treat `subpath` as a direct model id/path (e.g., `google/umt5-xxl`).
    """
    resolved_subpath = subpath or default_subpath
    if os.path.isabs(resolved_subpath):
        return resolved_subpath
    local_candidate = os.path.join(base_path, resolved_subpath)
    if os.path.exists(local_candidate):
        return local_candidate
    return resolved_subpath


def _read_video_fps(video_path: str) -> Optional[float]:
    """Read FPS metadata from a video file with OpenCV."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps is None or not np.isfinite(fps) or fps <= 0:
        return None
    return float(fps)


def _build_time_resample_indices(src_len: int, src_fps: float, tgt_fps: float) -> np.ndarray:
    """Build time-aligned frame indices for FPS resampling."""
    if src_len <= 1 or src_fps <= 0 or tgt_fps <= 0:
        return np.array([0], dtype=np.int64)

    target_len = max(1, int(round(src_len * float(tgt_fps) / float(src_fps))))
    raw = np.linspace(0, src_len - 1, target_len)
    indices = np.clip(np.round(raw), 0, src_len - 1).astype(np.int64)
    # Use monotonic unique indices to avoid duplicated neighboring frames.
    indices = np.unique(indices)
    if indices.size == 0:
        return np.array([0], dtype=np.int64)
    return indices


def _apply_frame_indices(video_np_or_tensor, indices: np.ndarray, layout_spec: str):
    """Apply temporal frame indexing to video in supported layouts."""
    if layout_spec == "FHWC":
        return video_np_or_tensor[indices]
    if layout_spec == "BCFHW":
        return video_np_or_tensor[:, :, indices, :, :]
    raise ValueError(f"Unsupported layout_spec for frame indexing: {layout_spec}")


def _clip_or_pad_validation_video(video_np: np.ndarray, target_num_frames: int) -> np.ndarray:
    """Deterministically trim or tail-pad a validation video to a fixed length."""
    if video_np.shape[0] <= 0:
        raise ValueError("Validation video has no frames")
    if target_num_frames <= 0:
        raise ValueError(f"Invalid target_num_frames={target_num_frames}")

    if video_np.shape[0] > target_num_frames:
        return video_np[:target_num_frames]
    if video_np.shape[0] < target_num_frames:
        pad = np.repeat(video_np[-1:], target_num_frames - video_np.shape[0], axis=0)
        return np.concatenate([video_np, pad], axis=0)
    return video_np


def _align_and_normalize_validation_videos(
    gt_video: np.ndarray,
    condition_video_np: np.ndarray,
    hand_video_np: Optional[np.ndarray] = None,
    static_disparity_video_np: Optional[np.ndarray] = None,
    hand_disparity_video_np: Optional[np.ndarray] = None,
    target_num_frames: Optional[int] = None,
):
    """Align validation modalities to a common length, then optionally normalize to a fixed length."""
    temporal_lengths = [gt_video.shape[0], condition_video_np.shape[0]]
    if hand_video_np is not None:
        temporal_lengths.append(hand_video_np.shape[0])
    if static_disparity_video_np is not None:
        temporal_lengths.append(static_disparity_video_np.shape[0])
    if hand_disparity_video_np is not None:
        temporal_lengths.append(hand_disparity_video_np.shape[0])

    common_len = min(temporal_lengths)
    if common_len <= 0:
        raise ValueError("Validation sample has non-positive temporal length")

    if gt_video.shape[0] != common_len:
        gt_video = gt_video[:common_len]
    if condition_video_np.shape[0] != common_len:
        condition_video_np = condition_video_np[:common_len]
    if hand_video_np is not None and hand_video_np.shape[0] != common_len:
        hand_video_np = hand_video_np[:common_len]
    if static_disparity_video_np is not None and static_disparity_video_np.shape[0] != common_len:
        static_disparity_video_np = static_disparity_video_np[:common_len]
    if hand_disparity_video_np is not None and hand_disparity_video_np.shape[0] != common_len:
        hand_disparity_video_np = hand_disparity_video_np[:common_len]

    if target_num_frames is not None:
        gt_video = _clip_or_pad_validation_video(gt_video, target_num_frames)
        condition_video_np = _clip_or_pad_validation_video(condition_video_np, target_num_frames)
        if hand_video_np is not None:
            hand_video_np = _clip_or_pad_validation_video(hand_video_np, target_num_frames)
        if static_disparity_video_np is not None:
            static_disparity_video_np = _clip_or_pad_validation_video(static_disparity_video_np, target_num_frames)
        if hand_disparity_video_np is not None:
            hand_disparity_video_np = _clip_or_pad_validation_video(hand_disparity_video_np, target_num_frames)

    return (
        gt_video,
        condition_video_np,
        hand_video_np,
        static_disparity_video_np,
        hand_disparity_video_np,
    )


def _resize_and_pad_video(
    video,
    target_h: int,
    target_w: int,
    pad_value: float = 0.0,
    short_side_mode: bool = True,
    center_pad: bool = True,
    layout_spec: str = "FHWC",
):
    """Resize video preserving aspect ratio, then pad to exact target size."""
    if layout_spec == "FHWC":
        # [F, H, W, C] -> [F, C, H, W]
        tensor = torch.from_numpy(video).permute(0, 3, 1, 2).contiguous()
    elif layout_spec == "BCFHW":
        # [B, C, F, H, W] -> [B*F, C, H, W]
        b, c, f, h, w = video.shape
        tensor = video.permute(0, 2, 1, 3, 4).reshape(b * f, c, h, w).contiguous()
    else:
        raise ValueError(f"Unsupported layout_spec for resize/pad: {layout_spec}")

    _, _, src_h, src_w = tensor.shape
    if src_h <= 0 or src_w <= 0:
        raise ValueError(f"Invalid video shape for resize/pad: H={src_h}, W={src_w}")

    if short_side_mode:
        src_short = min(src_h, src_w)
        tgt_short = min(target_h, target_w)
        scale = float(tgt_short) / float(src_short)
        new_h = max(1, int(round(src_h * scale)))
        new_w = max(1, int(round(src_w * scale)))
        # Safety clamp in case rounded long side exceeds target.
        if new_h > target_h or new_w > target_w:
            scale = min(float(target_h) / float(src_h), float(target_w) / float(src_w))
            new_h = max(1, int(round(src_h * scale)))
            new_w = max(1, int(round(src_w * scale)))
    else:
        scale = min(float(target_h) / float(src_h), float(target_w) / float(src_w))
        new_h = max(1, int(round(src_h * scale)))
        new_w = max(1, int(round(src_w * scale)))

    resized = F.interpolate(tensor, size=(new_h, new_w), mode="bilinear", align_corners=False)

    pad_h = max(0, target_h - new_h)
    pad_w = max(0, target_w - new_w)
    if center_pad:
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
    else:
        pad_top, pad_left = 0, 0
        pad_bottom, pad_right = pad_h, pad_w

    padded = F.pad(
        resized,
        (pad_left, pad_right, pad_top, pad_bottom),
        mode="constant",
        value=float(pad_value),
    )

    if layout_spec == "FHWC":
        return padded.permute(0, 2, 3, 1).cpu().numpy().astype(video.dtype, copy=False)

    # BCFHW path: [B*F, C, H, W] -> [B, C, F, H, W]
    b, c, f, _, _ = video.shape
    return padded.reshape(b, f, c, target_h, target_w).permute(0, 2, 1, 3, 4).contiguous().to(video.dtype)


def _normalize_bucket_values(bucket_value: Optional[Any]) -> List[int]:
    if bucket_value is None:
        return []
    if isinstance(bucket_value, (list, tuple)):
        values = bucket_value
    else:
        values = [bucket_value]
    return [int(value) for value in values]


def log_validation(
    vae,
    text_encoder,
    tokenizer,
    clip_image_encoder,
    transformer3d,
    network,
    config,
    args,
    accelerator,
    weight_dtype,
    global_step,
    load_tensors: bool = False,
    vae_path: str = None,
    vae_kwargs: dict = None,
    text_encoder_path: str = None,
    text_encoder_kwargs: dict = None,
    tokenizer_path: str = None,
):
    """Run validation during training using dataset-based validation samples."""
    validation_stage = "initialization"
    transformer_for_val = None
    transformer_for_val_was_training = None
    restore_text_encoder_to_cpu = False
    try:
        logger.info("Running validation...")
        training_config = config.get('training', {})
        custom_settings = training_config.get("custom_settings", {})
        conditioning_mode_for_reload = training_config.get("conditioning_mode", "dwm")
        use_diffusers_i2v_backend_for_reload = bool(custom_settings.get("use_diffusers_i2v_backend", False))
        use_diffusers_i2v_backend_for_reload = (
            use_diffusers_i2v_backend_for_reload and conditioning_mode_for_reload == "i2v"
        )

        # If load_tensors=True, VAE/text_encoder were deleted during training. Reload them for validation.
        models_reloaded = False
        if load_tensors and vae is None:
            if vae_path is None:
                logger.warning("load_tensors=True but vae_path not provided. Skipping validation.")
                return
            logger.info(f"📦 Reloading models for validation...")

            # Reload VAE
            logger.info(f"   Loading VAE from {vae_path}...")
            if use_diffusers_i2v_backend_for_reload:
                vae = diffusers.AutoencoderKLWan.from_pretrained(
                    vae_path,
                    torch_dtype=weight_dtype,
                ).eval()
            else:
                Choosen_AutoencoderKL = {
                    "AutoencoderKLWan": AutoencoderKLWan,
                    "AutoencoderKLWan3_8": AutoencoderKLWan3_8
                }[vae_kwargs.get('vae_type', 'AutoencoderKLWan') if vae_kwargs else 'AutoencoderKLWan']
                vae = Choosen_AutoencoderKL.from_pretrained(
                    vae_path,
                    additional_kwargs=vae_kwargs or {},
                ).eval()
            vae = vae.to(accelerator.device, dtype=weight_dtype)

            # Reload text_encoder and tokenizer
            if text_encoder_path is not None and tokenizer_path is not None:
                logger.info(f"   Loading text encoder from {text_encoder_path}...")
                if use_diffusers_i2v_backend_for_reload:
                    text_encoder = UMT5EncoderModel.from_pretrained(
                        text_encoder_path,
                        dtype=weight_dtype,
                    ).eval()
                else:
                    text_encoder = WanT5EncoderModel.from_pretrained(
                        text_encoder_path,
                        additional_kwargs=text_encoder_kwargs or {},
                        torch_dtype=weight_dtype,
                    ).eval()
                text_encoder = text_encoder.to(accelerator.device)

                logger.info(f"   Loading tokenizer from {tokenizer_path}...")
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

            models_reloaded = True
            logger.info("   Models reloaded successfully.")
        elif text_encoder is not None:
            text_encoder_device = next(text_encoder.parameters()).device
            if text_encoder_device.type == "cpu":
                logger.info("📦 Moving text encoder to GPU for validation...")
                text_encoder = text_encoder.to(accelerator.device)
                restore_text_encoder_to_cpu = True

        # Extract config values
        pipeline_config = config.get('pipeline', {})
        training_config = config.get('training', {})
        custom_settings = training_config.get("custom_settings", {})
        transformer_config = config.get('transformer', {})
        data_config = config.get('data', {})
        scheduler_kwargs = config.get('scheduler_kwargs', {})
        transformer_additional_kwargs = config.get('transformer_additional_kwargs', {})
        validation_prompt_mode = resolve_validation_prompt_mode(training_config)
        validation_lightx2v = bool(custom_settings.get("validation_lightx2v", False))
        validation_lightx2v_lora_path = custom_settings.get(
            "validation_lightx2v_lora_path", "ckpts/loras/lightx2v_rank64.safetensors"
        )
        validation_lightx2v_lora_scale = float(custom_settings.get("validation_lightx2v_lora_scale", 1.0))
        validation_lightx2v_compat_mode = str(custom_settings.get("lightx2v_compat_mode", "auto")).lower()
        validation_lightx2v_compat_strict = bool(custom_settings.get("lightx2v_compat_strict", True))
        validation_lightx2v_apply_non_block_diff = bool(custom_settings.get("lightx2v_apply_non_block_diff", False))

        # Propagate fixed FPS (used for RoPE scaling) into transformer loading kwargs
        if pipeline_config.get("fps") is not None:
            transformer_additional_kwargs["fps"] = int(pipeline_config["fps"])
        
        # Check if WAN 2.2 (needed for transformer loading and CLIP encoder)
        is_wan2_2 = "2.2" in pipeline_config.get("type", "")
        pipeline_type = pipeline_config.get("type", "")

        # Validation runtime adaptation (default off for backward compatibility)
        validation_runtime_adapt = bool(data_config.get("validation_runtime_adapt", False))
        validation_inference_size = data_config.get("validation_inference_size", None)
        validation_resize_mode = data_config.get("validation_resize_mode", "short_side_pad")
        validation_pad_value = float(data_config.get("validation_pad_value", 0))
        validation_center_pad = bool(data_config.get("validation_center_pad", True))
        validation_target_fps = data_config.get("validation_target_fps", None)
        validation_fps_mode = data_config.get("validation_fps_mode", "time_resample")
        validation_resize_to_target = False
        validation_target_h: Optional[int] = None
        validation_target_w: Optional[int] = None
        validation_target_num_frames: Optional[int] = None

        if validation_target_fps is not None:
            try:
                validation_target_fps = float(validation_target_fps)
                if validation_target_fps <= 0:
                    logger.warning(
                        f"Invalid validation_target_fps={validation_target_fps}. Disabling FPS adaptation."
                    )
                    validation_target_fps = None
            except Exception:
                logger.warning(
                    f"Failed to parse validation_target_fps={validation_target_fps}. Disabling FPS adaptation."
                )
                validation_target_fps = None

        inferred_validation_height_buckets = sorted(set(_normalize_bucket_values(data_config.get("height_buckets"))))
        inferred_validation_width_buckets = sorted(set(_normalize_bucket_values(data_config.get("width_buckets"))))
        inferred_validation_frame_buckets = sorted(set(_normalize_bucket_values(data_config.get("frame_buckets"))))

        raw_validation_target_num_frames = data_config.get("max_num_frames", custom_settings.get("max_num_frames"))
        if raw_validation_target_num_frames is not None:
            try:
                validation_target_num_frames = int(raw_validation_target_num_frames)
                if validation_target_num_frames <= 0:
                    raise ValueError
            except Exception:
                logger.warning(
                    f"Invalid validation max_num_frames={raw_validation_target_num_frames}. "
                    "Disabling temporal length normalization."
                )
                validation_target_num_frames = None
        elif len(inferred_validation_frame_buckets) == 1:
            validation_target_num_frames = inferred_validation_frame_buckets[0]
            logger.info(
                "Validation frame-count fallback: using single training frame bucket %d.",
                validation_target_num_frames,
            )

        if validation_target_fps is None:
            if "2.1" in pipeline_type:
                validation_target_fps = 16.0
            elif is_wan2_2 and pipeline_config.get("fps") is not None:
                validation_target_fps = float(pipeline_config.get("fps"))

        if validation_runtime_adapt:
            if (
                not isinstance(validation_inference_size, (list, tuple))
                or len(validation_inference_size) != 2
            ):
                logger.warning(
                    "validation_runtime_adapt=True but validation_inference_size is missing/invalid. "
                    "Falling back to legacy validation behavior."
                )
                validation_runtime_adapt = False
            else:
                try:
                    validation_target_h = int(validation_inference_size[0])
                    validation_target_w = int(validation_inference_size[1])
                    if validation_target_h <= 0 or validation_target_w <= 0:
                        raise ValueError
                except Exception:
                    logger.warning(
                        f"Invalid validation_inference_size={validation_inference_size}. "
                        "Falling back to legacy validation behavior."
                    )
                    validation_runtime_adapt = False
                else:
                    validation_resize_to_target = True

            if validation_resize_mode != "short_side_pad":
                logger.warning(
                    f"Unsupported validation_resize_mode={validation_resize_mode}. "
                    "Using short_side_pad."
                )
                validation_resize_mode = "short_side_pad"

            if validation_fps_mode != "time_resample":
                logger.warning(
                    f"Unsupported validation_fps_mode={validation_fps_mode}. "
                    "Disabling FPS adaptation."
                )
                validation_target_fps = None

            if validation_runtime_adapt and is_wan2_2 and (validation_target_w % 32 != 0):
                logger.warning(
                    f"WAN 2.2 with validation_inference_size width={validation_target_w} is not divisible by 32. "
                    "This may degrade compatibility."
                )
        if (
            not validation_resize_to_target
            and len(inferred_validation_height_buckets) == 1
            and len(inferred_validation_width_buckets) == 1
        ):
            validation_target_h = inferred_validation_height_buckets[0]
            validation_target_w = inferred_validation_width_buckets[0]
            validation_resize_to_target = True
            logger.info(
                "Validation size fallback: using single training bucket %dx%d.",
                validation_target_h,
                validation_target_w,
            )
        elif (
            not validation_resize_to_target
            and (len(inferred_validation_height_buckets) > 1 or len(inferred_validation_width_buckets) > 1)
        ):
            logger.warning(
                "Validation uses multiple spatial buckets and validation_runtime_adapt is disabled. "
                "Skipping automatic validation resize; set validation_inference_size explicitly if needed."
            )

        validation_save_fps = 16
        if validation_runtime_adapt and validation_target_fps is not None:
            validation_save_fps = max(1, int(round(validation_target_fps)))
            logger.info(
                f"🎯 Validation runtime adaptation enabled: size={validation_inference_size}, "
                f"target_fps={validation_target_fps}, save_fps={validation_save_fps}"
            )
        
        # Load validation set from config
        validation_stage = "validation_set_loading"
        validation_entries = data_config.get("validation_set")
        if validation_entries is None:
            logger.warning("No validation_set specified in config. Skipping validation.")
            return
        
        data_root = data_config.get("data_root", args.train_data_dir)
        static_video_source_mode = str(data_config.get("static_video_source_mode", "directory")).strip().lower()
        exclude_videos_file = data_config.get("exclude_videos_file")
        excluded_videos = load_excluded_video_entries(data_root, exclude_videos_file)
        
        validation_sampling_mode = str(data_config.get("validation_sampling_mode", "combined")).strip().lower()
        if validation_sampling_mode not in {"combined", "per_file"}:
            raise ValueError(
                f"Unsupported validation_sampling_mode: {validation_sampling_mode}. "
                "Use one of: combined, per_file."
            )
        validation_missing_prompt_mode = str(
            data_config.get("validation_missing_prompt_mode", "skip")
        ).strip().lower()
        if validation_missing_prompt_mode not in {"skip", "empty"}:
            raise ValueError(
                f"Unsupported validation_missing_prompt_mode: {validation_missing_prompt_mode}. "
                "Use one of: skip, empty."
            )

        validation_groups = _load_named_validation_sets(validation_entries, data_root)
        if not validation_groups:
            logger.warning("No validation set files could be loaded. Skipping validation.")
            return

        if excluded_videos:
            original_validation_count = sum(len(paths) for _, paths in validation_groups)
            validation_groups = [
                (entry_name, [video for video in videos if video not in excluded_videos])
                for entry_name, videos in validation_groups
            ]
            filtered_validation_count = sum(len(paths) for _, paths in validation_groups)
            excluded_count = original_validation_count - filtered_validation_count
            if excluded_count:
                logger.info("Excluded %d validation videos using %s", excluded_count, exclude_videos_file)
        
        # Limit validation samples
        max_validation_videos = data_config.get("max_validation_videos", 2)
        
        # For multi-GPU training, multiply by number of GPUs to ensure each GPU gets different videos
        total_validation_videos = max_validation_videos * accelerator.num_processes
        random_validation = bool(data_config.get("random_validation", False))
        validation_pool_size = sum(len(paths) for _, paths in validation_groups)

        if validation_sampling_mode == "per_file" and len(validation_groups) > 1:
            selected_validation_groups: List[tuple[str, List[str]]] = []
            for group_index, (entry_name, videos) in enumerate(validation_groups):
                entry_pool_size = len(videos)
                if entry_pool_size > total_validation_videos:
                    if random_validation:
                        entry_seed = int(global_step) + group_index * 1000003
                        entry_rng = random.Random(entry_seed)
                        selected_videos = entry_rng.sample(videos, total_validation_videos)
                        logger.info(
                            "Validation selection [%s]: per-file random sample enabled "
                            "(seed=%d, selected=%d/%d)",
                            entry_name,
                            entry_seed,
                            total_validation_videos,
                            entry_pool_size,
                        )
                    else:
                        selected_videos = videos[:total_validation_videos]
                        logger.info(
                            "Validation selection [%s]: per-file sequential prefix (selected=%d/%d)",
                            entry_name,
                            total_validation_videos,
                            entry_pool_size,
                        )
                else:
                    selected_videos = videos[:total_validation_videos]
                    logger.info(
                        "Validation selection [%s]: using full available set (selected=%d/%d)",
                        entry_name,
                        len(selected_videos),
                        entry_pool_size,
                    )
                selected_validation_groups.append((entry_name, selected_videos))
            validation_set = _interleave_validation_groups(selected_validation_groups)
            logger.info(
                "Validation selection: per-file mode enabled across %d validation files "
                "(combined selected=%d/%d)",
                len(selected_validation_groups),
                len(validation_set),
                validation_pool_size,
            )
        else:
            validation_set = [video for _, videos in validation_groups for video in videos]
            # Select validation videos (randomized or sequential)
            if len(validation_set) > total_validation_videos:
                if random_validation:
                    validation_seed = int(global_step)
                    validation_rng = random.Random(validation_seed)
                    validation_set = validation_rng.sample(validation_set, total_validation_videos)
                    logger.info(
                        "Validation selection: random sample enabled (seed=%d, selected=%d/%d)",
                        validation_seed,
                        total_validation_videos,
                        validation_pool_size,
                    )
                else:
                    validation_set = validation_set[:total_validation_videos]
                    logger.info(
                        "Validation selection: sequential prefix (selected=%d/%d)",
                        total_validation_videos,
                        validation_pool_size,
                    )
            else:
                validation_set = validation_set[:total_validation_videos]
                logger.info(
                    "Validation selection: using full available set (selected=%d/%d)",
                    len(validation_set),
                    validation_pool_size,
                )
        
        # Distribute videos across GPUs - each GPU gets different videos
        videos_per_gpu = len(validation_set) // accelerator.num_processes
        start_idx = accelerator.process_index * videos_per_gpu
        end_idx = start_idx + videos_per_gpu if accelerator.process_index < accelerator.num_processes - 1 else len(validation_set)
        
        # Each GPU gets its own subset of videos
        validation_set = validation_set[start_idx:end_idx]
        
        if not validation_set:
            logger.warning("No validation videos found. Skipping validation.")
            return
        
        logger.info(f"🎯 Multi-GPU Validation Strategy:")
        logger.info(f"   - Total GPUs: {accelerator.num_processes}")
        logger.info(f"   - Videos per GPU: {videos_per_gpu}")
        logger.info(f"   - GPU {accelerator.process_index}: Processing {len(validation_set)} validation videos (indices {start_idx}-{end_idx-1})")

        # Determine conditioning mode
        conditioning_mode = training_config.get("conditioning_mode", "dwm")
        i2v_use_hand_condition = bool(training_config.get("i2v_use_hand_condition", False))
        use_diffusers_i2v_backend = bool(custom_settings.get("use_diffusers_i2v_backend", False))
        transformer_type = transformer_config.get("class", "WanTransformer3DModelWithConcat")
        is_wan2_2 = "2.2" in pipeline_config.get("type", "")
        lightx2v_allowed = conditioning_mode == "i2v" and use_diffusers_i2v_backend

        # Reuse the training transformer for validation to avoid duplicating 14B weights on GPU.
        validation_stage = "model_build"
        transformer_for_val = accelerator.unwrap_model(transformer3d)
        transformer_for_val_was_training = transformer_for_val.training
        transformer_for_val.eval()
        torch.cuda.empty_cache()
        logger.info(
            "Validation transformer reuse: allocated=%.2f GiB, reserved=%.2f GiB",
            torch.cuda.memory_allocated() / (1024 ** 3),
            torch.cuda.memory_reserved() / (1024 ** 3),
        )

        validation_stage = "pipeline_build"
        scheduler = FlowMatchEulerDiscreteScheduler(
            **filter_kwargs(FlowMatchEulerDiscreteScheduler, scheduler_kwargs)
        )

        if is_wan2_2:
            # WAN 2.2: No CLIP encoder needed
            logger.info("🔧 Using WAN 2.2 pipeline for validation")
            pipeline = Wan2_2FunInpaintHandConcatPipeline(
                vae=accelerator.unwrap_model(vae),
                text_encoder=accelerator.unwrap_model(text_encoder),
                tokenizer=tokenizer,
                transformer=transformer_for_val,
                transformer_2=None,  # Can be loaded from config if needed
                scheduler=scheduler,
            )
        else:
            # WAN 2.1: CLIP encoder required
            clip_image_encoder_for_pipeline = clip_image_encoder
            if clip_image_encoder_for_pipeline is None:
                logger.warning("⚠️  Creating dummy CLIP encoder for pipeline compatibility")
                # Create a minimal dummy CLIP encoder
                clip_image_encoder_for_pipeline = CLIPModel()
                clip_image_encoder_for_pipeline.eval()
                clip_image_encoder_for_pipeline.requires_grad_(False)
            
            if pipeline_type == "wan2.1_fun_inp":
                logger.info("🔧 Using WAN 2.1 base pipeline for validation")
                pipeline = WanFunInpaintPipeline(
                    vae=accelerator.unwrap_model(vae),
                    text_encoder=accelerator.unwrap_model(text_encoder),
                    tokenizer=tokenizer,
                    transformer=transformer_for_val,
                    scheduler=scheduler,
                    clip_image_encoder=clip_image_encoder_for_pipeline,
                )
            elif pipeline_type in ("wan2.1_fun_inp_hand_concat", "wan2.1_fun_inp_hand_concat_i2v_diffusers"):
                use_diffusers_hand_concat_pipeline = (
                    pipeline_type == "wan2.1_fun_inp_hand_concat_i2v_diffusers"
                    or lightx2v_allowed
                )
                if use_diffusers_hand_concat_pipeline:
                    logger.info("🔧 Using WAN 2.1 diffusers hand concat pipeline for validation (i2v backend)")
                    base_ckpt = args.pretrained_model_name_or_path
                    image_processor_path = os.path.join(base_ckpt, "image_processor")
                    if os.path.isdir(image_processor_path):
                        image_processor_for_pipeline = CLIPImageProcessor.from_pretrained(image_processor_path)
                    else:
                        logger.warning(
                            "CLIP image_processor subfolder not found at %s. Using default CLIPImageProcessor().",
                            image_processor_path,
                        )
                        image_processor_for_pipeline = CLIPImageProcessor()

                    pipeline = WanI2VDiffusersHandConcatPipeline(
                        vae=accelerator.unwrap_model(vae),
                        text_encoder=accelerator.unwrap_model(text_encoder),
                        tokenizer=tokenizer,
                        image_encoder=clip_image_encoder_for_pipeline,
                        image_processor=image_processor_for_pipeline,
                        transformer=transformer_for_val,
                        scheduler=scheduler,
                    )
                else:
                    logger.info("🔧 Using WAN 2.1 hand concat pipeline for validation")
                    pipeline = WanFunInpaintHandConcatPipeline(
                        vae=accelerator.unwrap_model(vae),
                        text_encoder=accelerator.unwrap_model(text_encoder),
                        tokenizer=tokenizer,
                        transformer=transformer_for_val,
                        scheduler=scheduler,
                        clip_image_encoder=clip_image_encoder_for_pipeline,
                    )
            elif pipeline_type == "wan2.1_fun_inp_hand_vace":
                logger.info("🔧 Using WAN 2.1 VACE pipeline for validation")
                pipeline = WanFunInpaintHandVacePipeline(
                    vae=accelerator.unwrap_model(vae),
                    text_encoder=accelerator.unwrap_model(text_encoder),
                    tokenizer=tokenizer,
                    transformer=transformer_for_val,
                    scheduler=scheduler,
                    clip_image_encoder=clip_image_encoder_for_pipeline,
                )
            else:
                raise ValueError(f"Unsupported WAN 2.1 pipeline type: {pipeline_type}")
        pipeline = pipeline.to(accelerator.device, dtype=weight_dtype)
        validation_guidance_scale = 6.0
        validation_num_inference_steps = None
        validation_attention_kwargs = None
        if validation_lightx2v:
            if not validation_lightx2v_lora_path:
                raise ValueError(
                    "validation_lightx2v=true requires a non-empty validation_lightx2v_lora_path."
                )

            peft_capable = hasattr(transformer_for_val, "load_lora_adapter") or hasattr(transformer_for_val, "add_adapter")
            logger.info(
                "Validation transformer PEFT capability check (i2v diffusers backend): %s (%s)",
                peft_capable,
                type(transformer_for_val).__name__,
            )
            if lightx2v_allowed and not peft_capable:
                raise TypeError(
                    f"Expected a PEFT-capable transformer for i2v diffusers backend, got {type(transformer_for_val).__name__}"
                )

            logger.info("Loading LightX2V LoRA for validation: %s", validation_lightx2v_lora_path)
            logger.info(
                "LightX2V compat settings: mode=%s, strict=%s",
                validation_lightx2v_compat_mode,
                validation_lightx2v_compat_strict,
            )
            lightx2v_method, lightx2v_stats = load_lightx2v_with_fallback(
                pipeline=pipeline,
                lora_path=validation_lightx2v_lora_path,
                lora_scale=validation_lightx2v_lora_scale,
                compat_mode=validation_lightx2v_compat_mode,
                compat_strict=validation_lightx2v_compat_strict,
                expected_blocks=None,
                apply_non_block_diff=validation_lightx2v_apply_non_block_diff,
                logger=logger,
            )
            logger.info("LightX2V load method: %s", lightx2v_method)
            if lightx2v_method == "compat":
                logger.info(
                    "LightX2V compat stats: lora_pairs=%s diff=%s diff_b=%s",
                    lightx2v_stats.get("applied_lora_pairs"),
                    lightx2v_stats.get("applied_diff"),
                    lightx2v_stats.get("applied_diff_b"),
                )
            validation_attention_kwargs = {"scale": validation_lightx2v_lora_scale}
            logger.info(
                "LightX2V scale will be applied via attention_kwargs: %s",
                validation_attention_kwargs,
            )

            if hasattr(pipeline, "get_active_adapters"):
                logger.info("Validation pipeline active adapters: %s", pipeline.get_active_adapters())

            logger.info(
                "LightX2V validation enabled. Overriding guidance_scale %.1f -> 1.0 and num_inference_steps -> 4",
                validation_guidance_scale,
            )
            validation_guidance_scale = 1.0
            validation_num_inference_steps = 4

        if validation_lightx2v and validation_attention_kwargs is not None:
            logger.info("LightX2V attention scale path: attention_kwargs.")

        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
        
        # Get subdirectory names from config
        prompt_subdir = data_config.get("prompt_subdir", "prompts")
        hand_video_subdir = data_config.get("hand_video_subdir", "videos_hands")
        static_video_subdir = data_config.get("static_video_subdir", "videos_static")
        static_disparity_subdir = data_config.get("static_disparity_subdir")
        hand_disparity_subdir = data_config.get("hand_disparity_subdir")

        # Run validation for each sample
        logged_attention_kwargs_passthrough = False
        for i, video_path in enumerate(validation_set):
            validation_stage = f"sampling:{i+1}/{len(validation_set)}"
            video_path_obj = Path(video_path)
            video_name = video_path_obj.stem
            video_save_id = _build_validation_video_save_id(video_path_obj)
            
            # Construct paths for condition videos and prompt
            full_video_path = Path(data_root) / video_path
            prompt_path = full_video_path.parent.parent / prompt_subdir / f"{video_name}.txt"
            hand_video_path = (
                full_video_path.parent.parent / hand_video_subdir / video_path_obj.name
                if hand_video_subdir else None
            )
            static_video_path = full_video_path.parent.parent / static_video_subdir / video_path_obj.name
            static_disparity_video_path = None
            if static_disparity_subdir:
                static_disparity_video_path = (
                    full_video_path.parent.parent / static_disparity_subdir / video_path_obj.name
                )
            hand_disparity_video_path = None
            if hand_disparity_subdir:
                hand_disparity_video_path = (
                    full_video_path.parent.parent / hand_disparity_subdir / video_path_obj.name
                )
            
            # Check if required files exist
            if validation_prompt_mode == PROMPT_MODE_EMPTY:
                prompt_text = ""
            else:
                if not prompt_path.exists():
                    if validation_missing_prompt_mode == "empty":
                        logger.warning("Prompt not found: %s. Using empty prompt.", prompt_path)
                        prompt_text = ""
                    else:
                        logger.warning(f"Prompt not found: {prompt_path}. Skipping.")
                        continue
                else:
                    with open(prompt_path, "r") as f:
                        prompt_text = f.read().strip()

            if (
                conditioning_mode == "dwm"
                and static_video_source_mode != "copy_first_frame"
                and not static_video_path.exists()
            ):
                logger.warning(f"Static video not found: {static_video_path}. Skipping.")
                continue

            # Load GT video for comparison
            gt_video = _safe_read_validation_video(full_video_path, "ground-truth")
            if gt_video is None:
                continue
            
            # Load condition source for validation.
            if conditioning_mode == "dwm":
                if static_video_source_mode == "copy_first_frame":
                    condition_video_np = np.repeat(gt_video[:1], gt_video.shape[0], axis=0).copy()
                else:
                    condition_video_np = _safe_read_validation_video(static_video_path, "static condition")
                    if condition_video_np is None:
                        continue
            elif conditioning_mode == "i2v":
                # I2V condition is GT frame 0 only.
                condition_video_np = gt_video.copy()
                condition_video_np[1:] = 0.0
            else:
                raise ValueError(f"Unsupported conditioning_mode for validation: {conditioning_mode}")
            
            # Load hand video if exists
            use_hand_for_validation = (conditioning_mode == "dwm") or i2v_use_hand_condition
            if use_hand_for_validation and hand_video_path is not None and hand_video_path.exists():
                hand_video_np = _safe_read_validation_video(hand_video_path, "hand condition")
            else:
                hand_video_np = None

            if static_disparity_video_path is not None:
                if static_disparity_video_path.exists():
                    static_disparity_video_np = _safe_read_validation_video(
                        static_disparity_video_path, "static disparity"
                    )
                    if static_disparity_video_np is None:
                        continue
                else:
                    logger.warning(
                        f"Static disparity video not found: {static_disparity_video_path}. Skipping."
                    )
                    continue
            else:
                static_disparity_video_np = None

            if hand_disparity_video_path is not None:
                if hand_disparity_video_path.exists():
                    hand_disparity_video_np = _safe_read_validation_video(
                        hand_disparity_video_path, "hand disparity"
                    )
                    if hand_disparity_video_np is None:
                        continue
                else:
                    logger.warning(
                        f"Hand disparity video not found: {hand_disparity_video_path}. Skipping."
                    )
                    continue
            else:
                hand_disparity_video_np = None

            if validation_runtime_adapt:
                source_fps = _read_video_fps(str(full_video_path))
                if source_fps is None and validation_target_fps is not None:
                    logger.warning(
                        f"Could not read FPS from {full_video_path}. Skipping temporal FPS adaptation for this sample."
                    )

                if (
                    validation_target_fps is not None
                    and source_fps is not None
                    and validation_fps_mode == "time_resample"
                ):
                    temporal_lengths = [gt_video.shape[0], condition_video_np.shape[0]]
                    if hand_video_np is not None:
                        temporal_lengths.append(hand_video_np.shape[0])
                    if static_disparity_video_np is not None:
                        temporal_lengths.append(static_disparity_video_np.shape[0])
                    if hand_disparity_video_np is not None:
                        temporal_lengths.append(hand_disparity_video_np.shape[0])
                    common_len = min(temporal_lengths)
                    if common_len <= 0:
                        logger.warning(
                            f"Invalid frame length detected for {video_name}. Skipping temporal FPS adaptation."
                        )
                    else:
                        if gt_video.shape[0] != common_len:
                            gt_video = gt_video[:common_len]
                        if condition_video_np.shape[0] != common_len:
                            condition_video_np = condition_video_np[:common_len]
                        if hand_video_np is not None and hand_video_np.shape[0] != common_len:
                            hand_video_np = hand_video_np[:common_len]
                        if static_disparity_video_np is not None and static_disparity_video_np.shape[0] != common_len:
                            static_disparity_video_np = static_disparity_video_np[:common_len]
                        if hand_disparity_video_np is not None and hand_disparity_video_np.shape[0] != common_len:
                            hand_disparity_video_np = hand_disparity_video_np[:common_len]

                        frame_indices = _build_time_resample_indices(
                            src_len=common_len,
                            src_fps=source_fps,
                            tgt_fps=validation_target_fps,
                        )
                        gt_video = _apply_frame_indices(gt_video, frame_indices, layout_spec="FHWC")
                        condition_video_np = _apply_frame_indices(condition_video_np, frame_indices, layout_spec="FHWC")
                        if hand_video_np is not None:
                            hand_video_np = _apply_frame_indices(hand_video_np, frame_indices, layout_spec="FHWC")
                        if static_disparity_video_np is not None:
                            static_disparity_video_np = _apply_frame_indices(
                                static_disparity_video_np, frame_indices, layout_spec="FHWC"
                            )
                        if hand_disparity_video_np is not None:
                            hand_disparity_video_np = _apply_frame_indices(
                                hand_disparity_video_np, frame_indices, layout_spec="FHWC"
                            )

            try:
                (
                    gt_video,
                    condition_video_np,
                    hand_video_np,
                    static_disparity_video_np,
                    hand_disparity_video_np,
                ) = _align_and_normalize_validation_videos(
                    gt_video=gt_video,
                    condition_video_np=condition_video_np,
                    hand_video_np=hand_video_np,
                    static_disparity_video_np=static_disparity_video_np,
                    hand_disparity_video_np=hand_disparity_video_np,
                    target_num_frames=validation_target_num_frames,
                )
            except ValueError as e:
                logger.warning(
                    "Failed to normalize validation sample %s temporally: %s. Skipping.",
                    video_name,
                    e,
                )
                continue

            if validation_resize_to_target:
                gt_video = _resize_and_pad_video(
                    gt_video,
                    target_h=validation_target_h,
                    target_w=validation_target_w,
                    pad_value=validation_pad_value,
                    short_side_mode=(validation_resize_mode == "short_side_pad"),
                    center_pad=validation_center_pad,
                    layout_spec="FHWC",
                )
                condition_video_np = _resize_and_pad_video(
                    condition_video_np,
                    target_h=validation_target_h,
                    target_w=validation_target_w,
                    pad_value=validation_pad_value,
                    short_side_mode=(validation_resize_mode == "short_side_pad"),
                    center_pad=validation_center_pad,
                    layout_spec="FHWC",
                )
                if hand_video_np is not None:
                    hand_video_np = _resize_and_pad_video(
                        hand_video_np,
                        target_h=validation_target_h,
                        target_w=validation_target_w,
                        pad_value=validation_pad_value,
                        short_side_mode=(validation_resize_mode == "short_side_pad"),
                        center_pad=validation_center_pad,
                        layout_spec="FHWC",
                    )
                if static_disparity_video_np is not None:
                    static_disparity_video_np = _resize_and_pad_video(
                        static_disparity_video_np,
                        target_h=validation_target_h,
                        target_w=validation_target_w,
                        pad_value=validation_pad_value,
                        short_side_mode=(validation_resize_mode == "short_side_pad"),
                        center_pad=validation_center_pad,
                        layout_spec="FHWC",
                    )
                if hand_disparity_video_np is not None:
                    hand_disparity_video_np = _resize_and_pad_video(
                        hand_disparity_video_np,
                        target_h=validation_target_h,
                        target_w=validation_target_w,
                        pad_value=validation_pad_value,
                        short_side_mode=(validation_resize_mode == "short_side_pad"),
                        center_pad=validation_center_pad,
                        layout_spec="FHWC",
                    )

            # Rebuild first-frame-only condition after all runtime adaptations.
            if conditioning_mode == "i2v":
                i2v_condition = np.zeros_like(gt_video)
                i2v_condition[0] = gt_video[0]
                condition_video_np = i2v_condition

            condition_video = torch.from_numpy(condition_video_np).permute(3, 0, 1, 2).unsqueeze(0)  # [1, c, f, h, w]
            if hand_video_np is not None:
                hand_video = torch.from_numpy(hand_video_np).permute(3, 0, 1, 2).unsqueeze(0)  # [1, c, f, h, w]
            else:
                hand_video = None
            static_disparity_video = (
                torch.from_numpy(static_disparity_video_np).permute(3, 0, 1, 2).unsqueeze(0)
                if static_disparity_video_np is not None else None
            )
            hand_disparity_video = (
                torch.from_numpy(hand_disparity_video_np).permute(3, 0, 1, 2).unsqueeze(0)
                if hand_disparity_video_np is not None else None
            )

            # Align width to 32 for WAN 2.2 compatibility (legacy path only)
            if is_wan2_2 and not validation_resize_to_target:
                # Get original dimensions
                num_frames = gt_video.shape[0]
                orig_height, orig_width = gt_video.shape[1], gt_video.shape[2]
                
                # Calculate aligned width (round to nearest multiple of 32)
                aligned_width = round(orig_width / 32) * 32
                
                if aligned_width != orig_width:
                    # Resize GT video: [f, h, w, c] -> resize -> [f, h, w, c]
                    gt_video_tensor = torch.from_numpy(gt_video).permute(0, 3, 1, 2)  # [f, c, h, w]
                    gt_video_tensor = F.interpolate(
                        gt_video_tensor, 
                        size=(orig_height, aligned_width), 
                        mode='bilinear', 
                        align_corners=False
                    )
                    gt_video = gt_video_tensor.permute(0, 2, 3, 1).numpy()  # [f, h, w, c]
                    
                    # Resize condition video: [1, c, f, h, w] -> [1, c, f, h, w]
                    condition_video = F.interpolate(
                        condition_video.squeeze(0),  # [c, f, h, w]
                        size=(orig_height, aligned_width),
                        mode='bilinear',
                        align_corners=False
                    ).unsqueeze(0)  # [1, c, f, h, w]
                    
                    # Resize hand video if exists: [1, c, f, h, w] -> [1, c, f, h, w]
                    if hand_video is not None:
                        hand_video = F.interpolate(
                            hand_video.squeeze(0),  # [c, f, h, w]
                            size=(orig_height, aligned_width),
                            mode='bilinear',
                            align_corners=False
                        ).unsqueeze(0)  # [1, c, f, h, w]
                    if static_disparity_video is not None:
                        static_disparity_video = F.interpolate(
                            static_disparity_video.squeeze(0),
                            size=(orig_height, aligned_width),
                            mode='bilinear',
                            align_corners=False
                        ).unsqueeze(0)
                    if hand_disparity_video is not None:
                        hand_disparity_video = F.interpolate(
                            hand_disparity_video.squeeze(0),
                            size=(orig_height, aligned_width),
                            mode='bilinear',
                            align_corners=False
                        ).unsqueeze(0)
                    
                    height, width = orig_height, aligned_width
                else:
                    height, width = orig_height, orig_width
            else:
                num_frames = gt_video.shape[0]
                height, width = gt_video.shape[1], gt_video.shape[2]
            
            prompt_variants = get_validation_prompt_variants(prompt_text, validation_prompt_mode)
            with torch.no_grad():
                with torch.autocast("cuda", dtype=weight_dtype):
                    for prompt_variant_label, validation_prompt_text in prompt_variants:
                        mask_video = torch.zeros(1, 1, num_frames, height, width)
                        clip_image_for_i2v = None
                        if conditioning_mode == "i2v":
                            # Match original VideoX-Fun behavior: always build CLIP conditioning from
                            # the first frame for I2V validation when a real first frame is available.
                            first_frame = condition_video[0, :, 0].permute(1, 2, 0).cpu().numpy()
                            first_frame = np.clip(first_frame * 255.0, 0, 255).astype(np.uint8)
                            clip_image_for_i2v = Image.fromarray(first_frame)
                        common_sampling_kwargs = {
                            "num_frames": num_frames,
                            "negative_prompt": "bad detailed",
                            "height": height,
                            "width": width,
                            "guidance_scale": validation_guidance_scale,
                            "generator": generator,
                        }
                        if validation_num_inference_steps is not None:
                            common_sampling_kwargs["num_inference_steps"] = validation_num_inference_steps
                        if validation_attention_kwargs is not None:
                            common_sampling_kwargs["attention_kwargs"] = validation_attention_kwargs
                            if not logged_attention_kwargs_passthrough and lightx2v_allowed:
                                logger.info(
                                    "Validation sampling uses attention_kwargs passthrough for LightX2V: %s",
                                    validation_attention_kwargs,
                                )
                                logged_attention_kwargs_passthrough = True

                        if transformer_type == "WanTransformer3DModel":
                            # Base pipeline uses 'video' parameter instead of 'static_video'
                            if conditioning_mode == "i2v":
                                mask_video[:, :, 1:, :, :] = 255
                            sample = pipeline(
                                validation_prompt_text,
                                video=condition_video.to(accelerator.device, dtype=weight_dtype),
                                mask_video=mask_video.to(accelerator.device, dtype=weight_dtype),
                                clip_image=clip_image_for_i2v,
                                **common_sampling_kwargs,
                            ).videos
                        elif transformer_type == "WanTransformer3DModelWithConcat":
                            # Concat pipeline uses 'static_video' and 'hand_video'
                            if conditioning_mode == "i2v":
                                mask_video[:, :, 1:, :, :] = 255
                            concat_sampling_kwargs = {}
                            if isinstance(pipeline, WanFunInpaintHandConcatPipeline):
                                concat_sampling_kwargs["static_disparity_video"] = (
                                    static_disparity_video.to(accelerator.device, dtype=weight_dtype)
                                    if static_disparity_video is not None else None
                                )
                                concat_sampling_kwargs["hand_disparity_video"] = (
                                    hand_disparity_video.to(accelerator.device, dtype=weight_dtype)
                                    if hand_disparity_video is not None else None
                                )
                            sample = pipeline(
                                validation_prompt_text,
                                static_video=condition_video.to(accelerator.device, dtype=weight_dtype),
                                mask_video=mask_video.to(accelerator.device, dtype=weight_dtype),
                                hand_video=hand_video.to(accelerator.device, dtype=weight_dtype) if hand_video is not None else None,
                                clip_image=clip_image_for_i2v,
                                **concat_sampling_kwargs,
                                **common_sampling_kwargs,
                            ).videos
                        elif transformer_type == "WanTransformer3DVace":
                            # VACE pipeline uses 'static_video', 'hand_video', and 'vace_context_scale'
                            vace_context_scale = training_config.get("vace_context_scale", 1.0)
                            if conditioning_mode == "i2v":
                                mask_video[:, :, 1:, :, :] = 255
                            sample = pipeline(
                                validation_prompt_text,
                                static_video=condition_video.to(accelerator.device, dtype=weight_dtype),
                                mask_video=mask_video.to(accelerator.device, dtype=weight_dtype),
                                hand_video=hand_video.to(accelerator.device, dtype=weight_dtype) if hand_video is not None else None,
                                vace_context_scale=vace_context_scale,
                                clip_image=clip_image_for_i2v,
                                **common_sampling_kwargs,
                            ).videos
                        else:
                            raise ValueError(f"Unsupported transformer type: {transformer_type}")

                        # Prepare videos for comparison: condition | hand | generated | gt
                        # All videos should be [1, c, f, h, w] format
                        static_tensor = condition_video.cpu()  # Already [1, c, f, h, w]
                        if conditioning_mode == "i2v":
                            # For comparison visualization, show a fixed first-frame condition across time.
                            static_tensor = static_tensor[:, :, 0:1, :, :].repeat(1, 1, static_tensor.shape[2], 1, 1)

                        if hand_video is not None:
                            hand_tensor = hand_video.cpu()  # Already [1, c, f, h, w]
                        else:
                            hand_tensor = None
                        static_disparity_tensor = static_disparity_video.cpu() if static_disparity_video is not None else None
                        hand_disparity_tensor = hand_disparity_video.cpu() if hand_disparity_video is not None else None

                        sample_tensor = sample.cpu()  # Already [1, c, f, h, w]
                        gt_tensor = torch.from_numpy(gt_video).permute(3, 0, 1, 2).unsqueeze(0)  # [1, c, f, h, w]

                        # Ensure all videos have the same number of frames
                        num_frames_actual = min(
                            static_tensor.shape[2],
                            sample_tensor.shape[2],
                            gt_tensor.shape[2]
                        )
                        if hand_tensor is not None:
                            num_frames_actual = min(num_frames_actual, hand_tensor.shape[2])
                        if static_disparity_tensor is not None:
                            num_frames_actual = min(num_frames_actual, static_disparity_tensor.shape[2])
                        if hand_disparity_tensor is not None:
                            num_frames_actual = min(num_frames_actual, hand_disparity_tensor.shape[2])
                        static_tensor = static_tensor[:, :, :num_frames_actual, :, :]
                        if hand_tensor is not None:
                            hand_tensor = hand_tensor[:, :, :num_frames_actual, :, :]
                        if static_disparity_tensor is not None:
                            static_disparity_tensor = static_disparity_tensor[:, :, :num_frames_actual, :, :]
                        if hand_disparity_tensor is not None:
                            hand_disparity_tensor = hand_disparity_tensor[:, :, :num_frames_actual, :, :]
                        sample_tensor = sample_tensor[:, :, :num_frames_actual, :, :]
                        gt_tensor = gt_tensor[:, :, :num_frames_actual, :, :]

                        comparison_parts = [static_tensor]
                        comparison_labels = ["condition"]
                        if static_disparity_tensor is not None:
                            comparison_parts.append(static_disparity_tensor)
                            comparison_labels.append("static_disparity")
                        if hand_tensor is not None:
                            comparison_parts.append(hand_tensor)
                            comparison_labels.append("hand")
                        if hand_disparity_tensor is not None:
                            comparison_parts.append(hand_disparity_tensor)
                            comparison_labels.append("hand_disparity")
                        comparison_parts.extend([sample_tensor, gt_tensor])
                        comparison_labels.extend(["generated", "gt"])
                        comparison_caption = "(" + "|".join(comparison_labels) + ")"
                        comparison = torch.cat(comparison_parts, dim=4)

                        save_name = f"{video_save_id}_{prompt_variant_label}"
                        gpu_suffix = f"_gpu{accelerator.process_index}" if accelerator.num_processes > 1 else ""

                        os.makedirs(args.output_dir, exist_ok=True)

                        generated_filename = os.path.join(
                            args.output_dir,
                            f"step_{global_step}_generated_{save_name}{gpu_suffix}.mp4",
                        )
                        save_videos_grid(sample_tensor, generated_filename, fps=validation_save_fps)

                        comparison_filename = os.path.join(
                            args.output_dir,
                            f"step_{global_step}_comparison_{save_name}{gpu_suffix}.mp4",
                        )
                        save_videos_grid(comparison, comparison_filename, fps=validation_save_fps)

                        logger.info(
                            f"📹 GPU {accelerator.process_index}: Saved validation videos for {video_path_obj} "
                            f"(save_id={video_save_id}, prompt_mode={prompt_variant_label})"
                        )
                        logger.info(f"   Generated: {generated_filename}")
                        logger.info(f"   Comparison {comparison_caption}: {comparison_filename}")

                        if accelerator.is_main_process:
                            for tracker in accelerator.trackers:
                                if tracker.name == "wandb" and HAS_WANDB:
                                    caption = (
                                        f"step_{global_step}_{video_save_id}_{prompt_variant_label}_comparison "
                                        f"{comparison_caption}"
                                    )
                                    tracker.log({
                                        "validation": [
                                            wandb.Video(comparison_filename, caption=caption, fps=validation_save_fps)
                                        ]
                                    }, step=global_step)

                        logger.info(
                            f"Validation {i+1}/{len(validation_set)}: {video_path_obj} saved "
                            f"(save_id={video_save_id}, prompt_mode={prompt_variant_label})"
                        )

        # Explicitly delete all validation-specific objects
        del pipeline
        del scheduler
        # Clean up reloaded models if they were loaded for this validation run
        if models_reloaded:
            logger.info("📦 Cleaning up reloaded models after validation...")
            del vae
            if text_encoder is not None:
                del text_encoder
            if tokenizer is not None:
                del tokenizer
        elif restore_text_encoder_to_cpu and text_encoder is not None:
            logger.info("📦 Moving text encoder back to CPU after validation...")
            text_encoder = text_encoder.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    except Exception as e:
        if restore_text_encoder_to_cpu and text_encoder is not None:
            text_encoder = text_encoder.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        logger.error(f"Validation error during stage '{validation_stage}': {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        if transformer_for_val is not None and transformer_for_val_was_training:
            transformer_for_val.train()


def save_model_lora(output_dir, network, weight_dtype, accelerator, transformer3d=None, trainable_parameter_patterns=None):
    """Save LoRA weights and non-LoRA trainable weights."""
    if accelerator.is_main_process:
        # Save LoRA weights
        safetensor_save_path = os.path.join(output_dir, "lora_diffusion_pytorch_model.safetensors")
        accelerator.unwrap_model(network).save_weights(safetensor_save_path, weight_dtype, None)
        logger.info(f"Saved LoRA weights to {safetensor_save_path}")
        
        # Save non-LoRA trainable weights (e.g., patch_embedding)
        if transformer3d is not None and trainable_parameter_patterns:
            non_lora_state_dict = {}
            model = accelerator.unwrap_model(transformer3d)
            for name, param in model.named_parameters():
                for pattern in trainable_parameter_patterns:
                    if re.search(pattern, name):
                        non_lora_state_dict[name] = param.data.clone()
                        break
            
            if non_lora_state_dict:
                non_lora_path = os.path.join(output_dir, "non_lora_weights.safetensors")
                save_file(non_lora_state_dict, non_lora_path)
                logger.info(f"Saved non-LoRA weights ({len(non_lora_state_dict)} params) to {non_lora_path}")


def save_model_sft(output_dir, transformer3d, accelerator):
    """Save SFT model weights."""
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(transformer3d)
        model.save_pretrained(os.path.join(output_dir, "transformer"), safe_serialization=True, max_shard_size="5GB")
        logger.info(f"Saved transformer to {output_dir}/transformer")


def save_model_vace(output_dir, transformer3d, accelerator):
    """Save VACE weights (vace_blocks + vace_patch_embedding only)."""
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(transformer3d)

        # Extract only VACE-related weights
        vace_state_dict = {}
        for name, param in model.named_parameters():
            if "vace_blocks" in name or "vace_patch_embedding" in name:
                vace_state_dict[name] = param.data.clone()

        if vace_state_dict:
            vace_path = os.path.join(output_dir, "vace_weights.safetensors")
            save_file(vace_state_dict, vace_path)
            logger.info(f"Saved VACE weights ({len(vace_state_dict)} params) to {vace_path}")
        else:
            logger.warning("No VACE weights found to save!")


def save_train_mode_weights(
    save_path,
    args,
    network,
    weight_dtype,
    accelerator,
    transformer3d,
    trainable_parameter_patterns,
):
    """Save evaluation-friendly model weights for the active training mode."""
    os.makedirs(save_path, exist_ok=True)

    if args.train_mode == "lora":
        save_model_lora(save_path, network, weight_dtype, accelerator, transformer3d, trainable_parameter_patterns)
    elif args.train_mode == "vace_lora":
        save_model_lora(save_path, network, weight_dtype, accelerator, transformer3d, trainable_parameter_patterns=None)
        save_model_vace(save_path, transformer3d, accelerator)
    elif args.train_mode == "vace":
        save_model_vace(save_path, transformer3d, accelerator)
    else:
        save_model_sft(save_path, transformer3d, accelerator)

    logger.info(f"Saved checkpoint to {save_path}")


def load_or_create_empty_prompt_embeds(
    empty_prompt_embeds_path,
    tokenizer,
    text_encoder,
    text_encoder_kwargs,
    accelerator,
    weight_dtype,
):
    if empty_prompt_embeds_path and os.path.exists(empty_prompt_embeds_path):
        logger.info("Loading cached empty prompt embeds from %s", empty_prompt_embeds_path)
        return torch.load(empty_prompt_embeds_path, map_location="cpu")

    if tokenizer is None or text_encoder is None:
        raise ValueError(
            "prompt_mode=empty requires tokenizer/text_encoder when empty_prompt_embeds cache is missing."
        )

    logger.info("Cached empty prompt embeds not found. Building from text encoder.")
    max_length = int((text_encoder_kwargs or {}).get("text_length", 512))
    text_encoder_was_training = text_encoder.training
    text_encoder.eval()

    text_encoder_device = next(text_encoder.parameters()).device
    moved_to_device = False
    if text_encoder_device.type == "cpu":
        text_encoder = text_encoder.to(accelerator.device, dtype=weight_dtype)
        moved_to_device = True

    tokenized = tokenizer(
        [""],
        padding="max_length",
        max_length=max_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = tokenized.input_ids.to(accelerator.device)
    prompt_attention_mask = tokenized.attention_mask.to(accelerator.device)
    seq_len = int(prompt_attention_mask[0].gt(0).sum().item())

    with torch.no_grad():
        prompt_embeds = text_encoder(text_input_ids, attention_mask=prompt_attention_mask)[0]

    empty_prompt_embeds = prompt_embeds[0, :seq_len].detach().to("cpu").contiguous()

    if text_encoder_was_training:
        text_encoder.train()
    if moved_to_device:
        text_encoder.to("cpu")
        torch.cuda.empty_cache()

    if empty_prompt_embeds_path and accelerator.is_main_process:
        os.makedirs(os.path.dirname(empty_prompt_embeds_path), exist_ok=True)
        torch.save(empty_prompt_embeds, empty_prompt_embeds_path)
        logger.info("Saved empty prompt embeds cache to %s", empty_prompt_embeds_path)

    accelerator.wait_for_everyone()
    return empty_prompt_embeds


def main():
    from training.wan.args import add_all_args
    
    parser = argparse.ArgumentParser(description="WAN-based DWM Training Script")
    # Core arguments
    parser.add_argument("--experiment_config", type=str, required=True,
                        help="Path to experiment YAML config file")
    parser.add_argument("--override", type=str, nargs="*",
                        help="Override config values (key=value format)")
    parser.add_argument("--mode", type=str, choices=["debug", "slurm_test", "slurm", "batch"], default="slurm",
                        help="Training mode: debug, slurm_test, slurm, or batch")
    # # Add all other arguments from args.py
    add_all_args(parser)
    args = parser.parse_args()
    
    # ========== Load Configuration ==========
    print("🚀 Loading experiment configuration...")
    config = load_experiment_config(args.experiment_config, args.override)
    
    # Extract configuration sections
    experiment_config = config.get("experiment", {})
    training_config = config.get("training", {})
    pipeline_config = config.get("pipeline", {})
    transformer_config = config.get("transformer", {})
    data_config = config.get("data", {})
    prompt_mode = resolve_prompt_mode(training_config)
    validation_prompt_mode = resolve_validation_prompt_mode(training_config)
    i2v_condition_latent_type = str(training_config.get("i2v_condition_latent_type", "image")).strip().lower()
    if i2v_condition_latent_type not in {"image", "fun_inp"}:
        raise ValueError(
            f"Unsupported i2v_condition_latent_type: {i2v_condition_latent_type}. "
            "Use one of: image, fun_inp."
        )
    empty_prompt_embeds_path = data_config.get(
        "empty_prompt_embeds_path",
        "data/empty_prompt_embeds_wan.pt",
    )

    # Print training FPS (used for RoPE scaling in WanTransformer3DModel)
    if pipeline_config.get("fps") is not None:
        print(f"🎞️ Training FPS: {int(pipeline_config['fps'])} (from pipeline_config)")
    else:
        print("🎞️ Training FPS: 16 (default)")
    
    # Model-related configs from pipeline config (merged from configs/pipelines/*.yaml)
    text_encoder_kwargs = config.get("text_encoder_kwargs", {})
    vae_kwargs = config.get("vae_kwargs", {})
    scheduler_kwargs = config.get("scheduler_kwargs", {})
    transformer_additional_kwargs = config.get("transformer_additional_kwargs", {})
    image_encoder_kwargs = config.get("image_encoder_kwargs", {})

    # Propagate fixed FPS (used for RoPE scaling) into transformer loading kwargs
    if pipeline_config.get("fps") is not None:
        transformer_additional_kwargs["fps"] = int(pipeline_config["fps"])
    
    # Override args with config values (config values take priority over command line args)
    # Use "is not None" check to ensure config values (including 0, False, empty string) override args
    if training_config.get("learning_rate") is not None:
        args.learning_rate = float(training_config["learning_rate"])
    if training_config.get("batch_size") is not None:
        args.train_batch_size = int(training_config["batch_size"])
    if training_config.get("max_train_steps") is not None:
        args.max_train_steps = int(training_config["max_train_steps"])
    if training_config.get("gradient_accumulation_steps") is not None:
        args.gradient_accumulation_steps = int(training_config["gradient_accumulation_steps"])
    # Note: args.pretrained_model_name_or_path is NOT overridden by pipeline_config
    # to allow specifying SFT checkpoint path via CLI (--pretrained_model_name_or_path)
    # pipeline_config["base_model_name_or_path"] is used for model structure definition
    if args.pretrained_model_name_or_path is None and pipeline_config.get("base_model_name_or_path") is not None:
        args.pretrained_model_name_or_path = pipeline_config["base_model_name_or_path"]
    if data_config.get("data_root") is not None:
        args.train_data_dir = data_config["data_root"]
    if data_config.get("dataset_file") is not None:
        args.train_data_meta = data_config["dataset_file"]
    # Determine transformer type
    transformer_type = transformer_config.get("class", "WanTransformer3DModelWithConcat")
    conditioning_mode = training_config.get("conditioning_mode", "dwm")
    custom_settings = training_config.get("custom_settings", {})
    use_diffusers_i2v_backend = bool(custom_settings.get("use_diffusers_i2v_backend", False))
    keep_full_state_checkpoints = int(
        custom_settings.get("keep_full_state_checkpoints", args.keep_full_state_checkpoints)
    )
    if keep_full_state_checkpoints < 0:
        raise ValueError("training.custom_settings.keep_full_state_checkpoints must be >= 0")
    args.keep_full_state_checkpoints = keep_full_state_checkpoints
    validate_i2v_condition_latent_type(
        pipeline_config.get("type", ""),
        conditioning_mode,
        i2v_condition_latent_type,
        use_diffusers_i2v_backend=use_diffusers_i2v_backend,
    )
    if conditioning_mode != "i2v" and use_diffusers_i2v_backend:
        print(
            "⚠️ training.custom_settings.use_diffusers_i2v_backend=true is only used when "
            f"conditioning_mode='i2v'. Ignoring for conditioning_mode='{conditioning_mode}'."
        )
        use_diffusers_i2v_backend = False
    if use_diffusers_i2v_backend and transformer_type != "WanTransformer3DModelWithConcat":
        print(
            "⚠️ use_diffusers_i2v_backend=true expects "
            "transformer.class='WanTransformer3DModelWithConcat'. "
            f"Got '{transformer_type}'; disabling diffusers i2v backend."
        )
        use_diffusers_i2v_backend = False
    if conditioning_mode == "i2v":
        print(f"ℹ️ I2V backend selection: use_diffusers_i2v_backend={use_diffusers_i2v_backend}")
    if transformer_config.get("condition_channels") is not None:
        args.condition_channels = int(transformer_config["condition_channels"])
    if training_config.get("mode") is not None:
        args.train_mode = training_config["mode"]
    if training_config.get("lora_rank") is not None:
        args.rank = int(training_config["lora_rank"])
    if training_config.get("lora_alpha") is not None:
        args.network_alpha = int(training_config["lora_alpha"])
    
    # ========== Setup Output Directory with SLURM Support ==========
    # Check for resume configuration
    resume_from_checkpoint = training_config.get("resume_from_checkpoint") or args.resume_from_checkpoint
    if resume_from_checkpoint:
        print(f"🔄 Resume mode enabled: resume_from_checkpoint = {resume_from_checkpoint}")
    
    # Setup output directory based on mode
    exp_name = experiment_config.get("name", "wan_training")
    exp_date = experiment_config.get("date", "unknown")
    
    # Extract date suffix
    if exp_date != "unknown":
        try:
            parsed_date = dt.strptime(exp_date, "%Y-%m-%d")
            date_suffix = parsed_date.strftime("%y%m%d")
        except:
            date_suffix = "unknown"
    else:
        date_suffix = "unknown"
    
    # Construct output directory based on mode
    base_output_dir = f"outputs/{date_suffix}/{exp_name}"
    if args.mode == "debug":
        args.output_dir = f"{base_output_dir}_debug"
        print(f"🔧 Debug mode: Output directory set to {args.output_dir}")
    elif args.mode == "slurm_test":
        slurm_job_id = training_config.get("slurm_job_id")
        if resume_from_checkpoint and slurm_job_id:
            print(f"🔄 Resume mode: Using SLURM Job ID from config: {slurm_job_id}")
            args.output_dir = f"{base_output_dir}_slurm_{slurm_job_id}"
        else:
            args.output_dir = f"{base_output_dir}_slurm_test"
        print(f"🧪 SLURM test mode: Output directory set to {args.output_dir}")
    elif args.mode == "slurm":
        if resume_from_checkpoint:
            slurm_job_id = training_config.get("slurm_job_id")
            if slurm_job_id:
                print(f"🔄 Resume mode: Using SLURM Job ID from config: {slurm_job_id}")
            else:
                slurm_job_id = os.environ.get("SLURM_JOB_ID", "unknown")
                print(f"🔄 Resume mode: Using current SLURM Job ID: {slurm_job_id}")
        else:
            slurm_job_id = os.environ.get("SLURM_JOB_ID", "unknown")
            print(f"🚀 New training: Using current SLURM Job ID: {slurm_job_id}")
        args.output_dir = f"{base_output_dir}_slurm_{slurm_job_id}"
        print(f"📁 SLURM mode: Output directory set to {args.output_dir}")
    elif args.mode == "batch":
        # Batch mode: add timestamp to output directory (use env var so all processes get same dir)
        timestamp = os.environ.get("BATCH_TIMESTAMP")
        if not timestamp:
            timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"{base_output_dir}_batch_{timestamp}"
        print(f"🚀 Batch mode: Output directory set to {args.output_dir}")
    
    print(f"📁 Final output directory: {args.output_dir}")
    print(f"   - Base: {base_output_dir}")
    print(f"   - Experiment: {exp_name}")
    print(f"   - Date: {exp_date} -> {date_suffix}")
    print(f"   - Mode: {args.mode}")
    
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    init_process_group_kwargs = InitProcessGroupKwargs(backend="nccl", timeout=timedelta(seconds=1800))

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=training_config.get("custom_settings", {}).get("mixed_precision", args.mixed_precision),
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs, init_process_group_kwargs],
    )
    
    # ========== Detect DeepSpeed / FSDP ==========
    deepspeed_plugin = accelerator.state.deepspeed_plugin if hasattr(accelerator.state, "deepspeed_plugin") else None
    fsdp_plugin = accelerator.state.fsdp_plugin if hasattr(accelerator.state, "fsdp_plugin") else None
    
    if deepspeed_plugin is not None:
        zero_stage = int(deepspeed_plugin.zero_stage)
        fsdp_stage = 0
        print(f"🚀 Using DeepSpeed Zero stage: {zero_stage}")
        args.use_deepspeed = True
    elif fsdp_plugin is not None:
        from torch.distributed.fsdp import ShardingStrategy
        zero_stage = 0
        if fsdp_plugin.sharding_strategy is ShardingStrategy.FULL_SHARD:
            fsdp_stage = 3
        elif fsdp_plugin.sharding_strategy is None:
            fsdp_stage = 3
        elif fsdp_plugin.sharding_strategy is ShardingStrategy.SHARD_GRAD_OP:
            fsdp_stage = 2
        else:
            fsdp_stage = 0
        print(f"🚀 Using FSDP stage: {fsdp_stage}")
        args.use_fsdp = True
    else:
        zero_stage = 0
        fsdp_stage = 0
        print("🚀 DeepSpeed/FSDP is not enabled.")

    # Create output directory and copy experiment config
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Copy experiment config YAML to output directory
        if args.experiment_config:
            config_filename = os.path.basename(args.experiment_config)
            output_config_path = os.path.join(args.output_dir, config_filename)
            shutil.copy2(args.experiment_config, output_config_path)
            print(f"📋 Copied experiment config to: {output_config_path}")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Set seed
    if args.seed is not None:
        set_seed(args.seed)
        rng = np.random.default_rng(np.random.PCG64(args.seed + accelerator.process_index))
        torch_rng = torch.Generator(accelerator.device).manual_seed(args.seed + accelerator.process_index)
    else:
        rng = None
        torch_rng = None

    # Create output directory
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # Determine weight dtype
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # ========== Load Models ==========
    print("🔧 Loading models...")
    
    # Load scheduler
    noise_scheduler = FlowMatchEulerDiscreteScheduler(
        **filter_kwargs(FlowMatchEulerDiscreteScheduler, scheduler_kwargs)
    )

    # Load tokenizer / text encoder / VAE
    if conditioning_mode == "i2v" and use_diffusers_i2v_backend:
        # Video-As-Prompt style: load from diffusers base checkpoint with subfolders.
        base_ckpt = args.pretrained_model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(base_ckpt, subfolder="tokenizer")

        text_encoder = UMT5EncoderModel.from_pretrained(
            base_ckpt,
            subfolder="text_encoder",
            dtype=weight_dtype,
        ).eval()

        vae = diffusers.AutoencoderKLWan.from_pretrained(
            base_ckpt,
            subfolder="vae",
            torch_dtype=weight_dtype,
        ).eval()

        # Keep paths for validation reload.
        tokenizer_path = os.path.join(base_ckpt, "tokenizer")
        text_encoder_path = os.path.join(base_ckpt, "text_encoder")
        vae_path = os.path.join(base_ckpt, "vae")
    else:
        tokenizer_path = _resolve_component_path(
            args.pretrained_model_name_or_path,
            text_encoder_kwargs.get('tokenizer_subpath'),
            'tokenizer',
        )
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        text_encoder_path = _resolve_component_path(
            args.pretrained_model_name_or_path,
            text_encoder_kwargs.get('text_encoder_subpath'),
            'text_encoder',
        )
        text_encoder = WanT5EncoderModel.from_pretrained(
            text_encoder_path,
            additional_kwargs=text_encoder_kwargs,
            torch_dtype=weight_dtype,
        ).eval()

        vae_path = os.path.join(args.pretrained_model_name_or_path, vae_kwargs.get('vae_subpath', 'vae'))
        Choosen_AutoencoderKL = {
            "AutoencoderKLWan": AutoencoderKLWan,
            "AutoencoderKLWan3_8": AutoencoderKLWan3_8
        }[vae_kwargs.get('vae_type', 'AutoencoderKLWan')]
        vae = Choosen_AutoencoderKL.from_pretrained(
            vae_path,
            additional_kwargs=vae_kwargs,
        ).eval()

    # Check if WAN 2.2 (needed for transformer loading and CLIP encoder)
    is_wan2_2 = "2.2" in pipeline_config.get("type", "")
    
    # Load CLIP image encoder (skip only for WAN 2.2)
    if is_wan2_2:
        logger.info("⚠️  Skipping CLIP image encoder loading for WAN 2.2 backend")
        clip_image_encoder = None
    else:
        image_encoder_subpath = image_encoder_kwargs.get("image_encoder_subpath", "image_encoder")
        image_encoder_path = os.path.join(args.pretrained_model_name_or_path, image_encoder_subpath)
        if conditioning_mode == "i2v" and use_diffusers_i2v_backend:
            logger.info(
                "🔧 Loading HF CLIPVision image encoder for i2v-diffusers backend "
                "(base=%s, subfolder=%s)",
                args.pretrained_model_name_or_path,
                image_encoder_subpath,
            )
            try:
                # Video-As-Prompt style: load from base checkpoint + subfolder.
                clip_image_encoder = CLIPVisionModel.from_pretrained(
                    args.pretrained_model_name_or_path,
                    subfolder=image_encoder_subpath,
                    torch_dtype=weight_dtype,
                ).eval()
            except Exception as e:
                if os.path.isdir(image_encoder_path):
                    logger.warning(
                        "Failed subfolder loading, fallback to direct directory loading: %s (%s)",
                        image_encoder_path,
                        e,
                    )
                    clip_image_encoder = CLIPVisionModel.from_pretrained(
                        image_encoder_path,
                        torch_dtype=weight_dtype,
                    ).eval()
                else:
                    raise RuntimeError(
                        f"Failed to load CLIPVision image encoder for i2v-diffusers backend. "
                        f"base={args.pretrained_model_name_or_path}, subfolder={image_encoder_subpath}, "
                        f"resolved_path={image_encoder_path}"
                    ) from e
        else:
            clip_image_encoder = CLIPModel.from_pretrained(image_encoder_path).eval()

    # Load transformer
    if transformer_type == "WanTransformer3DModel":
        print(f"🔧 Loading WanTransformer3DModel (base model)...")
        transformer3d = WanTransformer3DModel.from_pretrained(
            os.path.join(args.pretrained_model_name_or_path, transformer_additional_kwargs.get('transformer_subpath', 'transformer')),
            transformer_additional_kwargs=transformer_additional_kwargs,
            torch_dtype=weight_dtype,
        )
    elif transformer_type == "WanTransformer3DModelWithConcat":
        if conditioning_mode == "i2v" and use_diffusers_i2v_backend:
            transformer_dir = os.path.join(
                args.pretrained_model_name_or_path,
                transformer_additional_kwargs.get('transformer_subpath', 'transformer'),
            )
            print(
                "🔧 Loading WanI2VTransformer3DModelWithConcat "
                f"(diffusers i2v backend, base load -> channel expand) with {args.condition_channels} condition channels..."
            )
            transformer3d = WanI2VTransformer3DModelWithConcat.from_pretrained(
                transformer_dir,
                condition_channels=args.condition_channels,
                torch_dtype=weight_dtype,
            )
        else:
            print(f"🔧 Loading WanTransformer3DModelWithConcat with {args.condition_channels} condition channels...")
            transformer3d = WanTransformer3DModelWithConcat.from_pretrained(
                os.path.join(args.pretrained_model_name_or_path, transformer_additional_kwargs.get('transformer_subpath', 'transformer')),
                transformer_additional_kwargs={
                    "condition_channels": args.condition_channels,
                    "is_wan2_2": is_wan2_2,
                    **transformer_additional_kwargs,
                },
                torch_dtype=weight_dtype,
            )
    elif transformer_type == "WanTransformer3DVace":
        # Get VACE-specific configuration
        vace_layers = transformer_config.get('vace_layers', None)  # e.g., [0, 2, 4, 6, ...] or None for default
        vace_in_dim = transformer_config.get('vace_in_dim', 16)  # Hand latent channels

        print(f"🔧 Loading WanTransformer3DVace with VACE conditioning...")
        print(f"   vace_layers: {vace_layers if vace_layers else 'default (every 2nd layer)'}")
        print(f"   vace_in_dim: {vace_in_dim}")

        transformer3d = WanTransformer3DVace.from_pretrained(
            os.path.join(args.pretrained_model_name_or_path, transformer_additional_kwargs.get('transformer_subpath', 'transformer')),
            transformer_additional_kwargs={
                "vace_layers": vace_layers,
                "vace_in_dim": vace_in_dim,
                "is_wan2_2": is_wan2_2,
                **transformer_additional_kwargs,
            },
            torch_dtype=weight_dtype,
        )
    else:
        raise ValueError(f"Unsupported transformer type: {transformer_type}")

    # Ensure VACE modules are in the correct dtype (they're not loaded from checkpoint)
    if transformer_type == "WanTransformer3DVace":
        if hasattr(transformer3d, 'vace_blocks'):
            for block in transformer3d.vace_blocks:
                block.to(weight_dtype)
            print(f"✅ Converted vace_blocks to {weight_dtype}")
        if hasattr(transformer3d, 'vace_patch_embedding'):
            transformer3d.vace_patch_embedding.to(weight_dtype)
            print(f"✅ Converted vace_patch_embedding to {weight_dtype}")

    # Load custom weights if provided
    if args.transformer_path is not None:
        print(f"Loading transformer from: {args.transformer_path}")
        if args.transformer_path.endswith("safetensors"):
            state_dict = load_file(args.transformer_path)
        else:
            state_dict = torch.load(args.transformer_path, map_location="cpu")
        state_dict = state_dict.get("state_dict", state_dict)
        m, u = transformer3d.load_state_dict(state_dict, strict=False)
        print(f"Missing keys: {len(m)}, Unexpected keys: {len(u)}")

    if args.vae_path is not None:
        print(f"Loading VAE from: {args.vae_path}")
        if args.vae_path.endswith("safetensors"):
            state_dict = load_file(args.vae_path)
        else:
            state_dict = torch.load(args.vae_path, map_location="cpu")
        state_dict = state_dict.get("state_dict", state_dict)
        m, u = vae.load_state_dict(state_dict, strict=False)
        print(f"Missing keys: {len(m)}, Unexpected keys: {len(u)}")

    # ========== Setup Training Mode ==========
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    if clip_image_encoder is not None:
        clip_image_encoder.requires_grad_(False)
    transformer3d.requires_grad_(False)

    network = None
    trainable_params = None
    vace_trainable_params = []

    # Get trainable_parameter_patterns from config (for non-LoRA trainable params like patch_embedding)
    trainable_parameter_patterns = training_config.get("trainable_parameter_patterns", [])
    
    if args.train_mode == "lora":
        print("🔧 Setting up LoRA training...")
        
        # Create LoRA network using VideoX-Fun style
        network = create_network(
            1.0,
            args.rank,
            args.network_alpha,
            text_encoder,
            transformer3d,
            neuron_dropout=None,
            skip_name=args.lora_skip_name,
        )
        network.apply_to(text_encoder, transformer3d, args.train_text_encoder, True)
        
        # Enable additional trainable parameters (e.g., patch_embedding for channel expansion)
        non_lora_trainable_params = []
        if trainable_parameter_patterns:
            print(f"📋 Trainable parameter patterns: {trainable_parameter_patterns}")
            for name, param in transformer3d.named_parameters():
                for pattern in trainable_parameter_patterns:
                    if re.search(pattern, name):
                        param.requires_grad = True
                        non_lora_trainable_params.append(param)
                        print(f"   ✓ Enabled: {name}")
                        break
        
        trainable_params = list(filter(lambda p: p.requires_grad, network.parameters()))
        trainable_params_optim = network.prepare_optimizer_params(
            args.learning_rate / 2, args.learning_rate, args.learning_rate
        )
        
        # Add non-LoRA trainable params to optimizer
        if non_lora_trainable_params:
            trainable_params_optim.append({
                'params': non_lora_trainable_params,
                'lr': args.learning_rate,
            })
            trainable_params.extend(non_lora_trainable_params)
            print(f"📊 Non-LoRA trainable parameters: {sum(p.numel() for p in non_lora_trainable_params):,}")
        
        print(f"📊 LoRA trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    elif args.train_mode == "vace_lora":
        print("🔧 Setting up VACE + Backbone LoRA training (vace modules trainable + backbone LoRA)...")

        # Ensure we're using WanTransformer3DVace
        if transformer_type != "WanTransformer3DVace":
            raise ValueError(f"vace_lora training mode requires WanTransformer3DVace, but got {transformer_type}")

        # Freeze everything first
        transformer3d.requires_grad_(False)

        # In vace_lora mode, default skip to avoid injecting LoRA into vace_* modules
        lora_skip_name = args.lora_skip_name if args.lora_skip_name is not None else "vace_"
        if args.lora_skip_name is None:
            print("ℹ️  vace_lora mode: lora_skip_name not provided, defaulting to 'vace_'")
        else:
            print(f"ℹ️  vace_lora mode: using user-provided lora_skip_name='{args.lora_skip_name}'")

        # Create LoRA network for backbone
        network = create_network(
            1.0,
            args.rank,
            args.network_alpha,
            text_encoder,
            transformer3d,
            neuron_dropout=None,
            skip_name=lora_skip_name,
        )
        network.apply_to(text_encoder, transformer3d, args.train_text_encoder, True)

        # Enable only VACE-specific full-trainable modules
        for name, param in transformer3d.named_parameters():
            if "vace_blocks" in name or "vace_patch_embedding" in name:
                param.requires_grad = True
                vace_trainable_params.append(param)
                print(f"   ✓ VACE enabled: {name}")

        if not vace_trainable_params:
            raise ValueError("No VACE parameters found to train! Check if transformer has vace_blocks and vace_patch_embedding.")

        # LoRA optimizer params + VACE full-train params
        trainable_params_optim = network.prepare_optimizer_params(
            args.learning_rate / 2, args.learning_rate, args.learning_rate
        )
        trainable_params_optim.append({
            'params': vace_trainable_params,
            'lr': args.learning_rate,
        })

        lora_trainable_params = list(filter(lambda p: p.requires_grad, network.parameters()))
        trainable_params = lora_trainable_params + vace_trainable_params

        print(f"📊 LoRA trainable parameters: {sum(p.numel() for p in lora_trainable_params):,}")
        print(f"📊 VACE trainable parameters: {sum(p.numel() for p in vace_trainable_params):,}")
        print(f"📊 Total trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    elif args.train_mode == "sft":
        print("🔧 Setting up SFT training...")
        
        transformer3d.train()
        
        # Enable specific modules for training
        if args.trainable_modules:
            for name, param in transformer3d.named_parameters():
                for trainable_module_name in args.trainable_modules + args.trainable_modules_low_learning_rate:
                    if trainable_module_name in name:
                        param.requires_grad = True
                        break
        else:
            # Full fine-tuning
            transformer3d.requires_grad_(True)
        
        trainable_params = list(filter(lambda p: p.requires_grad, transformer3d.parameters()))
        
        # Setup optimizer params with different learning rates
        trainable_params_optim = [
            {'params': [], 'lr': args.learning_rate},
            {'params': [], 'lr': args.learning_rate / 2},
        ]
        in_already = []
        for name, param in transformer3d.named_parameters():
            if not param.requires_grad or name in in_already:
                continue
            
            high_lr = False
            if args.trainable_modules:
                for module_name in args.trainable_modules:
                    if module_name in name:
                        trainable_params_optim[0]['params'].append(param)
                        in_already.append(name)
                        high_lr = True
                        break
            
            if not high_lr:
                for module_name in args.trainable_modules_low_learning_rate:
                    if module_name in name:
                        trainable_params_optim[1]['params'].append(param)
                        in_already.append(name)
                        break
                else:
                    # Default to high learning rate if no match
                    if not args.trainable_modules:
                        trainable_params_optim[0]['params'].append(param)
        
        print(f"📊 SFT trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    elif args.train_mode == "vace":
        print("🔧 Setting up VACE training (backbone frozen, only vace_blocks + vace_patch_embedding trainable)...")

        # Ensure we're using WanTransformer3DVace
        if transformer_type != "WanTransformer3DVace":
            raise ValueError(f"VACE training mode requires WanTransformer3DVace, but got {transformer_type}")

        # Freeze everything first (already done above, but be explicit)
        transformer3d.requires_grad_(False)

        # Enable only VACE-specific modules
        vace_trainable_params = []
        for name, param in transformer3d.named_parameters():
            if "vace_blocks" in name or "vace_patch_embedding" in name:
                param.requires_grad = True
                vace_trainable_params.append(param)
                print(f"   ✓ Enabled: {name}")

        if not vace_trainable_params:
            raise ValueError("No VACE parameters found to train! Check if transformer has vace_blocks and vace_patch_embedding.")

        trainable_params = vace_trainable_params
        trainable_params_optim = [
            {'params': vace_trainable_params, 'lr': args.learning_rate},
        ]

        print(f"📊 VACE trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    # Enable gradient checkpointing (from args or config)
    gradient_checkpointing = args.gradient_checkpointing or training_config.get("custom_settings", {}).get("gradient_checkpointing", False)
    if gradient_checkpointing:
        transformer3d.enable_gradient_checkpointing()
        print("✅ Gradient checkpointing enabled")

    # Enable TF32
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # ========== Setup Optimizer ==========
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
        except ImportError:
            raise ImportError("Please install bitsandbytes: pip install bitsandbytes")
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        trainable_params_optim,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-08,
    )

    # ========== Setup Learning Rate Scheduler ==========
    use_deepspeed_scheduler = (
        accelerator.state.deepspeed_plugin is not None
        and "scheduler" in accelerator.state.deepspeed_plugin.deepspeed_config
    )

    if use_deepspeed_scheduler:
        lr_scheduler = DummyScheduler(
            name=training_config.get("lr_scheduler", "cosine"),
            optimizer=optimizer,
            total_num_steps=args.max_train_steps * accelerator.num_processes,
            num_warmup_steps=training_config.get("lr_warmup_steps", 0) * accelerator.num_processes,
        )
    else:
        lr_scheduler = get_scheduler(
            training_config.get("lr_scheduler", "cosine"),
            optimizer=optimizer,
            num_warmup_steps=training_config.get("lr_warmup_steps", 0) * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
            num_cycles=training_config.get("lr_num_cycles", 1),
        )

    # ========== Setup Dataset ==========
    print("🔧 Setting up dataset...")
    
    # Use is_wan2_2 already defined above (for width alignment to 32)
    align_width_to_32 = is_wan2_2
    if align_width_to_32:
        print("📐 WAN 2.2 detected: Aligning width to multiples of 32 for compatibility")
    
    # Setup dataset and dataloader
    # Read load_tensors from training.custom_settings (same as cogvideox pattern)
    load_tensors = training_config.get("custom_settings", {}).get("load_tensors", False)
    conditioning_mode = training_config.get("conditioning_mode", "dwm")
    i2v_use_hand_condition = bool(training_config.get("i2v_use_hand_condition", False))
    static_video_source_mode = str(data_config.get("static_video_source_mode", "directory")).strip().lower()
    if load_tensors:
        print("📦 Using pre-encoded latents (load_tensors=True)")

    dataset_init_kwargs = {
        "data_root": data_config["data_root"],
        "dataset_file": data_config["dataset_file"],
        "max_num_frames": data_config.get("max_num_frames", 49),
        "load_tensors": load_tensors,
        "random_flip": data_config.get("random_flip", False),
        "height_buckets": data_config.get("height_buckets", 480),
        "width_buckets": data_config.get("width_buckets", 720),
        "frame_buckets": data_config.get("frame_buckets", 49),
        "prompt_subdir": data_config.get("prompt_subdir", "prompts"),
        "prompt_embeds_subdir": data_config.get("prompt_embeds_subdir", "prompt_embeds_rewrite_wan"),
        "static_video_subdir": data_config.get("static_video_subdir", "videos_static"),
        "static_video_source_mode": static_video_source_mode,
        "hand_video_subdir": data_config.get("hand_video_subdir", "videos_hands"),
        "hand_video_latents_subdir": data_config.get("hand_video_latents_subdir", "hand_video_latents_wan"),
        "static_disparity_subdir": data_config.get("static_disparity_subdir"),
        "static_disparity_latents_subdir": data_config.get("static_disparity_latents_subdir"),
        "hand_disparity_subdir": data_config.get("hand_disparity_subdir"),
        "hand_disparity_latents_subdir": data_config.get("hand_disparity_latents_subdir"),
        "video_latents_subdir": data_config.get("video_latents_subdir", "video_latents_wan"),
        "static_video_latents_subdir": data_config.get("static_video_latents_subdir", "static_video_latents_wan"),
        "image_latents_subdir": data_config.get("image_latents_subdir", "image_latents_wan"),
        "fun_inp_i2v_latents_subdir": data_config.get("fun_inp_i2v_latents_subdir", "fun_inp_i2v_latents_wan"),
        "clip_image_embeds_subdir": data_config.get("clip_image_embeds_subdir", "clip_image_embeds"),
        "exclude_videos_file": data_config.get("exclude_videos_file"),
        "require_static_videos": conditioning_mode == "dwm",
        "align_width_to_32": align_width_to_32,
        "image_to_video": conditioning_mode == "i2v",
    }

    train_dataset = VideoDatasetWithConditionsAndResizing(**dataset_init_kwargs)

    def collate_fn(batch):
        def stack_or_none(key: str):
            if key in batch[0] and batch[0][key] is not None:
                first_tensor = batch[0][key]
                stacked_items = []
                for item in batch:
                    value = item.get(key)
                    if value is None:
                        stacked_items.append(torch.zeros_like(first_tensor))
                    else:
                        stacked_items.append(value)
                return torch.stack(stacked_items)
            return None

        return {
            "videos": torch.stack([item["video"] for item in batch]),
            "images": stack_or_none("image"),
            "fun_inp_i2v_latents": stack_or_none("fun_inp_i2v_latents"),
            "clip_image_embeds": stack_or_none("clip_image_embeds"),
            "prompts": (
                torch.stack([item["prompt"] for item in batch])
                if isinstance(batch[0]["prompt"], torch.Tensor)
                else [item["prompt"] for item in batch]
            ),
            "hand_videos": stack_or_none("hand_videos"),
            "static_videos": stack_or_none("static_videos"),
            "static_disparities": stack_or_none("static_disparities"),
            "hand_disparities": stack_or_none("hand_disparities"),
            "masks": stack_or_none("masks"),
        }
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_config["batch_size"],
        shuffle=True,
        drop_last=True,  # Drop last incomplete batch to ensure consistent batch size
        collate_fn=collate_fn,
        num_workers=data_config.get("dataloader_num_workers", 0),
        pin_memory=data_config.get("pin_memory", True),
    )

    # ========== Prepare with Accelerator ==========
    use_deepspeed_or_fsdp = (fsdp_stage != 0) or (zero_stage != 0)
    
    if args.train_mode in ["lora", "vace_lora"]:
        if use_deepspeed_or_fsdp:
            # DeepSpeed/FSDP: Attach network to transformer so DeepSpeed can find all params
            transformer3d.network = network
            transformer3d = transformer3d.to(weight_dtype)
            transformer3d, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                transformer3d, optimizer, train_dataloader, lr_scheduler
            )
            print(f"📦 Prepared transformer3d with attached LoRA network (DeepSpeed/FSDP mode, train_mode={args.train_mode})")
        else:
            if args.train_mode == "vace_lora":
                # No DeepSpeed/FSDP: prepare both network and transformer
                network, transformer3d, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                    network, transformer3d, optimizer, train_dataloader, lr_scheduler
                )
            else:
                # No DeepSpeed/FSDP: Just prepare network
                network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                    network, optimizer, train_dataloader, lr_scheduler
                )
    else:
        transformer3d, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            transformer3d, optimizer, train_dataloader, lr_scheduler
        )

    # Move models to device (skip transformer3d if already prepared with DeepSpeed/FSDP)
    if not load_tensors:
        # Only keep VAE on device when encoding videos during training
        vae.to(accelerator.device, dtype=weight_dtype)
    if not (args.train_mode in ["lora", "vace_lora"] and use_deepspeed_or_fsdp):
        transformer3d.to(accelerator.device, dtype=weight_dtype)
    if not load_tensors:
        # Only keep text_encoder on device when encoding prompts during training
        text_encoder.to(accelerator.device)
    if clip_image_encoder is not None:
        clip_image_encoder.to(accelerator.device, dtype=weight_dtype)

    empty_prompt_embeds = None
    if prompt_mode == PROMPT_MODE_EMPTY:
        empty_prompt_embeds = load_or_create_empty_prompt_embeds(
            empty_prompt_embeds_path=empty_prompt_embeds_path,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            text_encoder_kwargs=text_encoder_kwargs,
            accelerator=accelerator,
            weight_dtype=weight_dtype,
        )
        if (not load_tensors) and text_encoder is not None:
            logger.info("📦 prompt_mode=empty: moving text_encoder to CPU for training.")
            text_encoder = text_encoder.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize(accelerator.device)

    # Free up memory when using pre-encoded latents
    if load_tensors:
        logger.info("📦 load_tensors=True: Deleting VAE and text_encoder to save memory")
        del vae
        vae = None
        del text_encoder
        del tokenizer
        text_encoder = None
        tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(accelerator.device)
        logger.info("   VAE and text_encoder deleted. GPU memory freed.")

    # Recalculate training steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # ========== Initialize Trackers (WandB) ==========
    if accelerator.is_main_process:
        # Create run name: {date}_{name}_{mode}_{job_id}
        slurm_job_id = os.environ.get("SLURM_JOB_ID", "local")
        run_name = f"{date_suffix}_{exp_name}_{args.mode}_{slurm_job_id}"
        
        # Prepare tracker config
        tracker_config = {
            "experiment": {
                "name": exp_name,
                "date": exp_date if 'exp_date' in dir() else "unknown",
                "output_dir": args.output_dir,
            },
            "training": {
                "mode": args.train_mode,
                "learning_rate": args.learning_rate,
                "batch_size": args.train_batch_size,
                "max_train_steps": args.max_train_steps,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
            },
            "model": {
                "base_model_name_or_path": args.pretrained_model_name_or_path,
                "condition_channels": args.condition_channels,
            },
            "system": {
                "num_gpus": accelerator.num_processes,
                "mixed_precision": args.mixed_precision,
            }
        }
        
        # Setup wandb init kwargs
        wandb_init_kwargs = {
            "name": run_name,
        }
        if args.tracker_entity_name:
            wandb_init_kwargs["entity"] = args.tracker_entity_name

        # Prevent frequent startup timeout in restricted/offline environments.
        wandb_init_timeout = float(custom_settings.get("wandb_init_timeout", 180))
        if HAS_WANDB:
            wandb_init_kwargs["settings"] = wandb.Settings(init_timeout=wandb_init_timeout)
        
        # Set wandb to offline mode for debug and slurm_test modes; disabled for batch
        if args.mode in ["debug", "slurm_test"]:
            wandb_init_kwargs["mode"] = "offline"
            print(f"🔧 WandB set to offline mode for {args.mode}")
        elif args.mode == "batch":
            wandb_init_kwargs["mode"] = "disabled"
            print(f"🔧 Batch mode: WandB disabled")
        
        try:
            accelerator.init_trackers(
                project_name=args.tracker_project_name,
                config=tracker_config,
                init_kwargs={"wandb": wandb_init_kwargs}
            )
        except Exception as e:
            error_text = f"{type(e).__name__}: {e}"
            is_wandb_init_issue = (
                "wandb" in str(args.report_to).lower()
                and (
                    "timeout" in error_text.lower()
                    or "commerror" in error_text.lower()
                    or "timed out waiting for response" in error_text.lower()
                )
            )
            if not is_wandb_init_issue:
                raise

            logger.warning(
                "WandB init failed (%s). Falling back to disabled WandB tracking for this run.",
                error_text,
            )
            fallback_wandb_kwargs = {k: v for k, v in wandb_init_kwargs.items() if k != "settings"}
            fallback_wandb_kwargs["mode"] = "disabled"
            accelerator.init_trackers(
                project_name=args.tracker_project_name,
                config=tracker_config,
                init_kwargs={"wandb": fallback_wandb_kwargs},
            )
            print("🔧 WandB fallback applied: mode=disabled")
        
        print(f"🔗 Experiment initialized: {run_name}")

    # ========== Training Loop ==========
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    first_epoch = 0
    initial_global_step = 0

    # Resume from checkpoint (supports both config-based and arg-based)
    checkpoint_to_resume = resume_from_checkpoint if 'resume_from_checkpoint' in dir() and resume_from_checkpoint else args.resume_from_checkpoint
    
    if checkpoint_to_resume:
        if checkpoint_to_resume != "latest":
            # Use the provided checkpoint path directly
            checkpoint_path = checkpoint_to_resume
            
            if not os.path.exists(checkpoint_path):
                accelerator.print(f"Checkpoint '{checkpoint_path}' does not exist. Starting a new training run.")
                checkpoint_to_resume = None
                initial_global_step = 0
            else:
                accelerator.print(f"Resuming from checkpoint {checkpoint_path}")
                
                # Try to load full accelerator state first, fallback to weight-only loading if it fails
                try:
                    accelerator.load_state(checkpoint_path)
                    accelerator.print(f"Successfully loaded full accelerator state from {checkpoint_path}")
                except Exception as e:
                    accelerator.print(f"Failed to load full accelerator state: {e}")
                    accelerator.print(f"Falling back to weight-only loading...")
                    
                    if args.train_mode == "lora":
                        # Load LoRA weights
                        lora_path = os.path.join(checkpoint_path, "lora_diffusion_pytorch_model.safetensors")
                        if os.path.exists(lora_path):
                            state_dict = load_file(lora_path, device=str(accelerator.device))
                            m, u = accelerator.unwrap_model(network).load_state_dict(state_dict, strict=False)
                            print(f"Loaded LoRA: missing {len(m)}, unexpected {len(u)}")
                        
                        # Load non-LoRA trainable weights (e.g., patch_embedding)
                        non_lora_path = os.path.join(checkpoint_path, "non_lora_weights.safetensors")
                        if os.path.exists(non_lora_path):
                            non_lora_state_dict = load_file(non_lora_path, device=str(accelerator.device))
                            m, u = accelerator.unwrap_model(transformer3d).load_state_dict(non_lora_state_dict, strict=False)
                            print(f"Loaded non-LoRA weights: missing {len(m)}, unexpected {len(u)}")
                    elif args.train_mode == "vace":
                        # Load VACE weights (vace_blocks + vace_patch_embedding)
                        vace_path = os.path.join(checkpoint_path, "vace_weights.safetensors")
                        if os.path.exists(vace_path):
                            vace_state_dict = load_file(vace_path, device=str(accelerator.device))
                            m, u = accelerator.unwrap_model(transformer3d).load_state_dict(vace_state_dict, strict=False)
                            print(f"Loaded VACE weights: missing {len(m)}, unexpected {len(u)}")
                    elif args.train_mode == "vace_lora":
                        # Load LoRA + VACE weights
                        lora_path = os.path.join(checkpoint_path, "lora_diffusion_pytorch_model.safetensors")
                        if os.path.exists(lora_path):
                            state_dict = load_file(lora_path, device=str(accelerator.device))
                            m, u = accelerator.unwrap_model(network).load_state_dict(state_dict, strict=False)
                            print(f"Loaded LoRA: missing {len(m)}, unexpected {len(u)}")

                        vace_path = os.path.join(checkpoint_path, "vace_weights.safetensors")
                        if os.path.exists(vace_path):
                            vace_state_dict = load_file(vace_path, device=str(accelerator.device))
                            m, u = accelerator.unwrap_model(transformer3d).load_state_dict(vace_state_dict, strict=False)
                            print(f"Loaded VACE weights: missing {len(m)}, unexpected {len(u)}")
                    else:
                        # Load transformer weights for SFT
                        transformer_path = os.path.join(checkpoint_path, "transformer")
                        if os.path.exists(transformer_path):
                            if transformer_type == "WanTransformer3DModelWithConcat" and conditioning_mode == "i2v" and use_diffusers_i2v_backend:
                                state_dict = WanI2VTransformer3DModelWithConcat.from_pretrained(
                                    transformer_path,
                                    condition_channels=args.condition_channels,
                                ).state_dict()
                            else:
                                state_dict = WanTransformer3DModelWithConcat.from_pretrained(transformer_path).state_dict()
                            m, u = accelerator.unwrap_model(transformer3d).load_state_dict(state_dict, strict=False)
                            print(f"Loaded transformer: missing {len(m)}, unexpected {len(u)}")
                
                # Extract global step from checkpoint name
                checkpoint_name = os.path.basename(checkpoint_path.rstrip('/'))
                if checkpoint_name.startswith("checkpoint-"):
                    global_step = extract_checkpoint_step(checkpoint_name)
                    initial_global_step = global_step
                    first_epoch = global_step // num_update_steps_per_epoch
                else:
                    accelerator.print(f"Warning: Could not extract global step from checkpoint name '{checkpoint_name}'")
                    initial_global_step = 0
        else:
            # Get the most recent checkpoint from output_dir
            if not os.path.exists(args.output_dir):
                accelerator.print(f"Output directory '{args.output_dir}' does not exist. Starting a new training run.")
                checkpoint_to_resume = None
                initial_global_step = 0
            else:
                dirs = list_checkpoint_dirs(args.output_dir)
                if len(dirs) == 0:
                    accelerator.print("No checkpoint directories found. Starting a new training run.")
                    checkpoint_to_resume = None
                    initial_global_step = 0
                else:
                    latest_checkpoint = dirs[-1]
                    checkpoint_path = os.path.join(args.output_dir, latest_checkpoint)
                    accelerator.print(f"Resuming from latest checkpoint {checkpoint_path}")
                    
                    # Try to load full accelerator state first, fallback to weight-only loading if it fails
                    try:
                        accelerator.load_state(checkpoint_path)
                        accelerator.print(f"Successfully loaded full accelerator state from {checkpoint_path}")
                    except Exception as e:
                        accelerator.print(f"Failed to load full accelerator state: {e}")
                        accelerator.print(f"Falling back to weight-only loading...")
                        
                        if args.train_mode == "lora":
                            lora_path = os.path.join(checkpoint_path, "lora_diffusion_pytorch_model.safetensors")
                            if os.path.exists(lora_path):
                                state_dict = load_file(lora_path, device=str(accelerator.device))
                                m, u = accelerator.unwrap_model(network).load_state_dict(state_dict, strict=False)
                                print(f"Loaded LoRA: missing {len(m)}, unexpected {len(u)}")
                            
                            # Load non-LoRA trainable weights (e.g., patch_embedding)
                            non_lora_path = os.path.join(checkpoint_path, "non_lora_weights.safetensors")
                            if os.path.exists(non_lora_path):
                                non_lora_state_dict = load_file(non_lora_path, device=str(accelerator.device))
                                m, u = accelerator.unwrap_model(transformer3d).load_state_dict(non_lora_state_dict, strict=False)
                                print(f"Loaded non-LoRA weights: missing {len(m)}, unexpected {len(u)}")
                        elif args.train_mode == "vace":
                            # Load VACE weights (vace_blocks + vace_patch_embedding)
                            vace_path = os.path.join(checkpoint_path, "vace_weights.safetensors")
                            if os.path.exists(vace_path):
                                vace_state_dict = load_file(vace_path, device=str(accelerator.device))
                                m, u = accelerator.unwrap_model(transformer3d).load_state_dict(vace_state_dict, strict=False)
                                print(f"Loaded VACE weights: missing {len(m)}, unexpected {len(u)}")
                        elif args.train_mode == "vace_lora":
                            lora_path = os.path.join(checkpoint_path, "lora_diffusion_pytorch_model.safetensors")
                            if os.path.exists(lora_path):
                                state_dict = load_file(lora_path, device=str(accelerator.device))
                                m, u = accelerator.unwrap_model(network).load_state_dict(state_dict, strict=False)
                                print(f"Loaded LoRA: missing {len(m)}, unexpected {len(u)}")

                            vace_path = os.path.join(checkpoint_path, "vace_weights.safetensors")
                            if os.path.exists(vace_path):
                                vace_state_dict = load_file(vace_path, device=str(accelerator.device))
                                m, u = accelerator.unwrap_model(transformer3d).load_state_dict(vace_state_dict, strict=False)
                                print(f"Loaded VACE weights: missing {len(m)}, unexpected {len(u)}")
                        else:
                            # Load transformer weights for SFT
                            transformer_path = os.path.join(checkpoint_path, "transformer")
                            if os.path.exists(transformer_path):
                                if transformer_type == "WanTransformer3DModelWithConcat" and conditioning_mode == "i2v" and use_diffusers_i2v_backend:
                                    state_dict = WanI2VTransformer3DModelWithConcat.from_pretrained(
                                        transformer_path,
                                        condition_channels=args.condition_channels,
                                    ).state_dict()
                                else:
                                    state_dict = WanTransformer3DModelWithConcat.from_pretrained(transformer_path).state_dict()
                                m, u = accelerator.unwrap_model(transformer3d).load_state_dict(state_dict, strict=False)
                                print(f"Loaded transformer weights: missing {len(m)}, unexpected {len(u)}")
                    
                    global_step = extract_checkpoint_step(latest_checkpoint)
                    initial_global_step = global_step
                    first_epoch = global_step // num_update_steps_per_epoch

    # Get step intervals from config (with args as fallback)
    checkpointing_steps = training_config.get("custom_settings", {}).get("checkpointing_steps", args.checkpointing_steps)
    validation_steps = data_config.get("validation_steps", args.validation_steps)
    init_validation_steps = data_config.get("init_validation_steps", 100)
    
    logger.info(f"📊 Step intervals: checkpointing={checkpointing_steps}, validation={validation_steps}, init_validation={init_validation_steps}")
    logger.info(
        "📦 Full-state checkpoint retention: keep_full_state_checkpoints=%s",
        keep_full_state_checkpoints,
    )

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    # Initialize DiscreteSampling for uniform timestep sampling
    idx_sampling = DiscreteSampling(args.train_sampling_steps, uniform_sampling=args.uniform_sampling)

    # Get patch_size from transformer config (used for seq_len calculation)
    patch_size = accelerator.unwrap_model(transformer3d).config.patch_size

    # VAE mini-batch encoding for memory efficiency
    vae_mini_batch = getattr(args, 'vae_mini_batch', 32)
    
    def _batch_encode_vae(pixel_values):
        """Encode pixel values to latents in mini-batches to save memory."""
        bs = vae_mini_batch
        new_latents = []
        for i in range(0, pixel_values.shape[0], bs):
            pixel_values_bs = pixel_values[i:i + bs]
            latents_bs = vae.encode(pixel_values_bs)[0].sample()
            new_latents.append(latents_bs)
        return torch.cat(new_latents, dim=0)

    max_validation_videos = int(data_config.get("max_validation_videos", 2) or 0)
    run_initial_validation = bool(custom_settings.get("run_initial_validation", True))

    # Run initial validation at step 0 (before training starts)
    # Run on all GPUs to distribute validation videos (not just main process)
    if (
        run_initial_validation
        and data_config.get("validation_set") is not None
        and max_validation_videos > 0
    ):
        if accelerator.is_main_process:
            logger.info("🎬 Running initial validation at step 0 (before training starts)")
        log_validation(
            vae, text_encoder, tokenizer, clip_image_encoder,
            transformer3d, network, config, args, accelerator,
            weight_dtype, global_step=global_step,
            load_tensors=load_tensors, vae_path=vae_path, vae_kwargs=vae_kwargs,
            text_encoder_path=text_encoder_path, text_encoder_kwargs=text_encoder_kwargs,
            tokenizer_path=tokenizer_path,
        )
    elif (
        not run_initial_validation
        and accelerator.is_main_process
        and data_config.get("validation_set") is not None
        and max_validation_videos > 0
    ):
        logger.info("Skipping initial validation at step 0 because training.custom_settings.run_initial_validation=false")

    requires_hand_videos = transformer_type in ["WanTransformer3DModelWithConcat", "WanTransformer3DVace"]

    # Time-based checkpoint saving (47h45m); only when slurm and partition is not h100
    training_start_time = time.time()
    time_based_checkpoint_saved = False
    TIME_LIMIT_SECONDS = 47 * 3600 + 45 * 60  # 47 hours 45 minutes
    enable_time_limit_checkpoint = (
        args.mode == "slurm"
        and os.environ.get("SLURM_JOB_PARTITION", "") != "h100"
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            if args.train_mode == "vace_lora":
                models_to_accumulate = [network, transformer3d]
            elif args.train_mode == "lora":
                models_to_accumulate = [network]
            else:
                models_to_accumulate = [transformer3d]

            with accelerator.accumulate(models_to_accumulate):
                # Get batch data
                videos = batch["videos"].to(accelerator.device, dtype=weight_dtype)
                prompts = batch["prompts"]
                images = batch.get("images")
                fun_inp_i2v_latents = batch.get("fun_inp_i2v_latents")
                clip_image_embeds = batch.get("clip_image_embeds")
                hand_videos = batch.get("hand_videos")
                static_videos = batch.get("static_videos")
                static_disparities = batch.get("static_disparities")
                hand_disparities = batch.get("hand_disparities")

                # ===== Encode videos to latents =====
                videos_bcfhw = None
                if load_tensors:
                    # Pre-encoded latents: already [B, C, F', H', W'] from dataset
                    # encode_with_wan.py already called .sample(), no need for VAE
                    latents = videos  # Already latents
                else:
                    # Raw videos: [B, F, C, H, W] -> encode to latents
                    videos_bcfhw = rearrange(videos, "b f c h w -> b c f h w")
                    with torch.no_grad():
                        latents = _batch_encode_vae(videos_bcfhw)

                # ===== Encode condition videos =====
                dropout_mask = None  # Initialize dropout_mask
                use_hand_condition = requires_hand_videos and (
                    conditioning_mode == "dwm" or i2v_use_hand_condition
                )
                if use_hand_condition and hand_videos is not None:
                    if load_tensors:
                        # Pre-encoded hand latents: already [B, C, F', H', W']
                        hand_latents = hand_videos.to(accelerator.device, dtype=weight_dtype)
                    else:
                        # Raw hand videos: encode to latents
                        hand_videos = hand_videos.to(accelerator.device, dtype=weight_dtype)
                        hand_videos = rearrange(hand_videos, "b f c h w -> b c f h w")
                        with torch.no_grad():
                            hand_latents = _batch_encode_vae(hand_videos)
                else:
                    hand_latents = None

                if static_disparities is not None:
                    if load_tensors:
                        static_disparity_latents = static_disparities.to(
                            accelerator.device, dtype=weight_dtype
                        )
                    else:
                        static_disparities = static_disparities.to(
                            accelerator.device, dtype=weight_dtype
                        )
                        static_disparities = rearrange(
                            static_disparities, "b f c h w -> b c f h w"
                        )
                        with torch.no_grad():
                            static_disparity_latents = _batch_encode_vae(static_disparities)
                else:
                    static_disparity_latents = None

                if hand_disparities is not None:
                    if load_tensors:
                        hand_disparity_latents = hand_disparities.to(
                            accelerator.device, dtype=weight_dtype
                        )
                    else:
                        hand_disparities = hand_disparities.to(
                            accelerator.device, dtype=weight_dtype
                        )
                        hand_disparities = rearrange(
                            hand_disparities, "b f c h w -> b c f h w"
                        )
                        with torch.no_grad():
                            hand_disparity_latents = _batch_encode_vae(hand_disparities)
                else:
                    hand_disparity_latents = None

                hand_condition_tensors = []
                if hand_latents is not None:
                    hand_condition_tensors.append(hand_latents)
                if hand_disparity_latents is not None:
                    hand_condition_tensors.append(hand_disparity_latents)

                hand_dropout_prob = training_config.get("hand_dropout_prob", 0.0)
                if hand_condition_tensors and hand_dropout_prob > 0:
                    hand_batch_size = hand_condition_tensors[0].shape[0]
                    dropout_mask = (
                        torch.rand(hand_batch_size, device=accelerator.device) < hand_dropout_prob
                    )

                    if dropout_mask.any():
                        for tensor in hand_condition_tensors:
                            tensor[dropout_mask] = torch.zeros_like(tensor[dropout_mask])
                        if global_step % 100 == 0:
                            num_dropped = dropout_mask.sum().item()
                            accelerator.log({"hand_dropout_count": num_dropped}, step=global_step)

                bsz, latent_channels, latent_frames, latent_height, latent_width = latents.size()
                static_latents = None

                if conditioning_mode == "dwm":
                    if static_videos is None:
                        if load_tensors or static_video_source_mode != "copy_first_frame":
                            raise ValueError("conditioning_mode='dwm' requires static_videos in batch")
                        static_videos = videos[:, 0:1, :, :, :].repeat(1, videos.shape[1], 1, 1, 1)

                    if load_tensors:
                        static_latents = static_videos.to(accelerator.device, dtype=weight_dtype)
                    else:
                        static_videos = static_videos.to(accelerator.device, dtype=weight_dtype)
                        static_videos = rearrange(static_videos, "b f c h w -> b c f h w")
                        with torch.no_grad():
                            static_latents = _batch_encode_vae(static_videos)

                    mask_latents = torch.ones(
                        bsz, 4, latent_frames, latent_height, latent_width,
                        device=accelerator.device, dtype=weight_dtype
                    )
                    inpaint_latents = torch.cat([mask_latents, static_latents], dim=1)
                elif conditioning_mode == "i2v":
                    if load_tensors and i2v_condition_latent_type == "fun_inp":
                        if fun_inp_i2v_latents is None:
                            raise ValueError(
                                "i2v_condition_latent_type='fun_inp' requires precomputed "
                                "fun_inp_i2v_latents. Check data.fun_inp_i2v_latents_subdir and re-encode data."
                            )
                        static_latents = fun_inp_i2v_latents.to(accelerator.device, dtype=weight_dtype)
                        if static_latents.ndim != 5:
                            raise ValueError(
                                f"Unexpected fun_inp_i2v_latents shape: {static_latents.shape}. "
                                "Expected [B, C, F, H, W]."
                            )
                        if static_latents.shape[2:] != (latent_frames, latent_height, latent_width):
                            raise ValueError(
                                "fun_inp_i2v_latents shape mismatch with target latents: "
                                f"got {tuple(static_latents.shape[2:])}, "
                                f"expected {(latent_frames, latent_height, latent_width)}."
                            )
                    elif (not load_tensors) and i2v_condition_latent_type == "fun_inp":
                        if videos_bcfhw is None:
                            raise ValueError(
                                "Raw i2v fun_inp conditioning expects decoded videos before VAE encoding."
                            )

                        # Match encode_fun_inp_i2v_video(): keep frame 0, zero later frames,
                        # then encode the full conditioning video with the causal VAE.
                        conditioned_videos = torch.zeros_like(videos_bcfhw)
                        conditioned_videos[:, :, 0:1, :, :] = videos_bcfhw[:, :, 0:1, :, :]
                        with torch.no_grad():
                            static_latents = _batch_encode_vae(conditioned_videos)

                        if static_latents.shape[2:] != (latent_frames, latent_height, latent_width):
                            raise ValueError(
                                "Raw fun_inp i2v latent shape mismatch with target latents: "
                                f"got {tuple(static_latents.shape[2:])}, "
                                f"expected {(latent_frames, latent_height, latent_width)}."
                            )
                    else:
                        if load_tensors:
                            # Prefer precomputed image latents when present, otherwise use first frame of video latents.
                            first_frame_latents = images if images is not None else latents
                            first_frame_latents = first_frame_latents.to(accelerator.device, dtype=weight_dtype)
                            if first_frame_latents.ndim == 4:
                                first_frame_latents = first_frame_latents.unsqueeze(2)
                            elif first_frame_latents.ndim == 5:
                                first_frame_latents = first_frame_latents[:, :, 0:1, :, :]
                            else:
                                raise ValueError(
                                    f"Unexpected latent shape for i2v first-frame conditioning: {first_frame_latents.shape}"
                                )
                        else:
                            if videos_bcfhw is None:
                                raise ValueError(
                                    "Raw i2v conditioning expects decoded videos before VAE encoding."
                                )

                            # Match original WAN I2V preprocessing: build a conditioning video
                            # that keeps frame 0 and zeroes later frames, then VAE-encode it.
                            conditioned_videos = torch.zeros_like(videos_bcfhw)
                            conditioned_videos[:, :, 0:1, :, :] = videos_bcfhw[:, :, 0:1, :, :]
                            with torch.no_grad():
                                static_latents = _batch_encode_vae(conditioned_videos)

                        if load_tensors:
                            if first_frame_latents.shape[-2:] != (latent_height, latent_width):
                                first_frame_latents = F.interpolate(
                                    first_frame_latents.squeeze(2),
                                    size=(latent_height, latent_width),
                                    mode="bilinear",
                                    align_corners=False,
                                ).unsqueeze(2)

                            static_latents = torch.zeros(
                                bsz, latent_channels, latent_frames, latent_height, latent_width,
                                device=accelerator.device, dtype=weight_dtype
                            )
                            static_latents[:, :, 0:1, :, :] = first_frame_latents[:, :, 0:1, :, :]

                    mask_latents = torch.zeros(
                        bsz, 4, latent_frames, latent_height, latent_width,
                        device=accelerator.device, dtype=weight_dtype
                    )
                    mask_latents[:, :, 0:1, :, :] = 1.0
                    inpaint_latents = torch.cat([mask_latents, static_latents], dim=1)
                else:
                    raise ValueError(f"Unsupported conditioning_mode: {conditioning_mode}")

                expected_condition_channels = int(getattr(args, "condition_channels", 0) or 0)
                condition_tensors = []
                if static_disparity_latents is not None:
                    condition_tensors.append(static_disparity_latents)
                condition_tensors.extend(hand_condition_tensors)
                condition_latents = torch.cat(condition_tensors, dim=1) if condition_tensors else None

                if expected_condition_channels > 0:
                    if condition_latents is None:
                        raise ValueError(
                            f"transformer.condition_channels={expected_condition_channels} "
                            "but no condition latents were loaded."
                        )
                    actual_condition_channels = int(condition_latents.shape[1])
                    if actual_condition_channels != expected_condition_channels:
                        raise ValueError(
                            "Condition channel mismatch: "
                            f"expected {expected_condition_channels}, got {actual_condition_channels}. "
                            "Check hand/static disparity latent subdir settings."
                        )

                use_empty_prompt = prompt_mode == PROMPT_MODE_EMPTY
                if use_empty_prompt and global_step % 100 == 0:
                    accelerator.log({"empty_prompt_count": int(bsz)}, step=global_step)

                if use_empty_prompt:
                    if empty_prompt_embeds is None:
                        raise ValueError("prompt_mode=empty but empty_prompt_embeds is not initialized.")

                    empty_prompt_embeds_device = empty_prompt_embeds.to(
                        accelerator.device, dtype=weight_dtype
                    )
                    if isinstance(prompts, list):
                        batch_prompt_count = len(prompts)
                    elif isinstance(prompts, torch.Tensor):
                        batch_prompt_count = int(prompts.shape[0])
                    else:
                        batch_prompt_count = int(bsz)
                    prompt_embeds = [empty_prompt_embeds_device.clone() for _ in range(batch_prompt_count)]
                else:
                    # Encode prompts only when prompt_mode=normal.
                    with torch.no_grad():
                        if load_tensors:
                            # Pre-encoded embeddings from dataset
                            # prompts is a list of tensors [seq_len_i, dim] from collate (variable lengths)
                            # or a stacked tensor [B, seq_len, dim] if same lengths
                            if isinstance(prompts, torch.Tensor):
                                # If stacked tensor, convert to list for WAN transformer
                                prompt_embeds = [p.to(accelerator.device, dtype=weight_dtype) for p in prompts]
                            else:
                                # Already a list of tensors
                                prompt_embeds = [p.to(accelerator.device, dtype=weight_dtype) for p in prompts]
                        elif isinstance(prompts, list):
                            prompt_ids = tokenizer(
                                prompts,
                                padding="max_length",
                                max_length=args.tokenizer_max_length,
                                truncation=True,
                                add_special_tokens=True,
                                return_tensors="pt"
                            )
                            text_input_ids = prompt_ids.input_ids.to(latents.device)
                            prompt_attention_mask = prompt_ids.attention_mask.to(latents.device)
                            seq_lens = prompt_attention_mask.gt(0).sum(dim=1).long()
                            prompt_embeds = text_encoder(text_input_ids, attention_mask=prompt_attention_mask)[0]
                            prompt_embeds = [u[:v].to(weight_dtype) for u, v in zip(prompt_embeds, seq_lens)]
                        else:
                            prompt_embeds = prompts.to(dtype=weight_dtype)

                clip_context = None
                if conditioning_mode == "i2v":
                    if clip_image_embeds is not None:
                        clip_context = clip_image_embeds.to(accelerator.device, dtype=weight_dtype)
                    elif use_diffusers_i2v_backend and (not load_tensors):
                        with torch.no_grad():
                            clip_context = encode_clip_context_batch(
                                clip_image_encoder,
                                images.to(accelerator.device, dtype=weight_dtype) if images is not None else None,
                                weight_dtype,
                            )

                # Sample noise
                bsz, channel, num_frames, height, width = latents.size()
                noise = torch.randn(latents.size(), device=latents.device, generator=torch_rng, dtype=weight_dtype)

                # Sample timesteps
                if not args.uniform_sampling:
                    # Use density-based sampling (weighting scheme based)
                    u = compute_density_for_timestep_sampling(
                        weighting_scheme=args.weighting_scheme,
                        batch_size=bsz,
                        logit_mean=args.logit_mean,
                        logit_std=args.logit_std,
                        mode_scale=args.mode_scale,
                    )
                    indices = (u * noise_scheduler.config.num_train_timesteps).long()
                else:
                    # Use DiscreteSampling for uniform sampling (supports distributed training)
                    indices = idx_sampling(bsz, generator=torch_rng, device=latents.device)
                    indices = indices.long().cpu()

                timesteps = noise_scheduler.timesteps[indices.cpu()].to(device=latents.device)

                # Get sigmas and create noisy latents (Flow Matching)
                sigmas = get_sigmas(noise_scheduler, timesteps, n_dim=latents.ndim, dtype=latents.dtype, device=accelerator.device)
                noisy_latents = (1.0 - sigmas) * latents + sigmas * noise

                # Flow Matching target
                # If hand_latents were dropped out, use static_latents as target instead of full latents
                # This teaches the model to generate static-only content when hand info is missing
                if dropout_mask is not None and dropout_mask.any() and static_latents is not None:
                    # For dropped samples, use static_latents as target
                    # target shape: [B, 16, F, H, W]
                    target = noise - latents.clone()
                    # Replace target for dropped samples with static-based target
                    target[dropout_mask] = noise[dropout_mask] - static_latents[dropout_mask]
                else:
                    target = noise - latents

                # Calculate seq_len for transformer (based on patched dimensions)
                seq_len = math.ceil(
                    (height / patch_size[1]) * (width / patch_size[2]) * (num_frames / patch_size[0])
                )

                # Forward pass
                # x: noisy_latents [B, 16, F, H, W]
                # y: inpaint_latents [B, 17, F, H, W] (mask + static)
                # condition_latents: concatenated extra conditions [B, C_cond, F, H, W]
                #
                # WanTransformer3DModelWithConcat.forward handles conversion to list format
                # and concatenation of condition_latents
                with torch.amp.autocast(device_type='cuda', dtype=weight_dtype):
                    if transformer_type == "WanTransformer3DModel":
                        # Base model: no condition_latents
                        noise_pred = transformer3d(
                            x=noisy_latents,
                            t=timesteps,
                            context=prompt_embeds,
                            seq_len=seq_len,
                            y=inpaint_latents,
                            clip_fea=clip_context,
                        )
                    elif transformer_type == "WanTransformer3DModelWithConcat":
                        # Concat model: with condition_latents
                        noise_pred = transformer3d(
                            x=noisy_latents,
                            t=timesteps,
                            context=prompt_embeds,
                            seq_len=seq_len,
                            y=inpaint_latents,
                            clip_fea=clip_context,
                            condition_latents=condition_latents,
                        )
                    elif transformer_type == "WanTransformer3DVace":
                        # VACE model: hand condition via vace_context (list format)
                        # Convert hand_latents to list format for VACE
                        if hand_latents is not None:
                            vace_context = [hand_latents[i] for i in range(hand_latents.shape[0])]
                        else:
                            vace_context = None

                        # Get vace_context_scale from config (default 1.0)
                        vace_context_scale = training_config.get("vace_context_scale", 1.0)

                        noise_pred = transformer3d(
                            x=noisy_latents,
                            t=timesteps,
                            context=prompt_embeds,
                            seq_len=seq_len,
                            y=inpaint_latents,
                            clip_fea=clip_context,
                            vace_context=vace_context,
                            vace_context_scale=vace_context_scale,
                        )
                    else:
                        raise ValueError(f"Unsupported transformer type: {transformer_type}")

                # Compute loss with weighting
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                loss = custom_mse_loss(noise_pred.float(), target.float(), weighting.float())
                loss = loss.mean()

                # Gather losses
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backward
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    if args.train_mode == "lora":
                        accelerator.clip_grad_norm_(network.parameters(), args.max_grad_norm)
                    elif args.train_mode == "vace_lora":
                        clip_params = list(network.parameters()) + vace_trainable_params
                        accelerator.clip_grad_norm_(clip_params, args.max_grad_norm)
                    else:
                        accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Update progress
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                # Save checkpoint (step interval or time limit; time limit only when slurm and partition != h100)
                elapsed_time = time.time() - training_start_time
                should_save_time_based_checkpoint = (
                    enable_time_limit_checkpoint
                    and not time_based_checkpoint_saved
                    and elapsed_time >= TIME_LIMIT_SECONDS
                )
                if should_save_time_based_checkpoint or global_step % checkpointing_steps == 0:
                    if accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED:
                        if should_save_time_based_checkpoint:
                            time_based_checkpoint_saved = True
                            elapsed_hours = elapsed_time / 3600
                            logger.info(f"⏰ Time-based checkpoint trigger: {elapsed_hours:.2f} hours elapsed (limit: 47.75 hours)")

                        # Remove old checkpoint directories before saving a new one.
                        if args.checkpoints_total_limit is not None:
                            checkpoints = list_checkpoint_dirs(args.output_dir)
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                if accelerator.is_main_process:
                                    for removing_checkpoint in checkpoints[:num_to_remove]:
                                        removing_path = os.path.join(args.output_dir, removing_checkpoint)
                                        logger.info(f"Removing old checkpoint: {removing_path}")
                                        shutil.rmtree(removing_path)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        save_training_checkpoint(
                            save_path,
                            args.output_dir,
                            accelerator,
                            keep_full_state_checkpoints,
                            lambda path: save_train_mode_weights(
                                path,
                                args,
                                network,
                                weight_dtype,
                                accelerator,
                                transformer3d,
                                trainable_parameter_patterns,
                            ),
                            logger,
                        )

                # Validation (runs if validation_set is defined in config)
                # Early phase: every init_validation_steps until first checkpoint; then every validation_steps (CogVideoX style)
                should_run_validation = data_config.get("validation_set") is not None and max_validation_videos > 0 and (
                    (global_step % init_validation_steps == 0 and global_step < checkpointing_steps)
                    or global_step % validation_steps == 0
                )
                if should_run_validation:
                    if accelerator.is_main_process:
                        logger.info(f"🎬 Running validation at step {global_step}")
                    log_validation(
                        vae, text_encoder, tokenizer, clip_image_encoder,
                        transformer3d, network, config, args, accelerator,
                        weight_dtype, global_step,
                        load_tensors=load_tensors, vae_path=vae_path, vae_kwargs=vae_kwargs,
                        text_encoder_path=text_encoder_path, text_encoder_kwargs=text_encoder_kwargs,
                        tokenizer_path=tokenizer_path,
                    )

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # Save final model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED:
        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
        save_training_checkpoint(
            save_path,
            args.output_dir,
            accelerator,
            keep_full_state_checkpoints,
            lambda path: save_train_mode_weights(
                path,
                args,
                network,
                weight_dtype,
                accelerator,
                transformer3d,
                trainable_parameter_patterns,
            ),
            logger,
        )

    if accelerator.is_main_process:
        logger.info(f"Training completed. Final checkpoint saved to {save_path}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
