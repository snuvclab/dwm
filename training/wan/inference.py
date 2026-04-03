#!/usr/bin/env python3
"""
WAN-based inference script for WanFunInpaintHandConcatPipeline.
Supports both LoRA and SFT checkpoints.
"""

import argparse
import json
import logging
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.wan.diffusers_compat import disable_broken_torchao

disable_broken_torchao()

import imageio.v3 as iio
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
import diffusers
from diffusers import FlowMatchEulerDiscreteScheduler
from safetensors.torch import load_file
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel

from training.dataset_layout_paths import build_output_stem
from training.wan.config_loader import load_experiment_config
from training.wan.models import (
    WanI2VTransformer3DModelWithConcat,
    WanTransformer3DModel,
    WanTransformer3DModelWithConcat,
)
from training.wan.pipeline import (
    WanFunInpaintHandConcatPipeline,
    WanFunInpaintPipeline,
    WanI2VDiffusersHandConcatPipeline,
)
from training.wan.utils.utils import (
    filter_kwargs,
    get_image_to_video_latent,
    save_videos_grid,
)
from training.wan.utils.lightx2v_compat import (
    load_lightx2v_with_fallback,
)
from training.wan.utils.lora_utils import merge_lora
from training.wan.models import AutoencoderKLWan, CLIPModel, WanT5EncoderModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_distributed():
    """Setup distributed training if available."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        
        return rank, world_size, local_rank
    else:
        return 0, 1, 0


def load_experiment_config_wrapper(config_path: str) -> Dict[str, Any]:
    """Load experiment configuration from YAML file."""
    return load_experiment_config(config_path, overrides=[])


def find_experiment_config(checkpoint_path: str) -> Optional[str]:
    """Find experiment config YAML near a checkpoint directory or packaged release folder."""
    checkpoint_dir = Path(checkpoint_path)
    search_dir = checkpoint_dir.parent if checkpoint_dir.name.startswith("checkpoint-") else checkpoint_dir
    
    yaml_files = list(search_dir.glob("*.yaml")) + list(search_dir.glob("*.yml"))
    
    if yaml_files:
        return str(yaml_files[0])
    
    return None


def _normalize_bucket_values(raw_value: Any) -> List[int]:
    """Normalize config bucket values into a sorted positive-int list."""
    if raw_value is None:
        return []
    if isinstance(raw_value, (list, tuple, set)):
        values = raw_value
    else:
        values = [raw_value]

    normalized = []
    for value in values:
        try:
            parsed = int(value)
        except Exception:
            continue
        if parsed > 0:
            normalized.append(parsed)
    return sorted(set(normalized))


def _resize_and_pad_video(
    video: Union[np.ndarray, torch.Tensor],
    target_h: int,
    target_w: int,
    pad_value: float = 0.0,
    short_side_mode: bool = True,
    center_pad: bool = True,
    layout_spec: str = "FHWC",
):
    """Resize video preserving aspect ratio, then pad to exact target size."""
    if layout_spec == "FHWC":
        tensor = torch.from_numpy(video).permute(0, 3, 1, 2).contiguous()
    elif layout_spec == "BCFHW":
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

    b, c, f, _, _ = video.shape
    return padded.reshape(b, f, c, target_h, target_w).permute(0, 2, 1, 3, 4).contiguous().to(video.dtype)


def _clip_or_pad_video_fhwc(video_np: np.ndarray, target_num_frames: int) -> np.ndarray:
    """Deterministically trim or tail-pad an FHWC video to a fixed length."""
    if video_np.shape[0] <= 0:
        raise ValueError("Video has no frames")
    if target_num_frames <= 0:
        raise ValueError(f"Invalid target_num_frames={target_num_frames}")

    if video_np.shape[0] > target_num_frames:
        return video_np[:target_num_frames]
    if video_np.shape[0] < target_num_frames:
        pad = np.repeat(video_np[-1:], target_num_frames - video_np.shape[0], axis=0)
        return np.concatenate([video_np, pad], axis=0)
    return video_np


def _clip_or_pad_video_bcfhw(video: torch.Tensor, target_num_frames: int) -> torch.Tensor:
    """Deterministically trim or tail-pad a BCFHW video to a fixed length."""
    if video.shape[2] <= 0:
        raise ValueError("Video has no frames")
    if target_num_frames <= 0:
        raise ValueError(f"Invalid target_num_frames={target_num_frames}")

    if video.shape[2] > target_num_frames:
        return video[:, :, :target_num_frames].contiguous()
    if video.shape[2] < target_num_frames:
        pad = video[:, :, -1:].repeat(1, 1, target_num_frames - video.shape[2], 1, 1)
        return torch.cat([video, pad], dim=2).contiguous()
    return video


def _resolve_inference_num_frames(config: Optional[Dict[str, Any]]) -> Optional[int]:
    """Resolve target num_frames using validation/training-style config fallbacks."""
    if config is None or not isinstance(config, dict):
        return None

    data_config = config.get("data", {})
    custom_settings = config.get("training", {}).get("custom_settings", {})
    raw_target_num_frames = data_config.get("max_num_frames", custom_settings.get("max_num_frames"))
    if raw_target_num_frames is not None:
        try:
            target_num_frames = int(raw_target_num_frames)
            if target_num_frames > 0:
                return target_num_frames
        except Exception:
            pass

    frame_buckets = _normalize_bucket_values(data_config.get("frame_buckets"))
    if len(frame_buckets) == 1:
        return frame_buckets[0]
    return None


def _resolve_inference_spatial_size(
    source_h: int,
    source_w: int,
    config: Optional[Dict[str, Any]],
) -> Optional[tuple[int, int]]:
    """Resolve inference target size using validation-style config fallbacks."""
    if config is None:
        return None

    data_config = config.get("data", {}) if isinstance(config, dict) else {}
    pipeline_type = str(config.get("pipeline", {}).get("type", "")) if isinstance(config, dict) else ""
    min_divisor = 32 if "2.2" in pipeline_type else 8

    validation_runtime_adapt = bool(data_config.get("validation_runtime_adapt", False))
    validation_inference_size = data_config.get("validation_inference_size")
    if (
        validation_runtime_adapt
        and isinstance(validation_inference_size, (list, tuple))
        and len(validation_inference_size) == 2
    ):
        try:
            target_h = int(validation_inference_size[0])
            target_w = int(validation_inference_size[1])
            if target_h > 0 and target_w > 0:
                return target_h, target_w
        except Exception:
            pass

    height_buckets = _normalize_bucket_values(data_config.get("height_buckets"))
    width_buckets = _normalize_bucket_values(data_config.get("width_buckets"))
    if len(height_buckets) == 1 and len(width_buckets) == 1:
        return height_buckets[0], width_buckets[0]

    if height_buckets and width_buckets and (source_h % min_divisor != 0 or source_w % min_divisor != 0):
        return min(
            ((height, width) for height in height_buckets for width in width_buckets),
            key=lambda size: abs(size[0] - source_h) + abs(size[1] - source_w),
        )

    if source_h % min_divisor != 0 or source_w % min_divisor != 0:
        target_h = max(min_divisor, int(round(source_h / min_divisor)) * min_divisor)
        target_w = max(min_divisor, int(round(source_w / min_divisor)) * min_divisor)
        return target_h, target_w

    return None


def build_pipeline(args: argparse.Namespace, config: Dict[str, Any]) -> Union[
    WanFunInpaintPipeline, WanFunInpaintHandConcatPipeline, WanI2VDiffusersHandConcatPipeline
]:
    """Build WAN pipeline from config and checkpoint."""
    
    pipeline_config = config.get("pipeline", {})
    transformer_config = config.get("transformer", {})
    model_config = config.get("model", {})
    training_config = config.get("training", {})
    custom_settings = training_config.get("custom_settings", {})
    conditioning_mode = training_config.get("conditioning_mode", "dwm")
    use_diffusers_i2v_backend = bool(custom_settings.get("use_diffusers_i2v_backend", False))
    use_diffusers_i2v_backend = use_diffusers_i2v_backend and conditioning_mode == "i2v"
    
    # Get base model path
    base_model_path = pipeline_config.get("base_model_name_or_path", args.base_model_path)
    if not base_model_path:
        raise ValueError("base_model_name_or_path must be provided in config or via --base_model_path")
    base_model_path = os.path.expanduser(base_model_path)
    
    logger.info(f"🔧 Building WAN pipeline from base model: {base_model_path}")
    
    # Determine weight dtype
    weight_dtype = torch.bfloat16
    
    # Get training mode
    training_mode = training_config.get("mode", "sft")
    logger.info(f"🔧 Training mode: {training_mode}")
    
    # Load components
    logger.info("📦 Loading model components...")

    text_encoder_kwargs = config.get("text_encoder_kwargs", {})
    vae_kwargs = config.get("vae_kwargs", {})
    image_encoder_kwargs = config.get("image_encoder_kwargs", {})

    if use_diffusers_i2v_backend:
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, subfolder="tokenizer")
        text_encoder = UMT5EncoderModel.from_pretrained(
            base_model_path,
            subfolder="text_encoder",
            dtype=weight_dtype,
        ).eval()
        vae = diffusers.AutoencoderKLWan.from_pretrained(
            base_model_path,
            subfolder="vae",
            torch_dtype=weight_dtype,
        ).eval()
        image_encoder_path = os.path.join(base_model_path, image_encoder_kwargs.get("image_encoder_subpath", "image_encoder"))
        clip_image_encoder = CLIPVisionModel.from_pretrained(image_encoder_path, torch_dtype=weight_dtype).eval()
        image_processor_path = os.path.join(base_model_path, "image_processor")
        if os.path.isdir(image_processor_path):
            clip_image_processor = CLIPImageProcessor.from_pretrained(image_processor_path)
        else:
            logger.warning("image_processor path not found at %s. Falling back to CLIPImageProcessor().", image_processor_path)
            clip_image_processor = CLIPImageProcessor()
    else:
        text_encoder = WanT5EncoderModel.from_pretrained(
            os.path.join(base_model_path, text_encoder_kwargs.get("text_encoder_subpath", "text_encoder")),
            additional_kwargs=text_encoder_kwargs,
            torch_dtype=weight_dtype,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(base_model_path, text_encoder_kwargs.get("tokenizer_subpath", "tokenizer")),
        )
        vae = AutoencoderKLWan.from_pretrained(
            os.path.join(base_model_path, vae_kwargs.get("vae_subpath", "vae")),
            additional_kwargs=vae_kwargs,
        ).eval()
        clip_image_encoder = CLIPModel.from_pretrained(
            os.path.join(base_model_path, image_encoder_kwargs.get("image_encoder_subpath", "image_encoder")),
        ).eval()
        clip_image_processor = None
    
    # Transformer
    transformer_additional_kwargs = config.get("transformer_additional_kwargs", {})
    transformer_subpath = transformer_additional_kwargs.get("transformer_subpath", "transformer")

    transformer_class_spec = transformer_config.get("class", WanTransformer3DModelWithConcat)
    if isinstance(transformer_class_spec, str):
        transformer_class_map = {
            "WanTransformer3DModel": WanTransformer3DModel,
            "WanTransformer3DModelWithConcat": WanTransformer3DModelWithConcat,
        }
        TransformerCls = transformer_class_map.get(transformer_class_spec)
        if TransformerCls is None:
            raise ValueError(
                f"Unsupported transformer_config.class={transformer_class_spec!r}. "
                f"Supported: {sorted(transformer_class_map.keys())}"
            )
    elif isinstance(transformer_class_spec, type):
        TransformerCls = transformer_class_spec
    else:
        raise TypeError(
            f"Unsupported transformer_config.class type: {type(transformer_class_spec)}. "
            "Expected str or class type."
        )

    # Model FPS (RoPE scaling). This is distinct from the FPS used when saving mp4 files.
    model_fps = int(getattr(args, "model_fps", 16))
    logger.info(f"🎞️ model_fps (transformer/RoPE): {model_fps}")

    # Extra kwargs for transformer init
    transformer_init_kwargs = {
        **transformer_additional_kwargs,
        "fps": model_fps,
    }
    if use_diffusers_i2v_backend and TransformerCls is WanTransformer3DModelWithConcat:
        TransformerCls = WanI2VTransformer3DModelWithConcat

    if TransformerCls is WanTransformer3DModelWithConcat:
        condition_channels = transformer_config.get("condition_channels", 16)
        transformer_init_kwargs["condition_channels"] = condition_channels

    if training_mode == "lora":
        # For LoRA, load base transformer
        if use_diffusers_i2v_backend:
            transformer3d = TransformerCls.from_pretrained(
                os.path.join(base_model_path, transformer_subpath),
                condition_channels=transformer_config.get("condition_channels", 16),
                torch_dtype=weight_dtype,
            )
        else:
            transformer3d = TransformerCls.from_pretrained(
                os.path.join(base_model_path, transformer_subpath),
                transformer_additional_kwargs=transformer_init_kwargs,
                torch_dtype=weight_dtype,
            )
    else:
        # For SFT, load from checkpoint if provided
        if args.checkpoint_path:
            if use_diffusers_i2v_backend:
                transformer3d = TransformerCls.from_pretrained(
                    os.path.join(args.checkpoint_path, transformer_subpath),
                    condition_channels=transformer_config.get("condition_channels", 16),
                    torch_dtype=weight_dtype,
                )
            else:
                transformer3d = TransformerCls.from_pretrained(
                    os.path.join(args.checkpoint_path, transformer_subpath),
                    transformer_additional_kwargs=transformer_init_kwargs,
                    torch_dtype=weight_dtype,
                )
        else:
            if use_diffusers_i2v_backend:
                transformer3d = TransformerCls.from_pretrained(
                    os.path.join(base_model_path, transformer_subpath),
                    condition_channels=transformer_config.get("condition_channels", 16),
                    torch_dtype=weight_dtype,
                )
            else:
                transformer3d = TransformerCls.from_pretrained(
                    os.path.join(base_model_path, transformer_subpath),
                    transformer_additional_kwargs=transformer_init_kwargs,
                    torch_dtype=weight_dtype,
                )
    
    # Scheduler
    scheduler_kwargs = filter_kwargs(FlowMatchEulerDiscreteScheduler, 
                                    config.get("scheduler_kwargs", {}))   
    scheduler = FlowMatchEulerDiscreteScheduler(**scheduler_kwargs)
    
    # Create pipeline (match training script behavior)
    pipeline_type = pipeline_config.get("type", "wan2.1_fun_inp_hand_concat")
    if pipeline_type == "wan2.1_fun_inp":
        logger.info("🔧 Using WAN 2.1 base pipeline for inference")
        pipeline = WanFunInpaintPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer3d,
            scheduler=scheduler,
            clip_image_encoder=clip_image_encoder,
        )
    elif pipeline_type in ("wan2.1_fun_inp_hand_concat", "wan2.1_fun_inp_hand_concat_i2v_diffusers"):
        if use_diffusers_i2v_backend:
            logger.info("🔧 Using WAN 2.1 diffusers hand concat pipeline for inference (i2v backend)")
            pipeline = WanI2VDiffusersHandConcatPipeline(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                image_encoder=clip_image_encoder,
                image_processor=clip_image_processor,
                transformer=transformer3d,
                scheduler=scheduler,
            )
        else:
            logger.info("🔧 Using WAN 2.1 hand concat pipeline for inference")
            pipeline = WanFunInpaintHandConcatPipeline(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                transformer=transformer3d,
                scheduler=scheduler,
                clip_image_encoder=clip_image_encoder,
            )
    else:
        raise ValueError(f"Unsupported WAN pipeline type for inference: {pipeline_type}")
    pipeline = pipeline.to(device, dtype=weight_dtype)
    
    # Load LoRA weights if in LoRA mode
    if training_mode == "lora" and args.checkpoint_path:
        logger.info("🔧 Loading LoRA weights via merge_lora...")
        lora_path = os.path.join(args.checkpoint_path, "lora_diffusion_pytorch_model.safetensors")
        if os.path.exists(lora_path):
            merge_lora(
                pipeline,
                lora_path,
                multiplier=1.0,
                device=device,
                dtype=weight_dtype,
            )
            logger.info("✅ LoRA weights loaded via merge_lora")
        else:
            logger.warning(f"⚠️  LoRA weights not found: {lora_path}")
        
        # Load non-LoRA trainable weights (e.g., patch_embedding)
        non_lora_path = os.path.join(args.checkpoint_path, "non_lora_weights.safetensors")
        if os.path.exists(non_lora_path):
            non_lora_state_dict = load_file(non_lora_path)
            transformer_state_dict = transformer3d.state_dict()
            loaded_keys = []
            for name, param_data in non_lora_state_dict.items():
                if name in transformer_state_dict:
                    transformer_state_dict[name].copy_(param_data.to(transformer_state_dict[name].dtype))
                    loaded_keys.append(name)
            transformer3d.load_state_dict(transformer_state_dict)
            logger.info(f"✅ Loaded non-LoRA weights: {len(loaded_keys)} parameters")

    lightx2v_lora_path = args.lightx2v_lora_path or custom_settings.get("lightx2v_lora_path")
    if lightx2v_lora_path:
        if args.lightx2v_lora_scale is not None:
            lightx2v_lora_scale = float(args.lightx2v_lora_scale)
        else:
            lightx2v_lora_scale = float(custom_settings.get("lightx2v_lora_scale", 1.0))

        if args.lightx2v_compat_mode is not None:
            lightx2v_compat_mode = str(args.lightx2v_compat_mode).lower()
        else:
            lightx2v_compat_mode = str(custom_settings.get("lightx2v_compat_mode", "auto")).lower()

        if args.lightx2v_compat_strict is not None:
            lightx2v_compat_strict = bool(args.lightx2v_compat_strict)
        else:
            lightx2v_compat_strict = bool(custom_settings.get("lightx2v_compat_strict", True))
        lightx2v_apply_non_block_diff = bool(custom_settings.get("lightx2v_apply_non_block_diff", False))

        logger.info(
            "🔧 Loading LightX2V LoRA for inference: path=%s scale=%.4f mode=%s strict=%s",
            lightx2v_lora_path,
            lightx2v_lora_scale,
            lightx2v_compat_mode,
            lightx2v_compat_strict,
        )
        load_method, load_stats = load_lightx2v_with_fallback(
            pipeline=pipeline,
            lora_path=lightx2v_lora_path,
            lora_scale=lightx2v_lora_scale,
            compat_mode=lightx2v_compat_mode,
            compat_strict=lightx2v_compat_strict,
            expected_blocks=None,
            apply_non_block_diff=lightx2v_apply_non_block_diff,
            logger=logger,
        )
        logger.info("✅ LightX2V loaded via %s path", load_method)
        args.lightx2v_attention_kwargs = {"scale": lightx2v_lora_scale}
        if load_method == "compat":
            logger.info(
                "LightX2V compat stats: lora_pairs=%s diff=%s diff_b=%s",
                load_stats.get("applied_lora_pairs"),
                load_stats.get("applied_diff"),
                load_stats.get("applied_diff_b"),
            )
    else:
        args.lightx2v_attention_kwargs = None
    
    logger.info("✅ Pipeline built successfully")
    return pipeline


def load_video_data(
    video_path: Optional[str] = None,
    static_video_path: Optional[str] = None,
    hand_video_path: Optional[str] = None,
    static_disparity_video_path: Optional[str] = None,
    hand_disparity_video_path: Optional[str] = None,
    prompt: Optional[str] = None,
    prompt_file: Optional[str] = None,
    data_root: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    prompt_subdir: Optional[str] = None,
) -> Dict[str, Any]:
    """Load video data from custom paths or auto-derive from video_path."""
    
    data = {}
    data_config = config.get("data", {}) if config else {}
    video_path_obj = Path(video_path) if video_path else None
    full_video_path = Path(data_root) / video_path if video_path and data_root else None
    static_video_source_mode = str(data_config.get("static_video_source_mode", "directory")).strip().lower()
    if static_video_source_mode not in {"directory", "copy_first_frame"}:
        raise ValueError(
            f"Unsupported static_video_source_mode: {static_video_source_mode}. "
            "Use one of: directory, copy_first_frame."
        )

    gt_video = None
    if full_video_path is not None and full_video_path.exists():
        gt_video = iio.imread(str(full_video_path)).astype(np.float32) / 255.0
    
    # Load prompt
    if prompt:
        data["prompt"] = prompt.strip()
    elif prompt_file and os.path.exists(prompt_file):
        with open(prompt_file, "r") as f:
            data["prompt"] = f.read().strip()
    elif full_video_path is not None:
        # Auto-derive prompt path
        video_name = video_path_obj.stem
        # Use provided prompt_subdir if available, otherwise use config, otherwise default
        if prompt_subdir is None:
            prompt_subdir = data_config.get("prompt_subdir", "prompts")
        prompt_path = full_video_path.parent.parent / prompt_subdir / f"{video_name}.txt"
        if prompt_path.exists():
            with open(prompt_path, "r") as f:
                data["prompt"] = f.read().strip()
        else:
            logger.warning(f"⚠️  Prompt not found: {prompt_path}")
            data["prompt"] = ""
    else:
        data["prompt"] = ""
    
    # Load static video
    if static_video_path:
        static_video = iio.imread(static_video_path).astype(np.float32) / 255.0
        data["static_video"] = torch.from_numpy(static_video).permute(3, 0, 1, 2).unsqueeze(0)  # [1, c, f, h, w]
    elif full_video_path is not None:
        # Auto-derive static video path
        if static_video_source_mode == "copy_first_frame":
            if gt_video is None:
                raise ValueError(
                    f"Ground-truth video not found for copy_first_frame mode: {full_video_path}"
                )
            if gt_video.shape[0] <= 0:
                raise ValueError(f"Ground-truth video has no frames: {full_video_path}")
            static_video = np.repeat(gt_video[:1], gt_video.shape[0], axis=0).copy()
            data["static_video"] = torch.from_numpy(static_video).permute(3, 0, 1, 2).unsqueeze(0)
        else:
            static_video_subdir = data_config.get("static_video_subdir", "videos_static")
            static_video_path = full_video_path.parent.parent / static_video_subdir / video_path_obj.name
            if static_video_path.exists():
                static_video = iio.imread(str(static_video_path)).astype(np.float32) / 255.0
                data["static_video"] = torch.from_numpy(static_video).permute(3, 0, 1, 2).unsqueeze(0)
            else:
                raise ValueError(f"Static video not found: {static_video_path}")
    else:
        raise ValueError("static_video_path or (video_path + data_root) must be provided")
    
    # Load hand video (optional)
    if hand_video_path:
        hand_video = iio.imread(hand_video_path).astype(np.float32) / 255.0
        data["hand_video"] = torch.from_numpy(hand_video).permute(3, 0, 1, 2).unsqueeze(0)  # [1, c, f, h, w]
    elif full_video_path is not None:
        # Auto-derive hand video path
        hand_video_subdir = data_config.get("hand_video_subdir", "videos_hands")
        if hand_video_subdir:
            hand_video_path = full_video_path.parent.parent / hand_video_subdir / video_path_obj.name
            if hand_video_path.exists():
                hand_video = iio.imread(str(hand_video_path)).astype(np.float32) / 255.0
                data["hand_video"] = torch.from_numpy(hand_video).permute(3, 0, 1, 2).unsqueeze(0)
            else:
                logger.warning(f"⚠️  Hand video not found: {hand_video_path}. Using None.")
                data["hand_video"] = None
        else:
            data["hand_video"] = None
    else:
        data["hand_video"] = None

    if static_disparity_video_path:
        static_disparity_video = iio.imread(static_disparity_video_path).astype(np.float32) / 255.0
        data["static_disparity_video"] = torch.from_numpy(static_disparity_video).permute(3, 0, 1, 2).unsqueeze(0)
    elif full_video_path is not None:
        static_disparity_subdir = data_config.get("static_disparity_subdir")
        if static_disparity_subdir:
            static_disparity_video_path = (
                full_video_path.parent.parent / static_disparity_subdir / video_path_obj.name
            )
            if static_disparity_video_path.exists():
                static_disparity_video = iio.imread(str(static_disparity_video_path)).astype(np.float32) / 255.0
                data["static_disparity_video"] = torch.from_numpy(static_disparity_video).permute(3, 0, 1, 2).unsqueeze(0)
            else:
                raise ValueError(f"Static disparity video not found: {static_disparity_video_path}")
        else:
            data["static_disparity_video"] = None
    else:
        data["static_disparity_video"] = None

    if hand_disparity_video_path:
        hand_disparity_video = iio.imread(hand_disparity_video_path).astype(np.float32) / 255.0
        data["hand_disparity_video"] = torch.from_numpy(hand_disparity_video).permute(3, 0, 1, 2).unsqueeze(0)
    elif full_video_path is not None:
        hand_disparity_subdir = data_config.get("hand_disparity_subdir")
        if hand_disparity_subdir:
            hand_disparity_video_path = (
                full_video_path.parent.parent / hand_disparity_subdir / video_path_obj.name
            )
            if hand_disparity_video_path.exists():
                hand_disparity_video = iio.imread(str(hand_disparity_video_path)).astype(np.float32) / 255.0
                data["hand_disparity_video"] = torch.from_numpy(hand_disparity_video).permute(3, 0, 1, 2).unsqueeze(0)
            else:
                raise ValueError(f"Hand disparity video not found: {hand_disparity_video_path}")
        else:
            data["hand_disparity_video"] = None
    else:
        data["hand_disparity_video"] = None
    
    # Load GT video if available (for comparison)
    if gt_video is not None:
        data["gt_video"] = gt_video
    else:
        data["gt_video"] = None

    temporal_lengths = []
    if data["gt_video"] is not None:
        temporal_lengths.append(int(data["gt_video"].shape[0]))
    for key in ("static_video", "hand_video", "static_disparity_video", "hand_disparity_video"):
        if data.get(key) is not None:
            temporal_lengths.append(int(data[key].shape[2]))

    if temporal_lengths:
        common_len = min(temporal_lengths)
        if common_len <= 0:
            raise ValueError("Inference sample has non-positive temporal length")
        if data["gt_video"] is not None and data["gt_video"].shape[0] != common_len:
            data["gt_video"] = data["gt_video"][:common_len]
        for key in ("static_video", "hand_video", "static_disparity_video", "hand_disparity_video"):
            if data.get(key) is not None and data[key].shape[2] != common_len:
                data[key] = data[key][:, :, :common_len].contiguous()

    target_num_frames = _resolve_inference_num_frames(config)
    if target_num_frames is not None:
        if data["gt_video"] is not None:
            data["gt_video"] = _clip_or_pad_video_fhwc(data["gt_video"], target_num_frames)
        for key in ("static_video", "hand_video", "static_disparity_video", "hand_disparity_video"):
            if data.get(key) is not None:
                data[key] = _clip_or_pad_video_bcfhw(data[key], target_num_frames)

    reference_h = None
    reference_w = None
    if data["gt_video"] is not None:
        reference_h = int(data["gt_video"].shape[1])
        reference_w = int(data["gt_video"].shape[2])
    elif data["static_video"] is not None:
        reference_h = int(data["static_video"].shape[-2])
        reference_w = int(data["static_video"].shape[-1])

    target_size = None
    if reference_h is not None and reference_w is not None:
        target_size = _resolve_inference_spatial_size(reference_h, reference_w, config)

    if target_size is not None and (reference_h, reference_w) != target_size:
        target_h, target_w = target_size
        validation_resize_mode = data_config.get("validation_resize_mode", "short_side_pad")
        validation_pad_value = float(data_config.get("validation_pad_value", 0.0))
        validation_center_pad = bool(data_config.get("validation_center_pad", True))
        short_side_mode = validation_resize_mode == "short_side_pad"
        logger.info(
            "📐 Resizing inference sample from %dx%d to %dx%d",
            reference_h,
            reference_w,
            target_h,
            target_w,
        )

        if data["gt_video"] is not None:
            data["gt_video"] = _resize_and_pad_video(
                data["gt_video"],
                target_h=target_h,
                target_w=target_w,
                pad_value=validation_pad_value,
                short_side_mode=short_side_mode,
                center_pad=validation_center_pad,
                layout_spec="FHWC",
            )
        for key in ("static_video", "hand_video", "static_disparity_video", "hand_disparity_video"):
            if data.get(key) is not None:
                data[key] = _resize_and_pad_video(
                    data[key],
                    target_h=target_h,
                    target_w=target_w,
                    pad_value=validation_pad_value,
                    short_side_mode=short_side_mode,
                    center_pad=validation_center_pad,
                    layout_spec="BCFHW",
                )
    
    return data


def load_variant_manifest(path: str) -> Dict[str, Any]:
    """Load and validate a single-sample multi-variant manifest."""
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if not isinstance(manifest, dict):
        raise ValueError(f"Variant manifest must be a JSON object: {path}")

    mode = str(manifest.get("mode", "")).strip().lower()
    if mode != "hand_video":
        raise ValueError(
            f"Unsupported variant manifest mode '{mode}'. Expected 'hand_video'."
        )

    shared = manifest.get("shared")
    if not isinstance(shared, dict):
        raise ValueError("Variant manifest must contain a 'shared' object.")

    variants = manifest.get("variants")
    if not isinstance(variants, list) or not variants:
        raise ValueError("Variant manifest must contain a non-empty 'variants' list.")

    normalized_shared: Dict[str, Any] = {}
    for key in (
        "video_path",
        "static_video_path",
        "static_disparity_video_path",
        "hand_disparity_video_path",
        "prompt",
        "prompt_file",
    ):
        value = shared.get(key)
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Variant manifest shared.{key} must be a non-empty string when provided.")
        normalized_shared[key] = value.strip()

    seen_labels = set()
    normalized_variants: List[Dict[str, str]] = []
    for index, variant in enumerate(variants):
        if not isinstance(variant, dict):
            raise ValueError(f"Variant entry at index {index} must be a JSON object.")

        label = variant.get("label")
        hand_video_path = variant.get("hand_video_path")
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"Variant entry at index {index} is missing a valid 'label'.")
        if not isinstance(hand_video_path, str) or not hand_video_path.strip():
            raise ValueError(
                f"Variant entry '{label}' is missing a valid 'hand_video_path'."
            )

        safe_label = re.sub(r"[^0-9A-Za-z._-]+", "_", label.strip()).strip("._-")
        if not safe_label:
            raise ValueError(f"Variant label '{label}' becomes empty after sanitization.")
        if safe_label in seen_labels:
            raise ValueError(f"Duplicate sanitized variant label '{safe_label}' in manifest.")
        seen_labels.add(safe_label)
        normalized_variants.append(
            {
                "label": label.strip(),
                "safe_label": safe_label,
                "hand_video_path": hand_video_path.strip(),
            }
        )

    return {
        "mode": mode,
        "shared": normalized_shared,
        "variants": normalized_variants,
    }


def has_explicit_custom_inputs(args: argparse.Namespace) -> bool:
    return any(
        getattr(args, name) is not None
        for name in (
            "static_video_path",
            "hand_video_path",
            "static_disparity_video_path",
            "hand_disparity_video_path",
            "prompt",
            "prompt_file",
            "variant_manifest_json",
        )
    )


def resolve_custom_input_base_name(
    video_path: Optional[str],
    static_video_path: Optional[str],
    hand_video_path: Optional[str],
) -> str:
    for candidate in (video_path, static_video_path, hand_video_path):
        if candidate:
            return Path(candidate).stem
    return "output"


def run_inference(
    pipeline: Union[WanFunInpaintPipeline, WanFunInpaintHandConcatPipeline, WanI2VDiffusersHandConcatPipeline],
    prompt: str,
    static_video: torch.Tensor,
    hand_video: Optional[torch.Tensor],
    static_disparity_video: Optional[torch.Tensor],
    hand_disparity_video: Optional[torch.Tensor],
    num_frames: int,
    height: int,
    width: int,
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    seed: Optional[int] = None,
    negative_prompt: str = "bad detailed",
    conditioning_mode: str = "dwm",
    device: Optional[torch.device] = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Run inference with WAN pipeline."""
    
    # Use provided device or infer from static_video
    if device is None:
        device = static_video.device if isinstance(static_video, torch.Tensor) else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    generator = torch.Generator(device=device).manual_seed(seed) if seed else None
    
    # Match validation behavior:
    # - dwm: provide the full condition video with an all-zero mask
    # - i2v: keep only frame 0 unmasked
    mask_video = torch.zeros(1, 1, num_frames, height, width, device=device, dtype=static_video.dtype)
    if conditioning_mode == "i2v":
        mask_video[:, :, 1:, :, :] = 255
    elif conditioning_mode != "dwm":
        raise ValueError(f"Unsupported conditioning_mode for inference: {conditioning_mode}")

    # Move videos to device
    static_video = static_video.to(device, dtype=static_video.dtype)
    if hand_video is not None:
        hand_video = hand_video.to(device, dtype=hand_video.dtype)
    if static_disparity_video is not None:
        static_disparity_video = static_disparity_video.to(device, dtype=static_disparity_video.dtype)
    if hand_disparity_video is not None:
        hand_disparity_video = hand_disparity_video.to(device, dtype=hand_disparity_video.dtype)
    
    start_time = time.time()
    with torch.no_grad():
        with torch.autocast("cuda", dtype=static_video.dtype):
            if isinstance(pipeline, (WanFunInpaintHandConcatPipeline, WanI2VDiffusersHandConcatPipeline)):
                call_kwargs = {}
                if attention_kwargs is not None:
                    call_kwargs["attention_kwargs"] = attention_kwargs
                if isinstance(pipeline, WanFunInpaintHandConcatPipeline):
                    call_kwargs["static_disparity_video"] = static_disparity_video
                    call_kwargs["hand_disparity_video"] = hand_disparity_video
                output = pipeline(
                    prompt=prompt,
                    num_frames=num_frames,
                    negative_prompt=negative_prompt,
                    height=height,
                    width=width,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                    static_video=static_video,
                    mask_video=mask_video,
                    hand_video=hand_video,
                    **call_kwargs,
                )
            else:
                # Base pipeline uses `video` instead of `static_video`, and no hand condition
                call_kwargs = {}
                if attention_kwargs is not None:
                    call_kwargs["attention_kwargs"] = attention_kwargs
                output = pipeline(
                    prompt=prompt,
                    num_frames=num_frames,
                    negative_prompt=negative_prompt,
                    height=height,
                    width=width,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                    video=static_video,
                    mask_video=mask_video,
                    **call_kwargs,
                )
    
    generation_time = time.time() - start_time
    logger.info(f"✅ Generated video in {generation_time:.2f}s")
    
    # Convert output.videos to [1, c, f, h, w] format for save_videos_grid
    video = output.videos
    if isinstance(video, np.ndarray):
        video = torch.from_numpy(video)
    
    # Handle different input shapes
    # Based on error: [3, 49, 480, 720] = [c, f, h, w]
    if video.ndim == 4:
        # Check if [c, f, h, w] (channel first) or [f, h, w, c] (channel last)
        if video.shape[0] in [3, 4]:  # [c, f, h, w] - channel first
            video = video.unsqueeze(0)  # [1, c, f, h, w]
        elif video.shape[-1] in [3, 4]:  # [f, h, w, c] - channel last
            video = video.permute(3, 0, 1, 2).unsqueeze(0)  # [1, c, f, h, w]
        else:
            # Try to infer: if first dim is small (3-4), assume [c, f, h, w]
            if video.shape[0] < 10:
                video = video.unsqueeze(0)  # [1, c, f, h, w]
            else:
                # Assume [f, h, w, c]
                video = video.permute(3, 0, 1, 2).unsqueeze(0)  # [1, c, f, h, w]
    elif video.ndim == 5:
        # Already [b, c, f, h, w] or [b, f, h, w, c]
        if video.shape[-1] in [3, 4]:  # [b, f, h, w, c]
            video = video.permute(0, 4, 1, 2, 3)  # [b, c, f, h, w]
        # Otherwise assume already [b, c, f, h, w]
    else:
        raise ValueError(f"Unexpected video shape: {video.shape}, expected 4D or 5D tensor")
    
    return video.cpu(), generation_time


def get_output_filename(video_path, suffix=""):
    """Generate output filename from video path."""
    video_path_obj = Path(video_path)
    try:
        base_name = build_output_stem(video_path_obj)
    except Exception:
        base_name = video_path_obj.stem
    
    # Add suffix to base_name if provided
    if suffix:
        base_name = f"{base_name}{suffix}"
    
    return f"{base_name}_generated.mp4"


def build_generated_output_path(output_dir: Union[str, Path], base_name: str, suffix: str = "") -> Path:
    stem = f"{base_name}{suffix}" if suffix else base_name
    return Path(output_dir) / f"{stem}_generated.mp4"


def check_output_exists(video_path, output_dir, args, suffix=""):
    """Check if output files already exist in subfolder or unified output_dir.
    
    For distributed modes, checks both:
    1. Subfolder (rank_{rank}) - current output location
    2. Unified output_dir (args.output_dir) - already moved location
    """
    if not args.skip_existing:
        return False
    filename = get_output_filename(video_path, suffix)
    output_dir = Path(output_dir)
    unified_output_dir = Path(args.output_dir)
    
    # Check in subfolder (current output location)
    subfolder_file = output_dir / filename
    if subfolder_file.exists():
        logger.info(f"⏭️  Skipping {video_path}: output exists in {subfolder_file}")
        return True
    
    # Check in unified output_dir (if different from subfolder)
    # This handles the case where files have already been moved
    if unified_output_dir != output_dir:
        unified_file = unified_output_dir / filename
        if unified_file.exists():
            logger.info(f"⏭️  Skipping {video_path}: output exists in unified dir {unified_file}")
            return True
    
    return False


def get_partition_bounds(total_items: int, partition_idx: int, num_partitions: int) -> tuple[int, int]:
    """Return the inclusive-exclusive slice for a partition."""
    if num_partitions <= 0:
        raise ValueError(f"num_partitions must be positive, got {num_partitions}")
    if partition_idx < 0 or partition_idx >= num_partitions:
        raise ValueError(
            f"partition_idx must satisfy 0 <= partition_idx < num_partitions, "
            f"got partition_idx={partition_idx}, num_partitions={num_partitions}"
        )

    items_per_partition = total_items // num_partitions
    remainder = total_items % num_partitions
    start_idx = partition_idx * items_per_partition + min(partition_idx, remainder)
    end_idx = start_idx + items_per_partition + (1 if partition_idx < remainder else 0)
    return start_idx, end_idx


def prepare_video_paths(video_paths: List[str], args: argparse.Namespace) -> List[str]:
    """Apply optional dataset shuffle and truncation before worker splitting."""
    prepared_paths = list(video_paths)

    if args.shuffle_dataset and len(prepared_paths) > 1:
        shuffle_seed = args.shuffle_seed if args.shuffle_seed is not None else args.seed
        order = np.random.default_rng(shuffle_seed).permutation(len(prepared_paths))
        prepared_paths = [prepared_paths[idx] for idx in order]
        logger.info(
            "🔀 Shuffled dataset entries with seed=%s (%d videos)",
            shuffle_seed,
            len(prepared_paths),
        )

    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError(f"--max_samples must be positive, got {args.max_samples}")
        original_count = len(prepared_paths)
        prepared_paths = prepared_paths[: args.max_samples]
        logger.info(
            "✂️  Applied max_samples=%d: %d -> %d videos",
            args.max_samples,
            original_count,
            len(prepared_paths),
        )

    return prepared_paths


def get_batch_summary_path(
    output_dir: Path,
    args: argparse.Namespace,
    rank: int,
    world_size: int,
) -> Path:
    """Choose a summary filename that avoids collisions across split modes."""
    if args.split_id is not None and world_size == 1:
        return output_dir / f"batch_summary_split_{args.split_id}.json"
    if world_size > 1:
        return output_dir / f"batch_summary_rank_{rank}.json"
    return output_dir / "batch_summary.json"


def _ensure_video_bcfhw(video: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    """Convert FHWC numpy or BCFHW/CFHW/FHWC tensor video to BCFHW torch."""
    if isinstance(video, np.ndarray):
        if video.ndim != 4:
            raise ValueError(f"Expected FHWC numpy video, got shape {video.shape}")
        return torch.from_numpy(video).permute(3, 0, 1, 2).unsqueeze(0)

    if not isinstance(video, torch.Tensor):
        raise TypeError(f"Unsupported video type: {type(video)}")

    if video.ndim == 5:
        return video
    if video.ndim != 4:
        raise ValueError(f"Expected 4D/5D tensor video, got shape {tuple(video.shape)}")

    if video.shape[0] in (3, 4):
        return video.unsqueeze(0)
    if video.shape[-1] in (3, 4):
        return video.permute(3, 0, 1, 2).unsqueeze(0)

    raise ValueError(f"Cannot infer tensor video layout from shape {tuple(video.shape)}")


def _prepare_comparison_parts(
    parts: List[torch.Tensor],
    target_h: int,
    target_w: int,
) -> List[torch.Tensor]:
    """Resize/pad comparison parts to a shared spatial size and trim to min frames."""
    normalized_parts = []
    for part in parts:
        if part.shape[3] != target_h or part.shape[4] != target_w:
            part = _resize_and_pad_video(
                part,
                target_h=target_h,
                target_w=target_w,
                layout_spec="BCFHW",
            )
        normalized_parts.append(part)

    min_frames = min(part.shape[2] for part in normalized_parts)
    return [part[:, :, :min_frames, :, :] for part in normalized_parts]


def save_outputs(
    generated_video: torch.Tensor,
    output_dir: Path,
    base_name: str,
    gt_video: Optional[np.ndarray] = None,
    static_video: Optional[torch.Tensor] = None,
    hand_video: Optional[torch.Tensor] = None,
    static_disparity_video: Optional[torch.Tensor] = None,
    hand_disparity_video: Optional[torch.Tensor] = None,
    generation_time: float = 0.0,
    args: Optional[argparse.Namespace] = None,
    suffix: str = "",
):
    """Save inference outputs using save_videos_grid for consistency."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Add suffix to base_name
    if suffix:
        base_name = f"{base_name}{suffix}"
    
    # Save generated video
    # generated_video is [1, c, f, h, w] format
    generated_path = output_dir / f"{base_name}_generated.mp4"
    save_fps = getattr(args, "save_fps", None)
    if save_fps is None:
        # Backward compatibility: fall back to legacy args.fps if present
        save_fps = getattr(args, "fps", 8) if args else 8
    save_videos_grid(generated_video, str(generated_path), fps=int(save_fps))
    logger.info(f"✅ Saved generated video: {generated_path}")
    
    # Save comparison video if GT available
    if gt_video is not None:
        # Convert GT video to torch tensor [1, c, f, h, w]
        gt_tensor = torch.from_numpy(gt_video).permute(3, 0, 1, 2).unsqueeze(0)  # [1, c, f, h, w]
        
        # Ensure same number of frames
        min_frames = min(generated_video.shape[2], gt_tensor.shape[2])
        generated_video_trimmed = generated_video[:, :, :min_frames, :, :]
        gt_tensor_trimmed = gt_tensor[:, :, :min_frames, :, :]
        
        # Concatenate along width: generated | gt
        comparison_video = torch.cat([generated_video_trimmed, gt_tensor_trimmed], dim=4)  # [1, c, f, h, 2*w]
        comparison_path = output_dir / f"{base_name}_comparison.mp4"
        save_videos_grid(comparison_video, str(comparison_path), fps=int(save_fps))
        logger.info(f"✅ Saved comparison video: {comparison_path}")

    conditioning_parts = []
    conditioning_labels = []
    comparison_dtype = generated_video.dtype
    for label, conditioning_video in (
        ("static", static_video),
        ("hand", hand_video),
        ("static_disp", static_disparity_video),
        ("hand_disp", hand_disparity_video),
    ):
        if conditioning_video is not None:
            conditioning_parts.append(_ensure_video_bcfhw(conditioning_video).cpu().to(dtype=comparison_dtype))
            conditioning_labels.append(label)

    if conditioning_parts:
        full_parts = conditioning_parts + [generated_video.cpu().to(dtype=comparison_dtype)]
        full_labels = conditioning_labels + ["generated"]
        if gt_video is not None:
            full_parts.append(_ensure_video_bcfhw(gt_video).cpu().to(dtype=comparison_dtype))
            full_labels.append("gt")

        full_parts = _prepare_comparison_parts(
            full_parts,
            target_h=int(generated_video.shape[3]),
            target_w=int(generated_video.shape[4]),
        )
        full_comparison_video = torch.cat(full_parts, dim=4)
        full_comparison_path = output_dir / f"{base_name}_full_comparison.mp4"
        save_videos_grid(full_comparison_video, str(full_comparison_path), fps=int(save_fps))
        logger.info(
            "✅ Saved full comparison video (%s): %s",
            " | ".join(full_labels),
            full_comparison_path,
        )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="WAN Inference Script")
    
    # Model configuration
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to checkpoint directory (required for LoRA, optional for SFT)",
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="",
        help="Path to base model (will be read from config if not provided)",
    )
    parser.add_argument(
        "--experiment_config",
        type=str,
        default=None,
        help="Path to experiment config YAML (auto-detected from checkpoint if not provided)",
    )
    
    # Input data
    parser.add_argument(
        "--dataset_file",
        type=str,
        default=None,
        help="Path to dataset file containing video paths (one per line)",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data",
        help="Root directory for dataset (required if using dataset_file)",
    )
    
    # Custom paths (alternative to dataset_file)
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="Custom video path (relative to data_root, for auto-deriving other paths)",
    )
    parser.add_argument(
        "--variant_manifest_json",
        type=str,
        default=None,
        help="JSON manifest for single-sample multi-variant inference. Currently supports mode='hand_video'.",
    )
    parser.add_argument(
        "--static_video_path",
        type=str,
        default=None,
        help="Custom static video path (absolute or relative)",
    )
    parser.add_argument(
        "--hand_video_path",
        type=str,
        default=None,
        help="Custom hand video path (optional)",
    )
    parser.add_argument(
        "--static_disparity_video_path",
        type=str,
        default=None,
        help="Custom static disparity video path (optional)",
    )
    parser.add_argument(
        "--hand_disparity_video_path",
        type=str,
        default=None,
        help="Custom hand disparity video path (optional)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom text prompt (string)",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default=None,
        help="Custom prompt file path",
    )
    parser.add_argument(
        "--prompt_subdir",
        type=str,
        default=None,
        help="Subdirectory name for prompts (overrides config value, default: 'prompts')",
    )
    parser.add_argument(
        "--prompt_prefix",
        type=str,
        default=None,
        help="Prefix to prepend to all prompts (e.g., 'first-person view, egocentric perspective,'). Default: None",
    )
    parser.add_argument(
        "--shuffle_dataset",
        dest="shuffle_dataset",
        action="store_true",
        help="Shuffle dataset entries before applying max_samples and worker splitting.",
    )
    parser.add_argument(
        "--no_shuffle_dataset",
        dest="shuffle_dataset",
        action="store_false",
        help="Disable dataset shuffling.",
    )
    parser.add_argument(
        "--shuffle_seed",
        type=int,
        default=None,
        help="Random seed used only for dataset shuffling (defaults to --seed).",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional cap on the number of dataset entries processed after optional shuffle.",
    )
    parser.set_defaults(shuffle_dataset=False)
    
    # Inference parameters
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of inference steps",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=6.0,
        help="Guidance scale",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Output height",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=720,
        help="Output width",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=49,
        help="Number of frames",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=8,
        help="(Deprecated; use --save_fps) FPS for output video",
    )
    parser.add_argument(
        "--save_fps",
        type=int,
        default=None,
        help="FPS for saved output videos (defaults to --fps).",
    )
    parser.add_argument(
        "--model_fps",
        type=int,
        default=None,
        help="FPS assumed by the model (RoPE scaling). Default: pipeline_config.fps if set, else 16.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="bad detailed",
        help="Negative prompt",
    )
    parser.add_argument(
        "--lightx2v_lora_path",
        type=str,
        default=None,
        help="Optional LightX2V safetensors path to apply during inference.",
    )
    parser.add_argument(
        "--lightx2v_lora_scale",
        type=float,
        default=None,
        help="LightX2V scale (defaults to training.custom_settings.lightx2v_lora_scale or 1.0).",
    )
    parser.add_argument(
        "--lightx2v_compat_mode",
        type=str,
        default=None,
        choices=["auto", "force_compat", "off"],
        help="LightX2V loading mode override: auto/native+fallback, force_compat, off(native-only).",
    )
    parser.add_argument(
        "--lightx2v_compat_strict",
        dest="lightx2v_compat_strict",
        action="store_true",
        help="Enable strict LightX2V compatibility checks.",
    )
    parser.add_argument(
        "--no_lightx2v_compat_strict",
        dest="lightx2v_compat_strict",
        action="store_false",
        help="Disable strict LightX2V compatibility checks.",
    )
    parser.set_defaults(lightx2v_compat_strict=None)
    
    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: {checkpoint_path}/eval)",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Suffix to append to output filenames",
    )
    
    # Multi-GPU support
    parser.add_argument(
        "--chunk_id",
        type=int,
        default=None,
        help="Optional chunk id for dataset-file inference (0-based).",
    )
    parser.add_argument(
        "--num_chunks",
        type=int,
        default=None,
        help="Total number of chunks when using --chunk_id.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
        help="Local rank for distributed training",
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=1,
        help="Number of processes for distributed training",
    )
    parser.add_argument(
        "--split_id",
        type=int,
        default=None,
        help="Optional explicit split id for single-process sharding (0-based).",
    )
    parser.add_argument(
        "--num_splits",
        type=int,
        default=None,
        help="Total number of splits when using --split_id or SLURM array mode.",
    )
    
    # Skip existing files
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="Skip inference if output files already exist (default: True)",
    )
    parser.add_argument(
        "--no_skip_existing",
        action="store_false",
        dest="skip_existing",
        help="Disable skip_existing (force re-run even if files exist)",
    )
    
    return parser.parse_args()


def main():
    """Main inference function."""
    args = parse_args()

    if args.chunk_id is None and args.split_id is not None:
        args.chunk_id = args.split_id
    if args.num_chunks is None and args.num_splits is not None:
        args.num_chunks = args.num_splits

    if args.chunk_id is None and "SLURM_ARRAY_TASK_ID" in os.environ:
        try:
            args.chunk_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
            logger.info(f"🔧 Auto-detected SLURM_ARRAY_TASK_ID: {args.chunk_id}")
        except ValueError:
            pass
    if args.num_chunks is None and "SLURM_ARRAY_TASK_COUNT" in os.environ:
        try:
            args.num_chunks = int(os.environ["SLURM_ARRAY_TASK_COUNT"])
            logger.info(f"🔧 Auto-detected SLURM_ARRAY_TASK_COUNT: {args.num_chunks}")
        except ValueError:
            pass

    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError(f"--max_samples must be positive, got {args.max_samples}")

    if args.chunk_id is not None or args.num_chunks is not None:
        if args.chunk_id is None or args.num_chunks is None:
            raise ValueError("Both --chunk_id and --num_chunks must be provided together.")
        if args.num_chunks <= 0:
            raise ValueError(f"--num_chunks must be positive, got {args.num_chunks}")
        if args.chunk_id < 0 or args.chunk_id >= args.num_chunks:
            raise ValueError(
                f"--chunk_id must satisfy 0 <= chunk_id < num_chunks, "
                f"got chunk_id={args.chunk_id}, num_chunks={args.num_chunks}"
            )

    # Setup distributed training if available
    rank, world_size, local_rank = setup_distributed()

    args.split_id = args.chunk_id
    args.num_splits = args.num_chunks
    if args.chunk_id is not None and rank == 0:
        logger.info(f"📊 Chunk mode enabled: chunk_id={args.chunk_id}, num_chunks={args.num_chunks}")
    
    # Set device based on distributed setup
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load experiment config
    if args.experiment_config is None and args.checkpoint_path:
        config_path = find_experiment_config(args.checkpoint_path)
        if config_path:
            if rank == 0:
                logger.info(f"🔧 Auto-detected config: {config_path}")
            args.experiment_config = config_path
            config = load_experiment_config_wrapper(config_path)
        else:
            if rank == 0:
                logger.warning("⚠️  Could not auto-detect config. Some settings may use defaults.")
            config = {}
    else:
        if args.experiment_config:
            config = load_experiment_config_wrapper(args.experiment_config)
        else:
            config = {}
    
    # Resolve model_fps and save_fps
    pipeline_fps = None
    try:
        pipeline_fps = config.get("pipeline", {}).get("fps") if isinstance(config, dict) else None
    except Exception:
        pipeline_fps = None

    if args.model_fps is None:
        args.model_fps = int(pipeline_fps) if pipeline_fps is not None else 16
    if args.save_fps is None:
        args.save_fps = int(args.fps)

    variant_manifest = None
    if args.variant_manifest_json is not None:
        if world_size > 1:
            raise ValueError("--variant_manifest_json currently supports single-process inference only.")
        if args.dataset_file is not None:
            raise ValueError("--variant_manifest_json cannot be combined with --dataset_file")
        if any(
            getattr(args, name) is not None
            for name in (
                "video_path",
                "static_video_path",
                "hand_video_path",
                "static_disparity_video_path",
                "hand_disparity_video_path",
                "prompt",
                "prompt_file",
            )
        ):
            raise ValueError(
                "--variant_manifest_json cannot be combined with explicit custom input paths or prompts."
            )
        variant_manifest = load_variant_manifest(args.variant_manifest_json)

    if rank == 0:
        logger.info(f"🎞️ model_fps={args.model_fps} (transformer/RoPE), save_fps={args.save_fps} (mp4)")

    # Set output directory
    if args.output_dir is None:
        if args.checkpoint_path:
            args.output_dir = os.path.join(args.checkpoint_path, "eval")
        else:
            args.output_dir = "outputs/inference"
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Output directory: {output_dir}")
    
    # Build pipeline
    pipeline = build_pipeline(args, config)
    
    # Load dataset file or use custom paths (only on rank 0, then split)
    if rank == 0 or world_size == 1:
        # Determine input mode
        if args.dataset_file:
            # Batch mode: process multiple videos from dataset file
            if not args.data_root:
                raise ValueError("--data_root is required when using --dataset_file")
            
            with open(args.dataset_file, "r") as f:
                video_paths = [line.strip() for line in f if line.strip()]
            
            if rank == 0:
                logger.info(f"📋 Loaded {len(video_paths)} videos from dataset file")
            video_paths = prepare_video_paths(video_paths, args)
            if rank == 0:
                logger.info(f"📋 Effective dataset size after ordering/filtering: {len(video_paths)} videos")
            if args.chunk_id is not None:
                total_videos = len(video_paths)
                start_idx, end_idx = get_partition_bounds(total_videos, args.chunk_id, args.num_chunks)
                video_paths = video_paths[start_idx:end_idx]
                if rank == 0:
                    logger.info(
                        "📦 Chunk %d/%d selected %d videos (indices %d-%d of %d)",
                        args.chunk_id,
                        args.num_chunks,
                        len(video_paths),
                        start_idx,
                        end_idx - 1,
                        total_videos,
                    )
        elif variant_manifest is not None or has_explicit_custom_inputs(args):
            video_paths = ["custom_input"]
            if rank == 0:
                if variant_manifest is not None:
                    logger.info(
                        "📋 Processing custom single sample with %d hand variants from manifest: %s",
                        len(variant_manifest["variants"]),
                        args.variant_manifest_json,
                    )
                else:
                    logger.info("📋 Processing custom inputs with explicit condition overrides")
        elif args.video_path is not None:
            video_paths = [args.video_path]
            if rank == 0:
                logger.info(f"📋 Processing single custom video path: {args.video_path}")
        else:
            # Single video mode: use custom paths (no dataset file)
            video_paths = ["custom_input"]
            if rank == 0:
                logger.info("📋 Processing custom inputs without dataset-derived video path")
    else:
        video_paths = []  # Will be received from rank 0

    # Split video_paths across GPUs for multi-GPU inference
    if world_size > 1:
        # Use GPU for broadcasting (nccl backend requires GPU tensors)
        broadcast_device = torch.device(f"cuda:{rank}")
        
        # Broadcast video_paths from rank 0 to all ranks
        if rank == 0:
            # Convert to list of strings for broadcasting
            video_paths_str = json.dumps(video_paths)
            video_paths_tensor = torch.tensor([ord(c) for c in video_paths_str], dtype=torch.int32, device=broadcast_device)
            video_paths_len = torch.tensor([len(video_paths_str)], dtype=torch.int32, device=broadcast_device)
        else:
            video_paths_tensor = torch.zeros(1, dtype=torch.int32, device=broadcast_device)
            video_paths_len = torch.zeros(1, dtype=torch.int32, device=broadcast_device)
        
        # Broadcast length first
        dist.broadcast(video_paths_len, src=0)
        if rank != 0:
            video_paths_tensor = torch.zeros(video_paths_len.item(), dtype=torch.int32, device=broadcast_device)
        
        # Broadcast video_paths
        dist.broadcast(video_paths_tensor, src=0)
        
        if rank != 0:
            video_paths_str = ''.join([chr(c.item()) for c in video_paths_tensor.cpu()])
            video_paths = json.loads(video_paths_str)
        else:
            # Move to CPU for processing
            video_paths_tensor = video_paths_tensor.cpu()
        
        # Split video_paths across ranks
        total_videos = len(video_paths)
        start_idx, end_idx = get_partition_bounds(total_videos, rank, world_size)
        my_video_paths = video_paths[start_idx:end_idx]
        
        logger.info(f"🎯 Rank {rank}/{world_size}: Processing {len(my_video_paths)} videos (indices {start_idx}-{end_idx-1} of {total_videos})")
    else:
        my_video_paths = video_paths
    
    # Create output directory based on mode
    if world_size > 1:
        # Distributed mode: use rank_{rank} directory
        rank_output_dir = output_dir / f"rank_{rank}"
    else:
        # Single GPU mode: use output_dir directly
        rank_output_dir = output_dir
    rank_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process videos
    results = []
    for video_path in my_video_paths:
        try:
            logger.info(f"🎯 Processing: {video_path}")

            if video_path == "custom_input" and variant_manifest is not None:
                shared = variant_manifest["shared"]
                shared_video_path = shared.get("video_path")
                shared_static_video_path = shared.get("static_video_path")
                base_name = resolve_custom_input_base_name(
                    shared_video_path,
                    shared_static_video_path,
                    None,
                )

                for variant in variant_manifest["variants"]:
                    variant_label = variant["label"]
                    variant_output_dir = rank_output_dir / variant["safe_label"]
                    variant_output_dir.mkdir(parents=True, exist_ok=True)
                    variant_generated_path = build_generated_output_path(
                        variant_output_dir,
                        base_name=base_name,
                        suffix=args.suffix,
                    )
                    if args.skip_existing and variant_generated_path.exists():
                        results.append(
                            {
                                "video_path": shared_video_path or "custom_input",
                                "variant_label": variant_label,
                                "success": True,
                                "skipped": True,
                            }
                        )
                        continue

                    logger.info("🎯 Processing hand variant: %s", variant_label)

                    data = load_video_data(
                        video_path=shared_video_path,
                        static_video_path=shared_static_video_path,
                        hand_video_path=variant["hand_video_path"],
                        static_disparity_video_path=shared.get("static_disparity_video_path"),
                        hand_disparity_video_path=shared.get("hand_disparity_video_path"),
                        prompt=shared.get("prompt"),
                        prompt_file=shared.get("prompt_file"),
                        data_root=args.data_root,
                        config=config,
                        prompt_subdir=args.prompt_subdir,
                    )

                    if data["gt_video"] is not None:
                        num_frames = data["gt_video"].shape[0]
                        height, width = data["gt_video"].shape[1], data["gt_video"].shape[2]
                    else:
                        num_frames = args.num_frames
                        height, width = args.height, args.width

                    final_prompt = data["prompt"]
                    if args.prompt_prefix:
                        final_prompt = f"{args.prompt_prefix} {final_prompt}".strip()
                        logger.info(f"📝 Prompt with prefix: {final_prompt[:100]}...")

                    conditioning_mode = config.get("training", {}).get("conditioning_mode", "dwm")
                    generated_video, gen_time = run_inference(
                        pipeline=pipeline,
                        prompt=final_prompt,
                        static_video=data["static_video"],
                        hand_video=data["hand_video"],
                        static_disparity_video=data["static_disparity_video"],
                        hand_disparity_video=data["hand_disparity_video"],
                        num_frames=num_frames,
                        height=height,
                        width=width,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        seed=args.seed,
                        negative_prompt=args.negative_prompt,
                        conditioning_mode=conditioning_mode,
                        device=device,
                        attention_kwargs=getattr(args, "lightx2v_attention_kwargs", None),
                    )

                    save_outputs(
                        generated_video=generated_video,
                        output_dir=variant_output_dir,
                        base_name=base_name,
                        gt_video=data["gt_video"],
                        static_video=data["static_video"],
                        hand_video=data["hand_video"],
                        static_disparity_video=data["static_disparity_video"],
                        hand_disparity_video=data["hand_disparity_video"],
                        generation_time=gen_time,
                        args=args,
                        suffix=args.suffix,
                    )

                    results.append(
                        {
                            "video_path": shared_video_path or "custom_input",
                            "variant_label": variant_label,
                            "success": True,
                            "skipped": False,
                        }
                    )
                continue

            if video_path == "custom_input":
                custom_base_name = resolve_custom_input_base_name(
                    args.video_path,
                    args.static_video_path,
                    args.hand_video_path,
                )
                custom_generated_path = build_generated_output_path(
                    rank_output_dir,
                    base_name=custom_base_name,
                    suffix=args.suffix,
                )
                if args.skip_existing and custom_generated_path.exists():
                    results.append(
                        {
                            "video_path": video_path,
                            "success": True,
                            "skipped": True,
                        }
                    )
                    continue

            # Check if output already exists (skip if exists)
            if check_output_exists(video_path, rank_output_dir, args, args.suffix):
                results.append({
                    "video_path": video_path,
                    "success": True,
                    "skipped": True,
                })
                continue
            
            # Load video data
            if video_path == "custom_input":
                # Single video mode: use custom paths
                data = load_video_data(
                    video_path=args.video_path,
                    static_video_path=args.static_video_path,
                    hand_video_path=args.hand_video_path,
                    static_disparity_video_path=args.static_disparity_video_path,
                    hand_disparity_video_path=args.hand_disparity_video_path,
                    prompt=args.prompt,
                    prompt_file=args.prompt_file,
                    data_root=args.data_root,
                    config=config,
                    prompt_subdir=args.prompt_subdir,
                )
            else:
                # Batch mode: auto-derive paths
                data = load_video_data(
                    video_path=video_path,
                    data_root=args.data_root,
                    config=config,
                    prompt_subdir=args.prompt_subdir,
                )
            
            # Determine video dimensions
            if data["gt_video"] is not None:
                num_frames = data["gt_video"].shape[0]
                height, width = data["gt_video"].shape[1], data["gt_video"].shape[2]
            else:
                num_frames = args.num_frames
                height, width = args.height, args.width
            
            # Apply prompt prefix if provided
            final_prompt = data["prompt"]
            if args.prompt_prefix:
                final_prompt = f"{args.prompt_prefix} {final_prompt}".strip()
                logger.info(f"📝 Prompt with prefix: {final_prompt[:100]}...")
            
            # Run inference
            conditioning_mode = config.get("training", {}).get("conditioning_mode", "dwm")
            generated_video, gen_time = run_inference(
                pipeline=pipeline,
                prompt=final_prompt,
                static_video=data["static_video"],
                hand_video=data["hand_video"],
                static_disparity_video=data["static_disparity_video"],
                hand_disparity_video=data["hand_disparity_video"],
                num_frames=num_frames,
                height=height,
                width=width,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                seed=args.seed,
                negative_prompt=args.negative_prompt,
                conditioning_mode=conditioning_mode,
                device=device,
                attention_kwargs=getattr(args, "lightx2v_attention_kwargs", None),
            )
            
            # Save outputs
            if video_path == "custom_input":
                base_name = "output"
                if args.video_path:
                    base_name = Path(args.video_path).stem
                elif args.static_video_path:
                    base_name = Path(args.static_video_path).stem
            else:
                base_name = Path(video_path).stem
            
            save_outputs(
                generated_video=generated_video,
                output_dir=rank_output_dir,
                base_name=base_name,
                gt_video=data["gt_video"],
                static_video=data["static_video"],
                hand_video=data["hand_video"],
                static_disparity_video=data["static_disparity_video"],
                hand_disparity_video=data["hand_disparity_video"],
                generation_time=gen_time,
                args=args,
                suffix=args.suffix,
            )
            
            results.append({"video_path": video_path, "success": True})
            
        except Exception as e:
            logger.error(f"❌ Failed to process {video_path}: {e}")
            results.append({"video_path": video_path, "success": False, "error": str(e)})
    
    # Save batch summary for this rank/split
    summary = {
        "total": len(my_video_paths),
        "successful": sum(1 for r in results if r["success"]),
        "failed": sum(1 for r in results if not r["success"]),
        "rank": rank,
        "world_size": world_size,
        "split_id": args.split_id,
        "num_splits": args.num_splits,
        "shuffle_dataset": args.shuffle_dataset,
        "shuffle_seed": args.shuffle_seed if args.shuffle_seed is not None else args.seed,
        "max_samples": args.max_samples,
        "results": results,
    }
    summary_path = get_batch_summary_path(rank_output_dir, args, rank, world_size)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    if rank == 0 or world_size == 1:
        logger.info(f"📄 Batch summary saved: {summary_path}")
    
    # Gather results from all ranks and merge (only on rank 0)
    if world_size > 1:
        # Wait for all ranks to finish
        dist.barrier()
        
        if rank == 0:
            logger.info(f"🔄 Merging results from {world_size} ranks...")
            
            # Collect all results
            all_results = []
            all_summaries = []
            
            for r in range(world_size):
                rank_dir = output_dir / f"rank_{r}"
                if rank_dir.exists():
                    # Load batch summary from each rank
                    summary_file = rank_dir / f"batch_summary_rank_{r}.json"
                    if summary_file.exists():
                        with summary_file.open('r') as f:
                            summary = json.load(f)
                        all_summaries.append(summary)
                        all_results.extend(summary.get("results", []))
            
            # Create merged summary
            merged_summary = {
                "total": sum(s.get("total", 0) for s in all_summaries),
                "successful": sum(s.get("successful", 0) for s in all_summaries),
                "failed": sum(s.get("failed", 0) for s in all_summaries),
                "num_ranks": world_size,
                "rank_summaries": all_summaries,
                "all_results": all_results,
            }
            
            # Save merged summary
            merged_dir = output_dir / "merged"
            merged_dir.mkdir(parents=True, exist_ok=True)
            merged_summary_file = merged_dir / "merged_summary.json"
            with merged_summary_file.open('w') as f:
                json.dump(merged_summary, f, indent=2)
            
            logger.info(f"✅ Merged summary saved: {merged_summary_file}")
            logger.info(f"📊 Total videos: {merged_summary['total']}")
            logger.info(f"✅ Successful: {merged_summary['successful']}")
            logger.info(f"❌ Failed: {merged_summary['failed']}")
            
            # Move results to output_dir
            logger.info(f"📦 Moving results from rank directories to {output_dir}...")
            for r in range(world_size):
                rank_dir = output_dir / f"rank_{r}"
                if rank_dir.exists():
                    for item in rank_dir.iterdir():
                        if item.suffix in ['.mp4', '.json']:
                            dest = output_dir / item.name
                            if dest.exists():
                                # Append rank number if file exists
                                dest = output_dir / f"{item.stem}_rank{r}{item.suffix}"
                            shutil.move(str(item), str(dest))
                    # Remove rank directory if empty
                    try:
                        rank_dir.rmdir()
                    except OSError:
                        pass
            logger.info(f"✅ All results moved to {output_dir}")
        
        # Cleanup distributed training
        dist.destroy_process_group()
    
    if rank == 0 or world_size == 1:
        logger.info("✅ Inference completed")


if __name__ == "__main__":
    main()
