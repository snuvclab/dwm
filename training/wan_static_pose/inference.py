#!/usr/bin/env python3
"""
WAN-based inference script for WanFunInpaintHandConcatPipeline.
Supports both LoRA and SFT checkpoints.
"""

import argparse
import json
import logging
import os
import time
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

import imageio.v3 as iio
import numpy as np
import torch
import torch.distributed as dist
import yaml
from diffusers import FlowMatchEulerDiscreteScheduler
from safetensors.torch import load_file
from transformers import AutoTokenizer

from training.wan_static_pose.config_loader import load_experiment_config
from training.wan_static_pose.models import WanTransformer3DModel, WanTransformer3DModelWithConcat
from training.wan_static_pose.pipeline import WanFunInpaintPipeline,WanFunInpaintHandConcatPipeline
from training.wan_static_pose.utils.lora_utils import merge_lora
from training.wan_static_pose.utils.utils import (
    filter_kwargs,
    get_image_to_video_latent,
    save_videos_grid,
)
from training.wan_static_pose.models import AutoencoderKLWan, CLIPModel, WanT5EncoderModel

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
    """Find experiment config YAML file in checkpoint parent directory."""
    checkpoint_dir = Path(checkpoint_path)
    parent_dir = checkpoint_dir.parent
    
    # Look for YAML files in parent directory
    yaml_files = list(parent_dir.glob("*.yaml")) + list(parent_dir.glob("*.yml"))
    
    if yaml_files:
        return str(yaml_files[0])
    
    return None


def build_pipeline(args: argparse.Namespace, config: Dict[str, Any]) -> Union[WanFunInpaintPipeline, WanFunInpaintHandConcatPipeline]:
    """Build WAN pipeline from config and checkpoint."""
    
    pipeline_config = config.get("pipeline", {})
    transformer_config = config.get("transformer", {})
    model_config = config.get("model", {})
    training_config = config.get("training", {})
    
    # Get base model path
    base_model_path = pipeline_config.get("base_model_name_or_path", args.base_model_path)
    if not base_model_path:
        raise ValueError("base_model_name_or_path must be provided in config or via --base_model_path")
    
    logger.info(f"🔧 Building WAN pipeline from base model: {base_model_path}")
    
    # Determine weight dtype
    weight_dtype = torch.bfloat16
    
    # Get training mode
    training_mode = training_config.get("mode", "sft")
    logger.info(f"🔧 Training mode: {training_mode}")
    
    # Load components
    logger.info("📦 Loading model components...")
    
    # Text encoder
    text_encoder_kwargs = config.get("text_encoder_kwargs", {})
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(base_model_path, text_encoder_kwargs.get("text_encoder_subpath", "text_encoder")),
        additional_kwargs=text_encoder_kwargs,
        torch_dtype=weight_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(base_model_path, text_encoder_kwargs.get("tokenizer_subpath", "tokenizer")),
    )
    
    # VAE
    vae_kwargs = config.get("vae_kwargs", {})
    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(base_model_path, vae_kwargs.get("vae_subpath", "vae")),
        additional_kwargs=vae_kwargs,
    ).eval()
    
    # CLIP image encoder
    image_encoder_kwargs = config.get("image_encoder_kwargs", {})
    clip_image_encoder = CLIPModel.from_pretrained(
        os.path.join(base_model_path, image_encoder_kwargs.get("image_encoder_subpath", "image_encoder")),
    ).eval()
    
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
    if TransformerCls is WanTransformer3DModelWithConcat:
        condition_channels = transformer_config.get("condition_channels", 16)
        transformer_init_kwargs["condition_channels"] = condition_channels

    if training_mode == "lora":
        # For LoRA, load base transformer
        transformer3d = TransformerCls.from_pretrained(
            os.path.join(base_model_path, transformer_subpath),
            transformer_additional_kwargs=transformer_init_kwargs,
            torch_dtype=weight_dtype,
        )
    else:
        # For SFT, load from checkpoint if provided
        if args.checkpoint_path:
            transformer3d = TransformerCls.from_pretrained(
                os.path.join(args.checkpoint_path, transformer_subpath),
                transformer_additional_kwargs=transformer_init_kwargs,
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
    elif pipeline_type == "wan2.1_fun_inp_hand_concat":
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
        logger.info("🔧 Loading LoRA weights...")
        lora_path = os.path.join(args.checkpoint_path, "lora_diffusion_pytorch_model.safetensors")
        if os.path.exists(lora_path):
            state_dict = load_file(lora_path)
            pipeline = merge_lora(
                pipeline, None, 1.0, device,
                state_dict=state_dict,
                transformer_only=True,
            )
            logger.info("✅ LoRA weights loaded")
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
    
    logger.info("✅ Pipeline built successfully")
    return pipeline


def load_video_data(
    video_path: Optional[str] = None,
    static_video_path: Optional[str] = None,
    hand_video_path: Optional[str] = None,
    prompt: Optional[str] = None,
    prompt_file: Optional[str] = None,
    data_root: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    prompt_subdir: Optional[str] = None,
) -> Dict[str, Any]:
    """Load video data from custom paths or auto-derive from video_path."""
    
    data = {}
    
    # Load prompt
    if prompt:
        data["prompt"] = prompt.strip()
    elif prompt_file and os.path.exists(prompt_file):
        with open(prompt_file, "r") as f:
            data["prompt"] = f.read().strip()
    elif video_path and data_root:
        # Auto-derive prompt path
        video_path_obj = Path(video_path)
        video_name = video_path_obj.stem
        data_config = config.get("data", {}) if config else {}
        # Use provided prompt_subdir if available, otherwise use config, otherwise default
        if prompt_subdir is None:
            prompt_subdir = data_config.get("prompt_subdir", "prompts")
        full_video_path = Path(data_root) / video_path
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
    elif video_path and data_root:
        # Auto-derive static video path
        video_path_obj = Path(video_path)
        data_config = config.get("data", {}) if config else {}
        static_video_subdir = data_config.get("static_video_subdir", "videos_static")
        full_video_path = Path(data_root) / video_path
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
    elif video_path and data_root:
        # Auto-derive hand video path
        video_path_obj = Path(video_path)
        data_config = config.get("data", {}) if config else {}
        hand_video_subdir = data_config.get("hand_video_subdir", "videos_hands")
        full_video_path = Path(data_root) / video_path
        hand_video_path = full_video_path.parent.parent / hand_video_subdir / video_path_obj.name
        if hand_video_path.exists():
            hand_video = iio.imread(str(hand_video_path)).astype(np.float32) / 255.0
            data["hand_video"] = torch.from_numpy(hand_video).permute(3, 0, 1, 2).unsqueeze(0)
        else:
            logger.warning(f"⚠️  Hand video not found: {hand_video_path}. Using None.")
            data["hand_video"] = None
    else:
        data["hand_video"] = None
    
    # Load GT video if available (for comparison)
    if video_path and data_root:
        full_video_path = Path(data_root) / video_path
        if full_video_path.exists():
            gt_video = iio.imread(str(full_video_path)).astype(np.float32) / 255.0
            data["gt_video"] = gt_video
        else:
            data["gt_video"] = None
    else:
        data["gt_video"] = None
    
    return data


def run_inference(
    pipeline: Union[WanFunInpaintPipeline, WanFunInpaintHandConcatPipeline],
    prompt: str,
    static_video: torch.Tensor,
    hand_video: Optional[torch.Tensor],
    num_frames: int,
    height: int,
    width: int,
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    seed: Optional[int] = None,
    negative_prompt: str = "bad detailed",
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Run inference with WAN pipeline."""
    
    # Use provided device or infer from static_video
    if device is None:
        device = static_video.device if isinstance(static_video, torch.Tensor) else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    generator = torch.Generator(device=device).manual_seed(seed) if seed else None
    
    # Create mask (all zeros for full generation)
    mask_video = torch.zeros(1, 1, num_frames, height, width, device=device, dtype=static_video.dtype)
    mask_video[:, :, 1:, :, :] = 255

    # Move videos to device
    static_video = static_video.to(device, dtype=static_video.dtype)
    if hand_video is not None:
        hand_video = hand_video.to(device, dtype=hand_video.dtype)
    
    start_time = time.time()
    with torch.no_grad():
        with torch.autocast("cuda", dtype=static_video.dtype):
            if isinstance(pipeline, WanFunInpaintHandConcatPipeline):
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
                )
            else:
                # Base pipeline uses `video` instead of `static_video`, and no hand condition
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
    video_name = video_path_obj.stem
    
    # Extract scene and action names from path if available
    # video_path: .../scene_name/action_name/processed2/videos/video_name.mp4
    try:
        base_path = video_path_obj.parent.parent  # videos/ -> processed2/
        action_name = base_path.parent.name
        scene_name = base_path.parent.parent.name
        base_name = f"{scene_name}_{action_name}_{video_name}"
    except:
        # Fallback to just video name
        base_name = video_name
    
    # Add suffix to base_name if provided
    if suffix:
        base_name = f"{base_name}{suffix}"
    
    return f"{base_name}_generated.mp4"


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


def save_outputs(
    generated_video: torch.Tensor,
    output_dir: Path,
    base_name: str,
    gt_video: Optional[np.ndarray] = None,
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
    
    # Setup distributed training if available
    rank, world_size, local_rank = setup_distributed()
    
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
                logger.info(f"📋 Processing {len(video_paths)} videos from dataset file")
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
        videos_per_rank = total_videos // world_size
        remainder = total_videos % world_size
        
        start_idx = rank * videos_per_rank + min(rank, remainder)
        end_idx = start_idx + videos_per_rank + (1 if rank < remainder else 0)
        
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
            # Check if output already exists (skip if exists)
            if check_output_exists(video_path, rank_output_dir, args, args.suffix):
                results.append({
                    "video_path": video_path,
                    "success": True,
                    "skipped": True,
                })
                continue
            
            logger.info(f"🎯 Processing: {video_path}")
            
            # Load video data
            if video_path == "custom_input":
                # Single video mode: use custom paths
                data = load_video_data(
                    video_path=args.video_path,
                    static_video_path=args.static_video_path,
                    hand_video_path=args.hand_video_path,
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
            generated_video, gen_time = run_inference(
                pipeline=pipeline,
                prompt=final_prompt,
                static_video=data["static_video"],
                hand_video=data["hand_video"],
                num_frames=num_frames,
                height=height,
                width=width,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                seed=args.seed,
                negative_prompt=args.negative_prompt,
                device=device,
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
                generation_time=gen_time,
                args=args,
                suffix=args.suffix,
            )
            
            results.append({"video_path": video_path, "success": True})
            
        except Exception as e:
            logger.error(f"❌ Failed to process {video_path}: {e}")
            results.append({"video_path": video_path, "success": False, "error": str(e)})
    
    # Save batch summary for this rank
    if len(my_video_paths) > 0:
        summary = {
            "total": len(my_video_paths),
            "successful": sum(1 for r in results if r["success"]),
            "failed": sum(1 for r in results if not r["success"]),
            "results": results,
        }
        summary_path = rank_output_dir / "batch_summary.json"
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
                    summary_file = rank_dir / "batch_summary.json"
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

