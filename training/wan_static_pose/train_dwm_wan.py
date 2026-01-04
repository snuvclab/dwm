# Copyright 2024 The HuggingFace Team.
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

import gc
import logging
import math
import os
import random
import shutil
import argparse
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import imageio.v3 as iio

import diffusers
import torch
import torch.nn.functional as F
import transformers
import wandb

# Try to import MS-SSIM for pixel-space loss
try:
    from pytorch_msssim import ms_ssim
    HAS_MS_SSIM = True
except ImportError:
    HAS_MS_SSIM = False
    logger_temp = logging.getLogger(__name__)
    logger_temp.warning("pytorch-msssim not found. Install with: pip install pytorch-msssim")
from accelerate import Accelerator, DistributedType, init_empty_weights
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    set_seed,
)
from training.wan_static_pose.pipeline.pipeline_wan_fun_inpaint import WanFunInpaintPipeline
from training.wan_static_pose.pipeline.pipeline_wan_fun_inpaint_pose_concat import WanFunInpaintPoseConcatPipeline

from training.cogvideox_static_pose.config_loader import load_experiment_config
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params
from diffusers.utils import export_to_video, convert_unet_state_dict_to_peft, load_image
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from huggingface_hub import create_repo, upload_folder
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, T5EncoderModel
import wandb

from args import get_args
from dataset import (BucketSampler, VideoDatasetWithResizing, VideoDatasetWithResizeAndRectangleCrop, 
                    VideoDatasetWithConditions, VideoDatasetWithConditionsAndResizing, VideoDatasetWithConditionsAndResizeAndRectangleCrop, 
                    VideoDatasetWithHumanMotions, VideoDatasetWithHumanMotionsAndResizing, VideoDatasetWithHumanMotionsAndResizeAndRectangleCrop)
from text_encoder import compute_prompt_embeddings
from utils import (
    get_gradient_norm,
    get_optimizer,
    prepare_rotary_positional_embeddings,
    print_memory,
    reset_memory,
    unwrap_model,
)  # isort:skip

logger = get_logger(__name__)


def log_validation_with_dataset(
    accelerator: Accelerator,
    pipe,
    config: Dict[str, Any],
    validation_video_path: str,
    validation_prompt: str,
    validation_warped_video_path: str = None,
    validation_mask_video_path: str = None,
    validation_hand_video_path: str = None,
    validation_static_video_path: str = None,
    validation_human_motions_path: str = None,
    is_final_validation: bool = False,
    step: int = 0,
    pipeline_type: str = None,
    validation_mode: str = None,
    warped_videos_prob: float = 0.0,
):
    """Log validation results with side-by-side comparison of generated and ground truth videos."""
    logger.info(
        f"Running validation with dataset... \n Generating video for: {validation_video_path}"
    )

    pipe = pipe.to(accelerator.device)

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(config.get("seed", 42)) if config.get("seed") else None

    # Load ground truth video
    gt_video = iio.imread(validation_video_path).astype(np.float32) / 255.0

    # Load condition data for WAN pipelines
    # For WAN pipelines, load hand and static videos
    hand_motions = None
    
    # Load single hand video
    if validation_hand_video_path and os.path.exists(validation_hand_video_path):
        hand_video = iio.imread(validation_hand_video_path).astype(np.float32) / 255.0
        hand_video = hand_video.transpose(3, 0, 1, 2)  # [F, H, W, C] -> [C, F, H, W]
        hand_video = hand_video[np.newaxis, :]  # Add batch dimension: [C, F, H, W] -> [1, C, F, H, W]
    else:
        hand_video = None
            
        if validation_static_video_path and os.path.exists(validation_static_video_path):
            static_video = iio.imread(validation_static_video_path).astype(np.float32) / 255.0
            static_video = static_video.transpose(3, 0, 1, 2)  # [F, H, W, C] -> [C, F, H, W]
            static_video = static_video[np.newaxis, :]  # Add batch dimension: [C, F, H, W] -> [1, C, F, H, W]
        else:
            static_video = None

    # Check condition control flags
    conditions_config = config.get("conditions", {})
    static_only = conditions_config.get("static_only", False)
    hand_only = conditions_config.get("hand_only", False)
    
    # Zero out conditions based on flags
    if static_only and hand_video is not None:
        print("🔧 Static-only mode: Zeroing out hand conditions")
        hand_video = np.zeros_like(hand_video)
    elif hand_only and static_video is not None:
        print("🔧 Hand-only mode: Zeroing out static conditions")
        static_video = np.zeros_like(static_video)

    # Generate video with conditions
    pipeline_args = {
        "prompt": validation_prompt,
        "guidance_scale": 6.0,  # Default guidance scale
        "use_dynamic_cfg": False,  # Disable dynamic cfg to match comparison script
        "height": 480,
        "width": 720,
        "num_frames": 49,
    }
    
    # Prepare mask_video for WAN Fun Inpaint pipelines
    # mask_video controls which frames to regenerate (255 = regenerate, 0 = keep)
    if pipeline_type in ["wan_fun_inpaint", "wan_fun_inpaint_pose_concat"]:
        # Create mask_video based on validation_mode
        batch_size = 1
        num_frames = 49
        height = 480
        width = 720
        
        if validation_mode == "image_to_video":
            # I2V mode: Only first frame is known (0), rest should be regenerated (255)
            # Shape: [B, C, F, H, W] where C=1 for mask
            mask_video = torch.zeros((batch_size, 1, num_frames, height, width), dtype=torch.uint8)
            mask_video[:, :, 1:, :, :] = 255  # Regenerate all frames except first
            pipeline_args["mask_video"] = mask_video
        elif validation_mode == "static_to_video":
            # Static-to-video mode: All frames are condition video (0 = keep all)
            # This leverages the Fun checkpoint's behavior to output cond video as-is
            mask_video = torch.zeros((batch_size, 1, num_frames, height, width), dtype=torch.uint8)
            pipeline_args["mask_video"] = mask_video
        elif validation_mode == "warped_to_video":
            # Warped-to-video mode: All frames are condition video (0 = keep all)
            # This leverages the Fun checkpoint's behavior to output cond video as-is
            validation_mask_video = 1 - iio.imread(validation_mask_video_path).astype(np.float32) / 255.0
            validation_mask_video = validation_mask_video.transpose(3, 0, 1, 2)  # [F, H, W, C] -> [C, F, H, W]
            validation_mask_video = validation_mask_video[np.newaxis, :]  # Add batch dimension: [C, F, H, W] -> [1, C, F, H, W]
            validation_mask_video = torch.from_numpy(validation_mask_video).float()
            pipeline_args["mask_video"] = validation_mask_video[:, :1]
    
    # Add pipeline-specific arguments
    if pipeline_type is None:
        pipeline_type = config.get("pipeline", {}).get("type", "wan_fun_inpaint_pose_concat")
    
    # WAN Fun Inpaint pipelines with video conditioning
    if pipeline_type in ["wan_fun_inpaint", "wan_fun_inpaint_pose_concat"]:
        # For these pipelines, we test both I2V and static-to-video modes
        if validation_mode == "image_to_video":
            # I2V mode: use first frame of GT video as image input
            if gt_video is not None and len(gt_video) > 0:
                first_frame = (gt_video[0] * 255).astype(np.uint8)  # Use first frame as image
                from PIL import Image
                if isinstance(first_frame, np.ndarray):
                    first_frame = Image.fromarray(first_frame)
                pipeline_args["image"] = first_frame
            else:
                # Create a dummy image if no GT video available
                dummy_image = np.zeros((480, 720, 3), dtype=np.uint8)
                from PIL import Image
                if isinstance(dummy_image, np.ndarray):
                    dummy_image = Image.fromarray(dummy_image)
                pipeline_args["image"] = dummy_image
            
            # For pose_concat, we still need hand_videos in I2V mode
            if pipeline_type == "wan_fun_inpaint_pose_concat":
                pipeline_args["hand_videos"] = hand_video
                # Don't pass static_videos in I2V mode (will use image instead)
        else:
            # Static-to-video mode: use full static video
            if pipeline_type == "wan_fun_inpaint":
                pipeline_args["static_videos"] = static_video
            elif pipeline_type == "wan_fun_inpaint_pose_concat":
                pipeline_args["static_videos"] = static_video
                pipeline_args["hand_videos"] = hand_video

    output = pipe(**pipeline_args, generator=generator)
    generated_video = output.frames[0]
    if isinstance(generated_video, torch.Tensor):
        generated_video = generated_video.cpu().numpy()

    # Ensure all videos have the same number of frames (49 frames at 8fps)
    # Only consider non-None videos for frame calculation
    frame_shapes = [generated_video.shape[0], gt_video.shape[0]]
    if static_video is not None:
        static_video = static_video[0].transpose(1, 2, 3, 0)  # [1, C, F, H, W] -> [C, F, H, W] -> [F, H, W, C]
        frame_shapes.append(static_video.shape[0])
    if hand_video is not None:
        hand_video = hand_video[0].transpose(1, 2, 3, 0)  # [1, C, F, H, W] -> [C, F, H, W] -> [F, H, W, C]
        frame_shapes.append(hand_video.shape[0])
    
    num_frames = min(*frame_shapes, 49)
    generated_video = generated_video[:num_frames]
    gt_video = gt_video[:num_frames]
    if hand_video is not None:
        hand_video = hand_video[:num_frames]
    if static_video is not None:
        static_video = static_video[:num_frames]

    # Create comparison video: only include non-None videos
    video_components = []
    if static_video is not None:
        video_components.append(static_video)
    if hand_video is not None:
        # Use hand video as is
        video_components.append(hand_video)
    video_components.extend([generated_video, gt_video])
    
    comparison_video = np.concatenate(video_components, axis=2)  # Concatenate along width

    # Save validation outputs
    phase_name = "test" if is_final_validation else "validation"
    
    # Extract filename from path
    validation_video_path = Path(validation_video_path)
    scene_code = validation_video_path.parent.parent.parent.parent.stem[:8]
    action_name = validation_video_path.parent.parent.parent.stem
    video_name = validation_video_path.stem
    save_name = f"{scene_code}_{action_name}_{video_name}"
    
    # Add GPU index to filename to avoid conflicts in multi-GPU training
    gpu_suffix = f"_gpu{accelerator.process_index}" if accelerator.num_processes > 1 else ""
    
    # Add validation mode suffix for static-to-video pipelines
    mode_suffix = f"_{validation_mode}" if validation_mode else ""
    
    # Ensure output directory exists
    output_dir = config["experiment"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    # Save generated video
    generated_filename = os.path.join(output_dir, f"step_{step}_{phase_name}_generated_{save_name}_{mode_suffix}{gpu_suffix}.mp4")
    export_to_video(generated_video, generated_filename, fps=8)
    
    # Save comparison video (static | hand | generated | gt)
    comparison_filename = os.path.join(output_dir, f"step_{step}_{phase_name}_comparison_{save_name}_{mode_suffix}{gpu_suffix}.mp4")
    export_to_video(comparison_video, comparison_filename, fps=8)
    
    print(f"📹 GPU {accelerator.process_index}: Saved validation videos for {save_name}")
    print(f"   Generated: {generated_filename}")
    print(f"   Comparison (static|hand|generated|gt): {comparison_filename}")

    # Log to wandb if available
    if accelerator.is_main_process:
        for tracker in accelerator.trackers:
            if tracker.name == "wandb":
                caption = f"step_{step}_{phase_name}_{video_name}_comparison{mode_suffix} (static|hand|generated|gt)"
                tracker.log(
                    {
                        phase_name: [
                            wandb.Video(comparison_filename, caption=caption)
                        ]
                    }
                )

    return generated_video


def run_validation(
    config: Dict[str, Any],
    accelerator: Accelerator,
    transformer,
    scheduler,
    model_config: Dict[str, Any],
    weight_dtype: torch.dtype,
    step: int = 0,
    should_run_max_validation: bool = False,
) -> None:
    """Run validation during training."""
    accelerator.print("===== Memory before validation =====")
    print_memory(accelerator.device)
    torch.cuda.synchronize(accelerator.device)

    # Setup pipeline for validation
    pipeline_config = config["pipeline"]
    model_config_dict = config["model"]
    data_config = config.get("data", {})
    
    if pipeline_config["type"] == "wan_fun_inpaint":
        # Setup WAN Fun Inpaint Pipeline for validation
        pipe = WanFunInpaintPipeline.from_pretrained(
            base_model_name_or_path=model_config_dict["base_model_name_or_path"],
            transformer=unwrap_model(accelerator, transformer),
            scheduler=scheduler,
            revision=model_config_dict.get("revision"),
            variant=model_config_dict.get("variant"),
            torch_dtype=weight_dtype,
        )
        
    elif pipeline_config["type"] == "wan_fun_inpaint_pose_concat":
        # Setup WAN Fun Inpaint Pose Concat Pipeline for validation
        condition_channels = pipeline_config.get("condition_channels", 16)
        pipe = WanFunInpaintPoseConcatPipeline.from_pretrained(
            base_model_name_or_path=model_config_dict["base_model_name_or_path"],
            transformer=unwrap_model(accelerator, transformer),
            scheduler=scheduler,
            revision=model_config_dict.get("revision"),
            variant=model_config_dict.get("variant"),
            torch_dtype=weight_dtype,
            condition_channels=condition_channels,
        )
    else:
        raise ValueError(f"Unsupported pipeline type: {pipeline_config['type']}")

    # Set dtype for all components to ensure consistency
    pipe.vae.to(dtype=weight_dtype)
    pipe.text_encoder.to(dtype=weight_dtype)
    pipe.transformer.to(dtype=weight_dtype)
    
    # Enable memory optimizations
    training_config = config["training"]
    if training_config.get("custom_settings", {}).get("enable_slicing", False):
        pipe.vae.enable_slicing()
    if training_config.get("custom_settings", {}).get("enable_tiling", False):
        pipe.vae.enable_tiling()
    if training_config.get("custom_settings", {}).get("enable_model_cpu_offload", False):
        pipe.enable_model_cpu_offload()

    # Load validation set if provided
    data_config = config["data"]
    validation_entries = data_config.get("validation_set")
    if validation_entries is not None:
        if isinstance(validation_entries, (list, tuple)):
            validation_set: List[str] = []
            for entry in validation_entries:
                validation_set_path = os.path.join(data_config["data_root"], entry)
                with open(validation_set_path, "r") as f:
                    validation_set.extend([video.strip() for video in f.readlines()])
        else:
            validation_set_path = os.path.join(data_config["data_root"], validation_entries)
            with open(validation_set_path, "r") as f:
                validation_set = [video.strip() for video in f.readlines()]
        
        # Apply validation stride
        validation_stride = training_config.get("custom_settings", {}).get("validation_stride", 1)
        if validation_stride > 1:
            validation_set = validation_set[::validation_stride]
        
        # Limit validation to just a few videos to speed up validation
        if should_run_max_validation:
            max_validation_videos = min(len(validation_set),data_config.get("max_validation_videos", 4))
        else:
            max_validation_videos = min(1, data_config.get("max_validation_videos", 4))
        
        # For multi-GPU training, multiply by number of GPUs to ensure each GPU gets different videos
        total_validation_videos = max_validation_videos * accelerator.num_processes
        
        # Select validation videos (randomized or sequential)
        if len(validation_set) > total_validation_videos:
            if data_config.get("random_validation", False):
                # Use step number to seed random selection for reproducibility
                random.seed(step // data_config.get("validation_steps", 1000))
                validation_set = random.sample(validation_set, total_validation_videos)
            else:
                # Take first N videos without randomization
                validation_set = validation_set[:total_validation_videos]
        else:
            validation_set = validation_set[:total_validation_videos]
        
        # Distribute videos across GPUs - each GPU gets different videos
        videos_per_gpu = len(validation_set) // accelerator.num_processes
        start_idx = accelerator.process_index * videos_per_gpu
        end_idx = start_idx + videos_per_gpu if accelerator.process_index < accelerator.num_processes - 1 else len(validation_set)
        
        # Each GPU gets its own subset of videos
        validation_set = validation_set[start_idx:end_idx]
        
        print(f"🎯 Multi-GPU Validation Strategy:")
        print(f"   - Total GPUs: {accelerator.num_processes}")
        print(f"   - Videos per GPU: {videos_per_gpu}")
        print(f"   - GPU {accelerator.process_index}: Processing {len(validation_set)} validation videos (indices {start_idx}-{end_idx-1})")
        
        # Extract just the filenames from the video paths
        validation_filenames = []
        for video_path in validation_set:
            # Extract filename from path like "trumans/ego_render_fov90/scene/action/processed2/videos/filename.mp4" -> "filename"
            action_name = video_path.split("/")[-4] if len(video_path.split("/")) >= 4 else "unknown"
            filename = video_path.split("/")[-1].replace(".mp4", "")
            validation_filenames.append(f"{action_name}_{filename}")
        
        # Print the specific video filenames this GPU will process
        print(f"📹 GPU {accelerator.process_index} validation videos:")
        for i, filename in enumerate(validation_filenames):
            print(f"   {i+1}. {filename}")

        # Construct paths for validation data
        
        validation_prompts = []
        validation_videos = [Path(data_config["data_root"]) / video_path for video_path in validation_set]
        
        # Derive prompt paths from video paths
        prompt_subdir = data_config.get("prompt_subdir", "prompts")
        for video_path in validation_set:
            # Convert video path to prompt path
            video_path_obj = Path(video_path)
            prompt_path = video_path_obj.parent.parent / prompt_subdir / f"{video_path_obj.stem}.txt"
            validation_prompts.append(Path(data_config["data_root"]) / prompt_path)
        
        # Derive hand and static video paths from main video paths
        validation_hand_videos = []
        validation_static_videos = []
        validation_warped_videos = []
        validation_warped_mask_videos = []
        hand_video_subdir = data_config.get("hand_video_subdir", "videos_hands")
        
        for video_path in validation_set:
            # Convert video path to hand and static video paths
            video_path_obj = Path(video_path)
            
            # Use single hand video path
            if data_config.get("use_gray_hand_videos", False):
                hand_path = video_path_obj.parent.parent / "videos_hands_gray" / video_path_obj.name
            else:
                # Use configurable hand_video_subdir for default mode
                hand_path = video_path_obj.parent.parent / hand_video_subdir / video_path_obj.name
            validation_hand_videos.append(Path(data_config["data_root"]) / hand_path)
            
            static_path = video_path_obj.parent.parent / "videos_static" / video_path_obj.name
            validation_static_videos.append(Path(data_config["data_root"]) / static_path)

            warped_video_path = video_path_obj.parent.parent / "warped_videos" / video_path_obj.name
            validation_warped_videos.append(Path(data_config["data_root"]) / warped_video_path)

            warped_mask_video_path = video_path_obj.parent.parent / "warped_mask_videos" / video_path_obj.name
            validation_warped_mask_videos.append(Path(data_config["data_root"]) / warped_mask_video_path)

        # Run validation for each video
        for validation_video, validation_prompt, validation_hand_video, validation_static_video, validation_warped_video, validation_warped_mask_video in zip(
            validation_videos, validation_prompts, validation_hand_videos, validation_static_videos, validation_warped_videos, validation_warped_mask_videos
        ):
            # Check required files for WAN pipeline
            # For WAN pipelines, check for hand/static videos
            required_files = [validation_video, validation_prompt, validation_static_video]
            
            # Check single hand video
            if validation_hand_video is not None:
                required_files.append(validation_hand_video)
            
            if not all(os.path.exists(f) for f in required_files):
                missing_files = [str(f) for f in required_files if not os.path.exists(f)]
                print(f"Warning: Missing validation files for {validation_video}: {missing_files}. Skipping.")
                continue
                
            # Load prompt
            with open(validation_prompt, "r") as f:
                prompt_text = f.read().strip()
            
            # Run validation with dataset comparison for WAN pipeline
            log_validation_with_dataset(
                pipe=pipe,
                config=config,
                accelerator=accelerator,
                validation_video_path=validation_video,
                validation_prompt=prompt_text,
                validation_hand_video_path=validation_hand_video,
                validation_static_video_path=validation_static_video,
                validation_human_motions_path=None,
                step=step,
                pipeline_type=config["pipeline"]["type"],
                validation_mode="static_to_video",
            )

    accelerator.print("===== Memory after validation =====")
    print_memory(accelerator.device)
    reset_memory(accelerator.device)

    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(accelerator.device)


def setup_pipeline_from_config(config: Dict[str, Any]):
    """
    Setup WAN pipeline based on configuration.
    
    Args:
        config: Merged configuration dictionary with structure like:
            {
                "pipeline": {"type": "wan_fun_inpaint" | "wan_fun_inpaint_pose_concat"},
                "concat": {
                    "condition_channels": 32  # Optional, defaults to vae.latent_channels * 2
                }
            }
        
    Returns:
        pipeline: Configured pipeline instance
    """
    pipeline_config = config["pipeline"]
    pipeline_type = pipeline_config.get("type", "wan_fun_inpaint_pose_concat")
    data_config = config.get("data", {})
    
    print(f"🔧 Setting up pipeline: {pipeline_type}")
    
    # Determine load dtype based on model size
    model_config = config["model"]
    model_path = model_config["base_model_name_or_path"]
    load_dtype = torch.bfloat16 if "5b" in model_path.lower() else torch.float16
    
    if pipeline_type == "wan_fun_inpaint":
        # Setup WAN Fun Inpaint Pipeline
        print("🔧 Setting up WAN Fun Inpaint pipeline")
        pipeline = WanFunInpaintPipeline.from_pretrained(
            base_model_name_or_path=model_path,
            torch_dtype=load_dtype,
            revision=model_config.get("revision"),
            variant=model_config.get("variant"),
        )
        
    elif pipeline_type == "wan_fun_inpaint_pose_concat":
        # Setup WAN Fun Inpaint Pose Concat Pipeline
        print("🔧 Setting up WAN Fun Inpaint Pose Concat pipeline")
        condition_channels = pipeline_config.get("condition_channels", 16)
        pipeline = WanFunInpaintPoseConcatPipeline.from_pretrained(
            base_model_name_or_path=model_path,
            torch_dtype=load_dtype,
            revision=model_config.get("revision"),
            variant=model_config.get("variant"),
            condition_channels=condition_channels,
        )
        
    else:
        raise ValueError(f"Unsupported pipeline type: {pipeline_type}")
    
    return pipeline


def setup_training_mode(transformer, config: Dict[str, Any]):
    """
    Setup training mode (full, lora, or partial) based on configuration.
    
    Args:
        transformer: Transformer model to configure
        config: Training configuration
        
    Returns:
        num_trainable_parameters: Number of trainable parameters
    """
    training_mode = config["training"]["mode"]
    
    print(f"🚀 Setting up training mode: {training_mode}")
    
    if training_mode == "full":
        return setup_full_training(transformer, config)
    elif training_mode == "lora":
        return setup_lora_training(transformer, config)
    elif training_mode == "partial":
        return setup_partial_training(transformer, config)
    else:
        raise ValueError(f"Unsupported training mode: {training_mode}")


def setup_full_training(transformer, config: Dict[str, Any]):
    """Setup full fine-tuning mode."""
    print("🔧 Setting up full fine-tuning...")
    
    # Check for frozen parameters before enabling all
    frozen_params = []
    for name, param in transformer.named_parameters():
        if not param.requires_grad:
            frozen_params.append((name, param.numel()))
    
    if frozen_params:
        print(f"⚠️  Found {len(frozen_params)} frozen parameter groups:")
        # Group by module prefix for better readability
        frozen_by_module = {}
        for name, numel in frozen_params:
            module_prefix = name.split('.')[0] if '.' in name else name
            if module_prefix not in frozen_by_module:
                frozen_by_module[module_prefix] = []
            frozen_by_module[module_prefix].append((name, numel))
        
        # Print summary by module
        total_frozen = sum(numel for _, numel in frozen_params)
        for module_prefix in sorted(frozen_by_module.keys()):
            module_params = frozen_by_module[module_prefix]
            module_total = sum(numel for _, numel in module_params)
            print(f"   📦 {module_prefix}: {len(module_params)} params, {module_total:,} elements")
            # Show first few parameter names as examples
            for name, numel in module_params[:3]:
                print(f"      - {name} ({numel:,} elements)")
            if len(module_params) > 3:
                print(f"      ... and {len(module_params) - 3} more")
        print(f"   📊 Total frozen: {total_frozen:,} elements")
    else:
        print("✅ All parameters are already trainable")
    
    # Explicitly enable all parameters for training
    for name, param in transformer.named_parameters():
        param.requires_grad_(True)
    
    # Enable gradient checkpointing if specified
    if config["training"].get("custom_settings", {}).get("gradient_checkpointing", False):
        transformer.enable_gradient_checkpointing()
        print("✅ Gradient checkpointing enabled")
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in transformer.parameters())
    print(f"📊 Total parameters: {total_params:,}")
    print(f"📊 Trainable parameters: {trainable_params:,} ({100.0 * trainable_params / total_params:.2f}%)")
    
    return trainable_params


def setup_lora_training(transformer, config: Dict[str, Any]):
    """Setup LoRA fine-tuning mode."""
    print("🔧 Setting up LoRA fine-tuning...")
    
    # Get LoRA parameters from config
    lora_rank = config["training"].get("lora_rank", 64)
    lora_alpha = config["training"].get("lora_alpha", 32)
    non_lora_lr_scale = config["training"].get("non_lora_lr_scale", None)  # Non-LoRA parameters learning rate scale
    
    # Get new flexible parameter patterns
    trainable_patterns = config["training"].get("trainable_parameter_patterns", [])
    frozen_patterns = config["training"].get("frozen_parameter_patterns", [])
    
    print(f"   LoRA Rank: {lora_rank}")
    print(f"   LoRA Alpha: {lora_alpha}")
    print(f"   Non-LoRA LR Scale: {non_lora_lr_scale}")
    print(f"   Trainable Patterns: {trainable_patterns}")
    print(f"   Frozen Patterns: {frozen_patterns}")
    
    # First, freeze all parameters
    for name, param in transformer.named_parameters():
        param.requires_grad_(False)
    
    # Enable gradient checkpointing BEFORE adding LoRA adapters (CRITICAL!)
    if config["training"].get("custom_settings", {}).get("gradient_checkpointing", False):
        transformer.enable_gradient_checkpointing()
        print("✅ Gradient checkpointing enabled BEFORE LoRA setup")
    else:
        print("⚠️  Gradient checkpointing disabled - this may cause high VRAM usage!")
    
    # Add LoRA to attention layers
    # Get target modules from config, with default fallback
    target_modules = config["training"].get("lora_target_modules", ["to_k", "to_q", "to_v", "to_out.0"])
    
    transformer_lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        init_lora_weights=True,
        target_modules=target_modules,
    )
    transformer.add_adapter(transformer_lora_config)
    
    # Apply parameter patterns if specified
    if trainable_patterns or frozen_patterns:
        print("🔧 Applying flexible parameter patterns...")
        _apply_parameter_patterns(transformer, trainable_patterns, frozen_patterns)
    
    # Count trainable parameters by category
    lora_params = 0
    other_trainable_params = 0
    
    for name, param in transformer.named_parameters():
        if param.requires_grad:
            if "lora" in name.lower():
                lora_params += param.numel()
            else:
                other_trainable_params += param.numel()
    
    # Count total trainable parameters
    trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    
    print(f"📊 LoRA parameters: {lora_params:,}")
    print(f"📊 Other trainable parameters: {other_trainable_params:,}")
    print(f"📊 Total trainable parameters: {trainable_params:,}")
    
    return trainable_params


def setup_partial_training(transformer, config: Dict[str, Any]):
    """Setup partial fine-tuning mode using only trainable_patterns."""
    print("🔧 Setting up partial fine-tuning...")
    
    # Get flexible parameter patterns
    trainable_patterns = config["training"].get("trainable_parameter_patterns", [])
    frozen_patterns = config["training"].get("frozen_parameter_patterns", [])
    
    print(f"   Trainable Patterns: {trainable_patterns}")
    print(f"   Frozen Patterns: {frozen_patterns}")
    
    # First, freeze all parameters
    for name, param in transformer.named_parameters():
        param.requires_grad_(False)
    
    # Enable gradient checkpointing if specified
    if config["training"].get("custom_settings", {}).get("gradient_checkpointing", False):
        transformer.enable_gradient_checkpointing()
        print("✅ Gradient checkpointing enabled")
    
    # Apply parameter patterns
    if trainable_patterns or frozen_patterns:
        print("🔧 Applying flexible parameter patterns...")
        _apply_parameter_patterns(transformer, trainable_patterns, frozen_patterns)
    else:
        print("⚠️  No trainable patterns specified - all parameters will remain frozen!")
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    
    print(f"📊 Total trainable parameters: {trainable_params:,}")
    
    return trainable_params


def _apply_parameter_patterns(transformer, trainable_patterns: list, frozen_patterns: list):
    """Apply flexible parameter patterns for trainable/frozen parameters."""
    import re
    
    # First apply frozen patterns
    for pattern in frozen_patterns:
        regex = re.compile(pattern)
        frozen_count = 0
        for name, param in transformer.named_parameters():
            if regex.match(name):
                param.requires_grad_(False)
                frozen_count += 1
        if frozen_count > 0:
            print(f"🔒 Frozen {frozen_count} parameters matching pattern: {pattern}")
    
    # Then apply trainable patterns
    for pattern in trainable_patterns:
        regex = re.compile(pattern)
        trainable_count = 0
        for name, param in transformer.named_parameters():
            if regex.match(name):
                print(f"✅ Set {name} trainable matching pattern: {pattern}")
                param.requires_grad_(True)
                trainable_count += 1
        if trainable_count > 0:
            print(f"✅ Set {trainable_count} parameters trainable matching pattern: {pattern}")


def create_save_hooks(accelerator, transformer, config: Dict[str, Any]):
    """Create save and load hooks for the model."""
    pipeline_type = config["pipeline"]["type"]
    training_config = config["training"]
    training_mode = config["training"]["mode"]
    
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                if isinstance(unwrap_model(accelerator, model), type(unwrap_model(accelerator, transformer))):
                    model = unwrap_model(accelerator, model)
                    
                    if training_mode == "lora":
                        # Save LoRA weights
                        transformer_lora_layers = get_peft_model_state_dict(model)
                        
                        # Save trainable parameters (non-LoRA weights) if trainable_parameter_patterns is specified
                        trainable_state_dict = None
                        if "trainable_parameter_patterns" in training_config:
                            import re
                            trainable_state_dict = {}
                            for name, param in model.named_parameters():
                                # Check if this parameter matches any trainable pattern using regex
                                for pattern in training_config["trainable_parameter_patterns"]:
                                    regex = re.compile(pattern)
                                    if regex.match(name):
                                        trainable_state_dict[name] = param.data
                                        break
                        # Save LoRA weights for WAN pipelines
                        if pipeline_type == "wan_fun_inpaint":
                            WanFunInpaintPipeline.save_lora_weights(
                                output_dir,
                                transformer_lora_layers=transformer_lora_layers,
                            )
                        elif pipeline_type == "wan_fun_inpaint_pose_concat":
                            WanFunInpaintPoseConcatPipeline.save_lora_weights(
                                output_dir,
                                transformer_lora_layers=transformer_lora_layers,
                            )
                        
                        # Save trainable parameters (non-LoRA weights)
                        if trainable_state_dict:
                            torch.save(trainable_state_dict, os.path.join(output_dir, "non_lora_weights.pt"))
                            saved_keys = list(trainable_state_dict.keys())
                            print(f"✅ Saved non-LoRA weights: {saved_keys}")
                        else:
                            print("⚠️ No trainable parameters to save")
                    elif training_mode == "partial":
                        # Save only trainable parameters for partial training
                        trainable_state_dict = {}
                        for name, param in model.named_parameters():
                            if param.requires_grad:
                                trainable_state_dict[name] = param.data
                        
                        # Save trainable parameters
                        torch.save(trainable_state_dict, os.path.join(output_dir, "trainable_parameters.pt"))
                        print(f"✅ Saved {len(trainable_state_dict)} trainable parameters")
                    else:
                        # Save full model
                        model.save_pretrained(
                            os.path.join(output_dir, "transformer"), 
                            safe_serialization=True, 
                            max_shard_size="5GB"
                        )
                    
                    # Remove from weights list
                    if weights:
                        weights.pop()
    
    def load_model_hook(models, input_dir):
        transformer_ = None
        
        # First, load the base transformer model based on pipeline type for WAN
        # This is the same logic regardless of DeepSpeed or training mode
        if pipeline_type == "wan_fun_inpaint":
            # For WAN Fun Inpaint, use base WAN transformer
            from training.wan_static_pose.models import WanTransformer3DModel
            transformer_ = WanTransformer3DModel.from_pretrained(
                config["model"]["base_model_name_or_path"],
                subfolder="transformer",
            )
        elif pipeline_type == "wan_fun_inpaint_pose_concat":
            # For WAN Fun Inpaint Pose Concat, use WAN transformer with concat
            from training.wan_static_pose.models.wan_transformer3d_with_conditions import WanTransformer3DModelWithConcat
            condition_channels = config["pipeline"].get("condition_channels", 16)
            transformer_ = WanTransformer3DModelWithConcat.from_pretrained(
                pretrained_model_name_or_path=None,  # Always start from base model
                base_model_name_or_path=config["model"]["base_model_name_or_path"],
                condition_channels=condition_channels,
            )
        else:
            raise ValueError(f"Unknown pipeline type: {pipeline_type}")
        
        # For non-DeepSpeed, replace the model in the models list
        if not accelerator.distributed_type == DistributedType.DEEPSPEED:
            while len(models) > 0:
                model = models.pop()
                if isinstance(unwrap_model(accelerator, model), type(unwrap_model(accelerator, transformer))):
                    # Replace the model with our loaded transformer
                    model.load_state_dict(transformer_.state_dict())
                    transformer_ = unwrap_model(accelerator, model)
                    break
        
        # Now apply training mode specific loading (LoRA, partial, or full model)
        if training_mode == "lora":
            # First, add PEFT adapter to transformer if it doesn't have one
            if not hasattr(transformer_, 'peft_config'):
                # Get LoRA parameters from config
                lora_rank = config["training"].get("lora_rank", 64)
                lora_alpha = config["training"].get("lora_alpha", 64)
                
                lora_config = LoraConfig(
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    target_modules=["to_k", "to_q", "to_v", "to_out.0"],
                )
                transformer_.add_adapter(lora_config)
            
            # Load LoRA weights for WAN pipelines
            if pipeline_type == "wan_fun_inpaint":
                lora_state_dict = WanFunInpaintPipeline.lora_state_dict(input_dir)
            elif pipeline_type == "wan_fun_inpaint_pose_concat":
                lora_state_dict = WanFunInpaintPoseConcatPipeline.lora_state_dict(input_dir)
            else:
                raise ValueError(f"Unknown pipeline type: {pipeline_type}")
            
            transformer_state_dict = {
                f'{k.replace("transformer.", "")}': v for k, v in lora_state_dict.items() if k.startswith("transformer.")
            }
            transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
            set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
            
            # Load non-LoRA weights (trainable parameters) if they exist
            non_lora_file = os.path.join(input_dir, "non_lora_weights.pt")
            if os.path.exists(non_lora_file):
                non_lora_state_dict = torch.load(non_lora_file, map_location="cpu")
                model_state_dict = transformer_.state_dict()
                loaded_keys = []
                for name, param_data in non_lora_state_dict.items():
                    if name in model_state_dict:
                        model_state_dict[name].copy_(param_data)
                        loaded_keys.append(name)
                print(f"✅ Loaded non-LoRA weights: {loaded_keys}")
        elif training_mode == "partial":
            # Load trainable parameters for partial training
            trainable_params_file = os.path.join(input_dir, "trainable_parameters.pt")
            if os.path.exists(trainable_params_file):
                trainable_state_dict = torch.load(trainable_params_file, map_location="cpu")
                
                # Apply trainable parameters to the model
                model_state_dict = transformer_.state_dict()
                loaded_keys = []
                for name, param_data in trainable_state_dict.items():
                    if name in model_state_dict:
                        model_state_dict[name].copy_(param_data)
                        loaded_keys.append(name)
                
                # Load the updated state dict
                transformer_.load_state_dict(model_state_dict)
                print(f"✅ Loaded {len(loaded_keys)} trainable parameters for partial training")
            else:
                print("⚠️ No trainable_parameters.pt found for partial training")
        else:
            # Load full model weights from checkpoint for WAN
            load_model = None
            if pipeline_type == "wan_fun_inpaint":
                # For WAN Fun Inpaint, use base WAN transformer
                from training.wan_static_pose.models import WanTransformer3DModel
                load_model = WanTransformer3DModel.from_pretrained(os.path.join(input_dir, "transformer"))
            elif pipeline_type == "wan_fun_inpaint_pose_concat":
                # For WAN Fun Inpaint Pose Concat, use WAN transformer with concat
                from training.wan_static_pose.models.wan_transformer3d_with_conditions import WanTransformer3DModelWithConcat
                condition_channels = config["pipeline"].get("condition_channels", 16)
                load_model = WanTransformer3DModelWithConcat.from_pretrained(
                    pretrained_model_name_or_path=os.path.join(input_dir, "transformer"),
                    base_model_name_or_path=config["model"]["base_model_name_or_path"],
                    condition_channels=condition_channels,
                )
            else:
                raise ValueError(f"Unknown pipeline type: {pipeline_type}")
            
            # Load the trained weights into our base transformer
            if load_model is not None:
                transformer_.register_to_config(**load_model.config)
                transformer_.load_state_dict(load_model.state_dict())
                del load_model
        
        # Cast training parameters if needed
        if config["training"].get("custom_settings", {}).get("mixed_precision") in ["fp16", "bf16"]:
            cast_training_params([transformer_])

    return save_model_hook, load_model_hook


def _process_condition_video(condition_video, vae, vae_scaling_factor, device, weight_dtype):
    """Process condition video: if it's raw video, encode it; if it's already latents, use as-is."""
    if condition_video is None:
        return None
    
    # Check if this is a raw video that needs encoding
    if hasattr(condition_video, '_needs_encoding') and condition_video._needs_encoding:
        logger.info(f"Encoding raw video to latents: {getattr(condition_video, '_source_path', 'unknown')}")
        
        # Move to device and dtype
        condition_video = condition_video.to(device=device, dtype=weight_dtype)
        
        # Encode to latents
        condition_video_permuted = condition_video.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
        latent_dist = vae.encode(condition_video_permuted).latent_dist
        latents = latent_dist.sample() * vae_scaling_factor
        
        # Convert back to [B, F, C, H, W] format
        latents = latents.permute(0, 2, 1, 3, 4)
        return latents
    else:
        # Already latents, just ensure correct format and device
        return condition_video.to(device=device, dtype=weight_dtype)


def _process_condition_videos(hand_videos, static_videos, vae, vae_scaling_factor, device, weight_dtype, load_tensors):
    """Process condition videos (hand and static) individually based on their type."""
    
    # Process hand videos
    if hand_videos is not None:
        hand_videos = hand_videos.to(device=device, dtype=vae.dtype)
        # Determine if this is raw video or latents based on channel count
        batch_size, channels, frames, height, width = hand_videos.shape
        
        if channels <= 3:  # Raw video (3 for RGB)
            # Raw video needs encoding
            with torch.no_grad():
                hand_latent = vae.encode(hand_videos)[0]
            hand_videos_latents = hand_latent.sample() * vae_scaling_factor
            hand_videos_latents = hand_videos_latents.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
        else:
            # Already latents (high channel count), need to sample from distribution
            hand_videos = hand_videos.to(device=device, dtype=weight_dtype)
            hand_latent_dist = DiagonalGaussianDistribution(hand_videos)
            hand_videos_latents = hand_latent_dist.sample() * vae_scaling_factor
            hand_videos_latents = hand_videos_latents.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
            hand_videos_latents = hand_videos_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)
    else:
        hand_videos_latents = None
    
    # Process static videos
    if static_videos is not None:
        static_videos = static_videos.to(device=device, dtype=vae.dtype)
        # Determine if this is raw video or latents based on channel count
        batch_size, channels, frames, height, width = static_videos.shape
        
        if channels <= 3:  # Raw video (3 for RGB)
            # Raw video needs encoding
            with torch.no_grad():
                static_latent = vae.encode(static_videos)[0]
            static_videos_latents = static_latent.sample() * vae_scaling_factor
            static_videos_latents = static_videos_latents.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
        else:
            # Already latents (high channel count), need to sample from distribution
            static_videos = static_videos.to(device=device, dtype=weight_dtype)
            static_latent_dist = DiagonalGaussianDistribution(static_videos)
            static_videos_latents = static_latent_dist.sample() * vae_scaling_factor
            static_videos_latents = static_videos_latents.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
            static_videos_latents = static_videos_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)
    else:
        static_videos_latents = None
    
    return hand_videos_latents, static_videos_latents


def _apply_condition_flags(hand_videos, static_videos, conditions_config):
    """Apply condition control flags (static_only, hand_only)."""
    static_only = conditions_config.get("static_only", False)
    hand_only = conditions_config.get("hand_only", False)
    
    if static_only:
        print("🔧 Static-only mode: Zeroing out hand conditions")
        hand_videos = None
    elif hand_only:
        print("🔧 Hand-only mode: Zeroing out static conditions")
        static_videos = None
    
    return hand_videos, static_videos


def _create_fun_mask_input(noisy_model_input, vae_scaling_factor=None):
    """Create mask input for VideoX-Fun pipelines.
    
    Args:
        noisy_model_input: [B, F_latent, C, H_latent, W_latent] tensor (latent space)
        vae_scaling_factor: Scaling factor for VAE (optional, for backward compatibility)
    
    Returns:
        mask_input: [B, F_latent, 1, H_latent, W_latent] tensor (matching noisy_model_input dimensions)
    """
    # Default: create ones mask matching noisy_model_input shape
    if vae_scaling_factor is not None:
        return torch.ones_like(noisy_model_input[:, :, :1]) * vae_scaling_factor
    else:
        return torch.ones_like(noisy_model_input[:, :, :1])



def main():
    parser = argparse.ArgumentParser(description="Unified CogVideoX Pose Training Script")
    parser.add_argument("--experiment_config", type=str, required=True,
                       help="Path to experiment YAML config file")
    parser.add_argument("--override", type=str, nargs="*",
                       help="Override config values (key=value format)")
    parser.add_argument("--mode", type=str, choices=["debug", "slurm_test", "slurm"], default="slurm",
                       help="Training mode: debug, slurm_test, or slurm")
    parser.add_argument("--test_dataloader", action="store_true",
                       help="Test dataloader only without training")
    parser.add_argument("--test_dataloader_samples", type=int, default=10000000,
                       help="Number of samples to test in dataloader test mode")
    args = parser.parse_args()
    
    # Load configuration
    print("🚀 Loading experiment configuration...")
    config = load_experiment_config(args.experiment_config, args.override)
    
    # Check for resume configuration
    resume_from_checkpoint = config.get("training", {}).get("resume_from_checkpoint")
    if resume_from_checkpoint:
        print(f"🔄 Resume mode enabled: resume_from_checkpoint = {resume_from_checkpoint}")
    
    # Extract configuration sections
    experiment_config = config["experiment"]
    pipeline_config = config["pipeline"]
    training_config = config["training"]
    data_config = config["data"]
    model_config = config["model"]
    logging_config = config["logging"]
    
    # Modify output directory based on mode and experiment info
    exp_name = experiment_config.get("name", "unknown_experiment")
    exp_date = experiment_config.get("date", "unknown_date")
    
    # Extract date from experiment date (e.g., "2025-09-01" -> "250901")
    if exp_date != "unknown_date":
        try:
            # Parse date and convert to YYMMDD format
            from datetime import datetime
            parsed_date = datetime.strptime(exp_date, "%Y-%m-%d")
            date_suffix = parsed_date.strftime("%y%m%d")
        except:
            date_suffix = "unknown"
    else:
        date_suffix = "unknown"
    
    # Construct output directory: outputs/{date}/{name}_{mode}
    base_output_dir = f"outputs/{date_suffix}/{exp_name}"
    if args.mode == "debug":
        experiment_config["output_dir"] = f"{base_output_dir}_debug"
        print(f"🔧 Debug mode: Output directory set to {experiment_config['output_dir']}")
    elif args.mode == "slurm_test":
        slurm_job_id = training_config.get("slurm_job_id")
        if resume_from_checkpoint and slurm_job_id:
            print(f"🔄 Resume mode: Using SLURM Job ID from config: {slurm_job_id}")
            experiment_config["output_dir"] = f"{base_output_dir}_slurm_{slurm_job_id}"
            print(f"📁 SLURM test mode: Output directory set to {experiment_config['output_dir']}")
        else:
            experiment_config["output_dir"] = f"{base_output_dir}_slurm_test"
            print(f"🧪 SLURM test mode: Output directory set to {experiment_config['output_dir']}")
        print(f"🧪 SLURM test mode: Output directory set to {experiment_config['output_dir']}")
    elif args.mode == "slurm":
        # Check if this is a resume situation
        if resume_from_checkpoint:
            # Resume mode: get SLURM job ID from config
            slurm_job_id = training_config.get("slurm_job_id")
            if slurm_job_id:
                print(f"🔄 Resume mode: Using SLURM Job ID from config: {slurm_job_id}")
            else:
                slurm_job_id = os.environ.get("SLURM_JOB_ID", "unknown")
                print(f"🔄 Resume mode: Using current SLURM Job ID: {slurm_job_id}")
        else:
            # New training: always use current environment job ID
            slurm_job_id = os.environ.get("SLURM_JOB_ID", "unknown")
            print(f"🚀 New training: Using current SLURM Job ID: {slurm_job_id}")
        
        experiment_config["output_dir"] = f"{base_output_dir}_slurm_{slurm_job_id}"
        print(f"📁 SLURM mode: Output directory set to {experiment_config['output_dir']}")
    
    print(f"📁 Final output directory: {experiment_config['output_dir']}")
    print(f"   - Base: {base_output_dir}")
    print(f"   - Experiment: {exp_name}")
    print(f"   - Date: {exp_date} -> {date_suffix}")
    print(f"   - Mode: {args.mode}")
    
    # Create output directory and copy experiment config
    os.makedirs(experiment_config["output_dir"], exist_ok=True)
    
    # Copy experiment config YAML to output directory
    config_filename = os.path.basename(args.experiment_config)
    output_config_path = os.path.join(experiment_config["output_dir"], config_filename)
    shutil.copy2(args.experiment_config, output_config_path)
    print(f"📋 Copied experiment config to: {output_config_path}")
    
    # Setup accelerator
    logging_dir = Path(experiment_config["output_dir"], "logs")
    accelerator_project_config = ProjectConfiguration(
        project_dir=experiment_config["output_dir"], 
        logging_dir=logging_dir
    )
    
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    init_process_group_kwargs = InitProcessGroupKwargs(backend="nccl", timeout=timedelta(seconds=1800))
    
    accelerator = Accelerator(
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        mixed_precision=training_config.get("custom_settings", {}).get("mixed_precision", "no"),
        log_with=logging_config["report_to"],
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs, init_process_group_kwargs],
    )
    
    # Update DeepSpeed gradient_accumulation_steps from YAML config if using DeepSpeed
    if accelerator.state.deepspeed_plugin is not None:
        yaml_gradient_accumulation_steps = training_config["gradient_accumulation_steps"]
        current_deepspeed_steps = accelerator.state.deepspeed_plugin.deepspeed_config.get("gradient_accumulation_steps")
        if current_deepspeed_steps != yaml_gradient_accumulation_steps:
            print(f"🔧 Updating DeepSpeed gradient_accumulation_steps: {current_deepspeed_steps} → {yaml_gradient_accumulation_steps}")
            accelerator.state.deepspeed_plugin.deepspeed_config["gradient_accumulation_steps"] = yaml_gradient_accumulation_steps
    
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
    if training_config.get("seed") is not None:
        set_seed(training_config["seed"])
    
    # Setup pipeline
    pipeline = setup_pipeline_from_config(config)
    transformer = pipeline.transformer
    scheduler = pipeline.scheduler
    vae = pipeline.vae
    text_encoder = pipeline.text_encoder
    tokenizer = pipeline.tokenizer
    
    # Setup training mode
    num_trainable_parameters = setup_training_mode(transformer, config)
    
    # Create save/load hooks
    save_model_hook, load_model_hook = create_save_hooks(accelerator, transformer, config)
    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)
    
    # Setup optimizer and scheduler
    if training_config["mode"] == "lora":
        # For LoRA training, check if we should use different learning rates for non-LoRA parameters
        non_lora_lr_scale = training_config.get("non_lora_lr_scale", None)
        
        if non_lora_lr_scale is not None:
            # Use different learning rates for LoRA and non-LoRA parameters
            base_lr = float(training_config["learning_rate"])  # Convert to float in case it's a string
            non_lora_lr = base_lr * non_lora_lr_scale
            
            # Separate parameters for different learning rates
            lora_params = []
            non_lora_params = []
            
            for name, param in transformer.named_parameters():
                if param.requires_grad:
                    if "lora" in name.lower():
                        lora_params.append(param)
                    else:
                        non_lora_params.append(param)  # All non-LoRA trainable parameters
            
            params_to_optimize = [
                {"params": lora_params, "lr": base_lr},
                {"params": non_lora_params, "lr": non_lora_lr}
            ]
            
            print(f"🔧 LoRA learning rate: {base_lr}")
            print(f"🔧 Non-LoRA learning rate: {non_lora_lr} (scale: {non_lora_lr_scale})")
            print(f"🔧 LoRA parameters: {len(lora_params)} parameters")
            print(f"🔧 Non-LoRA parameters: {len(non_lora_params)} parameters")
            
            # Print parameter names for debugging
            if len(lora_params) > 0:
                print("🔧 LoRA parameter names:")
                for param in lora_params[:5]:  # Show first 5 parameter names
                    for name, p in transformer.named_parameters():
                        if p is param:
                            print(f"   - {name}")
                            break
                if len(lora_params) > 5:
                    print(f"   ... and {len(lora_params) - 5} more")
            
            if len(non_lora_params) > 0:
                print("🔧 Non-LoRA parameter names:")
                for param in non_lora_params[:5]:  # Show first 5 parameter names
                    for name, p in transformer.named_parameters():
                        if p is param:
                            print(f"   - {name}")
                            break
                if len(non_lora_params) > 5:
                    print(f"   ... and {len(non_lora_params) - 5} more")
        else:
            # Use same learning rate for all parameters
            params_to_optimize = [{"params": list(filter(lambda p: p.requires_grad, transformer.parameters()))}]
            print(f"🔧 Using unified learning rate for all parameters: {training_config['learning_rate']}")
    elif training_config["mode"] == "partial":
        # For partial training, use same learning rate for all trainable parameters
        params_to_optimize = [{"params": list(filter(lambda p: p.requires_grad, transformer.parameters()))}]
        print(f"🔧 Using unified learning rate for partial training: {training_config['learning_rate']}")
    else:
        # Standard optimizer setup for full training
        params_to_optimize = [{"params": list(filter(lambda p: p.requires_grad, transformer.parameters()))}]
    
    optimizer = get_optimizer(
        params_to_optimize=params_to_optimize,
        optimizer_name=training_config.get("optimizer", "adamw"),
        learning_rate=training_config["learning_rate"],
        weight_decay=training_config.get("weight_decay", 0.01),
    )
    
    # Setup dataset and dataloader
    dataset_init_kwargs = {
        "data_root": data_config["data_root"],
        "dataset_file": data_config["dataset_file"],
        "max_num_frames": training_config.get("custom_settings", {}).get("max_num_frames", 48),
        "load_tensors": training_config.get("custom_settings", {}).get("load_tensors", False),
        "random_flip": training_config.get("custom_settings", {}).get("random_flip", False),
        "height_buckets": data_config.get("height_buckets", 480),
        "width_buckets": data_config.get("width_buckets", 720),
        "frame_buckets": data_config.get("frame_buckets", 49),
        "vae_scale_factor_temporal": data_config.get("vae_scale_factor_temporal", 4),
        "vae_scale_factor_spatial": data_config.get("vae_scale_factor_spatial", 8),
        "prompt_subdir": data_config.get("prompt_subdir", "prompts"),
        "prompt_embeds_subdir": data_config.get("prompt_embeds_subdir", "prompt_embeds"),
        "hand_video_subdir": data_config.get("hand_video_subdir", "videos_hands"),
        "hand_video_latents_subdir": data_config.get("hand_video_latents_subdir", "hand_video_latents"),
    }
    
    # Choose dataset class for WAN pipeline
    train_dataset = VideoDatasetWithConditionsAndResizing(**dataset_init_kwargs)
    print("🔧 Using VideoDatasetWithConditionsAndResizing for WAN training")
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=None,
        sampler=BucketSampler(train_dataset, batch_size=training_config["batch_size"], shuffle=True),
        collate_fn=lambda batch: {
            "videos": torch.stack([item["video"] for item in batch]),
            "prompts": (
                torch.stack([item["prompt"] for item in batch])
                if isinstance(batch[0]["prompt"], torch.Tensor)
                else [item["prompt"] for item in batch]
            ),
            "hand_videos": torch.stack([item["hand_videos"] for item in batch]) if "hand_videos" in batch[0] else None,
            "static_videos": torch.stack([item["static_videos"] for item in batch]) if "static_videos" in batch[0] else None,
            "human_motions": torch.stack([item["human_motions"] for item in batch]) if "human_motions" in batch[0] else None,
        },
        num_workers=data_config.get("dataloader_num_workers", 0),
        pin_memory=data_config.get("pin_memory", True),
    )

    # Setup learning rate scheduler
    # num_update_steps_per_epoch = batches_per_epoch / gradient_accumulation_steps
    # where batches_per_epoch = len(train_dataset) / batch_size
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_config["gradient_accumulation_steps"])
    
    # Debug info for training steps calculation
    print(f"📊 Training steps calculation:")
    print(f"   Dataset size: {len(train_dataset)}")
    print(f"   Batch size: {training_config['batch_size']}")
    print(f"   Dataloader length: {len(train_dataloader)}")
    print(f"   Expected batches per epoch: {len(train_dataset) // training_config['batch_size']}")
    print(f"   Gradient accumulation steps: {training_config['gradient_accumulation_steps']}")
    print(f"   Update steps per epoch: {num_update_steps_per_epoch}")
    
    # Verify the calculation
    expected_batches = len(train_dataset) // training_config['batch_size']
    if len(train_dataloader) != expected_batches:
        print(f"⚠️  WARNING: Dataloader length ({len(train_dataloader)}) != expected batches ({expected_batches})")
        print(f"   This suggests a custom sampler is being used")
    else:
        print(f"✅ Dataloader length matches expected batches")
    
    # Use max_train_steps directly if specified, otherwise calculate from epochs
    if "max_train_steps" in training_config:
        max_train_steps = training_config["max_train_steps"]
        # Set num_epochs to a large number to ensure we reach max_train_steps
        # The actual stopping will be controlled by global_step >= max_train_steps check
        num_epochs = max(100, math.ceil(max_train_steps / num_update_steps_per_epoch) * 2)
        print(f"   Max train steps specified: {max_train_steps}")
        print(f"   Calculated epochs (large): {num_epochs}")
        print(f"   Training will stop at step {max_train_steps} (controlled by global_step check)")
    else:
        num_epochs = training_config["num_epochs"]
        max_train_steps = num_epochs * num_update_steps_per_epoch
        print(f"   Num epochs specified: {num_epochs}")
        print(f"   Calculated max train steps: {max_train_steps}")
    
    # Determine weight dtype for mixed precision training
    weight_dtype = torch.float32
    if accelerator.state.deepspeed_plugin:
        # DeepSpeed is handling precision, use what's in the DeepSpeed config
        if (
            "fp16" in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config["fp16"]["enabled"]
        ):
            weight_dtype = torch.float16
        if (
            "bf16" in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config["bf16"]["enabled"]
        ):
            weight_dtype = torch.bfloat16
    else:
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
    
    # Setup learning rate scheduler
    use_cpu_offload_optimizer = training_config.get("custom_settings", {}).get("use_cpu_offload_optimizer", False)
    if use_cpu_offload_optimizer:
        lr_scheduler = None
        accelerator.print(
            "CPU Offload Optimizer cannot be used with DeepSpeed or builtin PyTorch LR Schedulers. If "
            "you are training with those settings, they will be ignored."
        )
    else:
        use_deepspeed_scheduler = (
            accelerator.state.deepspeed_plugin is not None
            and "scheduler" in accelerator.state.deepspeed_plugin.deepspeed_config
        )
        
        if use_deepspeed_scheduler:
            from accelerate.utils import DummyScheduler
            lr_scheduler = DummyScheduler(
                name=training_config.get("lr_scheduler", "cosine"),
                optimizer=optimizer,
                total_num_steps=max_train_steps * accelerator.num_processes,
                num_warmup_steps=training_config.get("lr_warmup_steps", 0) * accelerator.num_processes,
            )
        else:
            lr_scheduler = get_scheduler(
                training_config.get("lr_scheduler", "cosine"),
                optimizer=optimizer,
                num_warmup_steps=training_config.get("lr_warmup_steps", 0) * accelerator.num_processes,
                num_training_steps=max_train_steps * accelerator.num_processes,
                num_cycles=training_config.get("lr_num_cycles", 1),
            )
    
    # Move models to device and set dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    
    # Prepare everything with accelerator
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )
    
    # Initialize trackers
    if accelerator.distributed_type == DistributedType.DEEPSPEED or accelerator.is_main_process:
        # Create descriptive run name with date and experiment info
        exp_name = experiment_config.get("name", "unknown_experiment")
        exp_date = experiment_config.get("date", "unknown_date")
        training_mode = training_config.get("mode", "unknown")
        
        # Extract date suffix (e.g., "2025-09-01" -> "250901")
        if exp_date != "unknown_date":
            try:
                from datetime import datetime
                parsed_date = datetime.strptime(exp_date, "%Y-%m-%d")
                date_suffix = parsed_date.strftime("%y%m%d")
            except:
                date_suffix = "unknown"
        else:
            date_suffix = "unknown"
        
        # Create run name: {date}_{name}_{mode}_{job_id}
        slurm_job_id = os.environ.get("SLURM_JOB_ID", "local")
        run_name = f"{date_suffix}_{exp_name}_{args.mode}_{slurm_job_id}"
        
        # Initialize accelerator trackers (this will handle WandB initialization)
        project_name = logging_config.get("project_name", "world_model")
        entity_name = logging_config.get("entity_name", "vclab_2024")
        
        # Create custom config for accelerator with additional metadata
        accelerator_config = {
            "experiment": {
                "name": exp_name,
                "date": exp_date,
                "description": experiment_config.get("description", "No description"),
                "author": experiment_config.get("author", "Unknown"),
                "output_dir": experiment_config["output_dir"]
            },
            "pipeline": pipeline_config,
            "training": {
                "mode": training_mode,
                "learning_rate": training_config.get("learning_rate"),
                "batch_size": training_config.get("batch_size"),
                "max_train_steps": training_config.get("max_train_steps"),
                "optimizer": training_config.get("optimizer"),
                "lr_scheduler": training_config.get("lr_scheduler"),
                "prompt_dropout_prob": training_config.get("prompt_dropout_prob", 0.0),
            },
            "data": {
                "dataset_file": data_config.get("dataset_file"),
                "data_root": data_config.get("data_root")
            },
            "model": {
                "base_model_name_or_path": model_config.get("base_model_name_or_path"),
                "num_trainable_parameters": num_trainable_parameters
            },
            "system": {
                "num_gpus": accelerator.num_processes,
                "mixed_precision": training_config.get("custom_settings", {}).get("mixed_precision", "no"),
                "gradient_checkpointing": training_config.get("custom_settings", {}).get("gradient_checkpointing", False)
            }
        }
        
        # Initialize accelerator trackers with custom config and wandb settings
        # Set wandb to offline mode for debug and slurm_test modes
        wandb_init_kwargs = {
            "name": run_name,
            "entity": entity_name,
        }
        
        if args.mode in ["debug", "slurm_test"]:
            wandb_init_kwargs["mode"] = "offline"
            print(f"🔧 WandB set to offline mode for {args.mode}")
        
        accelerator.init_trackers(
            project_name=project_name,
            config=accelerator_config,
            init_kwargs={
                "wandb": wandb_init_kwargs
            }
        )
        
        # Print experiment info
        print(f"🔗 Experiment initialized: {run_name}")
        print(f"   Project: {project_name}")
        print(f"   Entity: {entity_name}")
        print(f"   Experiment: {exp_name}")
        print(f"   Date: {exp_date}")
        print(f"   Mode: {training_mode}")
        print(f"   Pipeline: {pipeline_config['type']}")
        print(f"   Trainable parameters: {num_trainable_parameters:,}")
        
        accelerator.print("===== Memory before training =====")
        reset_memory(accelerator.device)
        print_memory(accelerator.device)
    
    # Training loop
    total_batch_size = training_config["batch_size"] * accelerator.num_processes * training_config["gradient_accumulation_steps"]
    
    accelerator.print("***** Running training *****")
    accelerator.print(f"  Num trainable parameters = {num_trainable_parameters}")
    accelerator.print(f"  Num examples = {len(train_dataset)}")
    accelerator.print(f"  Num batches each epoch = {len(train_dataloader)}")
    accelerator.print(f"  Num epochs = {num_epochs}")
    accelerator.print(f"  Max train steps = {max_train_steps}")
    accelerator.print(f"  Instantaneous batch size per device = {training_config['batch_size']}")
    accelerator.print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    accelerator.print(f"  Gradient accumulation steps = {training_config['gradient_accumulation_steps']}")
    accelerator.print(f"  Total optimization steps = {max_train_steps}")
    
    # Print augmentation settings
    prompt_dropout_prob = training_config.get("prompt_dropout_prob", 0.0)
    hand_dropout_prob = training_config.get("hand_dropout_prob", 0.0)
    
    if prompt_dropout_prob > 0:
        accelerator.print(f"  🎲 Prompt dropout: {prompt_dropout_prob * 100:.1f}%")
    if hand_dropout_prob > 0:
        accelerator.print(f"  ✋ Hand dropout: {hand_dropout_prob * 100:.1f}%")
    
    print(f"🚀 Starting training for {max_train_steps} steps...")
    
    global_step = 0
    first_epoch = 0
    
    # Potentially load in the weights and states from a previous save
    if training_config.get("resume_from_checkpoint"):
        if training_config["resume_from_checkpoint"] != "latest":
            # Use the provided checkpoint path directly
            checkpoint_path = training_config["resume_from_checkpoint"]
            if not os.path.exists(checkpoint_path):
                accelerator.print(
                    f"Checkpoint '{checkpoint_path}' does not exist. Starting a new training run."
                )
                training_config["resume_from_checkpoint"] = None
                initial_global_step = 0
            else:
                accelerator.print(f"Resuming from checkpoint {checkpoint_path}")
                accelerator.load_state(checkpoint_path)
                # Extract global step from checkpoint path
                checkpoint_name = os.path.basename(checkpoint_path.rstrip('/'))
                if checkpoint_name.startswith("checkpoint-"):
                    global_step = int(checkpoint_name.split("-")[1])
                    initial_global_step = global_step
                    first_epoch = global_step // num_update_steps_per_epoch
                else:
                    accelerator.print(f"Warning: Could not extract global step from checkpoint name '{checkpoint_name}'")
                    initial_global_step = 0
        else:
            # Get the most recent checkpoint from output_dir
            output_dir = experiment_config["output_dir"]
            if not os.path.exists(output_dir):
                accelerator.print(f"Output directory '{output_dir}' does not exist. Starting a new training run.")
                training_config["resume_from_checkpoint"] = None
                initial_global_step = 0
            else:
                dirs = os.listdir(output_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                if len(dirs) == 0:
                    accelerator.print("No checkpoint directories found. Starting a new training run.")
                    training_config["resume_from_checkpoint"] = None
                    initial_global_step = 0
                else:
                    dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                    latest_checkpoint = dirs[-1]
                    checkpoint_path = os.path.join(output_dir, latest_checkpoint)
                    accelerator.print(f"Resuming from latest checkpoint {checkpoint_path}")
                    accelerator.load_state(checkpoint_path)
                    global_step = int(latest_checkpoint.split("-")[1])
                    initial_global_step = global_step
                    first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0
    
    # Initialize dataset's global_step for conditional i2v training
    if hasattr(train_dataset, 'set_global_step'):
        train_dataset.set_global_step(initial_global_step)

    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    # For DeepSpeed training
    transformer_config = transformer.module.config if hasattr(transformer, "module") else transformer.config

    # Enable TF32 for faster training on Ampere GPUs
    if training_config.get("custom_settings", {}).get("allow_tf32", False) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    # Scale learning rate if specified
    if training_config.get("scale_lr", False):
        training_config["learning_rate"] = (
            training_config["learning_rate"] * training_config["gradient_accumulation_steps"] * 
            training_config["batch_size"] * accelerator.num_processes
        )

    # Make sure the trainable params are in float32 for mixed precision
    if training_config.get("custom_settings", {}).get("mixed_precision") == "fp16":
        cast_training_params([transformer], dtype=torch.float32)

    # Get VAE scaling factors
    VAE_SCALING_FACTOR = vae.config.scaling_factor
    VAE_SCALE_FACTOR_SPATIAL = 2 ** (len(vae.config.block_out_channels) - 1)
    RoPE_BASE_HEIGHT = transformer_config.get("sample_height") * VAE_SCALE_FACTOR_SPATIAL
    RoPE_BASE_WIDTH = transformer_config.get("sample_width") * VAE_SCALE_FACTOR_SPATIAL

    # Get scheduler alphas_cumprod for loss weighting
    alphas_cumprod = scheduler.alphas_cumprod.to(accelerator.device, dtype=torch.float32)

    # Test dataloader if requested
    if args.test_dataloader:
        print("🧪 Testing dataloader...")
        print(f"Dataset length: {len(train_dataset)}")
        print(f"Dataloader length: {len(train_dataloader)}")
        
        # Test a few samples
        test_samples = min(args.test_dataloader_samples, len(train_dataloader))
        print(f"Testing {test_samples} samples...")
        
        try:
            for i, batch in enumerate(train_dataloader):
                if i >= test_samples:
                    break
                    
                print(f"\n--- Sample {i+1} ---")
                print(f"Batch keys: {list(batch.keys())}")
                
                if "videos" in batch:
                    print(f"Videos shape: {batch['videos'].shape}")
                    print(f"Videos dtype: {batch['videos'].dtype}")
                    print(f"Videos min/max: {batch['videos'].min():.3f}/{batch['videos'].max():.3f}")
                
                if "hand_videos" in batch and batch["hand_videos"] is not None:
                    print(f"Hand videos shape: {batch['hand_videos'].shape}")
                    print(f"Hand videos dtype: {batch['hand_videos'].dtype}")
                    print(f"Hand videos min/max: {batch['hand_videos'].min():.3f}/{batch['hand_videos'].max():.3f}")
                
                if "static_videos" in batch and batch["static_videos"] is not None:
                    print(f"Static videos shape: {batch['static_videos'].shape}")
                    print(f"Static videos dtype: {batch['static_videos'].dtype}")
                    print(f"Static videos min/max: {batch['static_videos'].min():.3f}/{batch['static_videos'].max():.3f}")
                
                if "prompts" in batch:
                    if isinstance(batch["prompts"], list):
                        print(f"Prompts (first 2): {batch['prompts'][:2]}")
                    else:
                        print(f"Prompts shape: {batch['prompts'].shape}")
                        print(f"Prompts dtype: {batch['prompts'].dtype}")
                
                print(f"Sample {i+1} loaded successfully!")
            
            print("\n✅ Dataloader test completed successfully!")
            
        except Exception as e:
            print(f"\n❌ Dataloader test failed with error:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            import traceback
            print(f"Traceback:")
            traceback.print_exc()
        
        print("Exiting without training...")
        return
    
    # Run initial validation at step 0 (before training starts)
    if data_config.get("validation_set") is not None:
        logger.info("Running initial validation at step 0 (before training starts)")
        should_run_max_validation = training_config.get("custom_settings", {}).get("should_run_max_validation", False)
        run_validation(
            config=config,
            accelerator=accelerator,
            transformer=transformer,
            scheduler=scheduler,
            model_config=transformer_config,
            weight_dtype=weight_dtype,
            step=initial_global_step,
            should_run_max_validation=should_run_max_validation
        )

    # Generate empty prompt embedding for prompt dropout (classifier-free guidance training)
    EMPTY_PROMPT_EMBED = None
    
    load_tensors = training_config.get("custom_settings", {}).get("load_tensors", False)
    
    if prompt_dropout_prob > 0:
        if text_encoder is None or tokenizer is None:
            logger.warning("⚠️  Prompt dropout requested but tokenizer/text_encoder not available; disabling.")
            prompt_dropout_prob = 0.0
        else:
            # Generate empty prompt embedding once before training loop
            with torch.no_grad():
                EMPTY_PROMPT_EMBED = get_t5_prompt_embeds(
                    prompt="",
                    tokenizer=tokenizer,
                    text_encoder=text_encoder,
                    device=accelerator.device,
                    dtype=weight_dtype,
                )
            logger.info(f"🎲 Prompt dropout enabled: {prompt_dropout_prob * 100:.1f}% probability")
            logger.info(f"   EMPTY_PROMPT_EMBED shape: {EMPTY_PROMPT_EMBED.shape}")

    # Free up memory by deleting vae and text_encoder when using preprocessed tensors
    if load_tensors:
        del vae, text_encoder
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(accelerator.device)

    # tempoary checkpoint save code here
    save_path = os.path.join(experiment_config["output_dir"], f"checkpoint-{global_step}")

    # Training loop - epoch based (like original CogVideoX)
    transformer.train()

    for epoch in range(first_epoch, num_epochs):
        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer]
            logs = {}

            with accelerator.accumulate(models_to_accumulate):
                # Update epoch for display
                current_epoch = epoch
                
                # Get videos (already decided at dataset level whether to use warped_videos)
                videos = batch["videos"].to(accelerator.device, non_blocking=True)
                
                images = batch.get("images")  # For I2V pipeline and prediction mode
                prompts = batch["prompts"]
                hand_videos = batch.get("hand_videos")
                static_videos = batch.get("static_videos")
                hand_motions = batch.get("hand_motions")
                if hand_motions is None:
                    hand_motions = batch.get("human_motions")  # Backward compatibility
                image_goal = batch.get("image_goal")  # For planning mode (future use)
                
                # Process condition videos once for all pipelines
                if load_tensors:
                    # When load_tensors=True, VAE is already deleted and videos are already latents
                    # Dataset returns latents in [C, F, H, W] format, collate_fn makes [B, C, F, H, W]
                    # Stored latents are distribution parameters (mean and logvar), need to sample
                    
                    # Process hand videos (already latents)
                    if hand_videos is not None:
                        hand_videos = hand_videos.to(device=accelerator.device, dtype=weight_dtype)
                        # Already latents: [B, C, F, H, W] format (distribution parameters)
                        # Sample from distribution
                        hand_latent_dist = DiagonalGaussianDistribution(hand_videos)
                        hand_videos_latents = hand_latent_dist.sample() * VAE_SCALING_FACTOR
                        
                        # Convert to [B, F, C, H, W] format
                        hand_videos_latents = hand_videos_latents.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
                        hand_videos_latents = hand_videos_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)
                    else:
                        hand_videos_latents = None
                    
                    # Process static videos (already latents)
                    if static_videos is not None:
                        static_videos = static_videos.to(device=accelerator.device, dtype=weight_dtype)
                        # Already latents: [B, C, F, H, W] format (distribution parameters)
                        # Sample from distribution (same as _process_condition_videos)
                        static_latent_dist = DiagonalGaussianDistribution(static_videos)
                        static_videos_latents = static_latent_dist.sample() * VAE_SCALING_FACTOR
                        static_videos_latents = static_videos_latents.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
                        static_videos_latents = static_videos_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)
                    else:
                        static_videos_latents = None
                else:
                    # When load_tensors=False, VAE is available for encoding
                    hand_videos_latents, static_videos_latents = _process_condition_videos(
                        hand_videos, static_videos, vae, VAE_SCALING_FACTOR, 
                        accelerator.device, weight_dtype, load_tensors
                    )
                
                # Apply hand dropout for classifier-free guidance training (helps prevent color drift)
                # This should be applied to hand_videos_latents directly since some pipelines use it directly
                if hand_videos_latents is not None and hand_dropout_prob > 0:
                    # Get batch size from hand_videos_latents shape
                    hand_batch_size = hand_videos_latents.shape[0]
                    # Sample dropout mask: each sample in batch has independent dropout probability
                    dropout_mask = torch.rand(hand_batch_size, device=accelerator.device) < hand_dropout_prob
                    
                    if dropout_mask.any():
                        # Zero out hand latents for dropped samples
                        # hand_videos_latents shape: [B, F, C, H, W]
                        hand_videos_latents[dropout_mask] = torch.zeros_like(hand_videos_latents[dropout_mask])
                        
                        # Log dropout statistics (only occasionally to avoid spam)
                        if global_step % 100 == 0:
                            num_dropped = dropout_mask.sum().item()
                            logs["hand_dropout_count"] = num_dropped

                # Encode videos
                if not load_tensors:
                    videos = videos.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
                    with torch.no_grad():
                        latent_dist = vae.encode(videos).latent_dist
                else:
                    latent_dist = DiagonalGaussianDistribution(videos)

                videos = latent_dist.sample() * VAE_SCALING_FACTOR
                videos = videos.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
                videos = videos.to(memory_format=torch.contiguous_format, dtype=weight_dtype)
                model_input = videos

                # Use already processed condition videos from _process_condition_videos
                # Note: hand_dropout is already applied to hand_videos_latents above
                if hand_videos_latents is not None:
                    hand_videos = hand_videos_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)
                else:
                    hand_videos = None
                    
                if static_videos_latents is not None:
                    static_videos = static_videos_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)
                else:
                    static_videos = None

                # Encode prompts
                if not training_config.get("custom_settings", {}).get("load_tensors", False):
                    # Handle case where prompts might be a list of strings
                    if isinstance(prompts, list):
                        prompt_embeds = compute_prompt_embeddings(
                            tokenizer,
                            text_encoder,
                            prompts,
                            model_config.max_text_seq_length,
                            accelerator.device,
                            weight_dtype,
                            requires_grad=False,
                        )
                    else:
                        prompt_embeds = prompts.to(dtype=weight_dtype)
                else:
                    # When load_tensors=True, prompts is already a batched tensor
                    prompt_embeds = prompts.to(dtype=weight_dtype)
                
                # Apply prompt dropout for classifier-free guidance training
                if prompt_dropout_prob > 0 and EMPTY_PROMPT_EMBED is not None:
                    # Sample dropout mask: each sample in batch has independent dropout probability
                    batch_size = prompt_embeds.shape[0]
                    dropout_mask = torch.rand(batch_size, device=accelerator.device) < prompt_dropout_prob
                    
                    if dropout_mask.any():
                        # Replace dropped prompts with empty prompt embedding
                        # EMPTY_PROMPT_EMBED shape: [1, seq_len, hidden_dim]
                        # prompt_embeds shape: [batch_size, seq_len, hidden_dim]
                        prompt_embeds[dropout_mask] = EMPTY_PROMPT_EMBED.to(
                            device=prompt_embeds.device, 
                            dtype=prompt_embeds.dtype
                        )
                        
                        # Log dropout statistics (only occasionally to avoid spam)
                        if global_step % 100 == 0:
                            num_dropped = dropout_mask.sum().item()
                            logs["prompt_dropout_count"] = num_dropped

                # Sample noise that will be added to the latents
                noise = torch.randn_like(model_input)
                batch_size, num_frames, num_channels, height, width = model_input.shape

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    scheduler.config.num_train_timesteps,
                    (batch_size,),
                    dtype=torch.int64,
                    device=model_input.device,
                )

                # Prepare rotary embeds
                image_rotary_emb = (
                    prepare_rotary_positional_embeddings(
                        height=height * VAE_SCALE_FACTOR_SPATIAL,
                        width=width * VAE_SCALE_FACTOR_SPATIAL,
                        num_frames=num_frames,
                        vae_scale_factor_spatial=VAE_SCALE_FACTOR_SPATIAL,
                        patch_size=transformer_config.patch_size,
                        patch_size_t=transformer_config.patch_size_t if hasattr(transformer_config, "patch_size_t") else None,
                        attention_head_dim=transformer_config.attention_head_dim,
                        device=accelerator.device,
                        base_height=RoPE_BASE_HEIGHT,
                        base_width=RoPE_BASE_WIDTH,
                    )
                    if transformer_config.use_rotary_positional_embeddings
                    else None
                )

                # Add noise to the model input according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_model_input = scheduler.add_noise(model_input, noise, timesteps)

                # Prepare condition latents for WAN pipeline
                if pipeline_config["type"] == "wan_fun_inpaint_pose_concat":
                    # For WAN Fun Inpaint Pose Concat pipeline, use both static and hand videos as conditions
                    # Use preprocessed condition latents
                    mask_input = _create_fun_mask_input(noisy_model_input, VAE_SCALING_FACTOR)
                    mask_input = mask_input.to(device=accelerator.device, dtype=weight_dtype)
                    transformer_input = torch.cat([noisy_model_input, mask_input, 
                                                   static_videos_latents, hand_videos_latents], dim=2)
                    condition_latents = None
                else:
                    # For basic WAN Fun Inpaint pipeline, use static video as condition
                    mask_input = _create_fun_mask_input(noisy_model_input, VAE_SCALING_FACTOR)
                    transformer_input = torch.cat([noisy_model_input, mask_input, static_videos_latents], dim=2)
                    condition_latents = None

                # Predict the noise residual
                if pipeline_config["type"] == "cogvideox_i2v":
                    # For basic I2V pipeline, use standard transformer forward
                    model_output = transformer(
                        hidden_states=transformer_input,
                        encoder_hidden_states=prompt_embeds,
                        timestep=timesteps,
                        image_rotary_emb=image_rotary_emb,
                        return_dict=False,
                    )[0]
                    
                elif pipeline_config["type"] == "cogvideox_pose_concat":
                    model_output = transformer(
                        hidden_states=transformer_input,
                        encoder_hidden_states=prompt_embeds,
                        timestep=timesteps,
                        image_rotary_emb=image_rotary_emb,
                        return_dict=False,
                    )[0]
                elif pipeline_config["type"] == "cogvideox_pose_adapter":
                    # For adapter pipeline, use the transformer's forward method with conditions
                    if condition_latents is not None:
                        model_output = transformer(
                            hidden_states=transformer_input,
                            encoder_hidden_states=prompt_embeds,
                            timestep=timesteps,
                            image_rotary_emb=image_rotary_emb,
                            hand_conditions=condition_latents["hand_videos_latents"],
                            static_conditions=condition_latents["static_videos_latents"],
                            return_dict=False,
                        )[0]
                    else:
                        model_output = transformer(
                            hidden_states=transformer_input,
                            encoder_hidden_states=prompt_embeds,
                            timestep=timesteps,
                            image_rotary_emb=image_rotary_emb,
                            return_dict=False,
                        )[0]
                elif pipeline_config["type"] in ["cogvideox_pose_adaln", "cogvideox_pose_adaln_perframe"]:
                    # For AdaLN pipeline, use the transformer's forward method with motion params
                    pose_params = None
                    if condition_latents is not None:
                        pose_params = condition_latents.get("hand_motions") or condition_latents.get("human_motions")
                    if pose_params is not None:
                        model_output = transformer(
                            hidden_states=transformer_input,
                            encoder_hidden_states=prompt_embeds,
                            timestep=timesteps,
                            image_rotary_emb=image_rotary_emb,
                            pose_params=pose_params,
                            return_dict=False,
                        )[0]
                    else:
                        model_output = transformer(
                            hidden_states=transformer_input,
                            encoder_hidden_states=prompt_embeds,
                            timestep=timesteps,
                            image_rotary_emb=image_rotary_emb,
                            return_dict=False,
                        )[0]
                elif pipeline_config["type"] in ["cogvideox_static_to_video", "cogvideox_static_to_video_pose_concat"]:
                    # For static-to-video pipelines, use standard transformer forward (conditions already concatenated)
                    transformer_input = torch.cat([noisy_model_input, static_videos_latents, hand_videos_latents], dim=2)
                    model_output = transformer(
                        hidden_states=transformer_input,
                        encoder_hidden_states=prompt_embeds,
                        timestep=timesteps,
                        image_rotary_emb=image_rotary_emb,
                        return_dict=False,
                    )[0]
                # For WAN pipelines, use standard transformer forward (conditions already concatenated)
                model_output = transformer(
                    hidden_states=transformer_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timesteps,
                    image_rotary_emb=image_rotary_emb,
                    return_dict=False,
                )[0]

                model_pred = scheduler.get_velocity(model_output, noisy_model_input, timesteps)

                weights = 1 / (1 - alphas_cumprod[timesteps])
                while len(weights.shape) < len(model_pred.shape):
                    weights = weights.unsqueeze(-1)

                target = model_input

                # Compute loss (standard velocity matching)
                loss = torch.mean(
                    (weights * (model_pred - target) ** 2).reshape(batch_size, -1),
                    dim=1,
                )
                loss = loss.mean()
                
                accelerator.backward(loss)

                if accelerator.sync_gradients and accelerator.distributed_type != DistributedType.DEEPSPEED:
                    gradient_norm_before_clip = get_gradient_norm(transformer.parameters())
                    accelerator.clip_grad_norm_(transformer.parameters(), training_config.get("max_grad_norm", 1.0))
                    gradient_norm_after_clip = get_gradient_norm(transformer.parameters())
                    logs.update(
                        {
                            "gradient_norm_before_clip": gradient_norm_before_clip,
                            "gradient_norm_after_clip": gradient_norm_after_clip,
                        }
                    )

                if accelerator.state.deepspeed_plugin is None:
                    optimizer.step()
                    optimizer.zero_grad()

                if lr_scheduler is not None and not training_config.get("custom_settings", {}).get("use_cpu_offload_optimizer", False):
                    lr_scheduler.step()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                # Update dataset's global_step for conditional i2v training
                if hasattr(train_dataset, 'set_global_step'):
                    train_dataset.set_global_step(global_step)

                # Log metrics
                last_lr = lr_scheduler.get_last_lr()[0] if lr_scheduler is not None else training_config["learning_rate"]
                logs.update(
                    {
                        "loss": loss.detach().item(),
                        "lr": last_lr,
                        "epoch": current_epoch,
                        "step": global_step,
                    }
                )
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                # Save checkpoint
                checkpointing_steps = training_config.get("custom_settings", {}).get("checkpointing_steps", 500)
                validation_steps = data_config.get("validation_steps", 500)
                init_validation_steps = data_config.get("init_validation_steps", 100)
                max_validation_steps = data_config.get("max_validation_steps", validation_steps * 2)
                if global_step % checkpointing_steps == 0:
                    if accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED:
                        
                        # Check if we need to remove old checkpoints
                        if training_config.get("custom_settings", {}).get("checkpoints_total_limit") is not None:
                            checkpoints = os.listdir(experiment_config["output_dir"])
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            if len(checkpoints) >= training_config["custom_settings"]["checkpoints_total_limit"]:
                                num_to_remove = len(checkpoints) - training_config["custom_settings"]["checkpoints_total_limit"] + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"Removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(experiment_config["output_dir"], removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(experiment_config["output_dir"], f"checkpoint-{global_step}")
                        
                        # Save checkpoint with memory optimization
                        try:
                            logger.info(f"💾 Saving checkpoint to {save_path}")
                            # Use save_state for full checkpoint saving (save_model_hook handles LoRA vs full model logic)
                            accelerator.save_state(save_path)
                            logger.info(f"✅ Saved state to {save_path}")
                        except RuntimeError as e:
                            if "out of memory" in str(e).lower():
                                logger.warning(f"⚠️ OOM during checkpoint save, retrying with CPU offload...")
                                # Additional cleanup and retry
                                gc.collect()
                                torch.cuda.empty_cache()
                                torch.cuda.synchronize(accelerator.device)
                                
                                # Try saving again
                                accelerator.save_state(save_path)
                                logger.info(f"✅ Saved state to {save_path} (after retry)")
                            else:
                                raise e
                        
                        # Clean up after checkpoint save
                        gc.collect()
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize(accelerator.device)

                # Validation during training steps
                should_run_validation = data_config.get("validation_set") is not None and (
                    (global_step % init_validation_steps == 0 and global_step < checkpointing_steps) 
                    or
                    global_step % validation_steps == 0
                )
                should_run_max_validation = global_step % max_validation_steps == 0
                if should_run_validation:
                    logger.info(f"Running validation at step {global_step}")
                    run_validation(
                        config=config,
                        accelerator=accelerator,
                        transformer=transformer,
                        scheduler=scheduler,
                        model_config=transformer_config,
                        weight_dtype=weight_dtype,
                        step=global_step,
                        should_run_max_validation=should_run_max_validation
                    )


                # Check if we've reached max_train_steps
                if global_step >= max_train_steps:
                    break

    accelerator.wait_for_everyone()
    
    # Save final model
    if accelerator.is_main_process:
        transformer = unwrap_model(accelerator, transformer)
        dtype = (
            torch.float16
            if training_config.get("custom_settings", {}).get("mixed_precision") == "fp16"
            else torch.bfloat16
            if training_config.get("custom_settings", {}).get("mixed_precision") == "bf16"
            else torch.float32
        )
        transformer = transformer.to(dtype)
        
        # Save the entire pipeline for WAN
        if pipeline_config["type"] == "wan_fun_inpaint":
            pipeline = WanFunInpaintPipeline(
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                vae=vae,
                transformer=transformer,
                scheduler=scheduler,
            )
        elif pipeline_config["type"] == "wan_fun_inpaint_pose_concat":
            # Get condition_channels from pipeline config
            condition_channels = pipeline_config.get("condition_channels", 16)
            pipeline = WanFunInpaintPoseConcatPipeline(
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                vae=vae,
                transformer=transformer,
                scheduler=scheduler,
                condition_channels=condition_channels,
            )
        else:
            raise ValueError(f"Unsupported pipeline type: {pipeline_config['type']}")
        
        pipeline.save_pretrained(experiment_config["output_dir"])
        
        # Final validation if validation set is provided
        if data_config.get("validation_set") is not None:
            logger.info("Running final validation...")
            run_validation(
                config=config,
                accelerator=accelerator,
                transformer=transformer,
                scheduler=scheduler,
                model_config=transformer_config,
                weight_dtype=weight_dtype,
                step=global_step
            )
        
        # Cleanup trained models to save memory
        if training_config.get("custom_settings", {}).get("load_tensors", False):
            del transformer
        else:
            del transformer, text_encoder, vae

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(accelerator.device)
        
        logger.info("Training completed!")
    
    accelerator.end_training()


if __name__ == "__main__":
    main()
