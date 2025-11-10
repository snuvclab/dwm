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
import numpy as np
import os
import random
import shutil
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict
import argparse
import diffusers
import torch
import transformers
import imageio.v3 as iio
import wandb
from accelerate import Accelerator, DistributedType, init_empty_weights
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    set_seed,
)
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    # CogVideoXImageToVideoPipeline,
    # CogVideoXTransformer3DModel,
)
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from videox_fun.models import (AutoencoderKLCogVideoX,
                              CogVideoXTransformer3DModel, T5EncoderModel,
                              T5Tokenizer)
from videox_fun.pipeline import (CogVideoXFunPipeline,
                                CogVideoXFunInpaintPipeline)
from videox_fun.utils.utils import get_video_to_video_latent, save_videos_grid

from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params
from diffusers.utils import convert_unet_state_dict_to_peft, export_to_video, load_image
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from huggingface_hub import create_repo, upload_folder
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
# from transformers import AutoTokenizer, T5EncoderModel
# from training.cogvideox_static_pose.config_loader import load_experiment_config
from config_loader import load_experiment_config


from args import get_args  # isort:skip
# from dataset import BucketSampler, VideoDatasetWithResizing, VideoDatasetWithResizeAndRectangleCrop  # isort:skip
from dataset import (BucketSampler, VideoDatasetWithResizing, VideoDatasetWithResizeAndRectangleCrop, 
                    VideoDatasetWithConditions, VideoDatasetWithConditionsAndResizing, VideoDatasetWithConditionsAndResizeAndRectangleCrop, 
                    VideoDatasetWithHumanMotions, VideoDatasetWithHumanMotionsAndResizing, VideoDatasetWithHumanMotionsAndResizeAndRectangleCrop)
from text_encoder import compute_prompt_embeddings  # isort:skip
from utils import (
    get_gradient_norm,
    get_optimizer,
    prepare_rotary_positional_embeddings,
    print_memory,
    reset_memory,
    unwrap_model,
)


logger = get_logger(__name__)


def save_model_card(
    repo_id: str,
    videos=None,
    base_model: str = None,
    validation_prompt=None,
    repo_folder=None,
    fps=8,
):
    widget_dict = []
    if videos is not None:
        for i, video in enumerate(videos):
            export_to_video(video, os.path.join(repo_folder, f"final_video_{i}.mp4", fps=fps))
            widget_dict.append(
                {
                    "text": validation_prompt if validation_prompt else " ",
                    "output": {"url": f"video_{i}.mp4"},
                }
            )

    model_description = f"""
# CogVideoX Full Finetune

<Gallery />

## Model description

This is a full finetune of the CogVideoX model `{base_model}`.

## License

Please adhere to the licensing terms as described [here](https://huggingface.co/THUDM/CogVideoX-5b-I2V/blob/main/LICENSE).
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="other",
        base_model=base_model,
        prompt=validation_prompt,
        model_description=model_description,
        widget=widget_dict,
    )
    tags = [
        "text-to-video",
        "image-to-video",
        "diffusers-training",
        "diffusers",
        "cogvideox",
        "cogvideox-diffusers",
    ]

    model_card = populate_model_card(model_card, tags=tags)
    model_card.save(os.path.join(repo_folder, "README.md"))


# Copied from diffusers.pipelines.cogvideo.pipeline_cogvideox.CogVideoXPipeline._get_t5_prompt_embeds
def get_t5_prompt_embeds(
    prompt = None,
    num_videos_per_prompt = 1,
    max_sequence_length = 226,
    device = None,
    dtype = None,
    tokenizer = None,
    text_encoder = None,
):
    device = device or text_encoder.device
    dtype = dtype or text_encoder.dtype

    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
        removed_text = tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
        logger.warning(
            "The following part of your input was truncated because `max_sequence_length` is set to "
            f" {max_sequence_length} tokens: {removed_text}"
        )

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

    return prompt_embeds


def log_validation_with_dataset(
    accelerator: Accelerator,
    pipe,
    config: Dict[str, Any],
    validation_video_path: str,
    validation_prompt: str,
    validation_hand_video_path: str = None,
    validation_static_video_path: str = None,
    validation_human_motions_path: str = None,
    validation_hand_video_left_path: str = None,
    validation_hand_video_right_path: str = None,
    validation_smpl_pos_map_path: str = None,
    validation_raymap_path: str = None,
    is_final_validation: bool = False,
    step: int = 0,
    pipeline_type: str = None,
    validation_mode: str = None,
    apply_blur_to_static: bool = False,
    blur_strength: float = 0.2,
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

    # Regular mode: load single hand video
    if validation_hand_video_path and os.path.exists(validation_hand_video_path):
        hand_video = iio.imread(validation_hand_video_path).astype(np.float32) / 255.0
        hand_video = hand_video.transpose(3, 0, 1, 2)  # [F, H, W, C] -> [C, F, H, W]
        hand_video = hand_video[np.newaxis, :]  # Add batch dimension: [C, F, H, W] -> [1, C, F, H, W]
    else:
        hand_video = None

    static_video = iio.imread(validation_static_video_path).astype(np.float32) / 255.0
    static_video = static_video.transpose(3, 0, 1, 2)  # [F, H, W, C] -> [C, F, H, W]
    static_video = static_video[np.newaxis, :]  # Add batch dimension: [C, F, H, W] -> [1, C, F, H, W]

    # Generate video with conditions
    pipeline_args = {
        "prompt": validation_prompt,
        "guidance_scale": 6.0,  # Default guidance scale
        "use_dynamic_cfg": False,  # Disable dynamic cfg to match comparison script
        "height": 480,
        "width": 720,
        "num_frames": 49,
    }

    batch_size = 1
    num_frames = 49
    height = 480
    width = 720
    mask_video = torch.zeros((batch_size, 1, num_frames, height, width), dtype=torch.uint8)
    pipeline_args["mask_video"] = mask_video
    pipeline_args["video"] = static_video
    pipeline_args["hand_video"] = hand_video

    output = pipe(**pipeline_args, generator=generator)
    generated_video = output.videos[0]
    generated_video = generated_video.permute(1, 2, 3, 0)  # [C, F, H, W] -> [F, H, W, C]
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
        # Handle split_hands case: convert 6-channel to RGB by assigning to R, G channels
        if hand_video.shape[3] == 6:  # split_hands mode: 6 channels (left 3 + right 3)
            # Convert 6-channel to 3-channel RGB
            # Left hand (first 3 channels) -> R channel, Right hand (last 3 channels) -> G channel, B channel = 0
            hand_left = hand_video[:, :, :, 0:3]  # [F, H, W, 3] - left hand
            hand_right = hand_video[:, :, :, 3:6]  # [F, H, W, 3] - right hand
            
            # Convert to grayscale and assign to R, G channels
            hand_left_gray = np.mean(hand_left, axis=3, keepdims=True)  # [F, H, W, 1]
            hand_right_gray = np.mean(hand_right, axis=3, keepdims=True)  # [F, H, W, 1]
            hand_b = np.zeros_like(hand_left_gray)  # [F, H, W, 1] - zero channel
            
            # Concatenate to create RGB: [F, H, W, 3]
            hand_video_rgb = np.concatenate([hand_left_gray, hand_right_gray, hand_b], axis=3)
            video_components.append(hand_video_rgb)
        else:
            # Regular mode: use as is
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
    
    # Add blur suffix if blur was applied
    if apply_blur_to_static:
        mode_suffix += "_blurred"
    
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



def log_validation(
    accelerator: Accelerator,
    pipe: CogVideoXFunInpaintPipeline,
    args: Dict[str, Any],
    pipeline_args: Dict[str, Any],
    is_final_validation: bool = False,
):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_videos} videos with prompt: {pipeline_args['prompt']}."
    )

    pipe = pipe.to(accelerator.device)

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None

    videos = []
    for _ in range(args.num_validation_videos):
        video = pipe(**pipeline_args, generator=generator, output_type="np").frames[0]
        videos.append(video)

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "wandb":
            video_filenames = []
            for i, video in enumerate(videos):
                prompt = (
                    pipeline_args["prompt"][:25]
                    .replace(" ", "_")
                    .replace(" ", "_")
                    .replace("'", "_")
                    .replace('"', "_")
                    .replace("/", "_")
                )
                filename = os.path.join(args.output_dir, f"{phase_name}_video_{i}_{prompt}.mp4")
                export_to_video(video, filename, fps=8)
                video_filenames.append(filename)

            tracker.log(
                {
                    phase_name: [
                        wandb.Video(filename, caption=f"{i}: {pipeline_args['prompt']}")
                        for i, filename in enumerate(video_filenames)
                    ]
                }
            )

    return videos


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
    accelerator.print("===== Memory before validation =====")
    print_memory(accelerator.device)
    torch.cuda.synchronize(accelerator.device)

    # Setup pipeline for validation
    pipeline_config = config["pipeline"]
    model_config_dict = config["model"]
    data_config = config.get("data", {})

    pipe = CogVideoXFunInpaintPipeline.from_pretrained(
        model_config_dict["base_model_name_or_path"],
        transformer=unwrap_model(accelerator, transformer),
        scheduler=scheduler,
        revision=model_config_dict.get("revision"),
        variant=model_config_dict.get("variant"),
        torch_dtype=weight_dtype,
    )

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
    if data_config.get("validation_set") is not None:
        validation_set_path = os.path.join(data_config["data_root"], data_config["validation_set"])
        with open(validation_set_path, "r") as f:
            validation_set = f.readlines()
        validation_set = [video.strip() for video in validation_set]
        
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
        hand_video_subdir = data_config.get("hand_video_subdir", "videos_hands")

        for video_path in validation_set:
            # Convert video path to prompt path
            video_path_obj = Path(video_path)
            prompt_path = video_path_obj.parent.parent / prompt_subdir / f"{video_path_obj.stem}.txt"
            validation_prompts.append(Path(data_config["data_root"]) / prompt_path)
        
        # Derive hand and static video paths from main video paths
        validation_hand_videos = []
        validation_static_videos = []

        for video_path in validation_set:
            # Convert video path to hand and static video paths
            video_path_obj = Path(video_path)
            
            # Regular mode: use single hand video path
            if data_config.get("use_gray_hand_videos", False):
                hand_path = video_path_obj.parent.parent / "videos_hands_gray" / video_path_obj.name
            else:
                # Use configurable hand_video_subdir for default mode
                hand_path = video_path_obj.parent.parent / hand_video_subdir / video_path_obj.name
            validation_hand_videos.append(Path(data_config["data_root"]) / hand_path)
            
            static_path = video_path_obj.parent.parent / "videos_static" / video_path_obj.name
            validation_static_videos.append(Path(data_config["data_root"]) / static_path)
        
        # Run validation for each video
        for validation_video, validation_prompt, validation_hand_video, validation_static_video in zip(
            validation_videos, validation_prompts, validation_hand_videos, validation_static_videos
        ):
            # Load prompt
            with open(validation_prompt, "r") as f:
                prompt_text = f.read().strip()
            
            log_validation_with_dataset(
                    pipe=pipe,
                    config=config,
                    accelerator=accelerator,
                    validation_video_path=validation_video,
                    validation_prompt=prompt_text,
                    validation_hand_video_path=validation_hand_video,
                    validation_static_video_path=validation_static_video,
                    validation_human_motions_path=None,
                    validation_hand_video_left_path=None,
                    validation_hand_video_right_path=None,
                    step=step,
                    pipeline_type=config["pipeline"]["type"],
                    validation_mode="static_to_video",
                    apply_blur_to_static=False,
                )
            
    accelerator.print("===== Memory after validation =====")
    print_memory(accelerator.device)
    reset_memory(accelerator.device)

    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(accelerator.device)


class CollateFunction:
    def __init__(self, weight_dtype: torch.dtype, load_tensors: bool) -> None:
        self.weight_dtype = weight_dtype
        self.load_tensors = load_tensors

    def __call__(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        prompts = [x["prompt"] for x in data[0]]

        if self.load_tensors:
            prompts = torch.stack(prompts).to(dtype=self.weight_dtype, non_blocking=True)

        images = [x["image"] for x in data[0]]
        images = torch.stack(images).to(dtype=self.weight_dtype, non_blocking=True)

        videos = [x["video"] for x in data[0]]
        videos = torch.stack(videos).to(dtype=self.weight_dtype, non_blocking=True)

        return {
            "images": images,
            "videos": videos,
            "prompts": prompts,
        }


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

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
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

    # # Handle the repository creation
    # if accelerator.is_main_process:
    #     if args.output_dir is not None:
    #         os.makedirs(args.output_dir, exist_ok=True)

    #     if args.push_to_hub:
    #         repo_id = create_repo(
    #             repo_id=args.hub_model_id or Path(args.output_dir).name,
    #             exist_ok=True,
    #         ).repo_id

    # Prepare models and scheduler
    model_config = config["model"]
    model_path = model_config["base_model_name_or_path"]
    load_dtype = torch.bfloat16 if "5b" in model_path.lower() else torch.float16

    tokenizer = T5Tokenizer.from_pretrained(
        model_path,
        subfolder="tokenizer",
        revision=model_config.get("revision"),
    )

    text_encoder = T5EncoderModel.from_pretrained(
        model_path,
        subfolder="text_encoder",
        revision=model_config.get("revision"),
    )

    # CogVideoX-2b weights are stored in float16
    # CogVideoX-5b and CogVideoX-5b-I2V weights are stored in bfloat16
    load_dtype = torch.bfloat16 if "5b" in model_path.lower() else torch.float16
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        model_path,
        subfolder="transformer",
        torch_dtype=load_dtype,
        # revision=model_config.get("revision"),
        # variant=model_config.get("variant"),
    )

    if model_config.get("ignore_learned_positional_embeddings"):
        del transformer.patch_embed.pos_embedding
        transformer.patch_embed.use_learned_positional_embeddings = False
        transformer.config.use_learned_positional_embeddings = False

    vae = AutoencoderKLCogVideoX.from_pretrained(
        model_path,
        subfolder="vae",
        # revision=model_config.get("revision"),
        # variant=model_config.get("variant"),
    )

    scheduler = CogVideoXDPMScheduler.from_pretrained(model_path, subfolder="scheduler")

    # Enable memory optimizations
    training_config = config["training"]
    if training_config.get("custom_settings", {}).get("enable_slicing", False):
        vae.enable_slicing()
    if training_config.get("custom_settings", {}).get("enable_tiling", False):
        vae.enable_tiling()

    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    transformer.requires_grad_(False)

    VAE_SCALING_FACTOR = vae.config.scaling_factor
    VAE_SCALE_FACTOR_SPATIAL = 2 ** (len(vae.config.block_out_channels) - 1)
    RoPE_BASE_HEIGHT = transformer.config.sample_height * VAE_SCALE_FACTOR_SPATIAL
    RoPE_BASE_WIDTH = transformer.config.sample_width * VAE_SCALE_FACTOR_SPATIAL

    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
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

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    text_encoder.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    if config["training"].get("custom_settings", {}).get("gradient_checkpointing", False):
        transformer.enable_gradient_checkpointing()
        print("✅ Gradient checkpointing enabled")

    # now we will add new LoRA weights to the attention layers
    transformer_lora_config = LoraConfig(
        r=config["training"].get("lora_rank", 64),
        lora_alpha=config["training"].get("lora_alpha", 64),
        init_lora_weights=True,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    transformer.add_adapter(transformer_lora_config)

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None
            projection_state_dict = None

            for model in models:
                if isinstance(unwrap_model(accelerator, model), type(unwrap_model(accelerator, transformer))):
                    model = unwrap_model(accelerator, model)
                    transformer_lora_layers_to_save = get_peft_model_state_dict(model)

                    # Concat models: save proj weights (modified existing proj)
                    if hasattr(model, 'patch_embed') and hasattr(model.patch_embed, 'proj'):
                        projection_state_dict = {
                            "transformer.patch_embed.proj.weight": model.patch_embed.proj.weight.data,
                            "transformer.patch_embed.proj.bias": model.patch_embed.proj.bias.data if model.patch_embed.proj.bias is not None else None,
                        }
                else:
                    raise ValueError(f"Unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                if weights:
                    weights.pop()

            CogVideoXFunInpaintPipeline.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save,
            )
            if projection_state_dict is not None:               
                torch.save(projection_state_dict, os.path.join(output_dir, "projection_layer_weights.pt"))
                print(f"✅ Saved concat projection weights: {list(projection_state_dict.keys())}")
                
    def load_model_hook(models, input_dir):
        transformer_ = None

        # This is a bit of a hack but I don't know any other solution.
        if not accelerator.distributed_type == DistributedType.DEEPSPEED:
            while len(models) > 0:
                model = models.pop()

                if isinstance(unwrap_model(accelerator, model), type(unwrap_model(accelerator, transformer))):
                    transformer_ = unwrap_model(accelerator, model)
                else:
                    raise ValueError(f"Unexpected save model: {unwrap_model(accelerator, model).__class__}")
        else:
            transformer_ = CogVideoXTransformer3DModel.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="transformer"
            )
            transformer_.add_adapter(transformer_lora_config)

        lora_state_dict = CogVideoXFunInpaintPipeline.lora_state_dict(input_dir)

        transformer_state_dict = {
            f'{k.replace("transformer.", "")}': v for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params([transformer_])


    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Modify Image projection Conv2D layer of transformer to take additional hand condition inputs
    with torch.no_grad():
        new_conv_in = torch.nn.Conv2d( # 16 for hand video condition latents
            transformer.patch_embed.proj.in_channels + 16, transformer.patch_embed.proj.out_channels, \
                transformer.patch_embed.proj.kernel_size, transformer.patch_embed.proj.stride, transformer.patch_embed.proj.padding
        )
        new_conv_in.weight.zero_()
        new_conv_in.weight[:, :transformer.patch_embed.proj.in_channels, :, :].copy_(transformer.patch_embed.proj.weight)
        new_conv_in.bias.zero_()
        new_conv_in.bias.copy_(transformer.patch_embed.proj.bias)
        transformer.patch_embed.proj = new_conv_in
    transformer.patch_embed.proj.weight.requires_grad_(True) # already enabled but just to be sure

    # Enable TF32 for faster training on Ampere GPUs
    if training_config.get("custom_settings", {}).get("allow_tf32", False) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    # Scale learning rate if specified
    training_config["learning_rate"] = float(training_config["learning_rate"])
    if training_config.get("scale_lr", False):
        training_config["learning_rate"] = (
            training_config["learning_rate"] * training_config["gradient_accumulation_steps"] * 
            training_config["batch_size"] * accelerator.num_processes
        )

    # Make sure the trainable params are in float32 for mixed precision
    if training_config.get("custom_settings", {}).get("mixed_precision") == "fp16":
        cast_training_params([transformer], dtype=torch.float32)

    transformer_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    # Optimization parameters
    transformer_parameters_with_lr = {
        "params": transformer_parameters,
        "lr": training_config["learning_rate"],
    }
    params_to_optimize = [transformer_parameters_with_lr]
    num_trainable_parameters = sum(param.numel() for model in params_to_optimize for param in model["params"])

    use_deepspeed_optimizer = (
        accelerator.state.deepspeed_plugin is not None
        and "optimizer" in accelerator.state.deepspeed_plugin.deepspeed_config
    )
    use_deepspeed_scheduler = (
        accelerator.state.deepspeed_plugin is not None
        and "scheduler" in accelerator.state.deepspeed_plugin.deepspeed_config
    )

    
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
        "image_to_video": data_config.get("image_to_video", False),
        "use_gray_hand_videos": data_config.get("use_gray_hand_videos", False),
        "split_hands": data_config.get("split_hands", False),
        "use_smpl_pos_map": data_config.get("use_smpl_pos_map", False),
        "compress_smpl_pos_map_temporal": data_config.get("compress_smpl_pos_map_temporal", False),
        "vae_scale_factor_temporal": data_config.get("vae_scale_factor_temporal", 4),
        "vae_scale_factor_spatial": data_config.get("vae_scale_factor_spatial", 8),
        "load_raymaps": data_config.get("load_raymaps", False),
        "load_image_goal": data_config.get("load_image_goal", False),
        "prompt_subdir": data_config.get("prompt_subdir", "prompts"),
        "prompt_embeds_subdir": data_config.get("prompt_embeds_subdir", "prompt_embeds"),
        "hand_video_subdir": data_config.get("hand_video_subdir", "videos_hands"),
        "hand_video_latents_subdir": data_config.get("hand_video_latents_subdir", "hand_video_latents"),
    }
    # Choose dataset class based on pipeline type
    if pipeline_config["type"] == "cogvideox_i2v":
        # For basic I2V pipeline, use standard dataset without pose conditioning
        train_dataset = VideoDatasetWithResizing(**dataset_init_kwargs)
        print("🔧 Using VideoDatasetWithResizing for basic I2V training")
    elif pipeline_config["type"] in ["cogvideox_pose_adaln", "cogvideox_pose_adaln_perframe"]:
        train_dataset = VideoDatasetWithHumanMotionsAndResizing(**dataset_init_kwargs)
        print("🔧 Using VideoDatasetWithHumanMotionsAndResizing for AdaLN pose training")
    else:
        train_dataset = VideoDatasetWithConditionsAndResizing(**dataset_init_kwargs)
        print("🔧 Using VideoDatasetWithConditionsAndResizing for concat/adapter training")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_config["batch_size"],
        shuffle=True,
        collate_fn=lambda batch: {
            "videos": torch.stack([item["video"] for item in batch]),
            "prompts": (
                torch.stack([item["prompt"] for item in batch])
                if isinstance(batch[0]["prompt"], torch.Tensor)
                else [item["prompt"] for item in batch]
            ),
            "images": torch.stack([item["image"] for item in batch]) if "image" in batch[0] and batch[0]["image"] is not None else None,
            "hand_videos": torch.stack([item["hand_videos"] for item in batch]) if "hand_videos" in batch[0] and batch[0]["hand_videos"] is not None else None,
            "static_videos": torch.stack([item["static_videos"] for item in batch]) if "static_videos" in batch[0] and batch[0]["static_videos"] is not None else None,
            "smpl_pos_map": torch.stack([item["smpl_pos_map"] for item in batch]) if "smpl_pos_map" in batch[0] and batch[0]["smpl_pos_map"] is not None else None,
            "human_motions": torch.stack([item["human_motions"] for item in batch]) if "human_motions" in batch[0] and batch[0]["human_motions"] is not None else None,
            "raymaps": torch.stack([item["raymap"] for item in batch]) if "raymap" in batch[0] and batch[0]["raymap"] is not None else None,
            "image_goal": torch.stack([item["image_goal"] for item in batch]) if "image_goal" in batch[0] and batch[0]["image_goal"] is not None else None,
        },
        num_workers=data_config.get("dataloader_num_workers", 0),
        pin_memory=data_config.get("pin_memory", True),
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_config["gradient_accumulation_steps"])
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


    # Prepare everything with our `accelerator`.
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    # num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    # if overrode_max_train_steps:
    #     args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # # Afterwards we recalculate our number of training epochs
    # args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
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
                "condition_blur_prob": training_config.get("condition_blur_prob", 0.0),
                "condition_blur_strength": training_config.get("condition_blur_strength", 0.2),
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

    # Train!
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

    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    
    EMPTY_PROMPT_EMBED = get_t5_prompt_embeds(
        prompt="",
        tokenizer=tokenizer,
        text_encoder=text_encoder,
    )

    # For DeepSpeed training
    model_config = transformer.module.config if hasattr(transformer, "module") else transformer.config

    load_tensors = training_config.get("custom_settings", {}).get("load_tensors", False)
    if load_tensors:
        del vae, text_encoder
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(accelerator.device)

    alphas_cumprod = scheduler.alphas_cumprod.to(accelerator.device, dtype=torch.float32)

    for epoch in range(first_epoch, num_epochs):
        transformer.train()
        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer]
            logs = {}

            with accelerator.accumulate(models_to_accumulate):
                videos = batch["videos"].to(accelerator.device, non_blocking=True)
                images = batch.get("images")  # For I2V pipeline and prediction mode
                prompts = batch["prompts"]
                hand_videos = batch.get("hand_videos")
                static_videos = batch.get("static_videos")

                prompts = EMPTY_PROMPT_EMBED.repeat(prompts.shape[0], 1, 1).to(prompts)

                # Encode videos
                if not load_tensors:
                    images = images.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
                    image_noise_sigma = torch.normal(
                        mean=-3.0, std=0.5, size=(images.size(0),), device=accelerator.device, dtype=weight_dtype
                    )
                    image_noise_sigma = torch.exp(image_noise_sigma)
                    noisy_images = images + torch.randn_like(images) * image_noise_sigma[:, None, None, None, None]
                    image_latent_dist = vae.encode(noisy_images).latent_dist

                    videos = videos.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
                    latent_dist = vae.encode(videos).latent_dist
                else:
                    # image_latent_dist = DiagonalGaussianDistribution(images)
                    latent_dist = DiagonalGaussianDistribution(videos)
                    hand_latent_dist = DiagonalGaussianDistribution(hand_videos) if hand_videos is not None else None
                    static_latent_dist = DiagonalGaussianDistribution(static_videos) if static_videos is not None else None

                # image_latents = image_latent_dist.sample() * VAE_SCALING_FACTOR
                # image_latents = image_latents.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
                # image_latents = image_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)

                video_latents = latent_dist.sample() * VAE_SCALING_FACTOR
                video_latents = video_latents.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
                video_latents = video_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)

                hand_latents = hand_latent_dist.sample() * VAE_SCALING_FACTOR if hand_latent_dist is not None else None
                if hand_latents is not None:
                    hand_latents = hand_latents.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
                    hand_latents = hand_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)

                static_latents = static_latent_dist.sample() * VAE_SCALING_FACTOR if static_latent_dist is not None else None
                if static_latents is not None:
                    static_latents = static_latents.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
                    static_latents = static_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)

                # # for latent debugging: decode and save video
                # latents = video_latents.permute(0, 2, 1, 3, 4)  # [batch_size, num_channels, num_frames, height, width]
                # latents = 1 / VAE_SCALING_FACTOR * latents
                # with torch.no_grad():
                #     vae = AutoencoderKLCogVideoX.from_pretrained(
                #         model_path,
                #         subfolder="vae",
                #         # revision=model_config.get("revision"),
                #         # variant=model_config.get("variant"),
                #     ).to(accelerator.device, dtype=weight_dtype)
                #     frames = vae.decode(latents).sample
                # frames = (frames / 2 + 0.5).clamp(0, 1)
                # # Return PyTorch tensor instead of numpy array for video_processor compatibility
                # frames = frames.cpu().float()
                # export_to_video(frames.numpy()[0].transpose(1, 2, 3, 0), "output.mp4")

                # if random.random() < args.noised_image_dropout:
                #     image_latents = torch.zeros_like(image_latents)

                mask_latents = torch.ones_like(video_latents)[:, :, :1].to(video_latents.device, video_latents.dtype) * VAE_SCALING_FACTOR
                masked_video_latents = static_latents

                inpaint_latents = torch.cat([mask_latents, masked_video_latents, hand_latents], dim=2).to(video_latents.dtype)

                # Encode prompts
                if not load_tensors:
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

                # Sample noise that will be added to the latents
                noise = torch.randn_like(video_latents)
                batch_size, num_frames, num_channels, height, width = video_latents.shape

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    scheduler.config.num_train_timesteps,
                    (batch_size,),
                    dtype=torch.int64,
                    device=accelerator.device,
                )

                # Prepare rotary embeds
                image_rotary_emb = (
                    prepare_rotary_positional_embeddings(
                        height=height * VAE_SCALE_FACTOR_SPATIAL,
                        width=width * VAE_SCALE_FACTOR_SPATIAL,
                        num_frames=num_frames,
                        vae_scale_factor_spatial=VAE_SCALE_FACTOR_SPATIAL,
                        patch_size=model_config.patch_size,
                        patch_size_t=model_config.patch_size_t if hasattr(model_config, "patch_size_t") else None,
                        attention_head_dim=model_config.attention_head_dim,
                        device=accelerator.device,
                        base_height=RoPE_BASE_HEIGHT,
                        base_width=RoPE_BASE_WIDTH,
                    )
                    if model_config.use_rotary_positional_embeddings
                    else None
                )

                # Add noise to the model input according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_video_latents = scheduler.add_noise(video_latents, noise, timesteps)
                noisy_model_input = noisy_video_latents
                model_config.patch_size_t if hasattr(model_config, "patch_size_t") else None,
                ofs_embed_dim = model_config.ofs_embed_dim if hasattr(model_config, "ofs_embed_dim") else None,
                ofs_emb = None if ofs_embed_dim is None else noisy_model_input.new_full((1,), fill_value=2.0)
                # Predict the noise residual
                model_output = transformer(
                    hidden_states=noisy_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timesteps,
                    image_rotary_emb=image_rotary_emb,
                    return_dict=False,
                    inpaint_latents=inpaint_latents,
                )[0]

                model_pred = scheduler.get_velocity(model_output, noisy_video_latents, timesteps)

                weights = 1 / (1 - alphas_cumprod[timesteps])
                while len(weights.shape) < len(model_pred.shape):
                    weights = weights.unsqueeze(-1)

                target = video_latents

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

                # Log metrics
                last_lr = lr_scheduler.get_last_lr()[0] if lr_scheduler is not None else training_config["learning_rate"]
                logs.update(
                    {
                        "loss": loss.detach().item(),
                        "lr": last_lr,
                        "epoch": epoch,
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
                if (global_step) % checkpointing_steps == 0:
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

                            # # save conv2d weights
                            # torch.save(transformer.patch_embed.proj.state_dict(), os.path.join(save_path, f"projection_layer_weights.pt"))
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
                    (global_step) % validation_steps == 0
                )
                should_run_max_validation = global_step % max_validation_steps == 0
                if should_run_validation:
                    logger.info(f"Running validation at step {global_step}")
                    run_validation(
                        config=config,
                        accelerator=accelerator,
                        transformer=transformer,
                        scheduler=scheduler,
                        model_config=model_config,
                        weight_dtype=weight_dtype,
                        step=global_step,
                        should_run_max_validation=should_run_max_validation
                    )


                # Check if we've reached max_train_steps
                if global_step >= max_train_steps:
                    break
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        transformer = unwrap_model(accelerator, transformer)
        dtype = (
            torch.float16
            if args.mixed_precision == "fp16"
            else torch.bfloat16
            if args.mixed_precision == "bf16"
            else torch.float32
        )
        transformer = transformer.to(dtype)

        transformer.save_pretrained(
            os.path.join(args.output_dir, "transformer"),
            safe_serialization=True,
            max_shard_size="5GB",
        )

        # Cleanup trained models to save memory
        if args.load_tensors:
            del transformer
        else:
            del transformer, text_encoder, vae

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(accelerator.device)

        accelerator.print("===== Memory before testing =====")
        print_memory(accelerator.device)
        reset_memory(accelerator.device)

        # Final test inference
        pipe = CogVideoXFunInpaintPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype,
        )
        pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config)

        if args.enable_slicing:
            pipe.vae.enable_slicing()
        if args.enable_tiling:
            pipe.vae.enable_tiling()
        if args.enable_model_cpu_offload:
            pipe.enable_model_cpu_offload()

        # Run inference
        validation_outputs = []
        if args.validation_prompt and args.num_validation_videos > 0:
            validation_prompts = args.validation_prompt.split(args.validation_prompt_separator)
            validation_images = args.validation_images.split(args.validation_prompt_separator)
            for validation_image, validation_prompt in zip(validation_images, validation_prompts):
                pipeline_args = {
                    "image": load_image(validation_image),
                    "prompt": validation_prompt,
                    "guidance_scale": args.guidance_scale,
                    "use_dynamic_cfg": args.use_dynamic_cfg,
                    "height": args.height,
                    "width": args.width,
                }

                video = log_validation(
                    accelerator=accelerator,
                    pipe=pipe,
                    args=args,
                    pipeline_args=pipeline_args,
                    is_final_validation=True,
                )
                validation_outputs.extend(video)

        accelerator.print("===== Memory after testing =====")
        print_memory(accelerator.device)
        reset_memory(accelerator.device)
        torch.cuda.synchronize(accelerator.device)

        if args.push_to_hub:
            save_model_card(
                repo_id,
                videos=validation_outputs,
                base_model=args.pretrained_model_name_or_path,
                validation_prompt=args.validation_prompt,
                repo_folder=args.output_dir,
                fps=args.fps,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    # args = get_args()
    main()