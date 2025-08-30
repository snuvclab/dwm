#!/usr/bin/env python3

import argparse
import functools
import json
import os
import pathlib
import queue
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Union
import glob
from natsort import natsorted

import torch
import torch.distributed as dist
from diffusers import AutoencoderKLCogVideoX
from diffusers.training_utils import set_seed
from diffusers.utils import export_to_video, get_logger
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer

import decord  # isort:skip
decord.bridge.set_bridge("torch")

logger = get_logger(__name__)

DTYPE_MAPPING = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def generate_trumans_filename(scene_name: str, action_name: str, video_name: str, extension: str) -> str:
    """Generate filename following Trumans naming convention:
    {first 8 chars of scene}_{action name}_{video name}.{ext}
    """
    # Extract first 8 characters of scene name
    scene_prefix = scene_name[:8]
    
    # Clean action name (replace @ with _ for filesystem compatibility)
    clean_action_name = action_name.replace("@", "_")
    
    # Generate filename
    filename = f"{scene_prefix}_{clean_action_name}_{video_name}.{extension}"
    return filename


def check_height(x: Any) -> int:
    x = int(x)
    if x % 16 != 0:
        raise argparse.ArgumentTypeError(
            f"`--height_buckets` must be divisible by 16, but got {x} which does not fit criteria."
        )
    return x


def check_width(x: Any) -> int:
    x = int(x)
    if x % 16 != 0:
        raise argparse.ArgumentTypeError(
            f"`--width_buckets` must be divisible by 16, but got {x} which does not fit criteria."
        )
    return x


def check_frames(x: Any) -> int:
    x = int(x)
    if x % 4 != 0 and x % 4 != 1:
        raise argparse.ArgumentTypeError(
            f"`--frames_buckets` must be of form `4 * k` or `4 * k + 1`, but got {x} which does not fit criteria."
        )
    return x


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Trumans dataset with batch processing for multiple scenes/actions")
    parser.add_argument(
        "--model_id",
        type=str,
        default="THUDM/CogVideoX-2b",
        help="Hugging Face model ID to use for tokenizer, text encoder and VAE.",
    )
    parser.add_argument("--data_root", type=str, required=True, help="Path to base directory containing scene folders.")
    parser.add_argument(
        "--caption_column",
        type=str,
        default="prompts.txt",
        help="Name of the file containing line-separated captions in each action sequence directory.",
    )
    parser.add_argument(
        "--video_column",
        type=str,
        default="videos.txt",
        help="Name of the file containing line-separated video paths in each action sequence directory.",
    )

    parser.add_argument(
        "--id_token",
        type=str,
        default=None,
        help="Identifier token appended to the start of each prompt if provided.",
    )
    parser.add_argument(
        "--height_buckets",
        nargs="+",
        type=check_height,
        default=[256, 320, 384, 480, 512, 576, 720, 768, 960, 1024, 1280, 1536],
    )
    parser.add_argument(
        "--width_buckets",
        nargs="+",
        type=check_width,
        default=[256, 320, 384, 480, 512, 576, 720, 768, 960, 1024, 1280, 1536],
    )
    parser.add_argument(
        "--frame_buckets",
        nargs="+",
        type=check_frames,
        default=[49],
    )
    parser.add_argument(
        "--random_flip",
        type=float,
        default=None,
        help="If random horizontal flip augmentation is to be used, this should be the flip probability.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        help="Whether or not to use the pinned memory setting in pytorch dataloader.",
    )
    parser.add_argument(
        "--video_reshape_mode",
        type=str,
        default=None,
        help="All input videos are reshaped to this mode. Choose between ['center', 'random', 'none']",
    )
    parser.add_argument(
        "--save_image_latents",
        action="store_true",
        help="Whether or not to encode and store image latents, which are required for image-to-video finetuning. The image latents are the first frame of input videos encoded with the VAE.",
    )
    parser.add_argument("--max_num_frames", type=int, default=49, help="Maximum number of frames in output video.")
    parser.add_argument(
        "--max_sequence_length", type=int, default=226, help="Max sequence length of prompt embeddings."
    )
    parser.add_argument("--target_fps", type=int, default=8, help="Frame rate of output videos.")
    parser.add_argument(
        "--save_latents_and_embeddings",
        action="store_true",
        help="Whether to encode videos/captions to latents/embeddings and save them in pytorch serializable format.",
    )
    parser.add_argument(
        "--use_slicing",
        action="store_true",
        help="Whether to enable sliced encoding/decoding in the VAE. Only used if `--save_latents_and_embeddings` is also used.",
    )
    parser.add_argument(
        "--use_tiling",
        action="store_true",
        help="Whether to enable tiled encoding/decoding in the VAE. Only used if `--save_latents_and_embeddings` is also used.",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Number of videos to process at once in the VAE.")
    parser.add_argument(
        "--num_decode_threads",
        type=int,
        default=0,
        help="Number of decoding threads for `decord` to use. The default `0` means to automatically determine required number of threads.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["fp32", "fp16", "bf16"],
        default="fp32",
        help="Data type to use when generating latents and prompt embeddings.",
    )
    parser.add_argument(
        "--disparity_format",
        type=str,
        choices=["npy", "npz", "video"],
        default="npz",
        help="Format for disparity data: 'npy' for NPY files, 'npz' for compressed NPZ files, 'video' for MP4 video files.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility.")
    parser.add_argument(
        "--num_artifact_workers", type=int, default=4, help="Number of worker threads for serializing artifacts."
    )
    parser.add_argument(
        "--sequences_dir",
        type=str,
        default="sequences",
        help="Name of the sequences directory under each action directory.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip processing if output files already exist.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["aether", "cogvideox_pose", "custom"],
        default="custom",
        help="Predefined file type combinations for different training models. 'aether' includes all file types for Aether training. 'cogvideox_pose' includes file types for CogVideoX pose training. 'custom' allows manual specification via --check_file_types.",
    )
    parser.add_argument(
        "--check_file_types",
        nargs="+",
        type=str,
        default=None,
        help="Only check these specific file types when using --skip_existing. If --model_type is not 'custom', this argument is ignored. Options: videos, images, images_goal, image_latents, image_goal_latents, video_latents, disparity, disparity_latents, raymaps, raymaps_abs, prompts, prompt_embeds, human_motions, hand_videos, hand_video_latents, hand_mask_videos, static_videos, static_video_latents",
    )
    parser.add_argument(
        "--scene_filter",
        nargs="+",
        type=str,
        default=None,
        help="Only process scenes that match these filter patterns. If not specified, all scenes are processed.",
    )
    parser.add_argument(
        "--save_prompt_embeds",
        action="store_true",
        help="Whether to save prompt embeddings. This is automatically enabled for cogvideox_pose model type.",
    )
    return parser.parse_args()


def get_model_file_types(model_type: str) -> List[str]:
    """Get predefined file types for different training models."""
    if model_type == "aether":
        return [
            "videos", "video_latents", "disparity", "disparity_latents",
            "image_goal_latents", "image_latents", "images", "images_goal",
            "prompts", "prompt_embeds", "raymaps", "raymaps_abs", "human_motions"
        ]
    elif model_type == "cogvideox_pose":
        return [
            "videos", "video_latents", "prompts", "prompt_embeds",
            "hand_videos", "hand_video_latents", "hand_mask_videos", "static_videos", "static_video_latents"
        ]
    else:  # custom
        return []


def get_expected_file_count(sequences_path: pathlib.Path) -> int:
    """Get the expected number of files for a scene-action pair by reading videos.txt."""
    videos_file = sequences_path / "videos.txt"
    if videos_file.exists():
        with open(videos_file, 'r') as f:
            return len(f.readlines())
    return 0


def find_scene_action_pairs(data_root: pathlib.Path, sequences_dir: str, scene_filter: Optional[List[str]] = None, rank: int = 0) -> List[tuple]:
    """Find all scene-action pairs that have valid sequence data."""
    scene_action_pairs = []
    
    for scene_dir in natsorted(data_root.iterdir(), key=lambda x: x.name):
        if not scene_dir.is_dir():
            continue
            
        scene_name = scene_dir.name
        
        # Apply scene filter if specified
        if scene_filter is not None:
            scene_matches = False
            for filter_pattern in scene_filter:
                if filter_pattern in scene_name:
                    scene_matches = True
                    break
            if not scene_matches:
                if rank == 0:
                    print(f"Skipping scene {scene_name} (does not match filter patterns: {scene_filter})")
                continue
        
        for action_dir in natsorted(scene_dir.iterdir(), key=lambda x: x.name):
            if not action_dir.is_dir():
                continue
                
            action_name = action_dir.name
            sequences_path = action_dir / sequences_dir
            
            # Check if sequences directory exists and has required files
            if sequences_path.exists() and sequences_path.is_dir():
                # Check for videos.txt and prompts.txt
                videos_file = sequences_path / "videos.txt"
                prompts_file = sequences_path / "prompts.txt"
                
                if videos_file.exists() and prompts_file.exists():
                    scene_action_pairs.append((scene_name, action_name, sequences_path))
                    if rank == 0:  # Only print on rank 0 to avoid duplicate messages
                        print(f"Found valid sequence: {scene_name}/{action_name}")
    
    return scene_action_pairs

def process_single_scene_action(
    scene_name: str, 
    action_name: str, 
    sequences_path: pathlib.Path,
    processed_dir: pathlib.Path,
    rank: int,
    args: argparse.Namespace,
    missing_file_types: Optional[List[str]] = None,
    vae: Optional[AutoencoderKLCogVideoX] = None,
    tokenizer: Optional[T5Tokenizer] = None,
    text_encoder: Optional[T5EncoderModel] = None,
    device: Optional[torch.device] = None,
    weight_dtype: Optional[torch.dtype] = None
):
    """Process a single scene-action pair using the provided models."""
    
    # Get actual video count for this action
    videos_file = sequences_path / "videos.txt"
    total_videos = None
    if videos_file.exists():
        with open(videos_file, 'r') as f:
            total_videos = len(f.readlines())
    
    print(f"\n{'='*60}")
    print(f"GPU {rank} - Processing: {scene_name}/{action_name}")
    print(f"Action path: {sequences_path}")
    if total_videos is not None:
        print(f"Expected videos: 00000 to {total_videos-1:05d} ({total_videos} total)")
    if missing_file_types:
        print(f"Selective processing: Only generating missing file types: {missing_file_types}")
    else:
        print(f"Full processing: Generating all file types")
    print(f"{'='*60}")
    
    # Create processed directory for this action based on model_type
    if args.model_type == "aether":
        # Aether uses 'processed' directory
        processed_dir = sequences_path.parent / "processed"
    else:
        # cogvideox_pose and custom use 'processed2' directory
        processed_dir = sequences_path.parent / "processed2"
    
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Build command to run the appropriate script based on model_type

    script_path = "data_processing/trumans/prepare_dataset_trumans.py"  # Default to unified script
    
    cmd_parts = [
        "python", script_path,
        "--model_id", args.model_id,
        "--data_root", str(sequences_path),
        "--output_dir", str(processed_dir),
        "--caption_column", args.caption_column,
        "--video_column", args.video_column,
        "--model_type", args.model_type,
        "--height_buckets"] + [str(h) for h in args.height_buckets] + [
        "--width_buckets"] + [str(w) for w in args.width_buckets] + [
        "--frame_buckets"] + [str(f) for f in args.frame_buckets] + [
        "--max_num_frames", str(args.max_num_frames),
        "--max_sequence_length", str(args.max_sequence_length),
        "--target_fps", str(args.target_fps),
        "--batch_size", str(args.batch_size),
        "--dtype", args.dtype,
        "--disparity_format", args.disparity_format,
        "--seed", str(args.seed),
        "--num_artifact_workers", str(args.num_artifact_workers),
        "--scene_name", scene_name,
        "--action_name", action_name,
    ]
    
    # Add skip_existing flag if provided
    if hasattr(args, 'skip_existing') and args.skip_existing:
        cmd_parts.append("--skip_existing")
    
            # Add selective processing flags based on missing file types
        if missing_file_types:
            # Map file types to their corresponding flags
            file_type_to_flag = {
                "images": "--save_image_latents",
                "image_latents": "--save_image_latents",
                "images_goal": "--save_image_latents", 
                "image_goal_latents": "--save_image_latents",
                "video_latents": "--save_latents_and_embeddings",
                "disparity_latents": "--save_latents_and_embeddings",
                "prompt_embeds": "--save_latents_and_embeddings",
                            "hand_videos": "--save_latents_and_embeddings",
            "hand_video_latents": "--save_latents_and_embeddings",
            "hand_mask_videos": "--save_latents_and_embeddings",
            "static_videos": "--save_latents_and_embeddings",
            "static_video_latents": "--save_latents_and_embeddings",
            }
            
            # Add flags for missing file types
            for file_type in missing_file_types:
                if file_type in file_type_to_flag:
                    flag = file_type_to_flag[file_type]
                    if flag not in cmd_parts:
                        cmd_parts.append(flag)
            
            # Always include basic processing flags for required file types
            # (videos, disparity, raymaps, prompts are always processed)
            if "images" in missing_file_types or "image_latents" in missing_file_types:
                if "--save_image_latents" not in cmd_parts:
                    cmd_parts.append("--save_image_latents")
            if ("video_latents" in missing_file_types or "disparity_latents" in missing_file_types or 
                "prompt_embeds" in missing_file_types or "hand_videos" in missing_file_types or
                "hand_video_latents" in missing_file_types or "hand_mask_videos" in missing_file_types or
                "static_videos" in missing_file_types or "static_video_latents" in missing_file_types):
                if "--save_latents_and_embeddings" not in cmd_parts:
                    cmd_parts.append("--save_latents_and_embeddings")
            
            # Add prompt embeddings flag for aether and cogvideox_pose model types or if explicitly requested
            if (args.model_type in ["aether", "cogvideox_pose"] or args.save_prompt_embeds) and "prompt_embeds" in missing_file_types:
                if "--save_prompt_embeds" not in cmd_parts:
                    cmd_parts.append("--save_prompt_embeds")
            
            # Add selective processing argument to only process missing file types
            cmd_parts.extend(["--selective_processing"] + missing_file_types)
    
    if args.id_token:
        cmd_parts.extend(["--id_token", args.id_token])
    if args.random_flip:
        cmd_parts.extend(["--random_flip", str(args.random_flip)])
    if args.pin_memory:
        cmd_parts.append("--pin_memory")
    if args.video_reshape_mode:
        cmd_parts.extend(["--video_reshape_mode", args.video_reshape_mode])
    if args.save_image_latents:
        cmd_parts.append("--save_image_latents")
    if args.save_latents_and_embeddings:
        cmd_parts.append("--save_latents_and_embeddings")
    if args.use_slicing:
        cmd_parts.append("--use_slicing")
    if args.use_tiling:
        cmd_parts.append("--use_tiling")
    
    # Add prompt embeddings flag for aether and cogvideox_pose model types or if explicitly requested
    if args.model_type in ["aether", "cogvideox_pose"] or args.save_prompt_embeds:
        cmd_parts.append("--save_prompt_embeds")
    if args.dataloader_num_workers:
        cmd_parts.extend(["--dataloader_num_workers", str(args.dataloader_num_workers)])
    if args.num_decode_threads:
        cmd_parts.extend(["--num_decode_threads", str(args.num_decode_threads)])
    
    # Run the command
    import subprocess
    try:
        print(f"Running command for {scene_name}/{action_name}...")
        # Create environment with the correct GPU assignment for this subprocess
        env = os.environ.copy()
        
        # Explicitly remove distributed environment variables to prevent inheritance
        if "LOCAL_RANK" in env:
            del env["LOCAL_RANK"]
        if "RANK" in env:
            del env["RANK"]
        if "WORLD_SIZE" in env:
            del env["WORLD_SIZE"]
        if "MASTER_ADDR" in env:
            del env["MASTER_ADDR"]
        if "MASTER_PORT" in env:
            del env["MASTER_PORT"]
        
        # Check GPU availability and assign the correct GPU
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            print(f"Available GPUs: {num_gpus}")
            if rank >= num_gpus:
                print(f"Warning: Rank {rank} >= num_gpus {num_gpus}, using rank {rank % num_gpus}")
                rank = rank % num_gpus
            
            # Set the environment variables to force this subprocess to use the current GPU
            env["CUDA_VISIBLE_DEVICES"] = str(rank)
            print(f"Setting CUDA_VISIBLE_DEVICES={rank} for subprocess")
        else:
            print("Warning: CUDA not available, using CPU")
            env["CUDA_VISIBLE_DEVICES"] = ""
        
        # Don't set LOCAL_RANK to prevent distributed initialization in subprocess
        # This forces the subprocess to use single-GPU mode
        # env["LOCAL_RANK"] = "0"  # Force single-GPU mode within the subprocess
        env["RANK"] = "0"
        env["WORLD_SIZE"] = "1"
        
        result = subprocess.run(cmd_parts, check=True, text=True, env=env)
        print(f"Successfully processed {scene_name}/{action_name}")
        
        # Force garbage collection and clear CUDA cache after each subprocess
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"  - Cleared GPU memory cache for GPU {rank}")
        
        if total_videos is not None:
            print(f"  - Processed videos: 00000 to {total_videos-1:05d} ({total_videos} videos)")
        print(f"  - Output saved to: {processed_dir}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error processing {scene_name}/{action_name}:")
        print(f"Return code: {e.returncode}")
        # Don't remove processed directory on error - keep files for debugging
        if processed_dir.exists():
            print(f"Keeping processed files in {processed_dir} for debugging")
        return False


def main():
    args = get_args()
    set_seed(args.seed)
    
    # Dynamically import dataset classes based on model_type
    if args.model_type == "aether":
        # Import Aether-specific dataset classes
        from training.aether.dataset import BucketSampler, VideoDatasetWithResizing, VideoDatasetWithResizeAndRectangleCrop
        print(f"Using Aether dataset classes from training/aether/dataset.py")
    elif args.model_type == "cogvideox_pose":
        # Import CogVideoX pose-specific dataset classes
        from training.cogvideox_static_pose.dataset import BucketSampler, VideoDatasetWithConditions, VideoDatasetWithConditionsAndResizing, VideoDatasetWithConditionsAndResizeAndRectangleCrop
        print(f"Using CogVideoX pose dataset classes from training/cogvideox_static_pose/dataset.py")
    else:  # custom
        # Default to unified dataset classes
        from training.cogvideox_static_pose.dataset import BucketSampler, VideoDatasetWithConditions, VideoDatasetWithConditionsAndResizing, VideoDatasetWithConditionsAndResizeAndRectangleCrop
        print(f"Using default dataset classes from training/cogvideox_static_pose/dataset.py")

    # Initialize distributed processing
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        
        # Check if the GPU exists
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            if local_rank >= num_gpus:
                print(f"Warning: LOCAL_RANK {local_rank} >= num_gpus {num_gpus}, using rank {local_rank % num_gpus}")
                local_rank = local_rank % num_gpus
                os.environ["LOCAL_RANK"] = str(local_rank)
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        # Single GPU
        local_rank = 0
        world_size = 1
        rank = 0
        torch.cuda.set_device(rank)

    # We won't use a central output directory - files will be kept in each action's sequences directory
    print(f"Files will be kept in each action's sequences/processed directory")

    # Find all scene-action pairs
    data_root = pathlib.Path(args.data_root)
    scene_action_pairs = find_scene_action_pairs(data_root, args.sequences_dir, args.scene_filter, rank)
    
    # Filter out already processed scene-action pairs if skip_existing is enabled
    if args.skip_existing:
        unprocessed_pairs = []
        for scene_name, action_name, sequences_path in scene_action_pairs:
            # Check if files have been processed in the sequences directory
            
            # Determine which file types to check
            if args.model_type != "custom":
                # Use predefined file types based on model type
                file_types_to_check = get_model_file_types(args.model_type)
                print(f"Using predefined file types for {args.model_type}: {file_types_to_check}")
            elif args.check_file_types:
                # Use only the specified file types
                file_types_to_check = args.check_file_types
                print(f"Checking only specified file types: {file_types_to_check}")
            else:
                # Check all file types that the script generates
                file_types_to_check = [
                    "videos", "images", "images_goal", "image_latents", "image_goal_latents",
                    "video_latents", "disparity", "disparity_latents", "raymaps", "raymaps_abs",
                    "prompts", "prompt_embeds", "human_motions"
                ]
            
            missing_file_types = []
            has_all_files = True
            
            # Get expected file count for this scene-action pair
            expected_file_count = get_expected_file_count(sequences_path)
            
            for file_type in file_types_to_check:
                # Check in the appropriate processed directory based on model_type
                if args.model_type == "aether":
                    processed_dir = sequences_path.parent / "processed" / file_type
                else:
                    processed_dir = sequences_path.parent / "processed2" / file_type
                
                if not processed_dir.exists():
                    missing_file_types.append(file_type)
                    has_all_files = False
                    continue
                
                # Look for files in the processed directory
                found_files = []
                for file_item in processed_dir.iterdir():
                    if file_item.is_file():
                        found_files.append(file_item.name)
                
                # Check if we have the expected number of files
                if len(found_files) < expected_file_count:
                    missing_file_types.append(file_type)
                    has_all_files = False
                    if rank == 0:
                        print(f"🔍 {scene_name}/{action_name} - {file_type}: Found {len(found_files)} files, expected {expected_file_count}")
                elif len(found_files) > expected_file_count:
                    if rank == 0:
                        print(f"⚠️  {scene_name}/{action_name} - {file_type}: Found {len(found_files)} files, expected {expected_file_count} (extra files)")
            
            if has_all_files:
                print(f"Skipping {scene_name}/{action_name}: Already processed (all checked file types found with expected counts)")
            else:
                if len(missing_file_types) <= 3:  # Show missing types if not too many
                    print(f"Processing {scene_name}/{action_name}: Missing {missing_file_types}")
                else:
                    print(f"Processing {scene_name}/{action_name}: Missing {len(missing_file_types)} file types")
                # Store missing file types with the pair for selective processing
                unprocessed_pairs.append((scene_name, action_name, sequences_path, missing_file_types))
        
        scene_action_pairs = unprocessed_pairs
    
    if rank == 0:
        print(f"Found {len(scene_action_pairs)} scene-action pairs to process")
        for pair_data in scene_action_pairs:
            # Handle both 3-tuple and 4-tuple formats
            if len(pair_data) == 3:
                scene_name, action_name, _ = pair_data
            else:
                scene_name, action_name, _, missing_file_types = pair_data
            print(f"  - {scene_name}/{action_name}")

    # Split scene-action pairs among GPUs
    if world_size > 1:
        pairs_per_gpu = len(scene_action_pairs) // world_size
        start_index = rank * pairs_per_gpu
        end_index = start_index + pairs_per_gpu
        if rank == world_size - 1:
            end_index = len(scene_action_pairs)  # Make sure the last GPU gets the remaining pairs
        
        # Get pairs for this GPU
        gpu_pairs = scene_action_pairs[start_index:end_index]
        print(f"GPU {rank}: Processing {len(gpu_pairs)} scene-action pairs (indices {start_index}-{end_index-1})")
    else:
        gpu_pairs = scene_action_pairs
        print(f"Single GPU: Processing {len(gpu_pairs)} scene-action pairs")

    # Process scene-action pairs assigned to this GPU
    print(f"\nStarting batch processing of {len(gpu_pairs)} scene-action pairs on GPU {rank}...")
    
    successful_pairs = []
    failed_pairs = []
    
    for i, pair_data in enumerate(gpu_pairs):
        # Handle both old format (3-tuple) and new format (4-tuple with missing file types)
        if len(pair_data) == 3:
            scene_name, action_name, sequences_path = pair_data
            missing_file_types = None  # Process everything
        else:
            scene_name, action_name, sequences_path, missing_file_types = pair_data
        
        # Create processed directory within this action's sequences directory based on model_type
        if args.model_type == "aether":
            processed_dir = sequences_path.parent / "processed"
        else:
            processed_dir = sequences_path.parent / "processed2"
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Process the scene-action pair
        success = process_single_scene_action(
            scene_name, action_name, sequences_path, processed_dir, rank, args, missing_file_types
        )
        
        if success:
            successful_pairs.append((scene_name, action_name, processed_dir))
            print(f"  - Successfully processed {scene_name}/{action_name}")
            print(f"  - Files saved to: {processed_dir}")
        else:
            failed_pairs.append((scene_name, action_name, processed_dir))
            print(f"  - Failed to process {scene_name}/{action_name}")
            continue

    # Complete distributed processing
    if world_size > 1:
        try:
            dist.destroy_process_group()
        except Exception as e:
            print(f"Warning: Failed to destroy process group: {e}")
            pass

    if rank == 0:
        print(f"\nCompleted batch processing.")
        print(f"Successful pairs: {len(successful_pairs)}")
        print(f"Failed pairs: {len(failed_pairs)}")
        print(f"All processed files have been saved to 'processed' directories within each action's sequences folder.")
        print(f"Each action now has a structure like: sequences/processed/[file_types]/")
    else:
        print(f"\nGPU {rank} completed processing.")


if __name__ == "__main__":
    main() 