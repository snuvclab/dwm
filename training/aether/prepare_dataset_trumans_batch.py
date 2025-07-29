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
from dataset import BucketSampler, VideoDatasetWithResizing, VideoDatasetWithResizeAndRectangleCrop  # isort:skip

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
        "--output_dir", type=str, required=True, help="Path to output directory where all processed data will be saved.",
    )
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
        choices=["npy", "video"],
        default="npy",
        help="Format for disparity data: 'npy' for NPY files, 'video' for MP4 video files.",
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
        "--simulate",
        action="store_true",
        help="Simulate processing without actual encoding. Creates dummy files to test file organization.",
    )
    return parser.parse_args()


def find_scene_action_pairs(data_root: pathlib.Path, sequences_dir: str, rank: int = 0) -> List[tuple]:
    """Find all scene-action pairs that have valid sequence data."""
    scene_action_pairs = []
    
    for scene_dir in sorted(data_root.iterdir()):
        if not scene_dir.is_dir():
            continue
            
        scene_name = scene_dir.name
        for action_dir in sorted(scene_dir.iterdir()):
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


def create_simulation_files(temp_output_dir: pathlib.Path, scene_name: str, action_name: str, rank: int):
    """Create dummy files to simulate the output structure of the actual processing."""
    # Create the directory structure directly in temp_output_dir (no tmp subdirectory)
    
    # Define the subdirectories that would be created by the actual script
    subdirs = [
        "images", "image_latents", "images_goal", "image_goal_latents",
        "videos", "video_latents", "disparity", "disparity_latents",
        "raymaps", "raymaps_abs", "prompts", "prompt_embeds"
    ]
    
    # Create subdirectories and dummy files directly in temp_output_dir
    for subdir_name in subdirs:
        subdir = temp_output_dir / subdir_name
        subdir.mkdir(parents=True, exist_ok=True)
        
        # Create a few dummy files with realistic names using the new convention
        if subdir_name in ["images", "images_goal"]:
            # Image files (PNG)
            for i in range(3):
                dummy_file = subdir / f"{scene_name}_{action_name}_video_{i:05d}.png"
                dummy_file.write_text(f"DUMMY_IMAGE_{scene_name}_{action_name}_{i}")
        elif subdir_name in ["videos", "disparity"]:
            # Video files (MP4)
            for i in range(2):
                dummy_file = subdir / f"{scene_name}_{action_name}_video_{i:05d}.mp4"
                dummy_file.write_text(f"DUMMY_VIDEO_{scene_name}_{action_name}_{i}")
        elif subdir_name in ["image_latents", "image_goal_latents", "video_latents", "disparity_latents", "prompt_embeds"]:
            # Tensor files (PT)
            for i in range(2):
                dummy_file = subdir / f"{scene_name}_{action_name}_video_{i:05d}.pt"
                dummy_file.write_text(f"DUMMY_TENSOR_{scene_name}_{action_name}_{i}")
        elif subdir_name in ["raymaps", "raymaps_abs"]:
            # Raymap files (PT) - using torch.save in real processing
            for i in range(2):
                dummy_file = subdir / f"{scene_name}_{action_name}_video_{i:05d}.pt"
                dummy_file.write_text(f"DUMMY_RAYMAP_{scene_name}_{action_name}_{i}")
        elif subdir_name == "prompts":
            # Text files (TXT)
            for i in range(2):
                dummy_file = subdir / f"{scene_name}_{action_name}_video_{i:05d}.txt"
                dummy_file.write_text(f"DUMMY_PROMPT_{scene_name}_{action_name}_{i}: A person walking in a scene")
    
    print(f"Created simulation files in {temp_output_dir}")
    print(f"  - {len(subdirs)} subdirectories")
    print(f"  - Rank {rank} files created")
    print(f"  - Files will be renamed to: {scene_name[:8]}_{action_name}_XXXXX.ext format")


def process_single_scene_action(
    scene_name: str, 
    action_name: str, 
    sequences_path: pathlib.Path,
    temp_output_dir: pathlib.Path,
    rank: int,
    args: argparse.Namespace,
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
    print(f"{'='*60}")
    
    # Create temporary output directory for this action
    temp_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build command to run the original script
    cmd_parts = [
        "python", "training/aether/prepare_dataset_trumans.py",
        "--model_id", args.model_id,
        "--data_root", str(sequences_path),
        "--output_dir", str(temp_output_dir),
        "--caption_column", args.caption_column,
        "--video_column", args.video_column,
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
    if args.dataloader_num_workers:
        cmd_parts.extend(["--dataloader_num_workers", str(args.dataloader_num_workers)])
    if args.num_decode_threads:
        cmd_parts.extend(["--num_decode_threads", str(args.num_decode_threads)])
    
    # Run the command or simulate
    import subprocess
    try:
        if args.simulate:
            print(f"SIMULATION MODE: Creating dummy files for {scene_name}/{action_name}...")
            # Create dummy files to simulate the output structure
            create_simulation_files(temp_output_dir, scene_name, action_name, rank)
            print(f"SIMULATION: Successfully created dummy files for {scene_name}/{action_name}")
        else:
            print(f"Running command for {scene_name}/{action_name}...")
            # Create environment with the correct GPU assignment for this subprocess
            env = os.environ.copy()
            # Set the environment variables to force this subprocess to use the current GPU
            env["CUDA_VISIBLE_DEVICES"] = str(rank)
            env["LOCAL_RANK"] = "0"  # Force single-GPU mode within the subprocess
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
        print(f"  - Output saved to: {temp_output_dir}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error processing {scene_name}/{action_name}:")
        print(f"Return code: {e.returncode}")
        # Don't remove temp directory on error - keep files for debugging
        if temp_output_dir.exists():
            print(f"Keeping temporary files in {temp_output_dir} for debugging")
        return False


def main():
    args = get_args()
    set_seed(args.seed)

    # Initialize distributed processing
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
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

    # For simulation mode, create a separate output directory to avoid interfering with existing files
    if args.simulate:
        output_dir = pathlib.Path(args.output_dir) / "SIMULATION_OUTPUT"
        print(f"SIMULATION MODE: Using separate output directory: {output_dir}")
    else:
        output_dir = pathlib.Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all scene-action pairs
    data_root = pathlib.Path(args.data_root)
    scene_action_pairs = find_scene_action_pairs(data_root, args.sequences_dir, rank)
    
    # Filter out already processed scene-action pairs if skip_existing is enabled
    if args.skip_existing:
        unprocessed_pairs = []
        for scene_name, action_name, sequences_path in scene_action_pairs:
            # Check if files have been moved to the main output directory
            scene_prefix = scene_name[:8]
            
            # Check if any files exist in the organized directories for this scene-action pair
            has_processed_files = False
            for file_type in ["videos", "images", "image_latents", "video_latents"]:
                type_dir = output_dir / file_type
                if type_dir.exists():
                    # Look for files that start with the scene prefix and action name
                    for file_item in type_dir.iterdir():
                        if file_item.is_file() and file_item.name.startswith(f"{scene_prefix}_{action_name}_"):
                            has_processed_files = True
                            break
                    if has_processed_files:
                        break
            
            if has_processed_files:
                print(f"Skipping {scene_name}/{action_name}: Already processed (files found in organized directories)")
            else:
                unprocessed_pairs.append((scene_name, action_name, sequences_path))
        
        scene_action_pairs = unprocessed_pairs
    
    if rank == 0:
        print(f"Found {len(scene_action_pairs)} scene-action pairs to process")
        for scene_name, action_name, _ in scene_action_pairs:
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
    
    for i, (scene_name, action_name, sequences_path) in enumerate(gpu_pairs):
        # Create temporary output directory for this action
        temp_output_dir = output_dir / f"temp_{scene_name}_{action_name}"
        
        # Process the scene-action pair
        success = process_single_scene_action(
            scene_name, action_name, sequences_path, temp_output_dir, rank, args
        )
        
        if success:
            successful_pairs.append((scene_name, action_name, temp_output_dir))
            
            # Move files from temp directory to main output directory
            # The temp directory contains organized data directly (no 'tmp' subdirectory)
            print(f"Checking temp directory structure: {temp_output_dir}")
            if temp_output_dir.exists():
                print(f"Temp directory contents: {list(temp_output_dir.iterdir())}")
            
            # Check if temp directory has the expected structure (direct subdirectories)
            has_valid_structure = False
            for item in temp_output_dir.iterdir():
                if item.is_dir() and item.name in ["images", "videos", "image_latents", "video_latents", "disparity", "disparity_latents", "raymaps", "raymaps_abs", "prompts", "prompt_embeds", "images_goal", "image_goal_latents"]:
                    has_valid_structure = True
                    break
            
            if has_valid_structure:
                
                # Create top-level directories for each file type
                file_type_dirs = {
                    "images": output_dir / "images",
                    "images_goal": output_dir / "images_goal", 
                    "image_latents": output_dir / "image_latents",
                    "image_goal_latents": output_dir / "image_goal_latents",
                    "videos": output_dir / "videos",
                    "video_latents": output_dir / "video_latents",
                    "disparity": output_dir / "disparity",
                    "disparity_latents": output_dir / "disparity_latents",
                    "raymaps": output_dir / "raymaps",
                    "raymaps_abs": output_dir / "raymaps_abs",
                    "prompts": output_dir / "prompts",
                    "prompt_embeds": output_dir / "prompt_embeds"
                }
                
                # Create all directories
                for dir_path in file_type_dirs.values():
                    dir_path.mkdir(parents=True, exist_ok=True)
                
                # Move all contents from temp directory to the appropriate type directories
                import shutil
                print(f"Moving contents from {temp_output_dir} to type-specific directories")
                moved_files = []
                
                # Extract scene prefix (first 8 characters)
                scene_prefix = scene_name[:8]
                
                for item in temp_output_dir.iterdir():
                    if item.is_dir():
                        file_type = item.name
                        if file_type in file_type_dirs:
                            target_dir = file_type_dirs[file_type]
                            
                            # Process files directly in the file type directory (no rank subdirectories)
                            for file_item in item.iterdir():
                                if file_item.is_file():
                                            # Extract the original filename and extension
                                            original_name = file_item.name
                                            
                                            # Parse the original filename to extract the video number
                                            # Actual format: 0a761819_2023-01-14@22-15-43_00000.mp4
                                            # Expected format: scene_prefix_action_name_XXXXX.ext
                                            
                                            # Split by underscore to get parts
                                            parts = original_name.split("_")
                                            if len(parts) >= 3:
                                                # Format: [scene_prefix, action_name, video_number.ext]
                                                scene_prefix_part = parts[0]
                                                action_name_part = parts[1]
                                                video_number_with_ext = parts[2]
                                                
                                                # Split video number and extension
                                                if "." in video_number_with_ext:
                                                    video_number = video_number_with_ext.split(".")[0]
                                                    extension = video_number_with_ext.split(".")[1]
                                                else:
                                                    video_number = video_number_with_ext
                                                    extension = ""
                                                    
                                                # Verify we have the expected scene prefix
                                                if scene_prefix_part != scene_prefix:
                                                    print(f"Warning: Scene prefix mismatch: expected {scene_prefix}, got {scene_prefix_part}")
                                                    continue
                                                    
                                                # Verify we have the expected action name
                                                if action_name_part != action_name:
                                                    print(f"Warning: Action name mismatch: expected {action_name}, got {action_name_part}")
                                                    continue
                                            else:
                                                # Fallback: try to extract number from end of filename
                                                if len(parts) >= 2:
                                                    last_part = parts[-1]
                                                    if "." in last_part:
                                                        video_number = last_part.split(".")[0]
                                                        extension = last_part.split(".")[1]
                                                    else:
                                                        video_number = last_part
                                                        extension = ""
                                                else:
                                                    video_number = "00000"
                                                    extension = "mp4"
                                            
                                            # Create new filename: scene_prefix_action_name_XXXXX.ext
                                            new_filename = f"{scene_prefix}_{action_name}_{video_number}.{extension}"
                                            target_file = target_dir / new_filename
                                            
                                            # Check if file already exists to prevent overwriting
                                            if target_file.exists():
                                                print(f"Warning: File already exists, skipping: {target_file}")
                                                print(f"  This could indicate duplicate scene-action combinations or filename conflicts")
                                                continue
                                            
                                            # Move the file
                                            shutil.move(str(file_item), str(target_file))
                                            moved_files.append(f"{file_type}: {original_name} -> {new_filename}")
                        else:
                            print(f"Warning: Unknown file type directory: {file_type}")
                    elif item.is_file():
                        # Handle any direct files (shouldn't happen with current structure)
                        print(f"Warning: Found direct file in tmp: {item.name}")
                
                print(f"Moved {len(moved_files)} files to type-specific directories")
                if moved_files:
                    print(f"  Sample moves: {moved_files[:5]}{'...' if len(moved_files) > 5 else ''}")
                
                # Verify that files were actually moved before removing temp directory
                if len(moved_files) == 0:
                    print(f"ERROR: No files were moved from {temp_output_dir}")
                    print(f"  This indicates a critical bug in the file moving logic")
                    print(f"  Keeping temporary directory for debugging: {temp_output_dir}")
                    print(f"  Please check the filename parsing logic and file structure")
                    continue
                
                # Keep temp directory for manual verification and cleanup
                print(f"  - Successfully moved {len(moved_files)} files")
                print(f"  - Temporary directory kept for manual verification: {temp_output_dir}")
                print(f"  - You can manually remove it after verifying the files are correct")
            else:
                print(f"Warning: No 'tmp' directory found in {temp_output_dir}")
                # Keep temp directory for debugging if no tmp subdirectory found
                print(f"  - Keeping temporary directory for debugging: {temp_output_dir}")
        else:
            failed_pairs.append((scene_name, action_name, temp_output_dir))
            # Keep temp directory for debugging when processing fails
            print(f"  - Keeping temporary directory for debugging: {temp_output_dir}")
            continue

    # Complete distributed processing
    if world_size > 1:
        try:
            dist.destroy_process_group()
        except Exception as e:
            print(f"Warning: Failed to destroy process group: {e}")
            pass

    if rank == 0:
        if args.simulate:
            print(f"\nSIMULATION COMPLETED. All simulation files saved to `{output_dir.as_posix()}`")
            print(f"Note: This was a simulation run - no actual processing was performed.")
        else:
            print(f"\nCompleted batch processing. All files saved to `{output_dir.as_posix()}`")
            print(f"Successful pairs: {len(successful_pairs)}")
            print(f"Failed pairs: {len(failed_pairs)}")
            
            # Check for temporary directories
            temp_dirs = list(output_dir.glob("temp_*"))
            if temp_dirs:
                print(f"\nTemporary directories found: {len(temp_dirs)}")
                print("These directories contain processed files that were successfully moved to the main output directory.")
                print("You can safely remove them after verifying the processing was successful.")
                print(f"To clean up, run: python cleanup_temp_dirs.py {output_dir}")
    else:
        print(f"\nGPU {rank} completed processing.")


if __name__ == "__main__":
    main() 