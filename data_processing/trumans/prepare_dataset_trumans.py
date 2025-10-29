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

import numpy as np
import torch
import torch.distributed as dist
from diffusers import AutoencoderKLCogVideoX
from diffusers.training_utils import set_seed
from diffusers.utils import export_to_video, get_logger
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path
from transformers import T5EncoderModel, T5Tokenizer


import decord  # isort:skip
# Import dataset classes based on model type
# These will be conditionally imported in the main function based on args.model_type

decord.bridge.set_bridge("torch")

logger = get_logger(__name__)

DTYPE_MAPPING = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def _get_t5_prompt_embeds(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: Union[str, List[str]],
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("`text_input_ids` must be provided when the tokenizer is not specified.")

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

    return prompt_embeds


def encode_prompt(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: Union[str, List[str]],
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt_embeds = _get_t5_prompt_embeds(
        tokenizer,
        text_encoder,
        prompt=prompt,
        num_videos_per_prompt=num_videos_per_prompt,
        max_sequence_length=max_sequence_length,
        device=device,
        dtype=dtype,
        text_input_ids=text_input_ids,
    )
    return prompt_embeds


def compute_prompt_embeddings(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompts: List[str],
    max_sequence_length: int,
    device: torch.device,
    dtype: torch.dtype,
    requires_grad: bool = False,
):
    if requires_grad:
        prompt_embeds = encode_prompt(
            tokenizer,
            text_encoder,
            prompts,
            num_videos_per_prompt=1,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )
    else:
        with torch.no_grad():
            prompt_embeds = encode_prompt(
                tokenizer,
                text_encoder,
                prompts,
                num_videos_per_prompt=1,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
    return prompt_embeds


def generate_trumans_filename(scene_name: str, action_name: str, video_name: str, extension: str) -> str:
    """Generate simple filename: {video_name}.{extension}
    
    Args:
        scene_name: Full scene name (e.g., 0a7618195-4647-8896747201b1)
        action_name: Action name (e.g., 20231-14@22-06-10)
        video_name: Video name without extension (e.g., '00000')
        extension: File extension (e.g., 'mp4', 'pt', 'png')
    
    Returns:
        Simple filename (e.g., '00000.mp4')
    """
    # Generate simple filename
    filename = f"{video_name}.{extension}"
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


def get_args() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description="Prepare Trumans dataset with descriptive naming")
    parser.add_argument(
        "--model_id",
        type=str,
        default="THUDM/CogVideoX-2b",
        help="Hugging Face model ID to use for tokenizer, text encoder and VAE.",
    )
    parser.add_argument("--data_root", type=str, required=True, help="Path to where training data is located.")
    parser.add_argument(
        "--dataset_file", type=str, default=None, help="Path to CSV file containing metadata about training data."
    )

    parser.add_argument(
        "--video_column",
        type=str,
        default="video",
        help="If using a CSV file via the `--dataset_file` argument, this should be the name of the column containing the video paths. If using the folder structure format for data loading, this should be the name of the file containing line-separated video paths (the file should be located in `--data_root`).",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="prompts.txt",
        help="Path to the file containing line-separated text prompts.",
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
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to output directory where preprocessed videos/latents/embeddings will be saved.",
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
        "--save_prompt_embeds",
        action="store_true",
        help="Whether to encode prompts to embeddings and save them in pytorch serializable format.",
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
    # Trumans-specific arguments
    parser.add_argument(
        "--scene_name",
        type=str,
        required=True,
        help="Scene name (e.g., 0a7618195-4647-889b-a726747201) to extract first 8 characters for naming.",
    )
    parser.add_argument(
        "--action_name",
        type=str,
        required=True,
        help="Action name (e.g., 20231-14@22-06-10) for naming convention.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip processing if output files already exist.",
    )
    parser.add_argument(
        "--selective_processing",
        nargs="+",
        type=str,
        default=None,
        help="Only process specific file types. Useful when some files already exist. Options: images, image_latents, images_goal, image_goal_latents, videos, video_latents, disparity, disparity_latents, raymaps, raymaps_abs, human_motions, hand_videos, hand_video_latents, static_videos, static_video_latents, prompts, prompt_embeds",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["aether", "cogvideox_pose"],
        default="aether",
        help="Model type to determine which dataset class and file types to use. 'aether' for Aether model, 'cogvideox_pose' for CogVideoX pose model.",
    )
    parser.add_argument(
        "--use_gray_hand_videos",
        action="store_true",
        help="Use gray hand videos instead of colored hand videos for CogVideoX pose model type.",
    )
    parser.add_argument(
        "--split_hands",
        action="store_true",
        help="Split hands into left and right hand videos for CogVideoX pose model type.",
    )
    parser.add_argument(
        "--hand_video_subdir",
        type=str,
        default="videos_hands",
        help="Subdirectory name for hand videos in default mode (not gray, not split). Default: 'videos_hands'",
    )
    parser.add_argument(
        "--hand_video_latents_subdir",
        type=str,
        default="hand_video_latents",
        help="Subdirectory name for hand video latents in default mode (not gray, not split). Default: 'hand_video_latents'",
    )
    return parser.parse_args()





to_pil_image = transforms.ToPILImage(mode="RGB")


def save_image(image: torch.Tensor, path: pathlib.Path) -> None:
    image = image.to(dtype=torch.float32).clamp(-1, 1)
    image = to_pil_image(image.float())
    image.save(path)


def save_video(video: torch.Tensor, path: pathlib.Path, fps: int = 8) -> None:
    video = video.to(dtype=torch.float32).clamp(-1, 1)
    video = [to_pil_image(frame) for frame in video]
    export_to_video(video, path, fps=fps)


def create_hand_mask(video: torch.Tensor) -> torch.Tensor:
    """
    Create hand mask from hand video by converting non-black pixels to white.
    Assumes hand video has black background with colored hand meshes.
    
    Args:
        video: Hand video tensor with shape [B, C, F, H, W] in range [0, 1]
    
    Returns:
        Hand mask tensor with shape [B, C, F, H, W] where hand pixels are white (1.0) and background is black (0.0)
    """
    # Convert to [0, 255] range for easier thresholding
    video_255 = (video * 255.0).to(torch.uint8)
    
    # Create mask: any pixel that is not black (0,0,0) becomes white (255,255,255)
    # Check if any channel is non-zero
    mask = (video_255.sum(dim=2, keepdim=True) > 100).float()  # [B, F, 1, H, W]
    
    # Expand to 3 channels
    mask = mask.expand(-1, -1, 3, -1, -1)  # [B, F, 3, H, W]
    
    return mask


def save_prompt(prompt: str, path: pathlib.Path) -> None:
    with open(path, "w", encoding="utf-8") as file:
        file.write(prompt)


def save_metadata(metadata: Dict[str, Any], path: pathlib.Path) -> None:
    with open(path, "w", encoding="utf-8") as file:
        file.write(json.dumps(metadata))


@torch.no_grad()
def serialize_artifacts(
    batch_size: int,
    fps: int,
    scene_name: str,
    action_name: str,
    video_paths: Optional[List[pathlib.Path]] = None,  # Add video paths parameter
    skip_existing: bool = False,  # Add skip_existing parameter
    selective_processing: Optional[List[str]] = None,  # Add selective processing parameter
    model_type: str = "aether",  # Add model type parameter
    images_dir: Optional[pathlib.Path] = None,
    image_latents_dir: Optional[pathlib.Path] = None,
    images_goal_dir: Optional[pathlib.Path] = None,
    image_goal_latents_dir: Optional[pathlib.Path] = None,
    videos_dir: Optional[pathlib.Path] = None,
    video_latents_dir: Optional[pathlib.Path] = None,
    disparity_dir: Optional[pathlib.Path] = None,
    disparity_latents_dir: Optional[pathlib.Path] = None,
    raymap_dir: Optional[pathlib.Path] = None,
    raymap_abs_dir: Optional[pathlib.Path] = None,
    prompts_dir: Optional[pathlib.Path] = None,
    prompt_embeds_dir: Optional[pathlib.Path] = None,
    human_motions_dir: Optional[pathlib.Path] = None,
    hand_videos_dir: Optional[pathlib.Path] = None,
    hand_video_latents_dir: Optional[pathlib.Path] = None,
    hand_videos_gray_dir: Optional[pathlib.Path] = None,
    hand_video_gray_latents_dir: Optional[pathlib.Path] = None,
    hand_videos_gray_left_dir: Optional[pathlib.Path] = None,
    hand_video_gray_left_latents_dir: Optional[pathlib.Path] = None,
    hand_videos_gray_right_dir: Optional[pathlib.Path] = None,
    hand_video_gray_right_latents_dir: Optional[pathlib.Path] = None,
    hand_mask_videos_dir: Optional[pathlib.Path] = None,
    static_videos_dir: Optional[pathlib.Path] = None,
    static_video_latents_dir: Optional[pathlib.Path] = None,
    smpl_pos_map_dir: Optional[pathlib.Path] = None,
    smpl_pos_map_latents_dir: Optional[pathlib.Path] = None,
    images: Optional[torch.Tensor] = None,
    image_latents: Optional[torch.Tensor] = None,
    images_goal: Optional[torch.Tensor] = None,
    image_goal_latents: Optional[torch.Tensor] = None,
    videos: Optional[torch.Tensor] = None,
    video_latents: Optional[torch.Tensor] = None,
    disparity: Optional[torch.Tensor] = None,
    disparity_latents: Optional[torch.Tensor] = None,
    raymap: Optional[torch.Tensor] = None,
    raymap_abs: Optional[torch.Tensor] = None,
    prompts: Optional[List[str]] = None,
    prompt_embeds: Optional[torch.Tensor] = None,
    human_motions: Optional[torch.Tensor] = None,
    hand_videos: Optional[torch.Tensor] = None,
    hand_video_latents: Optional[torch.Tensor] = None,
    hand_videos_gray: Optional[torch.Tensor] = None,
    hand_video_gray_latents: Optional[torch.Tensor] = None,
    hand_videos_gray_left: Optional[torch.Tensor] = None,
    hand_video_gray_left_latents: Optional[torch.Tensor] = None,
    hand_videos_gray_right: Optional[torch.Tensor] = None,
    hand_video_gray_right_latents: Optional[torch.Tensor] = None,
    hand_mask_videos: Optional[torch.Tensor] = None,
    static_videos: Optional[torch.Tensor] = None,
    static_video_latents: Optional[torch.Tensor] = None,
    smpl_pos_map: Optional[torch.Tensor] = None,
    smpl_pos_map_latents: Optional[torch.Tensor] = None,
) -> None:
    # Handle the case where videos is None (selective processing)
    if videos is not None:
        num_frames, height, width = videos.size(1), videos.size(3), videos.size(4)
        # Create metadata for each video in the batch
        metadata = [{"num_frames": num_frames, "height": height, "width": width} for _ in range(batch_size)]
    else:
        # For selective processing without videos, create dummy metadata
        metadata = [{"num_frames": 0, "height": 0, "width": 0} for _ in range(batch_size)]

    # Base data folder mapper list for Aether model
    aether_mapper_list = [
        (images, images_dir, lambda img, path: save_image(img[0], path), "png"),
        (image_latents, image_latents_dir, torch.save, "pt"),
        (images_goal, images_goal_dir, lambda img, path: save_image(img[0], path), "png"),
        (image_goal_latents, image_goal_latents_dir, torch.save, "pt"),
        (videos, videos_dir, functools.partial(save_video, fps=fps), "mp4"),
        (video_latents, video_latents_dir, torch.save, "pt"),
        (disparity, disparity_dir, functools.partial(save_video, fps=fps), "mp4"),
        (disparity_latents, disparity_latents_dir, torch.save, "pt"),
        (raymap, raymap_dir, torch.save, "pt"),
        (raymap_abs, raymap_abs_dir, torch.save, "pt"),
        (prompts, prompts_dir, save_prompt, "txt"),
        (prompt_embeds, prompt_embeds_dir, torch.save, "pt"),
        (human_motions, human_motions_dir, torch.save, "pt"),
        (metadata, videos_dir, save_metadata, "txt"),
    ]
    
    # CogVideoX pose data folder mapper list
    cogvideox_pose_mapper_list = [
        (videos, videos_dir, functools.partial(save_video, fps=fps), "mp4"),
        (video_latents, video_latents_dir, torch.save, "pt"),
        (hand_videos, hand_videos_dir, functools.partial(save_video, fps=fps), "mp4"),
        (hand_video_latents, hand_video_latents_dir, torch.save, "pt"),
        (hand_videos_gray, hand_videos_gray_dir, functools.partial(save_video, fps=fps), "mp4"),
        (hand_video_gray_latents, hand_video_gray_latents_dir, torch.save, "pt"),
        (hand_videos_gray_left, hand_videos_gray_left_dir, functools.partial(save_video, fps=fps), "mp4"),
        (hand_video_gray_left_latents, hand_video_gray_left_latents_dir, torch.save, "pt"),
        (hand_videos_gray_right, hand_videos_gray_right_dir, functools.partial(save_video, fps=fps), "mp4"),
        (hand_video_gray_right_latents, hand_video_gray_right_latents_dir, torch.save, "pt"),
        (hand_mask_videos, hand_mask_videos_dir, functools.partial(save_video, fps=fps), "mp4"),
        (static_videos, static_videos_dir, functools.partial(save_video, fps=fps), "mp4"),
        (static_video_latents, static_video_latents_dir, torch.save, "pt"),
        (smpl_pos_map, smpl_pos_map_dir, functools.partial(save_video, fps=fps), "mp4"),
        (smpl_pos_map_latents, smpl_pos_map_latents_dir, torch.save, "pt"),
        (prompts, prompts_dir, save_prompt, "txt"),
        (prompt_embeds, prompt_embeds_dir, torch.save, "pt"),
        (metadata, videos_dir, save_metadata, "txt"),
    ]
    
    # Choose mapper list based on model type
    if model_type == "cogvideox_pose":
        data_folder_mapper_list = cogvideox_pose_mapper_list
    else:  # aether
        data_folder_mapper_list = aether_mapper_list
    
    # Filter data_folder_mapper_list based on selective_processing
    if selective_processing:
        # Map file types to their index in data_folder_mapper_list
        if model_type == "cogvideox_pose":
            file_type_to_index = {
                "videos": 0,
                "video_latents": 1,
                "videos_hands": 2,
                "hand_video_latents": 3,
                "videos_hands_gray": 4,
                "hand_video_gray_latents": 5,
                "videos_hands_gray_left": 6,
                "hand_video_gray_left_latents": 7,
                "videos_hands_gray_right": 8,
                "hand_video_gray_right_latents": 9,
                "videos_hands_mask": 10,
                "videos_static": 11,
                "static_video_latents": 12,
                "smpl_pos_map_egoallo": 13,
                "smpl_pos_map_egoallo_latents": 14,
                "prompts": 15,
                "prompt_embeds": 16,
            }
        else:  # aether
            file_type_to_index = {
                "images": 0,
                "image_latents": 1,
                "images_goal": 2,
                "image_goal_latents": 3,
                "videos": 4,
                "video_latents": 5,
                "disparity": 6,
                "disparity_latents": 7,
                "raymaps": 8,
                "raymaps_abs": 9,
                "prompts": 10,
                "prompt_embeds": 11,
                "human_motions": 12,
            }
        
        # Keep only the requested file types
        filtered_mapper_list = []
        for file_type in selective_processing:
            # Handle gray hand video latents mapping for cogvideox_pose model
            if model_type == "cogvideox_pose":
                if file_type == "hand_video_gray_latents":
                    # Map hand_video_gray_latents to hand_video_latents for gray hand videos
                    mapped_file_type = "hand_video_latents"
                elif file_type == "hand_video_gray_left_latents":
                    # Map hand_video_gray_left_latents to hand_video_gray_left_latents
                    mapped_file_type = "hand_video_gray_left_latents"
                elif file_type == "hand_video_gray_right_latents":
                    # Map hand_video_gray_right_latents to hand_video_gray_right_latents
                    mapped_file_type = "hand_video_gray_right_latents"
                else:
                    mapped_file_type = file_type
            else:
                mapped_file_type = file_type
            
            if mapped_file_type in file_type_to_index:
                index = file_type_to_index[mapped_file_type]
                if index < len(data_folder_mapper_list):
                    filtered_mapper_list.append(data_folder_mapper_list[index])
        
        # Only include metadata if we're processing videos or other file types that need it
        if model_type == "cogvideox_pose":
            should_include_metadata = any(ft in selective_processing for ft in ["videos", "video_latents", "prompts", "prompt_embeds"])
            metadata_index = 9
        else:  # aether
            should_include_metadata = any(ft in selective_processing for ft in ["videos", "video_latents", "disparity", "disparity_latents", "images", "image_latents", "human_motions", "prompts", "prompt_embeds"])
            metadata_index = 13
        
        if should_include_metadata and metadata_index < len(data_folder_mapper_list):
            filtered_mapper_list.append(data_folder_mapper_list[metadata_index])
        
        data_folder_mapper_list = filtered_mapper_list
        print(f"Selective processing: Only saving {[item[3] for item in data_folder_mapper_list]} files")
    
    # Generate descriptive filenames using actual video names
    filenames = []
    for i in range(batch_size):
        if video_paths and i < len(video_paths):
            # Extract actual video name from the video path
            video_name = video_paths[i].stem  # Get filename without extension
            filename = generate_trumans_filename(scene_name, action_name, video_name, "mp4")
            # Remove extension for base filename
            base_filename = filename.replace(".mp4", "")
        else:
            raise ValueError(f"No video paths provided for batch index {i}")
        filenames.append(base_filename)    

    for data, folder, save_fn, extension in data_folder_mapper_list:
        if data is None or folder is None:
            if data is None:
                print(f"Skipping {folder} since data is None")
            if folder is None:
                print(f"Skipping {data} since folder is None")
            continue
        for slice, filename in zip(data, filenames):
            path = folder.joinpath(f"{filename}.{extension}")
            
            # Skip if file already exists and skip_existing is enabled
            if skip_existing and path.exists():
                print(f"Skipping existing file: {path.name}")
                continue
                
            if isinstance(slice, torch.Tensor):
                slice = slice.clone().to("cpu")
            save_fn(slice, path)
            print(f"Saved: {path.name}")


def save_intermediates(output_queue: queue.Queue) -> None:
    while True:
        try:
            item = output_queue.get(timeout=30)
            if item is None:
                break
            serialize_artifacts(**item)

        except queue.Empty:
            continue


@torch.no_grad()
def main():
    args = get_args()
    set_seed(args.seed)

    output_dir = pathlib.Path(args.output_dir)
    tmp_dir = output_dir.joinpath("tmp")

    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Create task queue for non-blocking serializing of artifacts
    output_queue = queue.Queue()
    save_thread = ThreadPoolExecutor(max_workers=args.num_artifact_workers)
    save_future = save_thread.submit(save_intermediates, output_queue)

    # Initialize distributed processing
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        try:
            dist.init_process_group(backend="nccl")
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        except Exception as e:
            print(f"Warning: Failed to initialize distributed processing: {e}")
            print("Falling back to single GPU mode")
            world_size = 1
            rank = 0
    else:
        # Single GPU
        local_rank = 0
        world_size = 1
        rank = 0
        torch.cuda.set_device(rank)

    # Create folders where intermediate tensors from each rank will be saved
    # Common folders for both model types
    videos_dir = tmp_dir.joinpath(f"videos/{rank}")
    video_latents_dir = tmp_dir.joinpath(f"video_latents/{rank}")
    
    # Aether-specific folders
    images_dir = tmp_dir.joinpath(f"images/{rank}")
    image_latents_dir = tmp_dir.joinpath(f"image_latents/{rank}")
    images_goal_dir = tmp_dir.joinpath(f"images_goal/{rank}")
    image_goal_latents_dir = tmp_dir.joinpath(f"image_goal_latents/{rank}")
    disparity_dir = tmp_dir.joinpath(f"disparity/{rank}")
    disparity_latents_dir = tmp_dir.joinpath(f"disparity_latents/{rank}")
    raymap_dir = tmp_dir.joinpath(f"raymaps/{rank}")
    raymap_abs_dir = tmp_dir.joinpath(f"raymaps_abs/{rank}")
    human_motions_dir = tmp_dir.joinpath(f"human_motions/{rank}")
    
    # CogVideoX pose-specific folders
    # Use configurable subdirectory names for default mode
    hand_videos_dir = tmp_dir.joinpath(f"{args.hand_video_subdir}/{rank}")
    hand_video_latents_dir = tmp_dir.joinpath(f"{args.hand_video_latents_subdir}/{rank}")
    hand_videos_gray_dir = tmp_dir.joinpath(f"videos_hands_gray/{rank}")
    hand_video_gray_latents_dir = tmp_dir.joinpath(f"hand_video_gray_latents/{rank}")
    hand_videos_gray_left_dir = tmp_dir.joinpath(f"videos_hands_gray_left/{rank}")
    hand_video_gray_left_latents_dir = tmp_dir.joinpath(f"hand_video_gray_left_latents/{rank}")
    hand_videos_gray_right_dir = tmp_dir.joinpath(f"videos_hands_gray_right/{rank}")
    hand_video_gray_right_latents_dir = tmp_dir.joinpath(f"hand_video_gray_right_latents/{rank}")
    hand_mask_videos_dir = tmp_dir.joinpath(f"videos_hands_mask/{rank}")
    static_videos_dir = tmp_dir.joinpath(f"videos_static/{rank}")
    static_video_latents_dir = tmp_dir.joinpath(f"static_video_latents/{rank}")
    smpl_pos_map_dir = tmp_dir.joinpath(f"smpl_pos_map_egoallo/{rank}")
    smpl_pos_map_latents_dir = tmp_dir.joinpath(f"smpl_pos_map_egoallo_latents/{rank}")

    # Create common folders
    videos_dir.mkdir(parents=True, exist_ok=True)
    video_latents_dir.mkdir(parents=True, exist_ok=True)
    
    # Create prompt-related folders
    prompts_dir = tmp_dir.joinpath(f"prompts/{rank}")
    prompt_embeds_dir = tmp_dir.joinpath(f"prompt_embeds/{rank}")
    prompts_dir.mkdir(parents=True, exist_ok=True)
    prompt_embeds_dir.mkdir(parents=True, exist_ok=True)
    
    # Create Aether-specific folders
    if args.model_type == "aether":
        images_dir.mkdir(parents=True, exist_ok=True)
        image_latents_dir.mkdir(parents=True, exist_ok=True)
        images_goal_dir.mkdir(parents=True, exist_ok=True)
        image_goal_latents_dir.mkdir(parents=True, exist_ok=True)
        disparity_dir.mkdir(parents=True, exist_ok=True)
        disparity_latents_dir.mkdir(parents=True, exist_ok=True)
        raymap_dir.mkdir(parents=True, exist_ok=True)
        raymap_abs_dir.mkdir(parents=True, exist_ok=True)
        human_motions_dir.mkdir(parents=True, exist_ok=True)
    
    # Create CogVideoX pose-specific folders
    if args.model_type == "cogvideox_pose":
        hand_videos_dir.mkdir(parents=True, exist_ok=True)
        hand_video_latents_dir.mkdir(parents=True, exist_ok=True)
        hand_videos_gray_dir.mkdir(parents=True, exist_ok=True)
        hand_video_gray_latents_dir.mkdir(parents=True, exist_ok=True)
        hand_videos_gray_left_dir.mkdir(parents=True, exist_ok=True)
        hand_video_gray_left_latents_dir.mkdir(parents=True, exist_ok=True)
        hand_videos_gray_right_dir.mkdir(parents=True, exist_ok=True)
        hand_video_gray_right_latents_dir.mkdir(parents=True, exist_ok=True)
        hand_mask_videos_dir.mkdir(parents=True, exist_ok=True)
        static_videos_dir.mkdir(parents=True, exist_ok=True)
        static_video_latents_dir.mkdir(parents=True, exist_ok=True)
        smpl_pos_map_dir.mkdir(parents=True, exist_ok=True)
        smpl_pos_map_latents_dir.mkdir(parents=True, exist_ok=True)

    weight_dtype = DTYPE_MAPPING[args.dtype]
    target_fps = args.target_fps

    # 1. Dataset
    # Import dataset classes based on model type
    if args.model_type == "aether":
        from training.aether.dataset import VideoDatasetWithResizing, VideoDatasetWithResizeAndRectangleCrop
    elif args.model_type == "cogvideox_pose":
        from training.cogvideox_static_pose.dataset import VideoDatasetWithConditionsAndResizing, VideoDatasetWithConditionsAndResizeAndRectangleCrop
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}. Must be 'aether' or 'cogvideox_pose'")
    
    dataset_init_kwargs = {
        "data_root": args.data_root,
        "dataset_file": args.dataset_file,
        "video_column": args.video_column,
        "caption_column": args.caption_column,
        "max_num_frames": args.max_num_frames,
        "id_token": args.id_token,
        "height_buckets": args.height_buckets,
        "width_buckets": args.width_buckets,
        "frame_buckets": args.frame_buckets,
        "load_tensors": False,
        "random_flip": args.random_flip,
    }

    if args.model_type == "cogvideox_pose":
        dataset_init_kwargs.update({
            "use_gray_hand_videos": args.use_gray_hand_videos,
            "split_hands": args.split_hands,  # Enable left/right hand video processing
            "hand_video_subdir": args.hand_video_subdir,
            "hand_video_latents_subdir": args.hand_video_latents_subdir,
        })
    
    # Add model-specific arguments
    if args.model_type == "aether":
        dataset_init_kwargs.update({
            "image_to_video": args.save_image_latents,
            "disparity_format": args.disparity_format,
        })
    
    # Choose dataset class based on model type
    if args.model_type == "cogvideox_pose":
        if args.video_reshape_mode is None:
            dataset = VideoDatasetWithConditionsAndResizing(**dataset_init_kwargs)
        else:
            dataset = VideoDatasetWithConditionsAndResizeAndRectangleCrop(
                video_reshape_mode=args.video_reshape_mode, **dataset_init_kwargs
            )
    else:  # aether
        if args.video_reshape_mode is None:
            dataset = VideoDatasetWithResizing(**dataset_init_kwargs)
        else:
            dataset = VideoDatasetWithResizeAndRectangleCrop(
                video_reshape_mode=args.video_reshape_mode, **dataset_init_kwargs
            )

    original_dataset_size = len(dataset)

    # Split data among GPUs
    if world_size > 1:
        samples_per_gpu = original_dataset_size // world_size
        start_index = rank * samples_per_gpu
        end_index = start_index + samples_per_gpu
        if rank == world_size - 1:
            end_index = original_dataset_size  # Make sure the last GPU gets the remaining data

        # Slice the data
        dataset.prompts = dataset.prompts[start_index:end_index]
        dataset.video_paths = dataset.video_paths[start_index:end_index]
        
        # Handle model-specific data splitting
        if args.model_type == "cogvideox_pose":
            if hasattr(dataset, 'hand_video_paths'):
                dataset.hand_video_paths = dataset.hand_video_paths[start_index:end_index]
            if hasattr(dataset, 'hand_video_gray_paths'):
                dataset.hand_video_gray_paths = dataset.hand_video_gray_paths[start_index:end_index]
            if hasattr(dataset, 'static_video_paths'):
                dataset.static_video_paths = dataset.static_video_paths[start_index:end_index]
    else:
        pass

    rank_dataset_size = len(dataset)
    
    # Check if dataset is empty for this rank
    if rank_dataset_size == 0:
        print(f"\n{'='*80}")
        print(f"⚠️  Rank {rank}: No valid data found. Skipping processing for this GPU.")
        print(f"{'='*80}")
        print(f"📂 Dataset data_root: {dataset.data_root}")
        print(f"🔧 Split hands enabled: {dataset.split_hands if hasattr(dataset, 'split_hands') else 'N/A'}")
        print(f"🔧 Use gray hand videos: {dataset.use_gray_hand_videos if hasattr(dataset, 'use_gray_hand_videos') else 'N/A'}")
        print(f"📊 Original dataset size: {original_dataset_size}")
        print(f"📊 Assigned range for this rank: {start_index if world_size > 1 else 0} to {end_index if world_size > 1 else original_dataset_size}")
        
        # Show what paths were expected
        if hasattr(dataset, 'video_paths') and dataset.video_paths is not None and len(dataset.video_paths) == 0:
            print(f"⚠️  Video paths list is empty")
        if hasattr(dataset, 'hand_video_gray_paths') and dataset.hand_video_gray_paths is not None and len(dataset.hand_video_gray_paths) == 0:
            print(f"⚠️  Hand video gray paths list is empty")
        if hasattr(dataset, 'hand_video_paths') and dataset.hand_video_paths is not None and len(dataset.hand_video_paths) == 0:
            print(f"⚠️  Hand video paths list is empty")
        if hasattr(dataset, 'static_video_paths') and dataset.static_video_paths is not None and len(dataset.static_video_paths) == 0:
            print(f"⚠️  Static video paths list is empty")
            
        print(f"\n💡 This can happen when:")
        print(f"   - split_hands=True and some GPUs don't have valid left/right hand videos")
        print(f"   - Data distribution across GPUs results in empty subsets")
        print(f"   - Required condition files (hand videos, static videos) are missing")
        print(f"{'='*80}\n")
        
        # Exit gracefully
        if world_size > 1:
            dist.destroy_process_group()
        return

    # Look for human_motions folder in the data_root
    human_motions_folder = Path(args.data_root) / "human_motions"
    if human_motions_folder.exists():
        print(f"📁 Found human_motions folder: {human_motions_folder}")
        human_motions_data = human_motions_folder
    else:
        print(f"ℹ️  Human motions folder not found: {human_motions_folder}")
        print(f"ℹ️  Expected location: {human_motions_folder}")
        human_motions_data = None

    # 2. Dataloader
    def collate_fn(batch):
        videos = torch.stack([item["video"] for item in batch]).to(dtype=weight_dtype, non_blocking=True)

        # Model-specific data collation
        if args.model_type == "aether":
            images = None
            images_goal = None
            if args.save_image_latents:
                images = torch.stack([item["image"] for item in batch]).to(dtype=weight_dtype, non_blocking=True)
                images_goal = torch.stack([item["image_goal"] for item in batch]).to(dtype=weight_dtype, non_blocking=True)

            disparity = torch.stack([item["disparity"] for item in batch]).to(dtype=weight_dtype, non_blocking=True)
            raymap = torch.stack([item["raymap"] for item in batch]).to(dtype=weight_dtype, non_blocking=True)
            raymap_abs = torch.stack([item["raymap_abs"] for item in batch]).to(dtype=weight_dtype, non_blocking=True)

            return {
                "images": images,
                "images_goal": images_goal,
                "videos": videos,
                "disparity": disparity,
                "raymap": raymap,
                "raymap_abs": raymap_abs,
            }
        else:  # cogvideox_pose
            # Add hand_videos, static_videos, and smpl_pos_map if they exist in the dataset
            hand_videos = None
            static_videos = None
            smpl_pos_map = None
            
            if "hand_videos" in batch[0] and batch[0]["hand_videos"] is not None:
                hand_videos = torch.stack([item["hand_videos"] for item in batch]).to(dtype=weight_dtype, non_blocking=True)
            
            if "static_videos" in batch[0] and batch[0]["static_videos"] is not None:
                static_videos = torch.stack([item["static_videos"] for item in batch]).to(dtype=weight_dtype, non_blocking=True)
            
            if "smpl_pos_map" in batch[0] and batch[0]["smpl_pos_map"] is not None:
                smpl_pos_map = torch.stack([item["smpl_pos_map"] for item in batch]).to(dtype=weight_dtype, non_blocking=True)

            return {
                "videos": videos,
                "hand_videos": hand_videos,
                "static_videos": static_videos,
                "smpl_pos_map": smpl_pos_map,
            }

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
        pin_memory=args.pin_memory,
    )

    # 3. Prepare models
    device = f"cuda:{rank}"

    if args.save_latents_and_embeddings:
        vae = AutoencoderKLCogVideoX.from_pretrained(args.model_id, subfolder="vae", torch_dtype=weight_dtype)
        vae = vae.to(device)

        if args.use_slicing:
            vae.enable_slicing()
        if args.use_tiling:
            vae.enable_tiling()
    
    # Initialize text encoder and tokenizer for prompt embeddings
    tokenizer = None
    text_encoder = None
    if args.save_prompt_embeds:
        tokenizer = T5Tokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
        text_encoder = T5EncoderModel.from_pretrained(
            args.model_id, subfolder="text_encoder", torch_dtype=weight_dtype
        )
        text_encoder = text_encoder.to(device)

    # 4. Compute latents and embeddings and save
    total_steps = (rank_dataset_size + args.batch_size - 1) // args.batch_size
    if rank == 0:
        iterator = tqdm(
            dataloader, desc="Encoding", total=total_steps
        )
    else:
        iterator = dataloader

    for step, batch in enumerate(iterator):
        # Calculate current video index for progress reporting
        current_video_idx = step * args.batch_size
        if rank == 0 and step % 10 == 0:  # Update every 10 steps to avoid spam
            print(f"Processing video {current_video_idx:05d} of {rank_dataset_size} (step {step}/{total_steps})")
        try:
            # Initialize variables for both model types
            images = None
            images_goal = None
            image_latents = None
            image_goal_latents = None
            video_latents = None
            disparity_latents = None
            raymap = None
            raymap_abs = None
            hand_videos = None
            hand_video_latents = None
            hand_mask_videos = None
            static_videos = None
            static_video_latents = None
            pose_params = None
            prompts = None
            prompt_embeds = None

            # Common processing for both models
            videos = batch["videos"].to(device, non_blocking=True)
            videos = videos.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]

            # Model-specific processing
            if args.model_type == "aether":
                # Aether-specific processing
                if args.save_image_latents:
                    images = batch["images"].to(device, non_blocking=True)
                    images = images.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]

                    images_goal = batch["images_goal"].to(device, non_blocking=True)
                    images_goal = images_goal.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]

                disparity = batch["disparity"].to(device, non_blocking=True)
                disparity = disparity.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]

                # Encode videos & images for Aether
                if args.save_latents_and_embeddings:
                    # Check if we should process image latents
                    should_process_images = (args.save_image_latents and 
                                           (args.selective_processing is None or 
                                            any(ft in args.selective_processing for ft in ["images", "image_latents", "images_goal", "image_goal_latents"])))
                    
                    # Check if we should process video latents
                    should_process_videos = (args.selective_processing is None or 
                                           any(ft in args.selective_processing for ft in ["video_latents", "disparity_latents", "prompt_embeds"]))
                    
                    if args.use_slicing:
                        if should_process_images and args.save_image_latents:
                            encoded_slices = [vae._encode(image_slice) for image_slice in images.split(1)]
                            image_latents = torch.cat(encoded_slices)
                            image_latents = image_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)

                        if should_process_videos:
                            encoded_slices = [vae._encode(video_slice) for video_slice in videos.split(1)]
                            video_latents = torch.cat(encoded_slices)

                    else:
                        if should_process_images and args.save_image_latents:
                            image_latents = vae._encode(images)
                            image_latents = image_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)

                            image_goal_latents = vae._encode(images_goal)
                            image_goal_latents = image_goal_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)

                        if should_process_videos:
                            video_latents = vae._encode(videos)

                    if should_process_videos:
                        video_latents = video_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)

                    # For video format, normalize to [0, 1] before sqrt operation
                    if args.disparity_format == "video" and disparity.max() > 1.0:
                        # sqrt already done in exr_to_disparity.py
                        disparity = disparity / 255.0

                    # Disparity latents
                    disparity_temp = disparity.clone()
                    disparity_temp = 2 * disparity_temp - 1  # Normalize to [-1, 1]
                    disparity_latents = vae._encode(disparity_temp)

                # Process raymaps if requested
                should_process_raymaps = (args.selective_processing is None or "raymaps" in args.selective_processing)
                should_process_raymaps_abs = (args.selective_processing is None or "raymaps_abs" in args.selective_processing)
                
                if should_process_raymaps:
                    raymap = batch["raymap"].to(device, non_blocking=True)
                else:
                    raymap = None
                    
                if should_process_raymaps_abs:
                    raymap_abs = batch["raymap_abs"].to(device, non_blocking=True)
                else:
                    raymap_abs = None

            else:  # cogvideox_pose
                # CogVideoX pose-specific processing
                if batch["hand_videos"] is not None:
                    hand_videos = batch["hand_videos"].to(device, non_blocking=True)
                
                if batch["static_videos"] is not None:
                    static_videos = batch["static_videos"].to(device, non_blocking=True)
                
                if batch["smpl_pos_map"] is not None:
                    smpl_pos_map = batch["smpl_pos_map"].to(device, non_blocking=True)

                # Encode videos for CogVideoX pose
                if args.save_latents_and_embeddings:
                    # Check if we should process video latents
                    should_process_videos = (args.selective_processing is None or 
                                           any(ft in args.selective_processing for ft in ["video_latents", "prompt_embeds"]))
                    
                    if args.use_slicing:
                        if should_process_videos:
                            encoded_slices = [vae._encode(video_slice) for video_slice in videos.split(1)]
                            video_latents = torch.cat(encoded_slices)
                    else:
                        if should_process_videos:
                            video_latents = vae._encode(videos)

                    if should_process_videos:
                        video_latents = video_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)

                        if hand_videos is not None:
                            hand_video_latents = vae._encode(hand_videos)
                            hand_video_latents = hand_video_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)
                        
                        if static_videos is not None:
                            static_video_latents = vae._encode(static_videos)
                            static_video_latents = static_video_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)
                        
                        if smpl_pos_map is not None:
                            smpl_pos_map_latents = vae._encode(smpl_pos_map)
                            smpl_pos_map_latents = smpl_pos_map_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)

            # Process images for Aether model
            should_process_images = False
            if args.model_type == "aether" and images is not None:
                # Only process images if they're in selective_processing or if no selective processing is specified
                should_process_images = (args.selective_processing is None or 
                                       any(ft in args.selective_processing for ft in ["images", "images_goal"]))
                
                if should_process_images:
                    images = (images.permute(0, 2, 1, 3, 4) + 1) / 2
                    images_goal = (images_goal.permute(0, 2, 1, 3, 4) + 1) / 2
                else:
                    # Set to None to skip saving
                    images = None
                    images_goal = None

            # Make all data types selective - only process what's requested
            # When selective_processing is None, process everything
            # When selective_processing is specified, only process the requested types
            should_process_videos = (args.selective_processing is None or "videos" in args.selective_processing)
            
            # Process videos if requested
            if should_process_videos:
                videos = (videos.permute(0, 2, 1, 3, 4) + 1) / 2
            else:
                videos = None

            # Model-specific selective processing
            if args.model_type == "aether":
                should_process_disparity = (args.selective_processing is None or "disparity" in args.selective_processing)
                should_process_raymaps = (args.selective_processing is None or "raymaps" in args.selective_processing)
                should_process_raymaps_abs = (args.selective_processing is None or "raymaps_abs" in args.selective_processing)
                
                # Process disparity if requested
                if should_process_disparity:
                    disparity = disparity.permute(0, 2, 1, 3, 4)
                else:
                    disparity = None
                
                # Process human_motions data if available
                human_motions = None
                if human_motions_data is not None:
                    # Check if we should process human_motions based on selective_processing
                    should_process_human_motions = (args.selective_processing is None or 
                                                  "human_motions" in args.selective_processing)
                    
                    if should_process_human_motions:
                        # Calculate the start index for this batch
                        batch_start_idx = step * args.batch_size
                        batch_end_idx = min(batch_start_idx + args.batch_size, len(dataset.video_paths))
                        
                        # Load human_motions data for this batch
                        human_motions_batch = []
                        for i in range(batch_start_idx, batch_end_idx):
                            human_motion_path = human_motions_data / f"{i:05}.npz"
                            if human_motion_path.exists():
                                try:
                                    # Load NPZ file and convert to tensor
                                    npz_data = np.load(human_motion_path)
                                    # Convert all arrays in the NPZ to tensors
                                    human_motion_tensors = {}
                                    for key in npz_data.files:
                                        human_motion_tensors[key] = torch.from_numpy(npz_data[key]).to(device, dtype=weight_dtype)
                                    human_motions_batch.append(human_motion_tensors)
                                except Exception as e:
                                    print(f"Warning: Could not load human_motion file {human_motion_path}: {e}")
                                    human_motions_batch.append(None)
                            else:
                                print(f"Warning: Human motion file not found: {human_motion_path}")
                                human_motions_batch.append(None)
                        
                        # Only set human_motions if we have valid data
                        if any(h is not None for h in human_motions_batch):
                            human_motions = human_motions_batch

            else:  # cogvideox_pose
                should_process_hand_videos = (args.selective_processing is None or "hand_video_latents" in args.selective_processing)
                should_process_hand_gray_left = (args.selective_processing is None or "hand_video_gray_left_latents" in args.selective_processing)
                should_process_hand_gray_right = (args.selective_processing is None or "hand_video_gray_right_latents" in args.selective_processing)
                should_process_static_videos = (args.selective_processing is None or "static_video_latents" in args.selective_processing)
                should_process_hand_masks = (args.selective_processing is None or "videos_hands_mask" in args.selective_processing)
                should_process_smpl_pos_map = (args.selective_processing is None or "smpl_pos_map_egoallo_latents" in args.selective_processing)
                
                # Need hand_videos for left/right processing even if not saving hand_video_latents
                need_hand_videos = should_process_hand_videos or should_process_hand_gray_left or should_process_hand_gray_right or should_process_hand_masks
                
                # Process hand_videos if needed
                if need_hand_videos and hand_videos is not None:
                    hand_videos_processed = (hand_videos.permute(0, 2, 1, 3, 4) + 1) / 2
                    
                    # Create hand mask from hand videos
                    if should_process_hand_masks:
                        hand_mask_videos = create_hand_mask(hand_videos_processed)
                        print(f"Created hand mask videos with shape: {hand_mask_videos.shape}")
                    else:
                        hand_mask_videos = None
                    
                    # Keep hand_videos_processed for later use (left/right splitting)
                    if should_process_hand_videos:
                        hand_videos = hand_videos_processed
                    else:
                        # Don't save hand_videos, but keep it for left/right splitting
                        hand_videos_for_splitting = hand_videos
                        hand_videos = None
                else:
                    hand_videos = None
                    hand_videos_for_splitting = None
                    hand_mask_videos = None
                
                # Process left/right gray hand videos if requested (additional to regular gray videos)
                # Use hand_videos if it's available, otherwise use hand_videos_for_splitting
                source_hand_videos = hand_videos if hand_videos is not None else hand_videos_for_splitting
                if should_process_hand_gray_left and source_hand_videos is not None:
                    # Split the concatenated hand videos back into left and right
                    batch_size, frames, total_channels, height, width = source_hand_videos.shape
                    channels_per_hand = total_channels // 2
                    
                    # Left hand videos (first half of channels)
                    hand_videos_gray_left = source_hand_videos[:, :, :channels_per_hand, :, :]
                    hand_videos_gray_left = (hand_videos_gray_left.permute(0, 2, 1, 3, 4) + 1) / 2
                    print(f"Created left hand videos with shape: {hand_videos_gray_left.shape}")
                else:
                    hand_videos_gray_left = None
                
                if should_process_hand_gray_right and source_hand_videos is not None:
                    # Right hand videos (second half of channels)
                    hand_videos_gray_right = source_hand_videos[:, :, channels_per_hand:, :, :]
                    hand_videos_gray_right = (hand_videos_gray_right.permute(0, 2, 1, 3, 4) + 1) / 2
                    print(f"Created right hand videos with shape: {hand_videos_gray_right.shape}")
                else:
                    hand_videos_gray_right = None
                    
                # Process static_videos if requested
                if should_process_static_videos and static_videos is not None:
                    static_videos = (static_videos.permute(0, 2, 1, 3, 4) + 1) / 2
                else:
                    static_videos = None
                
                # Process smpl_pos_map if requested
                if should_process_smpl_pos_map and smpl_pos_map is not None:
                    smpl_pos_map = (smpl_pos_map.permute(0, 2, 1, 3, 4) + 1) / 2
                    print(f"Processed SMPL pos map with shape: {smpl_pos_map.shape}")
                else:
                    smpl_pos_map = None

            # Encode videos for CogVideoX pose (after data processing so hand_videos_gray_left/right are available)
            if args.model_type == "cogvideox_pose" and args.save_latents_and_embeddings:
                # Check if we should process video latents
                should_process_videos = (args.selective_processing is None or 
                                       any(ft in args.selective_processing for ft in ["video_latents", "prompt_embeds"]))
                video_latents = None
                if should_process_videos:
                    if args.use_slicing:
                        encoded_slices = [vae._encode(video_slice) for video_slice in videos.split(1)]
                        video_latents = torch.cat(encoded_slices)
                    else:
                        video_latents = vae._encode(videos)
                    video_latents = video_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)

                should_process_hand_videos = (args.selective_processing is None or "hand_video_latents" in args.selective_processing)
                should_process_hand_gray_videos = (args.selective_processing is None or "hand_video_gray_latents" in args.selective_processing)
                should_process_hand_gray_left = (args.selective_processing is None or "hand_video_gray_left_latents" in args.selective_processing)
                should_process_hand_gray_right = (args.selective_processing is None or "hand_video_gray_right_latents" in args.selective_processing)
                should_process_static_videos_latents = (args.selective_processing is None or "static_video_latents" in args.selective_processing)
                should_process_smpl_pos_map_latents = (args.selective_processing is None or "smpl_pos_map_egoallo_latents" in args.selective_processing)

                # Process hand videos based on use_gray_hand_videos flag
                hand_video_latents = None
                hand_video_gray_latents = None
                hand_video_gray_left_latents = None
                hand_video_gray_right_latents = None
                if args.use_gray_hand_videos:
                    # Use gray hand videos (stored in hand_videos)
                    if should_process_hand_gray_videos and hand_videos is not None:
                        hand_video_gray_latents = vae._encode(hand_videos)
                        hand_video_gray_latents = hand_video_gray_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)
                        print(f"Encoded gray hand video latents with shape: {hand_video_gray_latents.shape}")
                    
                    # Additionally process left/right separately if requested
                    if should_process_hand_gray_left and hand_videos_gray_left is not None:
                        hand_video_gray_left_latents = vae._encode(hand_videos_gray_left)
                        hand_video_gray_left_latents = hand_video_gray_left_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)
                        print(f"Encoded left hand video latents with shape: {hand_video_gray_left_latents.shape}")
                    
                    if should_process_hand_gray_right and hand_videos_gray_right is not None:
                        hand_video_gray_right_latents = vae._encode(hand_videos_gray_right)
                        hand_video_gray_right_latents = hand_video_gray_right_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)
                        print(f"Encoded right hand video latents with shape: {hand_video_gray_right_latents.shape}")
                else:
                    # Use colored hand videos
                    if should_process_hand_videos and hand_videos is not None:
                        hand_video_latents = vae._encode(hand_videos)
                        hand_video_latents = hand_video_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)
                
                static_video_latents = None
                if should_process_static_videos_latents and static_videos is not None:
                    static_video_latents = vae._encode(static_videos)
                    static_video_latents = static_video_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)
                
                smpl_pos_map_latents = None
                if should_process_smpl_pos_map and smpl_pos_map is not None:
                    smpl_pos_map_latents = vae._encode(smpl_pos_map)
                    smpl_pos_map_latents = smpl_pos_map_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)
                    print(f"Encoded SMPL pos map latents with shape: {smpl_pos_map_latents.shape}")

            # Get video paths for this batch (always needed for filename generation)
            batch_video_paths = []
            if hasattr(dataset, 'video_paths'):
                # Calculate the start index for this batch
                batch_start_idx = step * args.batch_size
                batch_end_idx = min(batch_start_idx + args.batch_size, len(dataset.video_paths))
                batch_video_paths = dataset.video_paths[batch_start_idx:batch_end_idx]
            
            # Load prompts for this batch
            if args.save_prompt_embeds:
                batch_prompts = []
                for video_path in batch_video_paths:
                    # Convert video path to prompt path: sequences/videos/00000.mp4 -> sequences/prompts/00000.txt
                    video_name = video_path.stem  # Get filename without extension
                    prompt_path = video_path.parent.parent / "prompts" / f"{video_name}.txt"
                    
                    if prompt_path.exists():
                        try:
                            with open(prompt_path, 'r', encoding='utf-8') as f:
                                prompt = f.read().strip()
                            batch_prompts.append(prompt)
                        except Exception as e:
                            print(f"Warning: Could not read prompt file {prompt_path}: {e}")
                            batch_prompts.append("")  # Empty prompt as fallback
                    else:
                        print(f"Warning: Prompt file not found: {prompt_path}")
                        batch_prompts.append("")  # Empty prompt as fallback
                
                prompts = batch_prompts
                
                # Generate prompt embeddings
                if prompts and any(p.strip() for p in prompts):  # Only process if we have non-empty prompts
                    prompt_embeds = compute_prompt_embeddings(
                        tokenizer,
                        text_encoder,
                        prompts,
                        args.max_sequence_length,
                        device,
                        weight_dtype,
                        requires_grad=False,
                    )
            
            # Build output data dictionary with only processed data
            output_data = {
                "batch_size": len(batch["videos"]) if "videos" in batch else 1,
                "fps": target_fps,
                "scene_name": args.scene_name,
                "action_name": args.action_name,
                "video_paths": batch_video_paths,  # Pass video paths
                "skip_existing": args.skip_existing,  # Pass skip_existing flag
                "selective_processing": args.selective_processing, # Pass selective_processing
                "model_type": args.model_type,  # Pass model type
            }
            
            # Only include directories and data that were actually processed
            # When selective_processing is None, include all data
            # When selective_processing is specified, only include requested data
            if should_process_videos or args.selective_processing is None:
                output_data.update({
                    "videos_dir": videos_dir,
                    "video_latents_dir": video_latents_dir,
                    "videos": videos,
                    "video_latents": video_latents,
                })
            
            # Add prompt-related data
            should_process_prompts = (args.selective_processing is None or 
                                    any(ft in args.selective_processing for ft in ["prompts", "prompt_embeds"]))
            if should_process_prompts or args.selective_processing is None:
                output_data.update({
                    "prompts_dir": prompts_dir,
                    "prompt_embeds_dir": prompt_embeds_dir,
                    "prompts": prompts,
                    "prompt_embeds": prompt_embeds,
                })
            
            # Model-specific output data
            if args.model_type == "aether":
                if should_process_images or args.selective_processing is None:
                    output_data.update({
                        "images_dir": images_dir,
                        "image_latents_dir": image_latents_dir,
                        "images_goal_dir": images_goal_dir,
                        "image_goal_latents_dir": image_goal_latents_dir,
                        "images": images,
                        "image_latents": image_latents,
                        "images_goal": images_goal,
                        "image_goal_latents": image_goal_latents,
                    })
                
                if should_process_disparity or args.selective_processing is None:
                    output_data.update({
                        "disparity_dir": disparity_dir,
                        "disparity_latents_dir": disparity_latents_dir,
                        "disparity": disparity,
                        "disparity_latents": disparity_latents,
                    })
                
                if should_process_raymaps or args.selective_processing is None:
                    output_data.update({
                        "raymap_dir": raymap_dir,
                        "raymap": raymap,
                    })
                
                if should_process_raymaps_abs or args.selective_processing is None:
                    output_data.update({
                        "raymap_abs_dir": raymap_abs_dir,
                        "raymap_abs": raymap_abs,
                    })
                
                if should_process_human_motions or args.selective_processing is None:
                    output_data.update({
                        "human_motions_dir": human_motions_dir,
                        "human_motions": human_motions,
                    })
            
            else:  # cogvideox_pose
                # Define processing flags for cogvideox_pose
                should_process_hand_videos = (args.selective_processing is None or "hand_video_latents" in args.selective_processing)
                should_process_hand_gray_videos = (args.selective_processing is None or "hand_video_gray_latents" in args.selective_processing)
                should_process_hand_gray_left = (args.selective_processing is None or "hand_video_gray_left_latents" in args.selective_processing)
                should_process_hand_gray_right = (args.selective_processing is None or "hand_video_gray_right_latents" in args.selective_processing)
                should_process_hand_masks = (args.selective_processing is None or "videos_hands_mask" in args.selective_processing)
                should_process_static_videos = (args.selective_processing is None or "static_video_latents" in args.selective_processing)
                should_process_smpl_pos_map = (args.selective_processing is None or "smpl_pos_map_egoallo_latents" in args.selective_processing)
                
                # Process hand videos based on use_gray_hand_videos flag
                if args.use_gray_hand_videos:
                    # Use gray hand videos (stored in hand_videos)
                    if should_process_hand_gray_videos or args.selective_processing is None:
                        output_data.update({
                            "hand_videos_dir": hand_videos_gray_dir,  # Use gray directory
                            "hand_video_latents_dir": hand_video_gray_latents_dir,  # Use gray latents directory
                            "hand_videos": hand_videos,  # gray videos stored in hand_videos
                            "hand_video_latents": hand_video_gray_latents,  # gray latents stored in hand_video_latents
                        })
                    
                    # Additionally process left/right separately if requested
                    if should_process_hand_gray_left or args.selective_processing is None:
                        output_data.update({
                            "hand_videos_gray_left_dir": hand_videos_gray_left_dir,
                            "hand_video_gray_left_latents_dir": hand_video_gray_left_latents_dir,
                            "hand_videos_gray_left": hand_videos_gray_left.permute(0, 2, 1, 3, 4),
                            "hand_video_gray_left_latents": hand_video_gray_left_latents,
                        })
                    
                    if should_process_hand_gray_right or args.selective_processing is None:
                        output_data.update({
                            "hand_videos_gray_right_dir": hand_videos_gray_right_dir,
                            "hand_video_gray_right_latents_dir": hand_video_gray_right_latents_dir,
                            "hand_videos_gray_right": hand_videos_gray_right.permute(0, 2, 1, 3, 4),
                            "hand_video_gray_right_latents": hand_video_gray_right_latents,
                        })
                else:
                    # Use colored hand videos (with configurable subdirectory names)
                    if should_process_hand_videos or args.selective_processing is None:
                        output_data.update({
                            "hand_videos_dir": hand_videos_dir,  # Uses args.hand_video_subdir
                            "hand_video_latents_dir": hand_video_latents_dir,  # Uses args.hand_video_latents_subdir
                            "hand_videos": hand_videos.permute(0, 2, 1, 3, 4),
                            "hand_video_latents": hand_video_latents,
                        })
                
                if should_process_hand_masks or args.selective_processing is None:
                    output_data.update({
                        "hand_mask_videos_dir": hand_mask_videos_dir,
                        "hand_mask_videos": hand_mask_videos,
                    })
                
                if should_process_static_videos or args.selective_processing is None:
                    output_data.update({
                        "static_videos_dir": static_videos_dir,
                        "static_video_latents_dir": static_video_latents_dir,
                        "static_videos": static_videos,
                        "static_video_latents": static_video_latents,
                    })
                
                if should_process_smpl_pos_map or args.selective_processing is None:
                    output_data.update({
                        "smpl_pos_map_dir": smpl_pos_map_dir,
                        "smpl_pos_map_latents_dir": smpl_pos_map_latents_dir,
                        "smpl_pos_map": smpl_pos_map,
                        "smpl_pos_map_latents": smpl_pos_map_latents,
                    })
            output_queue.put(output_data)

        except Exception:
            print("-------------------------")
            print(f"An exception occurred while processing data: {rank=}, {world_size=}, {step=}")
            traceback.print_exc()
            print("-------------------------")

    # 5. Complete distributed processing
    if world_size > 1:
        dist.destroy_process_group()

    output_queue.put(None)
    save_thread.shutdown(wait=True)
    save_future.result()

    # 6. Combine results from each rank
    if rank == 0:
        print(
            f"Completed preprocessing latents and embeddings. Temporary files from all ranks saved to `{tmp_dir.as_posix()}`"
        )

        # Move files from each rank to common directory
        # Common folders for both model types
        common_folders = [
            ("videos", "mp4"),
            ("video_latents", "pt"),
            ("videos", "txt"),
            ("prompts", "txt"),
            ("prompt_embeds", "pt"),
        ]
        
        # Model-specific folders
        if args.model_type == "aether":
            model_folders = [
                ("images", "png"),
                ("image_latents", "pt"),
                ("images_goal", "png"),
                ("image_goal_latents", "pt"),
                ("disparity", "mp4"),
                ("disparity_latents", "pt"),
                ("raymaps", "pt"),
                ("raymaps_abs", "pt"),
                ("human_motions", "pt"),
            ]
        else:  # cogvideox_pose
            # Use configurable subdirectory names for default mode
            model_folders = [
                (args.hand_video_subdir, "mp4"),  # Configurable hand videos directory
                (args.hand_video_latents_subdir, "pt"),  # Configurable hand video latents directory
                ("videos_hands_gray", "mp4"),
                ("hand_video_gray_latents", "pt"),
                ("videos_hands_gray_left", "mp4"),
                ("hand_video_gray_left_latents", "pt"),
                ("videos_hands_gray_right", "mp4"),
                ("hand_video_gray_right_latents", "pt"),
                ("videos_hands_mask", "mp4"),
                ("videos_static", "mp4"),
                ("static_video_latents", "pt"),
                ("smpl_pos_map_egoallo", "mp4"),
                ("smpl_pos_map_egoallo_latents", "pt"),
            ]
        
        all_folders = common_folders + model_folders
        
        for subfolder, extension in all_folders:
            tmp_subfolder = tmp_dir.joinpath(subfolder)
            combined_subfolder = output_dir.joinpath(subfolder)
            combined_subfolder.mkdir(parents=True, exist_ok=True)
            pattern = f"*.{extension}"

            for file in tmp_subfolder.rglob(pattern):
                file.replace(combined_subfolder / file.name)

        # Remove temporary directories
        def rmdir_recursive(dir: pathlib.Path) -> None:
            for child in dir.iterdir():
                if child.is_file():
                    child.unlink()
                else:
                    rmdir_recursive(child)
            dir.rmdir()

        rmdir_recursive(tmp_dir)

        # Combine videos into individual text files and single jsonl
        videos_txt = output_dir.joinpath("videos.txt")
        data_jsonl = output_dir.joinpath("data.jsonl")

        # Get video stems from the videos directory
        videos_folder = output_dir.joinpath("videos")
        stems = []
        for filename in videos_folder.rglob("*.mp4"):
            stems.append(filename.stem)

        with open(videos_txt, "w") as file:
            for stem in stems:
                file.write(f"videos/{stem}.mp4\n")

        with open(data_jsonl, "w") as file:
            for stem in stems:
                video_metadata_txt = output_dir.joinpath(f"videos/{stem}.txt")
                with open(video_metadata_txt, "r", encoding="utf-8") as metadata_file:
                    metadata = json.loads(metadata_file.read())

                data = {
                    "metadata": metadata,
                }
                
                # Common file types for both models
                if args.selective_processing is None or "videos" in args.selective_processing:
                    data["video"] = f"videos/{stem}.mp4"
                if args.selective_processing is None or "video_latents" in args.selective_processing:
                    data["video_latent"] = f"video_latents/{stem}.pt"
                if args.selective_processing is None or "prompts" in args.selective_processing:
                    data["prompt"] = f"prompts/{stem}.txt"
                if args.selective_processing is None or "prompt_embeds" in args.selective_processing:
                    data["prompt_embed"] = f"prompt_embeds/{stem}.pt"
                
                # Model-specific file types
                if args.model_type == "aether":
                    if args.selective_processing is None or "disparity" in args.selective_processing:
                        data["disparity"] = f"disparity/{stem}.mp4"
                    if args.selective_processing is None or "raymaps" in args.selective_processing:
                        data["raymap"] = f"raymaps/{stem}.pt"
                    if args.selective_processing is None or "raymaps_abs" in args.selective_processing:
                        data["raymap_abs"] = f"raymaps_abs/{stem}.pt"
                    if args.selective_processing is None or "images" in args.selective_processing:
                        data["image"] = f"images/{stem}.png"
                    if args.selective_processing is None or "image_latents" in args.selective_processing:
                        data["image_latent"] = f"image_latents/{stem}.pt"
                    if args.selective_processing is None or "images_goal" in args.selective_processing:
                        data["image_goal"] = f"images_goal/{stem}.png"
                    if args.selective_processing is None or "image_goal_latents" in args.selective_processing:
                        data["image_goal_latent"] = f"image_goal_latents/{stem}.pt"
                    if args.selective_processing is None or "disparity_latents" in args.selective_processing:
                        data["disparity_latent"] = f"disparity_latents/{stem}.pt"
                    if args.selective_processing is None or "human_motions" in args.selective_processing:
                        data["human_motion"] = f"human_motions/{stem}.pt"
                
                else:  # cogvideox_pose
                    # Use configurable subdirectory names for default mode
                    if args.selective_processing is None or "hand_videos" in args.selective_processing:
                        data["hand_video"] = f"{args.hand_video_subdir}/{stem}.mp4"
                    if args.selective_processing is None or "hand_video_latents" in args.selective_processing:
                        data["hand_video_latent"] = f"{args.hand_video_latents_subdir}/{stem}.pt"
                    if args.selective_processing is None or "videos_hands_gray" in args.selective_processing:
                        data["hand_video_gray"] = f"videos_hands_gray/{stem}.mp4"
                    if args.selective_processing is None or "hand_video_gray_latents" in args.selective_processing:
                        data["hand_video_gray_latent"] = f"hand_video_gray_latents/{stem}.pt"
                    if args.selective_processing is None or "videos_hands_gray_left" in args.selective_processing:
                        data["hand_video_gray_left"] = f"videos_hands_gray_left/{stem}.mp4"
                    if args.selective_processing is None or "hand_video_gray_left_latents" in args.selective_processing:
                        data["hand_video_gray_left_latent"] = f"hand_video_gray_left_latents/{stem}.pt"
                    if args.selective_processing is None or "videos_hands_gray_right" in args.selective_processing:
                        data["hand_video_gray_right"] = f"videos_hands_gray_right/{stem}.mp4"
                    if args.selective_processing is None or "hand_video_gray_right_latents" in args.selective_processing:
                        data["hand_video_gray_right_latent"] = f"hand_video_gray_right_latents/{stem}.pt"
                    if args.selective_processing is None or "hand_mask_videos" in args.selective_processing:
                        data["hand_mask_video"] = f"videos_hands_mask/{stem}.mp4"
                    if args.selective_processing is None or "static_videos" in args.selective_processing:
                        data["static_video"] = f"videos_static/{stem}.mp4"
                    if args.selective_processing is None or "static_video_latents" in args.selective_processing:
                        data["static_video_latent"] = f"static_video_latents/{stem}.pt"
                    if args.selective_processing is None or "smpl_pos_map_egoallo" in args.selective_processing:
                        data["smpl_pos_map"] = f"smpl_pos_map_egoallo/{stem}.mp4"
                    if args.selective_processing is None or "smpl_pos_map_egoallo_latents" in args.selective_processing:
                        data["smpl_pos_map_latent"] = f"smpl_pos_map_egoallo_latents/{stem}.pt"
                
                file.write(json.dumps(data) + "\n")

        print(f"Completed preprocessing. All files saved to `{output_dir.as_posix()}`")
        print(f"Model type: {args.model_type}")
        print(f"Using Trumans naming convention: {args.scene_name[:8]}_{args.action_name}_XXXXX.ext")


if __name__ == "__main__":
    main() 