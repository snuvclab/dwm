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

import torch
import torch.distributed as dist
from diffusers import AutoencoderKLCogVideoX
from diffusers.training_utils import set_seed
from diffusers.utils import export_to_video, get_logger
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


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
    
    Args:
        scene_name: Full scene name (e.g., 0a7618195-4647-8896747201b1)
        action_name: Action name (e.g., 20231-14@22-06-10)
        video_name: Video name without extension (e.g., '00000')
        extension: File extension (e.g., 'mp4', 'pt', 'png')
    
    Returns:
        Formatted filename (e.g., 0a761819_20231-14_22-06-10_00000.mp4')
    """
    # Extract first 8 characters of scene name
    scene_prefix = scene_name[:8]
    
    # Generate filename
    filename = f"{scene_prefix}_{action_name}_{video_name}.{extension}"
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
        help="Only process specific file types. Useful when some files already exist. Options: images, image_latents, images_goal, image_goal_latents, videos, video_latents, disparity, disparity_latents, raymaps, raymaps_abs, prompts, human_motions",
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
) -> None:
    # Handle the case where videos is None (selective processing)
    if videos is not None:
        num_frames, height, width = videos.size(1), videos.size(3), videos.size(4)
        # Create metadata for each video in the batch
        metadata = [{"num_frames": num_frames, "height": height, "width": width} for _ in range(batch_size)]
    else:
        # For selective processing without videos, create dummy metadata
        metadata = [{"num_frames": 0, "height": 0, "width": 0} for _ in range(batch_size)]

    data_folder_mapper_list = [
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
        (human_motions, human_motions_dir, torch.save, "pt"),
        (metadata, videos_dir, save_metadata, "txt"),
    ]
    
    # Filter data_folder_mapper_list based on selective_processing
    if selective_processing:
        # Map file types to their index in data_folder_mapper_list
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
            "human_motions": 11,
        }
        
        # Keep only the requested file types
        filtered_mapper_list = []
        for file_type in selective_processing:
            if file_type in file_type_to_index:
                index = file_type_to_index[file_type]
                if index < len(data_folder_mapper_list):
                    filtered_mapper_list.append(data_folder_mapper_list[index])
        
        # Only include metadata if we're processing videos or other file types that need it
        # For human_motions only, we don't need metadata
        should_include_metadata = any(ft in selective_processing for ft in ["videos", "video_latents", "disparity", "disparity_latents", "images", "image_latents", "human_motions"])
        if should_include_metadata and 12 < len(data_folder_mapper_list):
            filtered_mapper_list.append(data_folder_mapper_list[12])
        
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
        if data is None:
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
        dist.init_process_group(backend="nccl")
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        # Single GPU
        local_rank = 0
        world_size = 1
        rank = 0
        torch.cuda.set_device(rank)

    # Create folders where intermediate tensors from each rank will be saved
    images_dir = tmp_dir.joinpath(f"images/{rank}")
    image_latents_dir = tmp_dir.joinpath(f"image_latents/{rank}")
    images_goal_dir = tmp_dir.joinpath(f"images_goal/{rank}")
    image_goal_latents_dir = tmp_dir.joinpath(f"image_goal_latents/{rank}")
    videos_dir = tmp_dir.joinpath(f"videos/{rank}")
    video_latents_dir = tmp_dir.joinpath(f"video_latents/{rank}")
    disparity_dir = tmp_dir.joinpath(f"disparity/{rank}")
    disparity_latents_dir = tmp_dir.joinpath(f"disparity_latents/{rank}")
    raymap_dir = tmp_dir.joinpath(f"raymaps/{rank}")
    raymap_abs_dir = tmp_dir.joinpath(f"raymaps_abs/{rank}")
    prompts_dir = tmp_dir.joinpath(f"prompts/{rank}")
    human_motions_dir = tmp_dir.joinpath(f"human_motions/{rank}")

    images_dir.mkdir(parents=True, exist_ok=True)
    image_latents_dir.mkdir(parents=True, exist_ok=True)
    images_goal_dir.mkdir(parents=True, exist_ok=True)
    image_goal_latents_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
    video_latents_dir.mkdir(parents=True, exist_ok=True)
    disparity_dir.mkdir(parents=True, exist_ok=True)
    disparity_latents_dir.mkdir(parents=True, exist_ok=True)
    raymap_dir.mkdir(parents=True, exist_ok=True)
    raymap_abs_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir.mkdir(parents=True, exist_ok=True)
    human_motions_dir.mkdir(parents=True, exist_ok=True)

    weight_dtype = DTYPE_MAPPING[args.dtype]
    target_fps = args.target_fps

    # 1. Dataset
    dataset_init_kwargs = {
        "data_root": args.data_root,
        "dataset_file": args.dataset_file,
        "video_column": args.video_column,
        "max_num_frames": args.max_num_frames,
        "id_token": args.id_token,
        "height_buckets": args.height_buckets,
        "width_buckets": args.width_buckets,
        "frame_buckets": args.frame_buckets,
        "load_tensors": False,
        "random_flip": args.random_flip,
        "image_to_video": args.save_image_latents,
        "disparity_format": args.disparity_format,
    }
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
    else:
        pass

    rank_dataset_size = len(dataset)

    # Look for human_motions folder in the data_root
    human_motions_folder = Path(args.data_root) / "sequences" / "human_motions"
    if human_motions_folder.exists():
        print(f"📁 Found human_motions folder: {human_motions_folder}")
        human_motions_data = human_motions_folder
    else:
        print(f"ℹ️  Human motions folder not found: {human_motions_folder}")
        print(f"ℹ️  Expected location: {human_motions_folder}")

    # 2. Dataloader
    def collate_fn(data):
        prompts = [x["prompt"] for x in data[0]]

        images = None
        images_goal = None
        if args.save_image_latents:
            images = [x["image"] for x in data[0]]
            images = torch.stack(images).to(dtype=weight_dtype, non_blocking=True)

            images_goal = [x["image_goal"] for x in data[0]]
            images_goal = torch.stack(images_goal).to(dtype=weight_dtype, non_blocking=True)

        videos = [x["video"] for x in data[0]]
        videos = torch.stack(videos).to(dtype=weight_dtype, non_blocking=True)

        disparity = [x["disparity"] for x in data[0]]
        disparity = torch.stack(disparity).to(dtype=weight_dtype, non_blocking=True)

        raymap = [x["raymap"] for x in data[0]]
        raymap = torch.stack(raymap).to(dtype=weight_dtype, non_blocking=True)

        raymap_abs = [x["raymap_abs"] for x in data[0]]
        raymap_abs = torch.stack(raymap_abs).to(dtype=weight_dtype, non_blocking=True)

        return {
            "images": images,
            "images_goal": images_goal,
            "videos": videos,
            "disparity": disparity,
            "raymap": raymap,
            "raymap_abs": raymap_abs,
            "prompts": prompts,
        }

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        sampler=BucketSampler(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False),
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
            images = None
            images_goal = None
            image_latents = None
            image_goal_latents = None
            video_latents = None
            disparity_latents = None
            raymap = None
            raymap_abs = None
            pose_params = None

            if args.save_image_latents:
                images = batch["images"].to(device, non_blocking=True)
                images = images.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]

                images_goal = batch["images_goal"].to(device, non_blocking=True)
                images_goal = images_goal.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]

            videos = batch["videos"].to(device, non_blocking=True)
            videos = videos.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]

            disparity = batch["disparity"].to(device, non_blocking=True)
            disparity = disparity.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]

            # Process prompts if requested
            should_process_prompts = (args.selective_processing is None or "prompts" in args.selective_processing)
            if should_process_prompts:
                prompts = batch["prompts"]
            else:
                prompts = None

            # Encode videos & images
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
                if should_process_raymaps:
                    raymap = batch["raymap"].to(device, non_blocking=True)
                else:
                    raymap = None
                    
                if should_process_raymaps_abs:
                    raymap_abs = batch["raymap_abs"].to(device, non_blocking=True)
                else:
                    raymap_abs = None
                
            if images is not None:
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
            should_process_disparity = (args.selective_processing is None or "disparity" in args.selective_processing)
            should_process_raymaps = (args.selective_processing is None or "raymaps" in args.selective_processing)
            should_process_raymaps_abs = (args.selective_processing is None or "raymaps_abs" in args.selective_processing)
            should_process_prompts = (args.selective_processing is None or "prompts" in args.selective_processing)

            
            # Process videos if requested
            if should_process_videos:
                videos = (videos.permute(0, 2, 1, 3, 4) + 1) / 2
            else:
                videos = None
                
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

            # Get video paths for this batch (always needed for filename generation)
            batch_video_paths = []
            if hasattr(dataset, 'video_paths'):
                # Calculate the start index for this batch
                batch_start_idx = step * args.batch_size
                batch_end_idx = min(batch_start_idx + args.batch_size, len(dataset.video_paths))
                batch_video_paths = dataset.video_paths[batch_start_idx:batch_end_idx]
            
            # Build output data dictionary with only processed data
            output_data = {
                "batch_size": len(prompts) if should_process_prompts else len(batch["videos"]) if "videos" in batch else 1,
                "fps": target_fps,
                "scene_name": args.scene_name,
                "action_name": args.action_name,
                "video_paths": batch_video_paths,  # Pass video paths
                "skip_existing": args.skip_existing,  # Pass skip_existing flag
                "selective_processing": args.selective_processing, # Pass selective_processing
            }
            
            # Only include directories and data that were actually processed
            # When selective_processing is None, include all data
            # When selective_processing is specified, only include requested data
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
            
            if should_process_videos or args.selective_processing is None:
                output_data.update({
                    "videos_dir": videos_dir,
                    "video_latents_dir": video_latents_dir,
                    "videos": videos,
                    "video_latents": video_latents,
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
            
            if should_process_prompts or args.selective_processing is None:
                output_data.update({
                    "prompts_dir": prompts_dir,
                    "prompts": prompts,
                })
            

            
            if should_process_human_motions or args.selective_processing is None:
                output_data.update({
                    "human_motions_dir": human_motions_dir,
                    "human_motions": human_motions,
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
        for subfolder, extension in [
            ("images", "png"),
            ("image_latents", "pt"),
            ("images_goal", "png"),
            ("image_goal_latents", "pt"),
            ("videos", "mp4"),
            ("video_latents", "pt"),
            ("disparity", "mp4"),
            ("disparity_latents", "pt"),
            ("raymaps", "pt"),
            ("raymaps_abs", "pt"),
            ("prompts", "txt"),
            ("human_motions", "pt"),
            ("videos", "txt"),
        ]:
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

        # Combine prompts and videos into individual text files and single jsonl
        prompts_folder = output_dir.joinpath("prompts")
        prompts = []
        stems = []

        for filename in prompts_folder.rglob("*.txt"):
            with open(filename, "r") as file:
                prompts.append(file.read().strip())
            stems.append(filename.stem)

        prompts_txt = output_dir.joinpath("prompts.txt")
        videos_txt = output_dir.joinpath("videos.txt")
        data_jsonl = output_dir.joinpath("data.jsonl")

        with open(prompts_txt, "w") as file:
            for prompt in prompts:
                file.write(f"{prompt}\n")

        with open(videos_txt, "w") as file:
            for stem in stems:
                file.write(f"videos/{stem}.mp4\n")

        with open(data_jsonl, "w") as file:
            for prompt, stem in zip(prompts, stems):
                video_metadata_txt = output_dir.joinpath(f"videos/{stem}.txt")
                with open(video_metadata_txt, "r", encoding="utf-8") as metadata_file:
                    metadata = json.loads(metadata_file.read())

                data = {
                    "prompt": prompt,
                    "metadata": metadata,
                }
                
                # Only add file types that were actually processed
                if args.selective_processing is None or "videos" in args.selective_processing:
                    data["video"] = f"videos/{stem}.mp4"
                if args.selective_processing is None or "disparity" in args.selective_processing:
                    data["disparity"] = f"disparity/{stem}.mp4"
                if args.selective_processing is None or "raymaps" in args.selective_processing:
                    data["raymap"] = f"raymaps/{stem}.pt"
                if args.selective_processing is None or "raymaps_abs" in args.selective_processing:
                    data["raymap_abs"] = f"raymaps_abs/{stem}.pt"

                
                # Only add file types that were actually processed
                if args.selective_processing is None or "images" in args.selective_processing:
                    data["image"] = f"images/{stem}.png"
                if args.selective_processing is None or "image_latents" in args.selective_processing:
                    data["image_latent"] = f"image_latents/{stem}.pt"
                if args.selective_processing is None or "images_goal" in args.selective_processing:
                    data["image_goal"] = f"images_goal/{stem}.png"
                if args.selective_processing is None or "image_goal_latents" in args.selective_processing:
                    data["image_goal_latent"] = f"image_goal_latents/{stem}.pt"
                if args.selective_processing is None or "video_latents" in args.selective_processing:
                    data["video_latent"] = f"video_latents/{stem}.pt"
                if args.selective_processing is None or "disparity_latents" in args.selective_processing:
                    data["disparity_latent"] = f"disparity_latents/{stem}.pt"

                if args.selective_processing is None or "human_motions" in args.selective_processing:
                    data["human_motion"] = f"human_motions/{stem}.pt"
                
                file.write(json.dumps(data) + "\n")

        print(f"Completed preprocessing. All files saved to `{output_dir.as_posix()}`")
        print(f"Using Trumans naming convention: {args.scene_name[:8]}_{args.action_name}_XXXXX.ext")


if __name__ == "__main__":
    main() 