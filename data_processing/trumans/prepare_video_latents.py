#!/usr/bin/env python3

import argparse
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import decord  # isort:skip
import torch
import torch.distributed as dist
from diffusers import AutoencoderKLCogVideoX
from diffusers.training_utils import set_seed
from tqdm.auto import tqdm

decord.bridge.set_bridge("torch")

logger = logging.getLogger(__name__)

DTYPE_MAPPING = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Encode videos from a dataset file into VAE latents."
    )
    parser.add_argument(
        "--dataset_file",
        type=str,
        required=True,
        help="Path to text file containing video paths (one per line).",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory containing the video data.",
    )
    parser.add_argument(
        "--video_subdir",
        type=str,
        default="videos_object",
        help="Input video subdirectory name (default: 'videos_object').",
    )
    parser.add_argument(
        "--video_latent_subdir",
        type=str,
        default="object_video_latents",
        help="Output latent subdirectory name (default: 'object_video_latents').",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["video", "image"],
        default="video",
        help="'video': encode full video. 'image': I2V setup — first frame only, pad remaining with -1 to num_frames then encode.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=49,
        help="Target frame count for image mode (first frame + -1 padding). Used only when --mode image.",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="THUDM/CogVideoX-5b",
        help="Hugging Face model ID for VAE.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for latents (defaults to data_root).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=tuple(DTYPE_MAPPING.keys()),
        default="fp32",
        help="Computation dtype for VAE.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for processing (currently only 1 supported).",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="Skip encoding if output file already exists (default: True).",
    )
    parser.add_argument(
        "--no_skip_existing",
        action="store_false",
        dest="skip_existing",
        help="Disable skip_existing (force re-run even if files exist).",
    )
    parser.add_argument(
        "--use_slicing",
        action="store_true",
        help="Enable VAE slicing for memory efficiency.",
    )
    parser.add_argument(
        "--use_tiling",
        action="store_true",
        help="Enable VAE tiling for memory efficiency.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
        help="Local rank for distributed training.",
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=1,
        help="Number of processes for distributed training.",
    )
    parser.add_argument(
        "--split_id",
        type=int,
        default=None,
        help="Split ID for SLURM array jobs (0-based, from SLURM_ARRAY_TASK_ID).",
    )
    parser.add_argument(
        "--num_splits",
        type=int,
        default=None,
        help="Total number of splits for SLURM array jobs.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )
    return parser.parse_args()


def setup_logging(log_level: str, rank: int = 0) -> logging.Logger:
    """Setup logging configuration."""
    # Create a custom formatter that includes rank
    class RankFormatter(logging.Formatter):
        def format(self, record):
            record.rank = rank
            return super().format(record)
    
    formatter = RankFormatter(
        "%(asctime)s | %(levelname)s | [Rank %(rank)s] %(message)s"
    )
    
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.addHandler(handler)
    logger.propagate = False
    
    return logger


def setup_distributed() -> Tuple[int, int, int]:
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


def load_dataset_file(dataset_file: str) -> List[str]:
    """Load video paths from dataset file."""
    with open(dataset_file, "r") as f:
        video_paths = [line.strip() for line in f.readlines() if line.strip()]
    return video_paths


def derive_input_video_path(
    video_path: str,
    data_root: str,
    video_subdir: str,
) -> Path:
    """Derive input video path from dataset video path.
    
    Args:
        video_path: Path from dataset file (e.g., "trumans/.../processed2/videos/00000.mp4")
        data_root: Root directory containing the data
        video_subdir: Subdirectory name for input videos (e.g., "videos_object")
    
    Returns:
        Absolute path to input video file
    
    Example:
        .../processed2/videos/00000.mp4 -> .../processed2/videos_object/00000.mp4
    """
    # Construct absolute path first
    if Path(video_path).is_absolute():
        video_path_obj = Path(video_path)
    else:
        video_path_obj = Path(data_root) / video_path
    
    # Get parent directory (e.g., .../processed2/videos)
    parent_dir = video_path_obj.parent
    # Get filename (e.g., 00000.mp4)
    filename = video_path_obj.name
    # Construct condition video path: parent_dir.parent / video_subdir / filename
    # Example: .../processed2/videos_object/00000.mp4
    condition_path = parent_dir.parent / video_subdir / filename
    
    return condition_path


def derive_output_path(
    video_path: str,
    data_root: str,
    video_latent_subdir: str,
    output_dir: Optional[str] = None,
) -> Path:
    """Derive output latent path from dataset video path.
    
    Args:
        video_path: Path from dataset file (e.g., "trumans/.../processed2/videos/00000.mp4")
        data_root: Root directory containing the data
        video_latent_subdir: Subdirectory name for output latents (e.g., "object_video_latents")
        output_dir: Optional output directory (defaults to data_root)
    
    Returns:
        Absolute path to output latent file
    
    Example:
        .../processed2/videos/00000.mp4 -> .../processed2/object_video_latents/00000.pt
    """
    # Construct absolute path first (same as derive_input_video_path)
    if Path(video_path).is_absolute():
        video_path_obj = Path(video_path)
    else:
        video_path_obj = Path(data_root) / video_path
    
    # Get parent directory (e.g., .../processed2/videos)
    parent_dir = video_path_obj.parent
    # Get filename without extension (e.g., 00000)
    filename_stem = video_path_obj.stem
    # Construct latent path: parent_dir.parent / video_latent_subdir / filename.pt
    # Example: .../processed2/object_video_latents/00000.pt
    latent_path = parent_dir.parent / video_latent_subdir / f"{filename_stem}.pt"
    
    # If output_dir is specified, adjust the path
    if output_dir:
        # Get relative path from data_root
        relative_path = video_path_obj.relative_to(Path(data_root))
        # Replace videos with video_latent_subdir in the relative path
        relative_parts = list(relative_path.parts)
        if "videos" in relative_parts:
            videos_idx = relative_parts.index("videos")
            relative_parts[videos_idx] = video_latent_subdir
        # Change extension to .pt
        relative_parts[-1] = filename_stem + ".pt"
        latent_path = Path(output_dir) / Path(*relative_parts)
    
    return latent_path


def split_dataset_for_rank(
    video_paths: List[str],
    rank: int,
    world_size: int,
) -> List[str]:
    """Split dataset for current rank in distributed training."""
    chunk_size = len(video_paths) // world_size
    start_idx = rank * chunk_size
    if rank == world_size - 1:
        # Last rank takes remaining items
        end_idx = len(video_paths)
    else:
        end_idx = start_idx + chunk_size
    return video_paths[start_idx:end_idx]


def split_dataset_for_array_job(
    video_paths: List[str],
    split_id: int,
    num_splits: int,
) -> List[str]:
    """Split dataset for SLURM array job."""
    chunk_size = len(video_paths) // num_splits
    start_idx = split_id * chunk_size
    if split_id == num_splits - 1:
        # Last split takes remaining items
        end_idx = len(video_paths)
    else:
        end_idx = start_idx + chunk_size
    return video_paths[start_idx:end_idx]


def load_video_tensor(path: Path) -> torch.Tensor:
    """Load video from file and convert to tensor format for VAE encoding.
    
    Returns:
        Video tensor in [C, F, H, W] format, normalized to [-1, 1]
    """
    reader = decord.VideoReader(str(path))
    frame_count = len(reader)
    if frame_count == 0:
        raise ValueError(f"Video has no frames: {path}")
    
    indices = list(range(frame_count))
    frames = reader.get_batch(indices)
    if isinstance(frames, torch.Tensor):
        frames = frames.to(dtype=torch.float32)
    else:
        frames = torch.from_numpy(frames.asnumpy()).float()
    
    # Normalize to [0, 1] then to [-1, 1]
    frames = frames / 255.0  # [F, H, W, C] in [0, 1]
    video = frames.permute(3, 0, 1, 2)  # [C, F, H, W]
    video = video * 2.0 - 1.0  # [-1, 1]
    return video.contiguous()


def load_image_mode_tensor(path: Path, num_frames: int) -> torch.Tensor:
    """Load first frame as image, pad remaining frames with -1 for I2V VAE encoding.

    Returns:
        Tensor [C, num_frames, H, W], normalized to [-1, 1]. Frame 0 = image, 1..num_frames-1 = -1.
    """
    video = load_video_tensor(path)
    # video: [C, F, H, W]
    first = video[:, :1, :, :]  # [C, 1, H, W]
    c, _, h, w = first.shape
    pad = torch.full((c, num_frames - 1, h, w), -1.0, dtype=first.dtype, device=first.device)
    out = torch.cat([first, pad], dim=1)
    return out.contiguous()


@torch.no_grad()
def encode_video(
    vae: AutoencoderKLCogVideoX,
    video: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Encode video using VAE.
    
    Args:
        vae: VAE model
        video: Video tensor in [C, F, H, W] format
        device: Device to run encoding on
        dtype: Data type for encoding
    
    Returns:
        Encoded latents in [C, F, H, W] format
    """
    batch = video.unsqueeze(0).to(device=device, dtype=dtype, non_blocking=True)
    latents = vae._encode(batch)
    return latents.squeeze(0).to("cpu").contiguous()


def process_video(
    input_video_path: Path,
    output_path: Path,
    vae: AutoencoderKLCogVideoX,
    device: torch.device,
    dtype: torch.dtype,
    skip_existing: bool = True,
    mode: str = "video",
    num_frames: int = 49,
) -> bool:
    """Process a single video or image-mode input: load, encode, and save.

    Video mode: load full video, encode, save.
    Image mode: first frame only, pad rest with -1 to num_frames, encode, save (I2V setup).
    
    Args:
        input_video_path: Path to input video file
        output_path: Path to output latent file
        vae: VAE model
        device: Device to run encoding on
        dtype: Data type for encoding
        skip_existing: Whether to skip if output file exists
        mode: "video" or "image"
        num_frames: Target frame count for image mode (used only when mode == "image")
    
    Returns:
        True if successful, False otherwise
    """
    if skip_existing and output_path.exists():
        logger.debug(f"Skipping existing file: {output_path}")
        return True

    try:
        if mode == "video":
            tensor = load_video_tensor(input_video_path)
        else:
            tensor = load_image_mode_tensor(input_video_path, num_frames)

        latents = encode_video(vae, tensor, device, dtype)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(latents, output_path)

        logger.debug(f"Saved latents: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to process {input_video_path}: {e}")
        return False


def main():
    """Main function."""
    args = parse_args()
    
    # Setup distributed first to get rank
    rank, world_size, local_rank = setup_distributed()
    
    # Setup logging with rank
    logger = setup_logging(args.log_level, rank)
    
    # Set device
    if world_size > 1:
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set seed
    set_seed(args.seed)
    
    # Log initial info
    if rank == 0:
        logger.info(f"Starting encoding with:")
        logger.info(f"  Mode: {args.mode}")
        if args.mode == "image":
            logger.info(f"  Num frames (image mode): {args.num_frames}")
        logger.info(f"  Dataset file: {args.dataset_file}")
        logger.info(f"  Data root: {args.data_root}")
        logger.info(f"  Video subdir: {args.video_subdir}")
        logger.info(f"  Video latent subdir: {args.video_latent_subdir}")
        logger.info(f"  Model ID: {args.model_id}")
        logger.info(f"  Device: {device}")
        logger.info(f"  Dtype: {args.dtype}")
        logger.info(f"  World size: {world_size}")
        if args.split_id is not None and args.num_splits is not None:
            logger.info(f"  SLURM array split: {args.split_id}/{args.num_splits}")
    
    # Load VAE
    dtype = DTYPE_MAPPING[args.dtype]
    vae = AutoencoderKLCogVideoX.from_pretrained(
        args.model_id,
        subfolder="vae",
        torch_dtype=dtype,
    )
    vae = vae.to(device)
    vae.eval()
    
    if args.use_slicing:
        vae.enable_slicing()
        if rank == 0:
            logger.info("Enabled VAE slicing")
    if args.use_tiling:
        vae.enable_tiling()
        if rank == 0:
            logger.info("Enabled VAE tiling")
    
    # Load dataset file
    video_paths = load_dataset_file(args.dataset_file)
    if rank == 0:
        logger.info(f"Loaded {len(video_paths)} video paths from dataset file")
    
    # Apply SLURM array job split if specified
    if args.split_id is not None and args.num_splits is not None:
        video_paths = split_dataset_for_array_job(
            video_paths, args.split_id, args.num_splits
        )
        if rank == 0:
            logger.info(
                f"After array split {args.split_id}/{args.num_splits}: "
                f"{len(video_paths)} videos to process"
            )
    
    # Apply distributed split if using multiple GPUs
    if world_size > 1:
        video_paths = split_dataset_for_rank(video_paths, rank, world_size)
        logger.info(f"Rank {rank}: Processing {len(video_paths)} videos")
    
    # Process videos
    successful = 0
    failed = 0
    skipped = 0
    successful_paths = []
    failed_paths = []
    skipped_paths = []
    
    progress_bar = tqdm(
        video_paths,
        desc=f"Rank {rank}",
        disable=(rank != 0) if world_size > 1 else False,
    )
    
    for video_path_str in progress_bar:
        # Derive paths
        input_video_path = derive_input_video_path(
            video_path_str, args.data_root, args.video_subdir
        )
        output_path = derive_output_path(
            video_path_str,
            args.data_root,
            args.video_latent_subdir,
            args.output_dir,
        )
        
        # Check if should skip
        if args.skip_existing and output_path.exists():
            skipped += 1
            skipped_paths.append(video_path_str)
            continue
        
        if process_video(
            input_video_path,
            output_path,
            vae,
            device,
            dtype,
            args.skip_existing,
            mode=args.mode,
            num_frames=args.num_frames,
        ):
            successful += 1
            successful_paths.append(video_path_str)
        else:
            failed += 1
            failed_paths.append(video_path_str)
        
        # Update progress
        if rank == 0 or world_size == 1:
            progress_bar.set_postfix(
                {
                    "success": successful,
                    "failed": failed,
                    "skipped": skipped,
                }
            )
    
    # Save failed paths to file
    if failed_paths:
        failed_file = Path(args.data_root) / f"failed_videos_rank{rank}.txt"
        if args.split_id is not None:
            failed_file = Path(args.data_root) / f"failed_videos_split{args.split_id}_rank{rank}.txt"
        
        failed_file.parent.mkdir(parents=True, exist_ok=True)
        with open(failed_file, "w") as f:
            for path in failed_paths:
                f.write(f"{path}\n")
        logger.warning(f"Failed videos saved to: {failed_file}")
    
    # Final summary
    if rank == 0 or world_size == 1:
        logger.info("=" * 60)
        logger.info("Processing complete!")
        logger.info(f"  Successful: {successful}")
        logger.info(f"  Failed: {failed}")
        logger.info(f"  Skipped: {skipped}")
        logger.info(f"  Total: {len(video_paths)}")
        logger.info("=" * 60)
        
        # Print successful paths (first 10, then summary)
        if successful_paths:
            logger.info("")
            logger.info("✅ Successfully processed videos:")
            for i, path in enumerate(successful_paths[:10]):
                logger.info(f"  {i+1}. {path}")
            if len(successful_paths) > 10:
                logger.info(f"  ... and {len(successful_paths) - 10} more")
        
        # Print failed paths
        if failed_paths:
            logger.info("")
            logger.error("❌ Failed videos:")
            for i, path in enumerate(failed_paths):
                logger.error(f"  {i+1}. {path}")
            failed_file = Path(args.data_root) / f"failed_videos_rank{rank}.txt"
            if args.split_id is not None:
                failed_file = Path(args.data_root) / f"failed_videos_split{args.split_id}_rank{rank}.txt"
            logger.error(f"  Full list saved to: {failed_file}")
        
        # Print skipped paths (optional, only if verbose)
        if skipped_paths and args.log_level == "DEBUG":
            logger.debug("")
            logger.debug("⏭️  Skipped videos (already exist):")
            for i, path in enumerate(skipped_paths[:10]):
                logger.debug(f"  {i+1}. {path}")
            if len(skipped_paths) > 10:
                logger.debug(f"  ... and {len(skipped_paths) - 10} more")
    
    # Cleanup distributed
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
