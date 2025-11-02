#!/usr/bin/env python3

import argparse
import os
import subprocess
import json
from pathlib import Path
import multiprocessing as mp
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for distributed inference."""
    parser = argparse.ArgumentParser(description="Distributed CogVideoX Inference")
    
    # Pipeline configuration
    parser.add_argument(
        "--pipeline_type",
        type=str,
        required=True,
        choices=["cogvideox_pose_concat", "cogvideox_pose_adapter", "cogvideox_pose_adaln", "cogvideox_i2v"],
        help="Type of pipeline to use for inference"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the trained checkpoint directory"
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="THUDM/CogVideoX-5b",
        help="Base model path (used for i2v and adaln pipelines)"
    )
    
    # Data configuration
    parser.add_argument(
        "--dataset_file",
        type=str,
        required=True,
        help="Path to dataset file containing video paths"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data",
        help="Root directory containing the dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to save the outputs"
    )
    
    # Distributed configuration
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for distributed inference"
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default=None,
        help="Comma-separated list of GPU IDs to use (e.g., '0,1,2,3')"
    )
    
    # Inference parameters
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of inference steps"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=6.0,
        help="Guidance scale for classifier-free guidance"
    )
    parser.add_argument(
        "--use_dynamic_cfg",
        action="store_true",
        default=True,
        help="Use dynamic cfg"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Height of the output video"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=720,
        help="Width of the output video"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=49,
        help="Number of frames to generate"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=8,
        choices=[8, 10, 12, 15, 24],
        help="Frames per second"
    )
    
    # Batch processing
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=None,
        help="Maximum number of files to process"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference (currently only 1 supported)"
    )
    
    # Other options
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--use_empty_prompts",
        action="store_true",
        default=False,
        help="Use empty prompts instead of provided text prompts"
    )
    parser.add_argument(
        "--compute_metrics",
        action="store_true",
        default=True,
        help="Compute video quality metrics (PSNR, SSIM, LPIPS)"
    )
    parser.add_argument(
        "--save_comparison_videos",
        action="store_true",
        default=True,
        help="Save side-by-side comparison videos"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def split_dataset_file(dataset_file: Path, num_splits: int) -> List[Path]:
    """Split dataset file into multiple parts for distributed processing."""
    
    # Read all video paths
    with dataset_file.open('r') as f:
        video_paths = [line.strip() for line in f if line.strip()]
    
    # Calculate split sizes
    total_videos = len(video_paths)
    videos_per_split = total_videos // num_splits
    remainder = total_videos % num_splits
    
    logger.info(f"📊 Splitting {total_videos} videos into {num_splits} parts")
    logger.info(f"   Videos per split: {videos_per_split}")
    if remainder > 0:
        logger.info(f"   Extra videos in first {remainder} splits")
    
    # Create split files
    split_files = []
    start_idx = 0
    
    for i in range(num_splits):
        # Calculate end index for this split
        split_size = videos_per_split + (1 if i < remainder else 0)
        end_idx = start_idx + split_size
        
        # Create split file
        split_file = dataset_file.parent / f"{dataset_file.stem}_split_{i}.txt"
        with split_file.open('w') as f:
            for video_path in video_paths[start_idx:end_idx]:
                f.write(f"{video_path}\n")
        
        split_files.append(split_file)
        logger.info(f"   Split {i}: {split_size} videos -> {split_file}")
        start_idx = end_idx
    
    return split_files


def run_single_gpu_inference(args, gpu_id, split_file, output_dir):
    """Run inference on a single GPU with a dataset split."""
    
    # Create GPU-specific output directory
    gpu_output_dir = Path(output_dir) / f"gpu_{gpu_id}"
    gpu_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build command
    cmd = [
        "python", "training/cogvideox_static_pose/inference_unified.py",
        "--pipeline_type", args.pipeline_type,
        "--checkpoint_path", args.checkpoint_path,
        "--base_model_path", args.base_model_path,
        "--dataset_file", str(split_file),
        "--data_root", args.data_root,
        "--output_dir", str(gpu_output_dir),
        "--num_inference_steps", str(args.num_inference_steps),
        "--guidance_scale", str(args.guidance_scale),
        "--height", str(args.height),
        "--width", str(args.width),
        "--num_frames", str(args.num_frames),
        "--fps", str(args.fps),
        "--seed", str(args.seed),
    ]
    
    # Add optional arguments
    if args.use_dynamic_cfg:
        cmd.append("--use_dynamic_cfg")
    if args.use_empty_prompts:
        cmd.append("--use_empty_prompts")
    if args.compute_metrics:
        cmd.append("--compute_metrics")
    if args.save_comparison_videos:
        cmd.append("--save_comparison_videos")
    if args.verbose:
        cmd.append("--verbose")
    if args.max_batch_size:
        cmd.extend(["--max_batch_size", str(args.max_batch_size)])
    
    # Set CUDA_VISIBLE_DEVICES
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    logger.info(f"🚀 Starting inference on GPU {gpu_id}")
    logger.info(f"   Command: {' '.join(cmd)}")
    logger.info(f"   Output: {gpu_output_dir}")
    
    # Run inference
    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            check=True
        )
        
        logger.info(f"✅ GPU {gpu_id} completed successfully")
        return {
            "gpu_id": gpu_id,
            "success": True,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "output_dir": str(gpu_output_dir),
        }
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ GPU {gpu_id} failed with return code {e.returncode}")
        logger.error(f"   stdout: {e.stdout}")
        logger.error(f"   stderr: {e.stderr}")
        return {
            "gpu_id": gpu_id,
            "success": False,
            "error": str(e),
            "returncode": e.returncode,
            "stdout": e.stdout,
            "stderr": e.stderr,
        }


def merge_results(output_dir: Path, num_gpus: int):
    """Merge results from multiple GPU outputs."""
    
    logger.info(f"🔄 Merging results from {num_gpus} GPUs")
    
    # Create merged output directory
    merged_dir = output_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all results
    all_results = []
    all_summaries = []
    
    for gpu_id in range(num_gpus):
        gpu_dir = output_dir / f"gpu_{gpu_id}"
        
        if not gpu_dir.exists():
            logger.warning(f"⚠️  GPU {gpu_id} output directory not found: {gpu_dir}")
            continue
        
        # Copy individual result files
        for result_file in gpu_dir.glob("*_result.json"):
            merged_result_file = merged_dir / f"gpu_{gpu_id}_{result_file.name}"
            merged_result_file.write_text(result_file.read_text())
            logger.info(f"   Copied: {result_file.name} -> {merged_result_file.name}")
        
        # Copy videos
        for video_file in gpu_dir.glob("*.mp4"):
            merged_video_file = merged_dir / f"gpu_{gpu_id}_{video_file.name}"
            merged_video_file.write_bytes(video_file.read_bytes())
            logger.info(f"   Copied: {video_file.name} -> {merged_video_file.name}")
        
        # Load batch summary
        summary_file = gpu_dir / "batch_summary.json"
        if summary_file.exists():
            with summary_file.open('r') as f:
                summary = json.load(f)
            all_summaries.append(summary)
            
            # Load individual results
            for result in summary.get("results", []):
                result["gpu_id"] = gpu_id
                all_results.append(result)
    
    # Create merged summary
    if all_summaries:
        merged_summary = {
            "total_videos": sum(s.get("total_videos", 0) for s in all_summaries),
            "successful": sum(s.get("successful", 0) for s in all_summaries),
            "failed": sum(s.get("failed", 0) for s in all_summaries),
            "success_rate": 0,
            "num_gpus": num_gpus,
            "gpu_summaries": all_summaries,
            "all_results": all_results,
        }
        
        if merged_summary["total_videos"] > 0:
            merged_summary["success_rate"] = merged_summary["successful"] / merged_summary["total_videos"]
        
        # Save merged summary
        merged_summary_file = merged_dir / "merged_summary.json"
        with merged_summary_file.open('w') as f:
            json.dump(merged_summary, f, indent=2)
        
        logger.info(f"✅ Merged summary saved: {merged_summary_file}")
        logger.info(f"📊 Total videos: {merged_summary['total_videos']}")
        logger.info(f"✅ Successful: {merged_summary['successful']}")
        logger.info(f"❌ Failed: {merged_summary['failed']}")
        logger.info(f"📊 Success rate: {merged_summary['success_rate']*100:.1f}%")
        
        # Compute overall metrics if available
        successful_results = [r for r in all_results if r.get("success") and r.get("metrics", {}).get("psnr") is not None]
        if successful_results:
            avg_psnr = sum(r["metrics"]["psnr"] for r in successful_results) / len(successful_results)
            avg_ssim = sum(r["metrics"]["ssim"] for r in successful_results) / len(successful_results)
            avg_lpips = sum(r["metrics"]["lpips"] for r in successful_results) / len(successful_results)
            
            logger.info(f"\n📊 Overall Metrics (n={len(successful_results)}):")
            logger.info(f"   PSNR: {avg_psnr:.3f}")
            logger.info(f"   SSIM: {avg_ssim:.3f}")
            logger.info(f"   LPIPS: {avg_lpips:.3f}")


def main():
    """Main distributed inference function."""
    
    args = parse_args()
    
    # Determine GPU IDs
    if args.gpu_ids:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
        num_gpus = len(gpu_ids)
    else:
        gpu_ids = list(range(args.num_gpus))
        num_gpus = args.num_gpus
    
    logger.info(f"🎯 Distributed Inference Setup")
    logger.info(f"   Pipeline: {args.pipeline_type}")
    logger.info(f"   Checkpoint: {args.checkpoint_path}")
    logger.info(f"   Dataset: {args.dataset_file}")
    logger.info(f"   Output: {args.output_dir}")
    logger.info(f"   GPUs: {gpu_ids}")
    
    # Check if dataset file exists
    dataset_file = Path(args.dataset_file)
    if not dataset_file.exists():
        raise ValueError(f"Dataset file does not exist: {dataset_file}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Split dataset file
    split_files = split_dataset_file(dataset_file, num_gpus)
    
    # Run inference on each GPU
    logger.info(f"🚀 Starting distributed inference on {num_gpus} GPUs")
    
    processes = []
    results = []
    
    for i, gpu_id in enumerate(gpu_ids):
        p = mp.Process(
            target=run_single_gpu_inference,
            args=(args, gpu_id, split_files[i], output_dir)
        )
        p.start()
        processes.append(p)
    
    # Wait for all processes to complete
    for i, p in enumerate(processes):
        p.join()
        logger.info(f"✅ GPU {gpu_ids[i]} process completed")
    
    # Merge results
    merge_results(output_dir, num_gpus)
    
    # Cleanup split files
    for split_file in split_files:
        split_file.unlink()
        logger.info(f"🗑️  Cleaned up split file: {split_file}")
    
    logger.info(f"🎉 Distributed inference completed!")


if __name__ == "__main__":
    main()


