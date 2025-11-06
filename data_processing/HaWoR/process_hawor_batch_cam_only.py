#!/usr/bin/env python3
"""
Batch processing script for HaWoR hand rendering - Camera Space Only version.
Uses process_hawor_single_cam_only.py with optimized model loading.

Processes videos from train_list.txt with multi-GPU support.
Model is loaded once per GPU worker to avoid repeated loading.

Input:
    - train_list.txt with relative paths (e.g., Bathroom/21412.mp4)
    - Input base directory: data/taste_rob/double_resized/

Output:
    - data/taste_rob/videos_hands/{video_name}.mp4
    - data/taste_rob/videos_hands_mask/{video_name}.mp4
    - data/taste_rob/videos_hands_overlay/{video_name}.mp4

Usage:
    python process_hawor_batch_cam_only.py \
        --train_list data/taste_rob/double_resized/train_list.txt \
        --input_base data/taste_rob/double_resized \
        --gpus 0,1,2,3
"""

import argparse
import subprocess
import sys
import os
import signal
from pathlib import Path
from tqdm import tqdm
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
import torch
import multiprocessing

# Import process_video function directly
sys.path.insert(0, os.path.dirname(__file__))
from process_hawor_single_cam_only import process_video
from scripts.scripts_test_video.hawor_video import load_hawor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_hawor_cam_only.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global variables for worker initialization (per process)
_worker_model = None
_worker_model_cfg = None
_worker_checkpoint = None


def process_single_video(video_rel_path, input_base, output_base, gpu_id, 
                         checkpoint, skip_existing, fps=None):
    """Process a single video on a specific GPU using pre-loaded model.
    
    Args:
        video_rel_path: Relative path from train_list (e.g., Bathroom/21412.mp4)
        input_base: Base directory for input videos
        output_base: Base directory for output videos
        gpu_id: GPU ID to use
        checkpoint: Path to checkpoint (for worker initialization)
        skip_existing: Skip if output exists
        fps: Frames per second for video extraction (optional)
    
    Returns:
        (success, video_name, error_message)
    """
    global _worker_model, _worker_model_cfg, _worker_checkpoint
    
    # Lazy load model on first call (per worker process)
    if _worker_model is None or _worker_checkpoint != checkpoint:
        logger.info(f"Worker (GPU {gpu_id}): Loading model from {checkpoint}...")
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        _worker_model, _worker_model_cfg = load_hawor(checkpoint)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _worker_model = _worker_model.to(device)
        _worker_model.eval()
        _worker_checkpoint = checkpoint
        logger.info(f"Worker (GPU {gpu_id}): Model loaded successfully")
    
    video_path = Path(input_base) / video_rel_path
    video_name = Path(video_rel_path).stem  # filename without extension
    
    if not video_path.exists():
        return False, video_name, f"Video not found: {video_path}"
    
    # Expected final output paths
    final_hands = Path(output_base) / 'videos_hands' / f'{video_name}.mp4'
    final_mask = Path(output_base) / 'videos_hands_mask' / f'{video_name}.mp4'
    final_overlay = Path(output_base) / 'videos_hands_overlay' / f'{video_name}.mp4'
    
    # Check if all outputs exist
    if skip_existing:
        if final_hands.exists() and final_mask.exists() and final_overlay.exists():
            return True, video_name, "skipped (already exists)"
    
    # Define output directory
    output_dir = Path(output_base) / 'videos_hands'
    
    try:
        # Call process_video directly with pre-loaded model
        success = process_video(
            str(video_path),
            checkpoint,  # Still needed for compatibility, but model is pre-loaded
            str(output_dir),
            keep_intermediates=False,
            fps=fps,
            img_focal=None,
            model=_worker_model,  # Use pre-loaded model
            model_cfg=_worker_model_cfg  # Use pre-loaded model config
        )
        
        if success:
            # Move files to expected locations
            # process_video outputs: 
            #   {video_name}_hand_mesh_cam_only.mp4
            #   {video_name}_hand_mask_cam_only.mp4
            #   {video_name}_hand_overlay_cam_only.mp4
            # We need: {video_name}.mp4 in different directories
            
            output_mesh = output_dir / f'{video_name}_hand_mesh_cam_only.mp4'
            output_mask = output_dir / f'{video_name}_hand_mask_cam_only.mp4'
            output_overlay = output_dir / f'{video_name}_hand_overlay_cam_only.mp4'
            
            logger.info(f"Moving output files for {video_name}...")
            if output_mesh.exists():
                output_mesh.rename(final_hands)
                logger.info(f"  Moved mesh: {final_hands}")
            if output_mask.exists():
                output_mask.rename(final_mask)
                logger.info(f"  Moved mask: {final_mask}")
            if output_overlay.exists():
                output_overlay.rename(final_overlay)
                logger.info(f"  Moved overlay: {final_overlay}")
            
            return True, video_name, "success"
        else:
            return False, video_name, "Process failed (see logs above)"
    
    except Exception as e:
        import traceback
        error_msg = f"Exception: {str(e)}\n{traceback.format_exc()}"
        logger.error(f"Error for {video_name} (GPU {gpu_id}):\n{error_msg}")
        return False, video_name, f"Exception: {str(e)}"


def load_video_list(train_list_path):
    """Load video list from train_list.txt."""
    train_list_file = Path(train_list_path)
    if not train_list_file.exists():
        logger.error(f"Train list file not found: {train_list_file}")
        return []
    
    videos = []
    with open(train_list_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Remove leading slash if present
            if line.startswith('/'):
                line = line[1:]
            
            videos.append(line)
    
    return videos


def get_available_gpus():
    """Get list of available GPU IDs."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--list-gpus'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            # Parse output like "GPU 0: NVIDIA ..."
            gpu_ids = []
            for line in result.stdout.strip().split('\n'):
                if line.startswith('GPU'):
                    gpu_id = int(line.split()[1].rstrip(':'))
                    gpu_ids.append(gpu_id)
            return gpu_ids
    except:
        pass
    
    # Fallback: assume GPU 0 exists
    logger.warning("Could not detect GPUs, assuming GPU 0")
    return [0]


# Global variable to track executor for cleanup
_executor = None
_shutdown_requested = False


def signal_handler(signum, frame):
    """Handle interrupt signals (Ctrl+C) gracefully."""
    global _executor, _shutdown_requested
    
    if _shutdown_requested:
        # Force exit if already shutting down
        logger.warning("Force exit requested...")
        os._exit(1)
    
    _shutdown_requested = True
    logger.warning(f"\n⚠️  Received signal {signum}. Shutting down gracefully...")
    
    if _executor is not None:
        logger.info("Shutting down worker processes...")
        try:
            # Shutdown executor immediately, cancel pending tasks
            _executor.shutdown(wait=False, cancel_futures=True)
            logger.info("Executor shutdown initiated")
        except Exception as e:
            logger.error(f"Error during executor shutdown: {e}")
    
    # Force kill any remaining child processes
    try:
        import psutil
        current_process = psutil.Process()
        children = current_process.children(recursive=True)
        if children:
            logger.info(f"Terminating {len(children)} child processes...")
            for child in children:
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            
            # Wait a bit, then kill if still alive
            gone, alive = psutil.wait_procs(children, timeout=2)
            for child in alive:
                try:
                    child.kill()
                    logger.info(f"Killed process {child.pid}")
                except psutil.NoSuchProcess:
                    pass
    except ImportError:
        # psutil not available, use os.kill as fallback
        logger.warning("psutil not available, using basic process cleanup")
        import subprocess
        try:
            # Kill child processes using pkill (less reliable but works without psutil)
            subprocess.run(['pkill', '-P', str(os.getpid())], 
                         capture_output=True, timeout=2)
        except:
            pass
    
    logger.info("Shutdown complete")
    sys.exit(0)


def main():
    global _executor
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    parser = argparse.ArgumentParser(description='Batch process HaWoR videos (Camera Space Only)')
    parser.add_argument('--train_list', type=str, required=True,
                        help='Path to train_list.txt')
    parser.add_argument('--input_base', type=str, default='data/taste_rob/double_resized',
                        help='Base directory for input videos')
    parser.add_argument('--output_base', type=str, default='data/taste_rob',
                        help='Base directory for output videos')
    parser.add_argument('--gpus', type=str, default=None,
                        help='Comma-separated GPU IDs (e.g., 0,1,2,3). Default: auto-detect all')
    parser.add_argument('--checkpoint', type=str, default='/virtual_lab/jhb_vclab/byungjun_vclab/world_model/weights/hawor/checkpoints/hawor.ckpt',
                        help='Path to HaWoR checkpoint')
    parser.add_argument('--no-skip-existing', action='store_true',
                        help='Force reprocessing even if output exists')
    parser.add_argument('--max_workers_per_gpu', type=int, default=1,
                        help='Maximum workers per GPU (default: 1)')
    parser.add_argument('--fps', type=int, default=10,
                        help='Frames per second for video extraction (default: auto-detect from video)')
    
    args = parser.parse_args()
    
    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    # This must be done before creating ProcessPoolExecutor
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # Already set, ignore
        pass
    
    skip_existing = not args.no_skip_existing
    
    # Get GPU list
    if args.gpus:
        gpu_ids = [int(g.strip()) for g in args.gpus.split(',')]
    else:
        gpu_ids = get_available_gpus()
    
    logger.info(f"Using GPUs: {gpu_ids}")
    
    # Load video list
    videos = load_video_list(args.train_list)
    logger.info(f"Loaded {len(videos)} videos from {args.train_list}")
    
    if len(videos) == 0:
        logger.error("No videos to process!")
        return
    
    # Create output directories
    output_hands_dir = Path(args.output_base) / 'videos_hands'
    output_mask_dir = Path(args.output_base) / 'videos_hands_mask'
    output_overlay_dir = Path(args.output_base) / 'videos_hands_overlay'
    output_hands_dir.mkdir(parents=True, exist_ok=True)
    output_mask_dir.mkdir(parents=True, exist_ok=True)
    output_overlay_dir.mkdir(parents=True, exist_ok=True)
    
    # Distribute videos across GPUs (round-robin)
    video_tasks = []
    for idx, video_rel_path in enumerate(videos):
        gpu_id = gpu_ids[idx % len(gpu_ids)]
        video_tasks.append((video_rel_path, gpu_id))
    
    # Process videos
    total = len(video_tasks)
    success_count = 0
    failed_count = 0
    skipped_count = 0
    
    # Statistics per GPU
    gpu_stats = defaultdict(lambda: {'success': 0, 'failed': 0, 'skipped': 0})
    
    logger.info(f"Starting batch processing with {len(gpu_ids)} GPU(s)...")
    logger.info(f"Total videos: {total}")
    logger.info(f"Skip existing: {skip_existing}")
    
    # Use ProcessPoolExecutor for parallel processing
    max_workers = len(gpu_ids) * args.max_workers_per_gpu
    logger.info(f"Max workers: {max_workers}")
    logger.info(f"Model will be loaded once per worker process (lazy loading on first video)")
    
    try:
        executor = ProcessPoolExecutor(max_workers=max_workers)
        _executor = executor  # Store globally for signal handler
        
        # Submit all tasks
        # Model will be loaded lazily on first call per worker
        future_to_task = {}
        for video_rel_path, gpu_id in video_tasks:
            if _shutdown_requested:
                logger.warning("Shutdown requested, stopping task submission")
                break
            future = executor.submit(
                process_single_video,
                video_rel_path,
                args.input_base,
                args.output_base,
                gpu_id,
                args.checkpoint,
                skip_existing,
                args.fps
            )
            future_to_task[future] = (video_rel_path, gpu_id)
        
        # Process results with progress bar
        with tqdm(total=total, desc="Processing videos") as pbar:
            for future in as_completed(future_to_task):
                if _shutdown_requested:
                    logger.warning("Shutdown requested, cancelling remaining tasks")
                    # Cancel remaining futures
                    for f in future_to_task:
                        if not f.done():
                            f.cancel()
                    break
                
                try:
                    video_rel_path, gpu_id = future_to_task[future]
                    # Use timeout only if shutdown is requested, otherwise wait normally
                    if _shutdown_requested:
                        success, video_name, message = future.result(timeout=0.1)
                    else:
                        success, video_name, message = future.result()
                    
                    if success:
                        if message == "skipped (already exists)":
                            skipped_count += 1
                            gpu_stats[gpu_id]['skipped'] += 1
                        else:
                            success_count += 1
                            gpu_stats[gpu_id]['success'] += 1
                    else:
                        failed_count += 1
                        gpu_stats[gpu_id]['failed'] += 1
                        logger.error(f"Failed: {video_name} (GPU {gpu_id}) - {message}")
                except Exception as e:
                    if not _shutdown_requested:
                        failed_count += 1
                        logger.error(f"Error processing task: {e}")
                
                pbar.update(1)
                
                # Update progress bar description
                pbar.set_description(
                    f"Processing (Success: {success_count}, "
                    f"Skipped: {skipped_count}, Failed: {failed_count})"
                )
    finally:
        # Ensure executor is properly shut down
        if executor is not None:
            logger.info("Shutting down executor...")
            executor.shutdown(wait=True, cancel_futures=True)
            _executor = None
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("BATCH PROCESSING COMPLETE")
    logger.info("="*60)
    logger.info(f"Total: {total}")
    logger.info(f"Success: {success_count}")
    logger.info(f"Skipped: {skipped_count}")
    logger.info(f"Failed: {failed_count}")
    logger.info("\nPer-GPU Statistics:")
    for gpu_id in sorted(gpu_stats.keys()):
        stats = gpu_stats[gpu_id]
        logger.info(f"  GPU {gpu_id}: {stats['success']} success, "
                   f"{stats['skipped']} skipped, {stats['failed']} failed")
    logger.info("="*60)


if __name__ == '__main__':
    main()

