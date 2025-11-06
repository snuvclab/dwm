#!/usr/bin/env python3
"""
Batch processing script for HaWoR hand rendering.

Processes videos from train_list.txt using process_hawor_single.py with multi-GPU support.

Input:
    - train_list.txt with relative paths (e.g., Bathroom/21412.mp4)
    - Input base directory: data/taste_rob/double_resized/

Output:
    - data/taste_rob/videos_hands/{video_name}.mp4
    - data/taste_rob/videos_hands_mask/{video_name}.mp4
    - data/taste_rob/videos_hands_overlay/{video_name}.mp4

Usage:
    python batch_process_hawor.py \
        --train_list data/taste_rob/double/train_list.txt \
        --input_base data/taste_rob/double_resized \
        --gpus 0,1,2,3
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
from tqdm import tqdm
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_hawor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def find_hawor_script():
    """Find the process_hawor_single.py script."""
    script_dir = Path(__file__).parent
    script_path = script_dir / 'process_hawor_single.py'
    
    if script_path.exists():
        return str(script_path)
    
    logger.error(f"Could not find process_hawor_single.py in {script_dir}")
    sys.exit(1)


def process_single_video(video_rel_path, input_base, output_base, gpu_id, 
                         checkpoint, infiller_weight, skip_existing, hawor_script):
    """Process a single video on a specific GPU.
    
    Args:
        video_rel_path: Relative path from train_list (e.g., Bathroom/21412.mp4)
        input_base: Base directory for input videos
        output_base: Base directory for output videos
        gpu_id: GPU ID to use
        checkpoint: Path to checkpoint
        infiller_weight: Path to infiller weight
        skip_existing: Skip if output exists
        hawor_script: Path to process_hawor_single.py
    
    Returns:
        (success, video_name, error_message)
    """
    video_path = Path(input_base) / video_rel_path
    video_name = Path(video_rel_path).stem  # filename without extension
    
    if not video_path.exists():
        return False, video_name, f"Video not found: {video_path}"
    
    # Define output paths
    output_hands = Path(output_base) / 'videos_hands' / f'{video_name}.mp4'
    output_mask = Path(output_base) / 'videos_hands_mask' / f'{video_name}.mp4'
    output_overlay = Path(output_base) / 'videos_hands_overlay' / f'{video_name}.mp4'
    
    # Check if all outputs exist
    if skip_existing:
        if output_hands.exists() and output_mask.exists() and output_overlay.exists():
            return True, video_name, "skipped (already exists)"
    
    # Build command
    output_dir = Path(output_hands).parent
    cmd = [
        sys.executable, hawor_script,
        '--video_path', str(video_path),
        '--output_dir', str(output_dir),
        '--checkpoint', checkpoint,
        '--infiller_weight', infiller_weight,
    ]
    
    # Set GPU and PYTHONPATH
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Add HaWoR directory and DROID-SLAM paths to PYTHONPATH
    # This is needed because masked_droid_slam.py uses relative paths:
    # sys.path.insert(0, 'thirdparty/DROID-SLAM/droid_slam')
    hawor_dir = Path(hawor_script).parent.resolve()  # Use absolute path
    droid_slam_dir = hawor_dir / 'thirdparty' / 'DROID-SLAM' / 'droid_slam'
    droid_dir = hawor_dir / 'thirdparty' / 'DROID-SLAM'
    
    # Build PYTHONPATH with all necessary paths (absolute paths)
    pythonpath_paths = [
        str(hawor_dir),  # HaWoR root
        str(droid_slam_dir),  # For 'from droid import Droid'
        str(droid_dir),  # For other DROID-SLAM imports
    ]
    
    current_pythonpath = env.get('PYTHONPATH', '')
    if current_pythonpath:
        pythonpath_paths.append(current_pythonpath)
    
    env['PYTHONPATH'] = ':'.join(pythonpath_paths)
    
    try:
        # Run with timeout (adjust based on video length)
        # cwd must be hawor_dir so that masked_droid_slam.py's relative paths work
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
            cwd=str(hawor_dir)  # Must be HaWoR directory for relative paths to work
        )
        
        if result.returncode == 0:
            # Move files to expected locations
            # process_hawor_single.py outputs: {video_name}_hand_mesh.mp4, {video_name}_hand_mask.mp4, {video_name}_hand_overlay.mp4
            # We need: {video_name}.mp4 in different directories
            output_dir = Path(output_hands).parent
            expected_overlay = output_dir / f'{video_name}_hand_overlay.mp4'
            expected_mesh = output_dir / f'{video_name}_hand_mesh.mp4'
            expected_mask = output_dir / f'{video_name}_hand_mask.mp4'
            
            # Move files to expected locations
            if expected_mesh.exists():
                expected_mesh.rename(output_hands)
            if expected_mask.exists():
                expected_mask.rename(output_mask)
            if expected_overlay.exists():
                expected_overlay.rename(output_overlay)
            
            return True, video_name, "success"
        else:
            # Collect full error message
            error_parts = []
            
            if result.stderr:
                # Try to find the last traceback or error message
                stderr_lines = result.stderr.split('\n')
                # Find last "Traceback" or "Error" line
                last_error_idx = -1
                for i in range(len(stderr_lines) - 1, -1, -1):
                    if 'Traceback' in stderr_lines[i] or 'Error' in stderr_lines[i] or 'Exception' in stderr_lines[i]:
                        last_error_idx = i
                        break
                
                if last_error_idx >= 0:
                    # Show from traceback to end, but limit to last 2000 chars
                    error_section = '\n'.join(stderr_lines[last_error_idx:])
                    if len(error_section) > 2000:
                        error_section = "..." + error_section[-2000:]
                    error_parts.append(error_section)
                else:
                    # No traceback found, show last 2000 chars
                    error_section = result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr
                    error_parts.append(error_section)
            
            if result.stdout:
                # Also check stdout for errors (sometimes errors go to stdout)
                stdout_lines = result.stdout.split('\n')
                error_lines = [line for line in stdout_lines if any(keyword in line.lower() for keyword in ['error', 'exception', 'traceback', 'failed'])]
                if error_lines:
                    error_parts.append("STDOUT errors:\n" + '\n'.join(error_lines[-20:]))  # Last 20 error lines from stdout
            
            if not error_parts:
                error_msg = f"Unknown error (return code: {result.returncode})"
            else:
                error_msg = '\n'.join(error_parts)
            
            # Log full error for debugging
            logger.error(f"Full error for {video_name} (GPU {gpu_id}):\n{error_msg}")
            
            return False, video_name, f"Process failed: {error_msg[:500]}"  # Still limit in return message
    
    except subprocess.TimeoutExpired:
        return False, video_name, "Timeout (1 hour)"
    except Exception as e:
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


def main():
    parser = argparse.ArgumentParser(description='Batch process HaWoR videos')
    parser.add_argument('--train_list', type=str, required=True,
                        help='Path to train_list.txt')
    parser.add_argument('--input_base', type=str, default='data/taste_rob/double_resized',
                        help='Base directory for input videos')
    parser.add_argument('--output_base', type=str, default='data/taste_rob',
                        help='Base directory for output videos')
    parser.add_argument('--gpus', type=str, default=None,
                        help='Comma-separated GPU IDs (e.g., 0,1,2,3). Default: auto-detect all')
    parser.add_argument('--checkpoint', type=str, default='./weights/hawor/checkpoints/hawor.ckpt',
                        help='Path to HaWoR checkpoint')
    parser.add_argument('--infiller_weight', type=str, default='./weights/hawor/checkpoints/infiller.pt',
                        help='Path to infiller weight')
    parser.add_argument('--no-skip-existing', action='store_true',
                        help='Force reprocessing even if output exists')
    parser.add_argument('--max_workers_per_gpu', type=int, default=1,
                        help='Maximum workers per GPU (default: 1)')
    parser.add_argument('--hawor_script', type=str, default=None,
                        help='Path to process_hawor_single.py (auto-detected if not specified)')
    
    args = parser.parse_args()
    
    skip_existing = not args.no_skip_existing
    
    # Find HaWoR script
    if args.hawor_script:
        hawor_script = args.hawor_script
    else:
        hawor_script = find_hawor_script()
    
    logger.info(f"Using HaWoR script: {hawor_script}")
    
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
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {}
        for video_rel_path, gpu_id in video_tasks:
            future = executor.submit(
                process_single_video,
                video_rel_path,
                args.input_base,
                args.output_base,
                gpu_id,
                args.checkpoint,
                args.infiller_weight,
                skip_existing,
                hawor_script
            )
            future_to_task[future] = (video_rel_path, gpu_id)
        
        # Process results with progress bar
        with tqdm(total=total, desc="Processing videos") as pbar:
            for future in as_completed(future_to_task):
                video_rel_path, gpu_id = future_to_task[future]
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
                
                pbar.update(1)
                
                # Update progress bar description
                pbar.set_description(
                    f"Processing (Success: {success_count}, "
                    f"Skipped: {skipped_count}, Failed: {failed_count})"
                )
    
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

