#!/usr/bin/env python3
"""
Script to check which videos need processing and generate a checklist.
This follows the same logic as inference_dataset_directory.py but only scans and reports.
"""

import argparse
import os
from pathlib import Path
from typing import List, Dict, Tuple


def get_prompt_save_path(video_path):
    """Convert video path to prompt save path.
    
    For video_path: data/trumans/ego_render_fov90/{scene_name}/{action_name}/sequences/videos/00000.mp4
    Return: data/trumans/ego_render_fov90/{scene_name}/{action_name}/sequences/prompts/00000.txt
    """
    video_path = Path(video_path)
    
    # Get the parent directory of 'videos' folder
    if 'videos' in video_path.parts:
        videos_idx = video_path.parts.index('videos')
        base_path = Path(*video_path.parts[:videos_idx])
    else:
        # Fallback: use the parent directory of the video file
        base_path = video_path.parent.parent
    
    # Get the video filename without extension
    video_name = video_path.stem
    
    # Create prompts directory and return the full path
    prompts_dir = base_path / "prompts"
    return prompts_dir / f"{video_name}.txt"


def find_video_files(data_root, dataset_name="trumans", render_type="ego_render_fov90"):
    """Find all video files in the directory structure.
    
    Expected structure:
    data_root/
    └── {dataset_name}/
        └── {render_type}/
            └── {scene_name}/
                └── {action_name}/
                    └── sequences/
                        └── videos/
                            ├── 00000.mp4
                            ├── 00001.mp4
                            └── ...
    """
    data_root = Path(data_root)
    video_files = []
    
    # Look for the dataset/render_type directory
    dataset_path = data_root / dataset_name / render_type
    if not dataset_path.exists():
        print(f"Dataset directory not found: {dataset_path}")
        return video_files
    
    print(f"Scanning for videos in: {dataset_path}")
    
    # Iterate through scene directories
    for scene_dir in sorted(dataset_path.iterdir()):
        if not scene_dir.is_dir():
            continue
            
        scene_name = scene_dir.name
        print(f"  Scanning scene: {scene_name}")
        
        # Iterate through action directories
        for action_dir in sorted(scene_dir.iterdir()):
            if not action_dir.is_dir():
                continue
                
            action_name = action_dir.name
            print(f"    Scanning action: {action_name}")
            
            # Look for sequences/videos directory
            videos_dir = action_dir / "sequences" / "videos"
            if not videos_dir.exists():
                print(f"      No videos directory found: {videos_dir}")
                continue
            
            # Find all MP4 files
            mp4_files = list(videos_dir.glob("*.mp4"))
            print(f"      Found {len(mp4_files)} video files")
            
            for video_file in sorted(mp4_files):
                video_files.append(str(video_file))
    
    print(f"Total video files found: {len(video_files)}")
    return video_files


def check_video_status(video_path: str, skip_existing: bool = True) -> Dict[str, any]:
    """Check the status of a single video file."""
    video_path = Path(video_path)
    save_path = get_prompt_save_path(video_path)
    
    status = {
        'video_path': str(video_path),
        'prompt_path': str(save_path),
        'video_exists': video_path.exists(),
        'prompt_exists': save_path.exists(),
        'prompt_empty': False,
        'needs_processing': False,
        'reason': ''
    }
    
    # Check if video exists
    if not status['video_exists']:
        status['needs_processing'] = False
        status['reason'] = 'Video file not found'
        return status
    
    # Check if prompt file exists and is not empty
    if status['prompt_exists']:
        try:
            existing_content = save_path.read_text().strip()
            status['prompt_empty'] = len(existing_content) == 0
            if existing_content and skip_existing:
                status['needs_processing'] = False
                status['reason'] = 'Prompt file exists and is not empty'
            else:
                status['needs_processing'] = True
                status['reason'] = 'Prompt file is empty' if status['prompt_empty'] else 'Prompt file exists but skip_existing=False'
        except Exception as e:
            status['needs_processing'] = True
            status['reason'] = f'Error reading prompt file: {str(e)}'
    else:
        status['needs_processing'] = True
        status['reason'] = 'No prompt file exists'
    
    return status


def generate_checklist(data_root: str, skip_existing: bool = True, output_file: str = None, dataset_name: str = "trumans", render_type: str = "ego_render_fov90", paths_only: bool = True) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Generate a checklist of videos to process."""
    print(f"Generating checklist for data root: {data_root}")
    print(f"Dataset: {dataset_name}")
    print(f"Render type: {render_type}")
    print(f"Skip existing: {skip_existing}")
    print("=" * 80)
    
    # Find all video files
    video_paths = find_video_files(data_root, dataset_name, render_type)
    
    if not video_paths:
        print("No video files found!")
        return [], [], []
    
    print(f"\nChecking status of {len(video_paths)} videos...")
    
    # Check status of each video
    all_videos = []
    videos_to_process = []
    videos_to_skip = []
    
    for i, video_path in enumerate(video_paths, 1):
        if i % 100 == 0:
            print(f"  Checked {i}/{len(video_paths)} videos...")
        
        status = check_video_status(video_path, skip_existing)
        all_videos.append(status)
        
        if status['needs_processing']:
            videos_to_process.append(status)
        else:
            videos_to_skip.append(status)
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total videos found: {len(all_videos)}")
    print(f"Videos to process: {len(videos_to_process)}")
    print(f"Videos to skip: {len(videos_to_skip)}")
    
    if videos_to_process:
        print(f"\nVideos that need processing:")
        for status in videos_to_process[:10]:  # Show first 10
            print(f"  {status['video_path']} - {status['reason']}")
        if len(videos_to_process) > 10:
            print(f"  ... and {len(videos_to_process) - 10} more")
    
    if videos_to_skip:
        print(f"\nVideos that will be skipped:")
        for status in videos_to_skip[:5]:  # Show first 5
            print(f"  {status['video_path']} - {status['reason']}")
        if len(videos_to_skip) > 5:
            print(f"  ... and {len(videos_to_skip) - 5} more")
    
    # Save to file if requested
    if output_file:
        save_checklist_to_file(all_videos, videos_to_process, videos_to_skip, output_file, data_root, paths_only)
    
    return all_videos, videos_to_process, videos_to_skip


def save_checklist_to_file(all_videos: List[Dict], videos_to_process: List[Dict], videos_to_skip: List[Dict], output_file: str, data_root: str, paths_only: bool = True):
    """Save the checklist to a file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data_root_path = Path(data_root).resolve()
    
    with open(output_path, 'w') as f:
        if paths_only:
            # Write only the video paths that need processing, relative to data_root
            for status in videos_to_process:
                video_path = Path(status['video_path']).resolve()
                # Get relative path from data_root
                try:
                    relative_path = video_path.relative_to(data_root_path)
                    f.write(f"{relative_path}\n")
                except ValueError:
                    # If video_path is not relative to data_root, use the full path
                    f.write(f"{status['video_path']}\n")
        else:
            # Write detailed checklist with header information
            f.write("VIDEO PROCESSING CHECKLIST\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Generated at: {Path().absolute()}\n")
            f.write(f"Data root: {data_root}\n")
            f.write(f"Total videos: {len(all_videos)}\n")
            f.write(f"Videos to process: {len(videos_to_process)}\n")
            f.write(f"Videos to skip: {len(videos_to_skip)}\n\n")
            
            f.write("VIDEOS TO PROCESS:\n")
            f.write("-" * 20 + "\n")
            for status in videos_to_process:
                video_path = Path(status['video_path']).resolve()
                try:
                    relative_path = video_path.relative_to(data_root_path)
                    f.write(f"{relative_path}\t{status['reason']}\n")
                except ValueError:
                    f.write(f"{status['video_path']}\t{status['reason']}\n")
            
            f.write("\nVIDEOS TO SKIP:\n")
            f.write("-" * 15 + "\n")
            for status in videos_to_skip:
                video_path = Path(status['video_path']).resolve()
                try:
                    relative_path = video_path.relative_to(data_root_path)
                    f.write(f"{relative_path}\t{status['reason']}\n")
                except ValueError:
                    f.write(f"{status['video_path']}\t{status['reason']}\n")
    
    print(f"\nChecklist saved to: {output_path}")
    if paths_only:
        print(f"Contains {len(videos_to_process)} video paths (relative to {data_root})")
    else:
        print(f"Contains detailed checklist with {len(videos_to_process)} videos to process")


def main():
    parser = argparse.ArgumentParser(description="Generate a checklist of videos that need processing")
    parser.add_argument("--data_root", type=str, default="./data",
                       help="Data root directory (default: ./data)")
    parser.add_argument("--dataset_name", type=str, default="trumans",
                       help="Dataset name (default: trumans)")
    parser.add_argument("--render_type", type=str, default="ego_render_fov90",
                       help="Render type (default: ego_render_fov90)")
    parser.add_argument("--skip_existing", action="store_true", default=True,
                       help="Skip videos that already have non-empty prompt files (default: True)")
    parser.add_argument("--no_skip_existing", action="store_false", dest="skip_existing",
                       help="Process all videos, overwriting existing prompt files")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Save checklist to this file (optional)")
    parser.add_argument("--paths_only", action="store_true",
                       help="When saving to file, only save video paths (no header info)")
    parser.add_argument("--verbose", action="store_true",
                       help="Show detailed information for each video")
    
    args = parser.parse_args()
    
    # Generate checklist
    all_videos, videos_to_process, videos_to_skip = generate_checklist(
        args.data_root, args.skip_existing, args.output_file, args.dataset_name, args.render_type, args.paths_only
    )
    
    # Show detailed information if requested
    if args.verbose and videos_to_process:
        print(f"\nDETAILED BREAKDOWN OF VIDEOS TO PROCESS:")
        print("=" * 60)
        for status in videos_to_process:
            print(f"Video: {status['video_path']}")
            print(f"  Prompt path: {status['prompt_path']}")
            print(f"  Video exists: {status['video_exists']}")
            print(f"  Prompt exists: {status['prompt_exists']}")
            print(f"  Prompt empty: {status['prompt_empty']}")
            print(f"  Reason: {status['reason']}")
            print()


if __name__ == "__main__":
    main()
