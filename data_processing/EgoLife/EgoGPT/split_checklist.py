#!/usr/bin/env python3
"""
Script to split a video checklist file into multiple parts.
"""

import argparse
import math
from pathlib import Path


def split_checklist(input_file, num_parts=2, output_prefix="video_checklist"):
    """
    Split a checklist file into multiple parts.
    
    Args:
        input_file (str): Path to input checklist file
        num_parts (int): Number of parts to split into
        output_prefix (str): Prefix for output files
    """
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return
    
    # Get the directory of the input file
    input_dir = input_path.parent
    input_stem = input_path.stem  # filename without extension
    
    # Read all lines from input file
    with open(input_path, 'r') as f:
        lines = f.readlines()
    
    # Filter out empty lines and comments
    video_lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith('#')]
    comment_lines = [line for line in lines if line.strip().startswith('#') or not line.strip()]
    
    total_videos = len(video_lines)
    if total_videos == 0:
        print("No video entries found in the checklist file.")
        return
    
    print(f"Found {total_videos} video entries")
    print(f"Splitting into {num_parts} parts...")
    print(f"Output directory: {input_dir}")
    
    # Calculate lines per part
    lines_per_part = math.ceil(total_videos / num_parts)
    
    # Split and write files
    for i in range(num_parts):
        start_idx = i * lines_per_part
        end_idx = min(start_idx + lines_per_part, total_videos)
        
        if start_idx >= total_videos:
            break
        
        part_lines = video_lines[start_idx:end_idx]
        output_file = input_dir / f"{input_stem}_part{i+1:02d}.txt"
        
        with open(output_file, 'w') as f:
            # Write header comments
            f.write(f"# Video checklist part {i+1}/{num_parts}\n")
            f.write(f"# Total videos in this part: {len(part_lines)}\n")
            f.write(f"# Original file: {input_path.name}\n")
            f.write("#\n")
            
            # Write video paths
            for line in part_lines:
                f.write(f"{line}\n")
        
        print(f"Created {output_file} with {len(part_lines)} videos (lines {start_idx+1}-{end_idx})")
    
    print(f"\nSplit complete! Created {num_parts} files in {input_dir}")


def split_by_gpu(input_file, num_gpus=2, output_prefix="video_checklist"):
    """
    Split a checklist file for multi-GPU processing.
    
    Args:
        input_file (str): Path to input checklist file
        num_gpus (int): Number of GPUs
        output_prefix (str): Prefix for output files (not used when saving in input directory)
    """
    split_checklist(input_file, num_gpus, output_prefix)


def main():
    parser = argparse.ArgumentParser(description="Split a video checklist file into multiple parts")
    parser.add_argument("input_file", help="Path to input checklist file")
    parser.add_argument("--num_parts", type=int, default=2, 
                       help="Number of parts to split into (default: 2)")
    parser.add_argument("--num_gpus", type=int, default=None,
                       help="Number of GPUs (creates GPU-specific files)")
    parser.add_argument("--output_prefix", type=str, default="video_checklist",
                       help="Prefix for output files (default: video_checklist, not used when saving in input directory)")
    
    args = parser.parse_args()
    
    if args.num_gpus:
        print(f"Splitting for {args.num_gpus} GPUs...")
        split_by_gpu(args.input_file, args.num_gpus, args.output_prefix)
    else:
        print(f"Splitting into {args.num_parts} parts...")
        split_checklist(args.input_file, args.num_parts, args.output_prefix)


if __name__ == "__main__":
    main()
