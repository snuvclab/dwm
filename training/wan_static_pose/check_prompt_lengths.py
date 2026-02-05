#!/usr/bin/env python3
"""
Check prompt files for tokenizer max_length violations.

Usage:
    python check_prompt_lengths.py --prompt_dir /path/to/prompts --max_length 226
    python check_prompt_lengths.py --dataset_file /path/to/dataset.txt --data_root /path/to/data --max_length 226
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple
from transformers import AutoTokenizer
import sys

# Add project root to path
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
sys.path.insert(0, project_root)


def load_tokenizer(tokenizer_path: str = "google/umt5-xxl"):
    """Load the tokenizer used in WAN training."""
    print(f"🔧 Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    print(f"✅ Tokenizer loaded: {tokenizer.__class__.__name__}")
    return tokenizer


def check_single_prompt(prompt_path: Path, tokenizer, max_length: int) -> Tuple[bool, int, str]:
    """
    Check if a single prompt file exceeds max_length.
    
    Returns:
        (is_valid, token_length, error_message)
    """
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt_text = f.read().strip()
        
        if not prompt_text:
            return True, 0, "Empty prompt"
        
        # Tokenize the prompt
        tokens = tokenizer(
            prompt_text,
            padding=False,
            truncation=False,
            add_special_tokens=True,
            return_tensors="pt"
        )
        
        token_length = tokens.input_ids.shape[1]
        
        if token_length > max_length:
            return False, token_length, f"Exceeds max_length: {token_length} > {max_length}"
        else:
            return True, token_length, "OK"
            
    except Exception as e:
        return False, -1, f"Error: {str(e)}"


def check_prompt_directory(prompt_dir: Path, tokenizer, max_length: int, extensions: List[str] = ['.txt']):
    """Check all prompt files in a directory."""
    results = []
    
    prompt_files = []
    for ext in extensions:
        prompt_files.extend(prompt_dir.rglob(f"*{ext}"))
    
    print(f"📋 Found {len(prompt_files)} prompt files in {prompt_dir}")
    
    for prompt_file in prompt_files:
        is_valid, token_length, message = check_single_prompt(prompt_file, tokenizer, max_length)
        results.append({
            'path': prompt_file,
            'is_valid': is_valid,
            'token_length': token_length,
            'message': message
        })
    
    return results


def check_dataset_file(dataset_file: Path, data_root: Path, tokenizer, max_length: int, 
                       prompt_subdir: str = "prompts"):
    """Check prompt files referenced in a dataset file."""
    results = []
    
    # Read dataset file
    with open(dataset_file, 'r') as f:
        video_paths = [line.strip() for line in f if line.strip()]
    
    print(f"📋 Found {len(video_paths)} video paths in dataset file")
    
    for video_path in video_paths:
        # Derive prompt path from video path
        # video_path: .../processed2/videos/{video_name}.mp4
        # prompt_path: .../processed2/{prompt_subdir}/{video_name}.txt
        video_path_obj = Path(video_path)
        video_name = video_path_obj.stem
        base_path = data_root / video_path_obj.parent.parent  # videos/ -> processed2/
        prompt_path = base_path / prompt_subdir / f"{video_name}.txt"
        
        if prompt_path.exists():
            is_valid, token_length, message = check_single_prompt(prompt_path, tokenizer, max_length)
            results.append({
                'path': prompt_path,
                'video_path': video_path,
                'is_valid': is_valid,
                'token_length': token_length,
                'message': message
            })
        else:
            results.append({
                'path': prompt_path,
                'video_path': video_path,
                'is_valid': False,
                'token_length': -1,
                'message': f"Prompt file not found"
            })
    
    return results


def print_results(results: List[dict], max_length: int):
    """Print check results in a readable format."""
    valid_count = sum(1 for r in results if r['is_valid'])
    invalid_count = len(results) - valid_count
    
    print("\n" + "=" * 80)
    print("📊 PROMPT LENGTH CHECK RESULTS")
    print("=" * 80)
    print(f"✅ Valid prompts (≤ {max_length} tokens): {valid_count}")
    print(f"❌ Invalid prompts (> {max_length} tokens): {invalid_count}")
    print(f"📊 Total checked: {len(results)}")
    print("=" * 80)
    
    if invalid_count > 0:
        print("\n❌ INVALID PROMPTS (> max_length):")
        print("-" * 80)
        for r in results:
            if not r['is_valid']:
                print(f"  {r['path']}")
                print(f"    Token length: {r['token_length']}")
                print(f"    Message: {r['message']}")
                if 'video_path' in r:
                    print(f"    Video: {r['video_path']}")
                print()
    
    # Show statistics
    valid_lengths = [r['token_length'] for r in results if r['is_valid'] and r['token_length'] >= 0]
    if valid_lengths:
        import statistics
        print("\n📈 STATISTICS (valid prompts only):")
        print(f"  Min: {min(valid_lengths)}")
        print(f"  Max: {max(valid_lengths)}")
        print(f"  Mean: {statistics.mean(valid_lengths):.1f}")
        print(f"  Median: {statistics.median(valid_lengths):.1f}")
        if len(valid_lengths) > 1:
            print(f"  Std: {statistics.stdev(valid_lengths):.1f}")


def main():
    parser = argparse.ArgumentParser(description="Check prompt files for tokenizer max_length violations")
    
    parser.add_argument(
        "--prompt_dir",
        type=str,
        default=None,
        help="Directory containing prompt files to check"
    )
    parser.add_argument(
        "--dataset_file",
        type=str,
        default=None,
        help="Dataset file containing video paths (will derive prompt paths)"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data",
        help="Root directory for data (used with --dataset_file)"
    )
    parser.add_argument(
        "--prompt_subdir",
        type=str,
        default="prompts",
        help="Subdirectory name for prompts (used with --dataset_file)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=226,
        help="Maximum token length (default: 226)"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="google/umt5-xxl",
        help="Tokenizer path (default: google/umt5-xxl)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output file to save invalid prompt paths (optional)"
    )
    
    args = parser.parse_args()
    
    # Load tokenizer
    tokenizer = load_tokenizer(args.tokenizer_path)
    
    # Check prompts
    if args.prompt_dir:
        prompt_dir = Path(args.prompt_dir)
        if not prompt_dir.exists():
            raise ValueError(f"Prompt directory does not exist: {prompt_dir}")
        results = check_prompt_directory(prompt_dir, tokenizer, args.max_length)
    elif args.dataset_file:
        dataset_file = Path(args.dataset_file)
        data_root = Path(args.data_root)
        if not dataset_file.exists():
            raise ValueError(f"Dataset file does not exist: {dataset_file}")
        if not data_root.exists():
            raise ValueError(f"Data root does not exist: {data_root}")
        results = check_dataset_file(dataset_file, data_root, tokenizer, args.max_length, args.prompt_subdir)
    else:
        raise ValueError("Either --prompt_dir or --dataset_file must be provided")
    
    # Print results
    print_results(results, args.max_length)
    
    # Save invalid prompts to file if requested
    if args.output_file:
        invalid_paths = [str(r['path']) for r in results if not r['is_valid']]
        with open(args.output_file, 'w') as f:
            for path in invalid_paths:
                f.write(f"{path}\n")
        print(f"\n💾 Saved {len(invalid_paths)} invalid prompt paths to: {args.output_file}")
    
    # Exit with error code if invalid prompts found
    invalid_count = sum(1 for r in results if not r['is_valid'])
    if invalid_count > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()





