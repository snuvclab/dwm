import argparse
import copy
import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import requests
import soundfile as sf
import torch
import torch.distributed as dist
import whisper
from decord import VideoReader, cpu
from egogpt.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_SPEECH_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    SPEECH_TOKEN_INDEX,
)
from egogpt.conversation import SeparatorStyle, conv_templates
from egogpt.mm_utils import get_model_name_from_path, process_images
from egogpt.model.builder import load_pretrained_model
from PIL import Image
from scipy.signal import resample


def setup(rank, world_size, port=None):
    os.environ["MASTER_ADDR"] = "localhost"
    if port is None:
        # Use a different port based on rank to avoid conflicts
        port = str(12355 + rank)
    os.environ["MASTER_PORT"] = port
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def load_video(video_path=None, audio_path=None, max_frames_num=16, fps=1):
    if audio_path is not None:
        speech, sample_rate = sf.read(audio_path)
        if sample_rate != 16000:
            target_length = int(len(speech) * 16000 / sample_rate)
            speech = resample(speech, target_length)
        if speech.ndim > 1:
            speech = np.mean(speech, axis=1)
        speech = whisper.pad_or_trim(speech.astype(np.float32))
        speech = whisper.log_mel_spectrogram(speech, n_mels=128).permute(1, 0)
        speech_lengths = torch.LongTensor([speech.shape[0]])
    else:
        speech = torch.zeros(3000, 128)
        speech_lengths = torch.LongTensor([3000])

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    avg_fps = round(vr.get_avg_fps() / fps)
    frame_idx = [i for i in range(0, total_frame_num, avg_fps)]
    if max_frames_num > 0 and len(frame_idx) > max_frames_num:
        uniform_sampled_frames = np.linspace(
            0, total_frame_num - 1, max_frames_num, dtype=int
        )
        frame_idx = uniform_sampled_frames.tolist()
    video = vr.get_batch(frame_idx).asnumpy()
    return video, speech, speech_lengths


def split_text(text, keywords):
    pattern = "(" + "|".join(map(re.escape, keywords)) + ")"
    parts = re.split(pattern, text)
    parts = [part for part in parts if part]
    return parts


def process_single_video(model, tokenizer, video_path, audio_path, query, device):
    """Process a single video and return the generated text."""
    conv_template = "qwen_1_5"
    question = f"<image>\n<speech>\n\n{query}"
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    video, speech, speech_lengths = load_video(
        video_path=video_path, audio_path=audio_path
    )
    speech = torch.stack([speech]).to(device).half()
    processor = model.get_vision_tower().image_processor
    processed_video = processor.preprocess(video, return_tensors="pt")["pixel_values"]
    image = [(processed_video, video[0].size, "video")]

    parts = split_text(prompt_question, ["<image>", "<speech>"])
    input_ids = []
    for part in parts:
        if part == "<image>":
            input_ids.append(IMAGE_TOKEN_INDEX)
        elif part == "<speech>":
            input_ids.append(SPEECH_TOKEN_INDEX)
        else:
            input_ids.extend(tokenizer(part).input_ids)

    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
    image_tensor = [image[0][0].half()]
    image_sizes = [image[0][1]]
    generate_kwargs = {"eos_token_id": tokenizer.eos_token_id}

    cont = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=image_sizes,
        speech=speech,
        speech_lengths=speech_lengths,
        do_sample=False,
        temperature=0.5,
        max_new_tokens=4096,
        modalities=["video"],
        **generate_kwargs,
    )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
    return text_outputs[0]


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


def load_video_checklist(checklist_file, data_root):
    """Load video paths from a checklist file and convert to absolute paths."""
    checklist_path = Path(checklist_file)
    if not checklist_path.exists():
        print(f"Checklist file not found: {checklist_path}")
        return []
    
    data_root_path = Path(data_root).resolve()
    video_files = []
    
    print(f"Loading video checklist from: {checklist_path}")
    
    with open(checklist_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Convert relative path to absolute path
            try:
                relative_path = Path(line)
                absolute_path = (data_root_path / relative_path).resolve()
                video_files.append(str(absolute_path))
            except Exception as e:
                print(f"Warning: Invalid path on line {line_num}: {line} - {e}")
                continue
    
    print(f"Loaded {len(video_files)} video paths from checklist")
    return video_files


def remove_from_checklist(checklist_file, video_path, data_root):
    """Remove a processed video from the checklist file."""
    checklist_path = Path(checklist_file)
    if not checklist_path.exists():
        return
    
    data_root_path = Path(data_root).resolve()
    video_path_resolved = Path(video_path).resolve()
    
    try:
        # Convert absolute video path to relative path for comparison
        relative_video_path = video_path_resolved.relative_to(data_root_path)
    except ValueError:
        # If video path is not relative to data root, use the full path
        relative_video_path = video_path_resolved
    
    # Read all lines and filter out the processed video
    with open(checklist_path, 'r') as f:
        lines = f.readlines()
    
    original_count = len(lines)
    filtered_lines = []
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped or line_stripped.startswith('#'):
            filtered_lines.append(line)
            continue
        
        try:
            line_path = Path(line_stripped)
            if line_path != relative_video_path:
                filtered_lines.append(line)
        except Exception:
            # Keep lines that can't be parsed as paths
            filtered_lines.append(line)
    
    # Write back the filtered lines
    with open(checklist_path, 'w') as f:
        f.writelines(filtered_lines)
    
    removed_count = original_count - len(filtered_lines)
    if removed_count > 0:
        print(f"Removed {removed_count} processed video(s) from checklist")


def main(
    pretrained_path="checkpoints/EgoGPT-7b-EgoIT-EgoLife",
    data_root="./data",
    audio_path=None,
    query="Please describe the video in detail.",
    dry_run=False,
    skip_existing=True,
    num_gpus=1,
    checklist_file=None,
    dataset_name="trumans",
    render_type="ego_render_fov90",
    remove_from_checklist_after_processing=False,
):
    warnings.filterwarnings("ignore")
    
    # Initialize distributed processing if using multiple GPUs
    if num_gpus > 1:
        if "LOCAL_RANK" in os.environ:
            local_rank = int(os.environ["LOCAL_RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            setup(local_rank, world_size)
            device = f"cuda:{local_rank}"
            device_map = f"cuda:{local_rank}"
        else:
            print("Warning: Multi-GPU requested but no distributed environment found. Using single GPU.")
            num_gpus = 1
            setup(0, 1)
            device = "cuda"
            device_map = "cuda"
    else:
        # Single GPU mode - use CUDA_VISIBLE_DEVICES as rank offset
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        try:
            gpu_offset = int(cuda_visible_devices.split(',')[0])
        except (ValueError, IndexError):
            gpu_offset = 0
        
        setup(gpu_offset, 1)
        device = "cuda"
        device_map = "cuda"
        local_rank = gpu_offset
        world_size = 1

    # Load model only if not in dry run mode
    if not dry_run:
        print(f"Loading model from {pretrained_path}...")
        print(f"Device map: {device_map}")
        print(f"Local rank: {local_rank}, World size: {world_size}")
        
        try:
            tokenizer, model, max_length = load_pretrained_model(
                pretrained_path, device_map=device_map
            )
            model.eval()
            print(f"Model loaded successfully on GPU {local_rank}!")
        except Exception as e:
            print(f"Error loading model on GPU {local_rank}: {str(e)}")
            raise
    else:
        print("DRY RUN MODE - No model loading required")
        tokenizer, model = None, None

    # Find all video files - either from checklist or by scanning directory
    if checklist_file:
        print(f"Using video checklist: {checklist_file}")
        video_paths = load_video_checklist(checklist_file, data_root)
    else:
        print(f"Scanning for videos in {data_root}...")
        video_paths = find_video_files(data_root, dataset_name, render_type)
    
    if not video_paths:
        print("No video files found!")
        return

    print(f"Found {len(video_paths)} videos to process")
    print(f"Data root: {data_root}")
    print(f"Using {num_gpus} GPU(s)")
    print(f"Local rank: {local_rank}, World size: {world_size}")

    # Filter videos that need processing (empty or missing prompts)
    videos_to_process = []
    for video_path in video_paths:
        save_path = get_prompt_save_path(video_path)
        
        # Check if prompt file exists and is not empty
        needs_processing = True
        if save_path.exists():
            try:
                existing_content = save_path.read_text().strip()
                if existing_content and skip_existing:
                    needs_processing = False
            except Exception:
                # If we can't read the file, assume it needs processing
                pass
        
        if needs_processing:
            videos_to_process.append(video_path)

    print(f"Videos that need processing: {len(videos_to_process)} out of {len(video_paths)}")

    # Split videos among GPUs
    if num_gpus > 1 and len(videos_to_process) > 0:
        videos_per_gpu = len(videos_to_process) // num_gpus
        start_idx = local_rank * videos_per_gpu
        end_idx = start_idx + videos_per_gpu
        if local_rank == num_gpus - 1:  # Last GPU gets remaining videos
            end_idx = len(videos_to_process)
        
        gpu_videos = videos_to_process[start_idx:end_idx]
        print(f"GPU {local_rank}: Processing {len(gpu_videos)} videos (indices {start_idx}-{end_idx-1})")
    else:
        gpu_videos = videos_to_process
        if num_gpus > 1:
            print(f"GPU {local_rank}: Processing {len(gpu_videos)} videos")

    # Process videos assigned to this GPU
    for i, video_path in enumerate(gpu_videos):
        try:
            print(f"Processing video {i+1}/{len(gpu_videos)}: {video_path}")
            
            # Check if video file exists
            if not os.path.exists(video_path):
                print(f"Video file not found: {video_path}")
                continue

            # Determine save path
            save_path = get_prompt_save_path(video_path)
            
            # Note: We already filtered videos that need processing, so we can skip the check here
            # But we'll keep it for safety in case the filtering logic changes
            if save_path.exists():
                try:
                    existing_content = save_path.read_text().strip()
                    if existing_content and skip_existing:
                        print(f"Skipping {video_path} - prompt file already exists and is not empty")
                        continue
                except Exception as e:
                    print(f"Error reading existing prompt file {save_path}: {str(e)}")
                    print(f"Processing {video_path} - will overwrite existing file")
            else:
                print(f"Processing {video_path} - no existing prompt file")

            if dry_run:
                print(f"[DRY RUN] Would process: {video_path}")
                print(f"[DRY RUN] Would save to: {save_path}")
            else:
                # Generate prompt for the video
                generated_text = process_single_video(
                    model, tokenizer, video_path, audio_path, query, device
                )
                
                # Create prompts directory if it doesn't exist
                save_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Save the generated prompt
                with open(save_path, 'w') as f:
                    f.write(generated_text)
                
                print(f"Saved prompt to: {save_path}")
                
                # Remove from checklist if requested
                if checklist_file and remove_from_checklist_after_processing:
                    remove_from_checklist(checklist_file, video_path, data_root)

        except Exception as e:
            print(f"Error processing video {video_path}: {str(e)}")
            continue

    # Clean up distributed processing
    if num_gpus > 1:
        try:
            dist.destroy_process_group()
        except Exception as e:
            print(f"Warning: Failed to destroy process group: {e}")

    if dry_run:
        print("DRY RUN completed! No files were actually processed.")
    else:
        print("Processing completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_path", type=str, default="lmms-lab/EgoGPT-7b-EgoIT-EgoLife"
    )
    parser.add_argument("--data_root", type=str, default="./data",
                       help="Data root directory (default: ./data)")
    parser.add_argument("--audio_path", type=str, default=None)
    parser.add_argument(
        "--query", type=str, default="Please describe the video in detail."
    )
    parser.add_argument("--dry_run", action="store_true", 
                       help="Dry run mode - check which files need processing without actually processing them")
    parser.add_argument("--skip_existing", action="store_true", default=True,
                       help="Skip videos that already have non-empty prompt files (default: True)")
    parser.add_argument("--no_skip_existing", action="store_false", dest="skip_existing",
                       help="Process all videos, overwriting existing prompt files")
    parser.add_argument("--num_gpus", type=int, default=1,
                       help="Number of GPUs to use for processing (default: 1)")
    parser.add_argument("--checklist_file", type=str, default=None,
                       help="Path to video checklist file (optional, if not provided will scan directory)")
    parser.add_argument("--dataset_name", type=str, default="trumans",
                       help="Dataset name (default: trumans)")
    parser.add_argument("--render_type", type=str, default="ego_render_fov90",
                       help="Render type (default: ego_render_fov90)")
    parser.add_argument("--remove_from_checklist", action="store_true",
                       help="Remove processed videos from checklist file after successful processing")
    args = parser.parse_args()
    main(args.pretrained_path, args.data_root, args.audio_path, args.query, args.dry_run, args.skip_existing, args.num_gpus, args.checklist_file, args.dataset_name, args.render_type, args.remove_from_checklist)
