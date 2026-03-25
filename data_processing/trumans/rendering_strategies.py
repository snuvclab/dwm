#!/usr/bin/env python3

import os
import shutil
import subprocess
import tempfile


def resolve_execution_strategy(*, auto_split=False, direct_clips=False, default=None):
    if auto_split and direct_clips:
        raise ValueError("Choose only one execution strategy")
    if direct_clips:
        return "direct_clips"
    if auto_split:
        return "auto_split"
    return default


def build_frame_numbers(render_start_frame, render_end_frame, frame_skip):
    return list(range(render_start_frame, render_end_frame + 1, max(1, frame_skip)))


def build_clip_start_indices(total_frames, clip_length, clip_stride):
    if clip_length <= 0 or clip_stride <= 0 or total_frames < clip_length:
        return []
    return list(range(0, total_frames - clip_length + 1, clip_stride))


def build_clip_windows(frames_to_render, clip_length, clip_stride):
    windows = []
    for start_index in build_clip_start_indices(len(frames_to_render), clip_length, clip_stride):
        windows.append((start_index, frames_to_render[start_index:start_index + clip_length]))
    return windows


def create_video_clip_from_frames(frame_files, output_path, fps=8, min_size=1024):
    if not frame_files:
        return False

    temp_dir = tempfile.mkdtemp(prefix="truman_clip_")
    try:
        for index, frame_file in enumerate(frame_files, start=1):
            if os.path.exists(frame_file):
                target_file = os.path.join(temp_dir, f"frame_{index:04d}.png")
                shutil.copy2(frame_file, target_file)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        command = [
            "ffmpeg",
            "-y",
            "-start_number",
            "1",
            "-framerate",
            str(fps),
            "-i",
            os.path.join(temp_dir, "frame_%04d.png"),
            "-frames:v",
            str(len(frame_files)),
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "medium",
            "-pix_fmt",
            "yuv420p",
            output_path,
        ]
        subprocess.run(command, capture_output=True, check=True)
        return os.path.exists(output_path) and os.path.getsize(output_path) >= min_size
    except subprocess.CalledProcessError as exc:
        print(f"Error creating video clip: {exc}")
        if exc.stderr:
            print(f"FFmpeg error: {exc.stderr.decode()}")
        return False
    except Exception as exc:
        print(f"Error creating video clip: {exc}")
        return False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
