#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPO_ROOT / "data"
DEFAULT_OUTPUT_ROOT = DEFAULT_DATA_ROOT / "custom_inputs"
DEFAULT_DATASET_FILE_ROOT = DEFAULT_DATA_ROOT / "dataset_files" / "custom_inputs"
DEFAULT_CAPTION_PROMPT = REPO_ROOT / "data_processing" / "video_caption" / "prompt" / "caption_ego.txt"
DEFAULT_REWRITE_PROMPT = REPO_ROOT / "data_processing" / "video_caption" / "prompt" / "rewrite.txt"
DEFAULT_CAPTION_MODEL = "OpenGVLab/InternVL2-40B-AWQ"
DEFAULT_REWRITE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    avg_frame_rate: str
    r_frame_rate: str
    duration_sec: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a single custom hand-object interaction video into the DWM sample layout."
    )
    parser.add_argument("--input_video", type=Path, required=True)
    parser.add_argument("--sample_name", type=str, default=None)
    parser.add_argument("--group_name", type=str, default="CustomInputs")
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset_file_root", type=Path, default=DEFAULT_DATASET_FILE_ROOT)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--target_fps", type=int, default=8)
    parser.add_argument("--target_frames", type=int, default=49)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--resize_policy", choices=["auto", "pad", "crop"], default="auto")
    parser.add_argument("--hand_backend", choices=["original", "mediapipe"], default="original")
    parser.add_argument("--ffmpeg_bin", type=str, default="ffmpeg")
    parser.add_argument("--ffprobe_bin", type=str, default="ffprobe")
    parser.add_argument("--caption_model_path", type=str, default=DEFAULT_CAPTION_MODEL)
    parser.add_argument("--caption_prompt_file", type=Path, default=DEFAULT_CAPTION_PROMPT)
    parser.add_argument("--caption_num_sampled_frames", type=int, default=16)
    parser.add_argument("--caption_tensor_parallel_size", type=int, default=None)
    parser.add_argument("--caption_gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--rewrite_model_name", type=str, default=DEFAULT_REWRITE_MODEL)
    parser.add_argument("--rewrite_prompt_file", type=Path, default=DEFAULT_REWRITE_PROMPT)
    parser.add_argument("--rewrite_engine", choices=["auto", "vllm", "transformers"], default="auto")
    parser.add_argument("--rewrite_tensor_parallel_size", type=int, default=None)
    parser.add_argument("--rewrite_gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def run_command(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    print(f"[CMD] {shlex.join(cmd)}")
    subprocess.run(
        cmd,
        check=True,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
    )


def parse_fraction(text: str) -> float:
    if not text or text == "0/0":
        return 0.0
    try:
        return float(Fraction(text))
    except (ValueError, ZeroDivisionError):
        return 0.0


def probe_video(video_path: Path, ffprobe_bin: str) -> VideoInfo:
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    stream = payload["streams"][0]
    avg_frame_rate = str(stream.get("avg_frame_rate", "0/0"))
    r_frame_rate = str(stream.get("r_frame_rate", "0/0"))
    fps = parse_fraction(avg_frame_rate) or parse_fraction(r_frame_rate)
    duration_raw = payload.get("format", {}).get("duration")
    duration_sec = float(duration_raw) if duration_raw not in (None, "N/A") else None
    return VideoInfo(
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps=fps,
        avg_frame_rate=avg_frame_rate,
        r_frame_rate=r_frame_rate,
        duration_sec=duration_sec,
    )


def pad_penalty(src_width: int, src_height: int, dst_width: int, dst_height: int) -> float:
    scale = min(dst_width / src_width, dst_height / src_height)
    scaled_area = (src_width * scale) * (src_height * scale)
    target_area = dst_width * dst_height
    return max(0.0, 1.0 - (scaled_area / target_area))


def crop_penalty(src_width: int, src_height: int, dst_width: int, dst_height: int) -> float:
    scale = max(dst_width / src_width, dst_height / src_height)
    covered_area = (src_width * scale) * (src_height * scale)
    target_area = dst_width * dst_height
    return max(0.0, 1.0 - (target_area / covered_area))


def choose_resize_mode(info: VideoInfo, dst_width: int, dst_height: int, resize_policy: str) -> tuple[str, float, float]:
    pad_loss = pad_penalty(info.width, info.height, dst_width, dst_height)
    crop_loss = crop_penalty(info.width, info.height, dst_width, dst_height)
    if resize_policy != "auto":
        return resize_policy, pad_loss, crop_loss
    if crop_loss < pad_loss:
        return "crop", pad_loss, crop_loss
    return "pad", pad_loss, crop_loss


def build_resize_filter(mode: str, dst_width: int, dst_height: int) -> str:
    if mode == "crop":
        return (
            f"scale={dst_width}:{dst_height}:force_original_aspect_ratio=increase,"
            f"crop={dst_width}:{dst_height},setsar=1"
        )
    return (
        f"scale={dst_width}:{dst_height}:force_original_aspect_ratio=decrease,"
        f"pad={dst_width}:{dst_height}:(ow-iw)/2:(oh-ih)/2:black,setsar=1"
    )


def prepare_clip(
    input_video: Path,
    output_video: Path,
    *,
    ffmpeg_bin: str,
    start_frame: int,
    target_fps: int,
    target_frames: int,
    width: int,
    height: int,
    resize_mode: str,
    overwrite: bool,
) -> None:
    output_video.parent.mkdir(parents=True, exist_ok=True)
    stop_duration = max(1.0, target_frames / max(1, target_fps))
    vf = ",".join(
        [
            f"trim=start_frame={start_frame}",
            "setpts=PTS-STARTPTS",
            f"fps={target_fps}",
            build_resize_filter(resize_mode, width, height),
            f"tpad=stop_mode=clone:stop_duration={stop_duration:.3f}",
        ]
    )
    cmd = [
        ffmpeg_bin,
        "-y" if overwrite else "-n",
        "-i",
        str(input_video),
        "-vf",
        vf,
        "-frames:v",
        str(target_frames),
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "medium",
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(output_video),
    ]
    run_command(cmd)


def prepare_static_video(
    processed_video: Path,
    output_video: Path,
    *,
    ffmpeg_bin: str,
    target_fps: int,
    target_frames: int,
    overwrite: bool,
) -> None:
    output_video.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dwm_custom_static_") as tmp_dir:
        first_frame = Path(tmp_dir) / "first_frame.png"
        extract_cmd = [
            ffmpeg_bin,
            "-y" if overwrite else "-n",
            "-i",
            str(processed_video),
            "-vf",
            "select=eq(n\\,0)",
            "-frames:v",
            "1",
            str(first_frame),
        ]
        run_command(extract_cmd)

        static_cmd = [
            ffmpeg_bin,
            "-y" if overwrite else "-n",
            "-loop",
            "1",
            "-framerate",
            str(target_fps),
            "-i",
            str(first_frame),
            "-frames:v",
            str(target_frames),
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "medium",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(output_video),
        ]
        run_command(static_cmd)


def build_runtime_env() -> dict[str, str]:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    repo_root_str = str(REPO_ROOT)
    env["PYTHONPATH"] = repo_root_str if not existing_pythonpath else f"{repo_root_str}:{existing_pythonpath}"
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    env["TOKENIZERS_PARALLELISM"] = "false"
    return env


def render_hand_video(
    *,
    data_root: Path,
    relative_video_path: str,
    backend: str,
    target_fps: int,
    skip_existing: bool,
    overwrite: bool,
) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", prefix="dwm_custom_hands_", delete=False, encoding="utf-8") as handle:
        handle.write(relative_video_path + "\n")
        video_list_file = Path(handle.name)

    try:
        if backend == "mediapipe":
            script = REPO_ROOT / "data_processing" / "hands" / "render_videos_hands_hamer.py"
            cmd = [
                sys.executable,
                str(script),
                "--data_root",
                str(data_root),
                "--video_list_file",
                str(video_list_file),
                "--fps",
                str(target_fps),
            ]
        else:
            script = REPO_ROOT / "data_processing" / "hands" / "render_videos_hands_hamer_original.py"
            cmd = [
                sys.executable,
                str(script),
                "--data_root",
                str(data_root),
                "--video_list_file",
                str(video_list_file),
                "--fps",
                str(target_fps),
            ]
        if skip_existing:
            cmd.append("--skip_existing")
        elif overwrite:
            cmd.append("--overwrite")
        run_command(cmd, env=build_runtime_env(), cwd=REPO_ROOT)
    finally:
        video_list_file.unlink(missing_ok=True)


def create_temp_caption_root(
    sample_name: str,
    *,
    processed_video: Path,
    prompts_dir: Path,
    prompts_rewrite_dir: Path,
) -> tempfile.TemporaryDirectory[str]:
    temp_dir = tempfile.TemporaryDirectory(prefix="dwm_custom_caption_root_")
    root = Path(temp_dir.name)
    sample_root = root / "CustomInputs" / sample_name
    videos_dir = sample_root / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    (videos_dir / processed_video.name).symlink_to(processed_video)
    (sample_root / "prompts").symlink_to(prompts_dir, target_is_directory=True)
    (sample_root / "prompts_rewrite").symlink_to(prompts_rewrite_dir, target_is_directory=True)
    return temp_dir


def run_captioning(
    *,
    temp_root: Path,
    caption_prompt_file: Path,
    caption_model_path: str,
    caption_num_sampled_frames: int,
    caption_tensor_parallel_size: int | None,
    caption_gpu_memory_utilization: float,
    skip_existing: bool,
) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "data_processing" / "video_caption" / "internvl2_video_recaptioning.py"),
        "--root_dir",
        str(temp_root),
        "--video_type",
        "egocentric",
        "--video_folder_name",
        "videos",
        "--output_folder_name",
        "prompts",
        "--input_prompt_file",
        str(caption_prompt_file),
        "--model_path",
        caption_model_path,
        "--num_sampled_frames",
        str(caption_num_sampled_frames),
        "--frame_sample_method",
        "uniform",
        "--batch_size",
        "1",
        "--num_workers",
        "4",
        "--dataset_type",
        "taste_rob",
        "--save_format",
        "txt",
        "--split_by",
        "video",
        "--num_splits",
        "1",
        "--gpu_memory_utilization",
        str(caption_gpu_memory_utilization),
    ]
    if caption_tensor_parallel_size is not None:
        cmd.extend(["--tensor_parallel_size", str(caption_tensor_parallel_size)])
    if skip_existing:
        cmd.append("--skip_existing")
    run_command(cmd, env=build_runtime_env(), cwd=REPO_ROOT)


def run_rewrite(
    *,
    temp_root: Path,
    rewrite_prompt_file: Path,
    rewrite_model_name: str,
    rewrite_engine: str,
    rewrite_tensor_parallel_size: int | None,
    rewrite_gpu_memory_utilization: float,
    skip_existing: bool,
) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "data_processing" / "video_caption" / "caption_rewrite.py"),
        "--root_dir",
        str(temp_root),
        "--prompt_subdir",
        "prompts",
        "--output_folder_name",
        "prompts_rewrite",
        "--prompt_file",
        str(rewrite_prompt_file),
        "--model_name",
        rewrite_model_name,
        "--engine",
        rewrite_engine,
        "--num_splits",
        "1",
        "--gpu_memory_utilization",
        str(rewrite_gpu_memory_utilization),
    ]
    if rewrite_tensor_parallel_size is not None:
        cmd.extend(["--tensor_parallel_size", str(rewrite_tensor_parallel_size)])
    if skip_existing:
        cmd.append("--skip_existing")
    run_command(cmd, env=build_runtime_env(), cwd=REPO_ROOT)


def write_dataset_file(dataset_file_root: Path, sample_name: str, relative_video_path: str) -> Path:
    dataset_file_root.mkdir(parents=True, exist_ok=True)
    dataset_file = dataset_file_root / f"{sample_name}.txt"
    dataset_file.write_text(relative_video_path + "\n", encoding="utf-8")
    return dataset_file


def relative_to_root(path: Path, root: Path, *, error_label: str) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"{error_label}: {path} is not under {root}."
        ) from exc


def maybe_prepare_processed_assets(
    *,
    args: argparse.Namespace,
    processed_video: Path,
    static_video: Path,
    resize_mode: str,
) -> None:
    if not (args.skip_existing and processed_video.exists()):
        prepare_clip(
            input_video=args.input_video,
            output_video=processed_video,
            ffmpeg_bin=args.ffmpeg_bin,
            start_frame=args.start_frame,
            target_fps=args.target_fps,
            target_frames=args.target_frames,
            width=args.width,
            height=args.height,
            resize_mode=resize_mode,
            overwrite=args.overwrite or not args.skip_existing,
        )

    if not (args.skip_existing and static_video.exists()):
        prepare_static_video(
            processed_video=processed_video,
            output_video=static_video,
            ffmpeg_bin=args.ffmpeg_bin,
            target_fps=args.target_fps,
            target_frames=args.target_frames,
            overwrite=args.overwrite or not args.skip_existing,
        )


def main() -> int:
    args = parse_args()
    if args.start_frame < 0:
        raise SystemExit("--start_frame must be >= 0")
    if args.target_fps < 1 or args.target_frames < 1:
        raise SystemExit("--target_fps and --target_frames must be >= 1")
    if args.width < 1 or args.height < 1:
        raise SystemExit("--width and --height must be >= 1")

    input_video = args.input_video.resolve()
    if not input_video.exists():
        raise SystemExit(f"input_video not found: {input_video}")
    if not args.caption_prompt_file.exists():
        raise SystemExit(f"caption_prompt_file not found: {args.caption_prompt_file}")
    if not args.rewrite_prompt_file.exists():
        raise SystemExit(f"rewrite_prompt_file not found: {args.rewrite_prompt_file}")

    sample_name = args.sample_name or input_video.stem
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    sample_root = output_root

    videos_dir = output_root / "videos"
    videos_static_dir = output_root / "videos_static"
    videos_hands_dir = output_root / "videos_hands"
    prompts_dir = output_root / "prompts"
    prompts_rewrite_dir = output_root / "prompts_rewrite"
    metadata_dir = output_root / "metadata"

    clip_name = f"{sample_name}.mp4"
    prompt_name = f"{sample_name}.txt"
    processed_video = videos_dir / clip_name
    static_video = videos_static_dir / clip_name
    hand_video = videos_hands_dir / clip_name
    prompt_path = prompts_dir / prompt_name
    rewritten_prompt_path = prompts_rewrite_dir / prompt_name

    for directory in (videos_dir, videos_static_dir, videos_hands_dir, prompts_dir, prompts_rewrite_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)

    video_info = probe_video(input_video, args.ffprobe_bin)
    resize_mode, pad_loss, crop_loss = choose_resize_mode(video_info, args.width, args.height, args.resize_policy)

    maybe_prepare_processed_assets(
        args=args,
        processed_video=processed_video,
        static_video=static_video,
        resize_mode=resize_mode,
    )

    relative_video_path = relative_to_root(
        processed_video,
        data_root,
        error_label="Prepared sample path must stay under data_root for seamless inference",
    )
    hand_relative_video_path = relative_to_root(
        processed_video,
        output_root,
        error_label="Prepared sample path must stay under output_root for HaMeR rendering",
    )
    render_hand_video(
        data_root=output_root,
        relative_video_path=hand_relative_video_path,
        backend=args.hand_backend,
        target_fps=args.target_fps,
        skip_existing=args.skip_existing,
        overwrite=args.overwrite or not args.skip_existing,
    )

    with create_temp_caption_root(
        sample_name,
        processed_video=processed_video,
        prompts_dir=prompts_dir,
        prompts_rewrite_dir=prompts_rewrite_dir,
    ) as temp_root_str:
        temp_root = Path(temp_root_str)
        run_captioning(
            temp_root=temp_root,
            caption_prompt_file=args.caption_prompt_file,
            caption_model_path=args.caption_model_path,
            caption_num_sampled_frames=args.caption_num_sampled_frames,
            caption_tensor_parallel_size=args.caption_tensor_parallel_size,
            caption_gpu_memory_utilization=args.caption_gpu_memory_utilization,
            skip_existing=args.skip_existing,
        )
        run_rewrite(
            temp_root=temp_root,
            rewrite_prompt_file=args.rewrite_prompt_file,
            rewrite_model_name=args.rewrite_model_name,
            rewrite_engine=args.rewrite_engine,
            rewrite_tensor_parallel_size=args.rewrite_tensor_parallel_size,
            rewrite_gpu_memory_utilization=args.rewrite_gpu_memory_utilization,
            skip_existing=args.skip_existing,
        )

    dataset_file = write_dataset_file(
        dataset_file_root=args.dataset_file_root.resolve(),
        sample_name=sample_name,
        relative_video_path=relative_video_path,
    )
    metadata_path = metadata_dir / f"{sample_name}.json"

    metadata = {
        "input_video": str(input_video),
        "sample_name": sample_name,
        "data_root": str(data_root),
        "sample_root": str(sample_root),
        "relative_video_path": relative_video_path,
        "dataset_file": str(dataset_file),
        "clip_stem": sample_name,
        "start_frame": args.start_frame,
        "target_fps": args.target_fps,
        "target_frames": args.target_frames,
        "width": args.width,
        "height": args.height,
        "resize_policy": args.resize_policy,
        "selected_resize_mode": resize_mode,
        "pad_loss": pad_loss,
        "crop_loss": crop_loss,
        "source_width": video_info.width,
        "source_height": video_info.height,
        "source_fps": video_info.fps,
        "source_avg_frame_rate": video_info.avg_frame_rate,
        "source_r_frame_rate": video_info.r_frame_rate,
        "source_duration_sec": video_info.duration_sec,
        "hand_backend": args.hand_backend,
        "caption_model_path": args.caption_model_path,
        "rewrite_model_name": args.rewrite_model_name,
        "paths": {
            "video": str(processed_video),
            "static_video": str(static_video),
            "hand_video": str(hand_video),
            "prompt": str(prompt_path),
            "prompt_rewrite": str(rewritten_prompt_path),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print("[DONE]")
    print(f"sample_root={sample_root}")
    print(f"video={processed_video}")
    print(f"static_video={static_video}")
    print(f"hand_video={hand_video}")
    print(f"prompt={prompt_path}")
    print(f"prompt_rewrite={rewritten_prompt_path}")
    print(f"metadata={metadata_path}")
    print(f"dataset_file={dataset_file}")
    print(f"relative_video_path={relative_video_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
