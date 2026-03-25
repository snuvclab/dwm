#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

try:
    from check_rendering_status import get_animation_names
except ImportError:
    from data_processing.trumans.check_rendering_status import get_animation_names


REPO_ROOT = Path(__file__).resolve().parents[2]


def find_blend_file(recordings_root: Path, scene_name: str | None) -> Path:
    blend_files = sorted(recordings_root.glob("*/*.blend"))
    if not blend_files:
        raise FileNotFoundError(f"No blend files found under {recordings_root}")
    if scene_name is None:
        return blend_files[0]
    for blend_file in blend_files:
        if blend_file.parent.name == scene_name:
            return blend_file
    raise FileNotFoundError(f"Scene not found: {scene_name}")


def resolve_animation(blend_file: Path, animation_name: str | None) -> tuple[int, str]:
    animation_names = get_animation_names(str(blend_file))
    if not animation_names:
        raise RuntimeError(f"No animations found in {blend_file}")
    if animation_name is None:
        return 0, animation_names[0]
    for index, candidate in enumerate(animation_names):
        if candidate == animation_name or candidate.replace(".pkl", "") == animation_name:
            return index, candidate
    raise ValueError(f"Animation {animation_name} not found in {blend_file}")


def run_blender(blend_file: Path, script_relpath: str, extra_args: list[str], gpu_id: int) -> None:
    script_path = REPO_ROOT / script_relpath
    env = dict(**os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cmd = [
        "blender",
        "-b",
        str(blend_file),
        "-P",
        str(script_path),
        "--",
        *extra_args,
    ]
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, env=env, check=True)


def copy_selected_clips(source_dir: Path, target_dir: Path, stems: Iterable[str]) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        src = source_dir / f"{stem}.mp4"
        if not src.exists():
            raise FileNotFoundError(f"Missing source clip: {src}")
        shutil.copy2(src, target_dir / src.name)


def copy_sorted_clips(source_dir: Path, target_dir: Path, clip_count: int) -> list[Path]:
    clips = sorted(source_dir.glob("*.mp4"))
    if len(clips) < clip_count:
        raise RuntimeError(f"Requested {clip_count} clips but only found {len(clips)} under {source_dir}")

    target_dir.mkdir(parents=True, exist_ok=True)
    selected = clips[:clip_count]
    for index, src in enumerate(selected):
        shutil.copy2(src, target_dir / f"{index:05d}.mp4")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a small TRUMANS smoke set into data_refactor.")
    parser.add_argument("--recordings_root", type=Path, default=Path("data/trumans/Recordings_blend"))
    parser.add_argument("--output_root", type=Path, default=Path("data_refactor/trumans"))
    parser.add_argument("--scene", type=str, default=None)
    parser.add_argument("--animation", type=str, default=None)
    parser.add_argument("--clip_count", type=int, default=6)
    parser.add_argument("--frame_skip", type=int, default=3)
    parser.add_argument("--clip_length", type=int, default=49)
    parser.add_argument("--clip_stride", type=int, default=25)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    blend_file = find_blend_file(args.recordings_root, args.scene)
    animation_index, animation_name = resolve_animation(blend_file, args.animation)
    scene_name = blend_file.parent.name
    animation_stem = animation_name.replace(".pkl", "")

    raw_root = args.output_root / "_raw_blender"
    final_root = args.output_root / scene_name / animation_stem

    # Only render the frame range needed to produce the requested number of overlapping clips.
    rendered_frames_needed = args.clip_length + max(0, args.clip_count - 1) * args.clip_stride
    start_frame = 0
    end_frame = max(0, (rendered_frames_needed - 1) * args.frame_skip)

    if args.dry_run:
        print(f"scene={scene_name}")
        print(f"animation={animation_name}")
        print(f"start_frame={start_frame}")
        print(f"end_frame={end_frame}")
        print(f"raw_root={raw_root}")
        print(f"final_root={final_root}")
        return

    rgb_src = raw_root / scene_name / animation_stem / "videos"
    static_src = raw_root / scene_name / animation_stem / "sequences" / "videos_static"
    hands_src = raw_root / scene_name / animation_stem / "processed2" / "videos_hands"

    if not args.dry_run:
        run_blender(
            blend_file,
            "data_processing/trumans/blender_ego_video_render.py",
            [
                "--animation_index", str(animation_index),
                "--start_frame", str(start_frame),
                "--end_frame", str(end_frame),
                "--samples", str(args.samples),
                "--save-path", str(raw_root),
                "--frame-skip", str(args.frame_skip),
                "--video-output",
                "--auto-split-clips",
                "--clip-length", str(args.clip_length),
                "--clip-stride", str(args.clip_stride),
                "--fps", str(args.fps),
                "--skip-existing",
            ],
            args.gpu,
        )
        run_blender(
            blend_file,
            "data_processing/trumans/blender_ego_static.py",
            [
                "--animation_index", str(animation_index),
                "--start_frame", str(start_frame),
                "--end_frame", str(end_frame),
                "--samples", str(args.samples),
                "--save-path", str(raw_root),
                "--frame-skip", str(args.frame_skip),
                "--stride", str(args.clip_stride),
                "--clip-length", str(args.clip_length),
                "--skip-existing",
            ],
            args.gpu,
        )
        run_blender(
            blend_file,
            "data_processing/trumans/blender_ego_hand.py",
            [
                "--animation_index", str(animation_index),
                "--start_frame", str(start_frame),
                "--end_frame", str(end_frame),
                "--samples", str(args.samples),
                "--save-path", str(raw_root),
                "--frame-skip", str(args.frame_skip),
                "--stride", str(args.clip_stride),
                "--clip-length", str(args.clip_length),
                "--fps", str(args.fps),
                "--skip-existing",
            ],
            args.gpu,
        )

    rgb_selected = copy_sorted_clips(rgb_src, final_root / "videos", args.clip_count)
    static_selected = copy_sorted_clips(static_src, final_root / "videos_static", args.clip_count)
    hands_selected = copy_sorted_clips(hands_src, final_root / "videos_hands", args.clip_count)

    print(f"Wrote smoke clips to {final_root}")
    print(f"selected_rgb={[path.name for path in rgb_selected]}")
    print(f"selected_static={[path.name for path in static_selected]}")
    print(f"selected_hands={[path.name for path in hands_selected]}")


if __name__ == "__main__":
    main()
