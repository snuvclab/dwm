#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from resize_videos_taste_rob import build_output_parent, build_scene_case_map, process_video


VALID_HAND_TYPES = ("SingleHand", "DoubleHand")


def discover_videos(input_root: Path) -> list[Path]:
    return sorted(path for path in input_root.rglob("*.mp4") if path.is_file())


def select_videos(videos: list[Path], input_root: Path, clip_count: int, min_per_type: int) -> list[Path]:
    by_type = {hand_type: [] for hand_type in VALID_HAND_TYPES}
    for video in videos:
        rel = video.relative_to(input_root)
        if rel.parts and rel.parts[0] in by_type:
            by_type[rel.parts[0]].append(video)

    selected: list[Path] = []
    for hand_type in VALID_HAND_TYPES:
        selected.extend(by_type[hand_type][:min_per_type])

    if len(selected) < clip_count:
        for video in videos:
            if video in selected:
                continue
            selected.append(video)
            if len(selected) >= clip_count:
                break

    return selected[:clip_count]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a small TASTE-Rob smoke set under data_refactor.")
    parser.add_argument("--input_dir", type=Path, default=Path("/virtual_lab/dataset/taste_rob"))
    parser.add_argument("--output_dir", type=Path, default=Path("data_refactor/taste_rob_resized"))
    parser.add_argument("--clip_count", type=int, default=6)
    parser.add_argument("--min_per_type", type=int, default=1)
    parser.add_argument("--target_width", type=int, default=720)
    parser.add_argument("--target_height", type=int, default=480)
    parser.add_argument("--target_frames", type=int, default=49)
    parser.add_argument("--output_fps", type=int, default=8)
    parser.add_argument("--video_codec", type=str, default="libx264")
    parser.add_argument("--crf", type=int, default=23)
    parser.add_argument("--preset", type=str, default="medium")
    parser.add_argument("--static_crf", type=int, default=23)
    parser.add_argument("--static_preset", type=str, default="medium")
    parser.add_argument("--merge_scene_prefix", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--render_hands_backend", choices=["none", "mediapipe", "original"], default="none")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    videos = discover_videos(args.input_dir)
    if not videos:
        raise FileNotFoundError(f"No videos found under {args.input_dir}")
    selected = select_videos(videos, args.input_dir, clip_count=args.clip_count, min_per_type=args.min_per_type)
    scene_case_map = build_scene_case_map(selected, input_root=args.input_dir, merge_scene_prefix=args.merge_scene_prefix)

    if args.dry_run:
        for path in selected:
            print(path.relative_to(args.input_dir))
        return

    for video in selected:
        ok, message = process_video(
            video,
            args.input_dir,
            args.output_dir,
            args.target_width,
            args.target_height,
            args.target_frames,
            args.output_fps,
            args.video_codec,
            args.crf,
            args.preset,
            args.static_crf,
            args.static_preset,
            args.skip_existing,
            args.merge_scene_prefix,
            scene_case_map,
        )
        print(message)
        if not ok:
            raise RuntimeError(message)

    selected_list = args.output_dir / "_logs" / "taste_rob_smoke_selected.txt"
    output_video_relpaths = []
    for video in selected:
        rel_parent = video.relative_to(args.input_dir).parent
        out_parent = build_output_parent(
            rel_parent=rel_parent,
            input_root=args.input_dir,
            output_root=args.output_dir,
            merge_scene_prefix=args.merge_scene_prefix,
            scene_case_map=scene_case_map,
        )
        output_video_relpaths.append(str((out_parent / "videos" / video.name).relative_to(args.output_dir)).replace("\\", "/"))
    selected_list.parent.mkdir(parents=True, exist_ok=True)
    selected_list.write_text(
        "\n".join(output_video_relpaths) + "\n",
        encoding="utf-8",
    )

    if args.render_hands_backend != "none":
        cmd = [
            "bash",
            "data_processing/hands/run_render_hands_hamer.sh",
            "--backend",
            args.render_hands_backend,
            "--data_root",
            str(args.output_dir),
            "--video_list_file",
            str(selected_list),
            "--skip_existing",
        ]
        print("[RUN]", " ".join(cmd))
        subprocess.run(cmd, check=True)

    print(f"Prepared smoke TASTE-Rob data at {args.output_dir}")


if __name__ == "__main__":
    main()
