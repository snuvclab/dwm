#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from backend_common import HamerSequenceResult
from mano_overlay_video import save_mano_render_video
from original_extractor import HamerOriginalExtractor


VALID_TYPES = {"SingleHand", "DoubleHand"}


@dataclass(frozen=True)
class Task:
    input_video: str
    output_video: str


def discover_videos(data_root: Path, video_dir_name: str) -> Iterable[Path]:
    pattern = f"**/{video_dir_name}/*.mp4"
    for video_path in sorted(data_root.glob(pattern)):
        if video_path.is_file():
            yield video_path


def classify_type(video_path: Path) -> str | None:
    for part in video_path.parts:
        if part in VALID_TYPES:
            return part
    return None


def extract_scene(video_path: Path, video_dir_name: str) -> str | None:
    if video_path.parent.name != video_dir_name:
        return None
    return video_path.parent.parent.name if video_path.parent.parent else None


def load_video_list_file(data_root: Path, list_file: Path) -> tuple[list[Path], int]:
    videos: list[Path] = []
    missing_count = 0
    if not list_file.exists():
        raise FileNotFoundError(f"video_list_file not found: {list_file}")
    with list_file.open("r", encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            rel = Path(line)
            if rel.is_absolute():
                print(f"[WARN] {list_file}:{lineno} absolute path is not allowed, skip: {line}")
                continue
            abs_path = (data_root / rel).resolve()
            if not abs_path.exists():
                missing_count += 1
                print(f"[WARN] {list_file}:{lineno} file not found, skip: {line}")
                continue
            videos.append(abs_path)
    return sorted(set(videos)), missing_count


def relative_to_root(path: Path, data_root: Path) -> str:
    return str(path.resolve().relative_to(data_root.resolve()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render videos_hands with original HaMeR (single GPU, model cached).")
    parser.add_argument("--data_root", type=Path, default=Path("data_refactor/taste_rob_resized"))
    parser.add_argument("--video_dir_name", type=str, default="videos")
    parser.add_argument("--output_dir_name", type=str, default="videos_hands")
    parser.add_argument("--hamer_root", type=Path, default=Path("third_party/hamer"))
    parser.add_argument("--hand_type", type=str, default="all", choices=["all", "SingleHand", "DoubleHand"])
    parser.add_argument("--scene", nargs="+", default=None)
    parser.add_argument("--video_list_file", type=Path, default=None)
    parser.add_argument("--save_video_list", type=Path, default=None)
    parser.add_argument("--progress_every", type=int, default=5)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--log_jsonl", type=Path, default=None)
    parser.add_argument("--split_index", type=int, default=0)
    parser.add_argument("--num_splits", type=int, default=1)
    parser.add_argument("--enable_split", action="store_true")
    parser.add_argument("--fail_fast", action="store_true")

    skip_group = parser.add_mutually_exclusive_group()
    skip_group.add_argument("--skip_existing", action="store_true", default=True)
    skip_group.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run_video_task(task: Task, extractor: HamerOriginalExtractor, fps: int | None) -> tuple[bool, str]:
    input_video = Path(task.input_video).resolve()
    output_video = Path(task.output_video).resolve()
    output_video.parent.mkdir(parents=True, exist_ok=True)
    sequence: HamerSequenceResult = extractor.run(input_video)
    save_mano_render_video(
        input_video,
        sequence,
        mano_faces=extractor.mano_faces,
        output_path=output_video,
        fps=fps,
        composite_on_video=False,
    )
    return True, f"rendered {input_video.name} -> {output_video}"


def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve()
    if not data_root.exists():
        raise SystemExit(f"data_root not found: {data_root}")
    log_jsonl = args.log_jsonl.resolve() if args.log_jsonl else (data_root / "_logs" / "hamer_render_original.jsonl")
    log_jsonl.parent.mkdir(parents=True, exist_ok=True)

    if args.video_list_file is None:
        candidates = list(discover_videos(data_root, args.video_dir_name))
        missing_from_list = 0
    else:
        candidates, missing_from_list = load_video_list_file(data_root, args.video_list_file)

    selected: list[Path] = []
    for video_path in candidates:
        hand_type = classify_type(video_path)
        if args.hand_type != "all" and hand_type != args.hand_type:
            continue
        scene = extract_scene(video_path, args.video_dir_name)
        if args.scene and scene not in set(args.scene):
            continue
        selected.append(video_path)

    if args.enable_split:
        selected = selected[args.split_index::args.num_splits]

    if args.save_video_list is not None:
        args.save_video_list.parent.mkdir(parents=True, exist_ok=True)
        args.save_video_list.write_text(
            "\n".join(relative_to_root(path, data_root) for path in selected) + "\n",
            encoding="utf-8",
        )

    extractor = HamerOriginalExtractor(hamer_root=args.hamer_root, checkpoint_path=args.checkpoint)

    processed = 0
    failed = 0
    with log_jsonl.open("a", encoding="utf-8") as log_handle:
        for index, input_video in enumerate(selected, start=1):
            output_video = input_video.parent.parent / args.output_dir_name / input_video.name
            if args.skip_existing and output_video.exists():
                continue
            try:
                ok, message = run_video_task(Task(str(input_video), str(output_video)), extractor, args.fps)
                processed += int(ok)
                print(message)
                log_handle.write(json.dumps({"video": str(input_video), "output": str(output_video), "ok": ok}) + "\n")
            except Exception as exc:
                failed += 1
                print(f"[ERROR] {input_video}: {exc}")
                log_handle.write(json.dumps({"video": str(input_video), "output": str(output_video), "ok": False, "error": str(exc)}) + "\n")
                if args.fail_fast:
                    raise
            if index % args.progress_every == 0:
                print(f"progress={index}/{len(selected)} processed={processed} failed={failed} missing_list={missing_from_list}")

    print(f"done processed={processed} failed={failed} selected={len(selected)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
