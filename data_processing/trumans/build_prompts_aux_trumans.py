#!/usr/bin/env python3
"""Build Trumans prompts_aux JSON with action_hints from Actions/*.txt."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ActionAnn:
    start: int
    end: int
    desc: str


@dataclass
class Stats:
    actions_seen: int = 0
    clips_seen: int = 0
    clips_updated: int = 0
    clips_missing_prompt: int = 0
    clips_skipped_existing: int = 0
    actions_missing_annotation: int = 0
    clips_non_numeric: int = 0


def natural_sort_key(path: Path):
    if path.stem.isdigit():
        return (0, int(path.stem))
    return (1, path.stem)


def load_scene_filter(scene_filter_file: Path | None) -> set[str] | None:
    if scene_filter_file is None:
        return None
    if not scene_filter_file.exists():
        raise FileNotFoundError(f"Scene filter file not found: {scene_filter_file}")
    scenes = set()
    for line in scene_filter_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        scenes.add(line)
    return scenes


def parse_action_file(path: Path) -> list[ActionAnn]:
    if not path.exists():
        return []
    anns: list[ActionAnn] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        try:
            start = int(parts[0])
            end = int(parts[1])
        except ValueError:
            continue
        desc = parts[2].strip()
        anns.append(ActionAnn(start=start, end=end, desc=desc))
    return anns


def clip_frame_range(clip_id: int, clip_length: int, clip_stride: int, frame_skip: int) -> tuple[int, int]:
    start_seq = clip_id * clip_stride
    end_seq = start_seq + clip_length - 1
    return start_seq * frame_skip, end_seq * frame_skip


def build_action_hints(
    anns: list[ActionAnn],
    start_frame: int,
    end_frame: int,
    clip_length: int,
) -> list[str]:
    overlaps = []
    for ann in anns:
        if end_frame < ann.start or start_frame > ann.end:
            continue
        overlap_start = max(start_frame, ann.start)
        overlap_end = min(end_frame, ann.end)
        overlap_duration = overlap_end - overlap_start + 1
        overlap_ratio = overlap_duration / max(1, clip_length)
        overlaps.append((overlap_ratio, ann.desc))

    overlaps.sort(key=lambda x: x[0], reverse=True)
    hints = []
    seen = set()
    for _ratio, desc in overlaps:
        if desc in seen:
            continue
        seen.add(desc)
        hints.append(desc)
    return hints


def iter_actions(root_dir: Path, scene_filter: set[str] | None):
    for scene_dir in sorted([p for p in root_dir.iterdir() if p.is_dir()]):
        if scene_filter is not None and scene_dir.name not in scene_filter:
            continue
        for action_dir in sorted([p for p in scene_dir.iterdir() if p.is_dir()]):
            yield action_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attach action_hints to Trumans prompts_aux JSON.")
    parser.add_argument("--root_dir", type=Path, required=True, help="Root like data/trumans/ego_render_fov90_new")
    parser.add_argument("--actions_root", type=Path, default=Path("data/trumans/Actions"))
    parser.add_argument("--third_prompt_dirname", type=str, default="prompts_aux")
    parser.add_argument("--third_video_dirname", type=str, default="videos_third")
    parser.add_argument("--clip_length", type=int, default=49)
    parser.add_argument("--clip_stride", type=int, default=25)
    parser.add_argument("--frame_skip", type=int, default=3)
    parser.add_argument("--scene_filter_file", type=Path, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.root_dir.exists():
        raise SystemExit(f"root_dir not found: {args.root_dir}")
    if not args.actions_root.exists():
        raise SystemExit(f"actions_root not found: {args.actions_root}")

    scene_filter = load_scene_filter(args.scene_filter_file)
    stats = Stats()

    for action_dir in iter_actions(args.root_dir, scene_filter):
        stats.actions_seen += 1
        prompt_dir = action_dir / args.third_prompt_dirname
        video_dir = action_dir / args.third_video_dirname
        if not video_dir.exists():
            continue

        action_file = args.actions_root / f"{action_dir.name}.txt"
        anns = parse_action_file(action_file)
        if not anns:
            stats.actions_missing_annotation += 1

        for video_file in sorted(video_dir.glob("*.mp4"), key=natural_sort_key):
            stats.clips_seen += 1
            if not video_file.stem.isdigit():
                stats.clips_non_numeric += 1
                continue

            clip_id = int(video_file.stem)
            prompt_file = prompt_dir / f"{clip_id:05d}.json"
            if not prompt_file.exists():
                stats.clips_missing_prompt += 1
                print(f"[MISS-PROMPT] {prompt_file}")
                continue

            try:
                data = json.loads(prompt_file.read_text(encoding="utf-8"))
            except Exception:
                data = {}

            if args.skip_existing and isinstance(data.get("action_hints"), list) and len(data["action_hints"]) > 0:
                stats.clips_skipped_existing += 1
                continue

            start_frame, end_frame = clip_frame_range(
                clip_id=clip_id,
                clip_length=args.clip_length,
                clip_stride=args.clip_stride,
                frame_skip=args.frame_skip,
            )
            hints = build_action_hints(
                anns=anns,
                start_frame=start_frame,
                end_frame=end_frame,
                clip_length=args.clip_length,
            )

            prompt = data.get("prompt", "")
            payload = {
                "prompt": prompt,
                "action_hints": hints,
            }

            if args.dry_run:
                stats.clips_updated += 1
                print(f"[DRY-RUN] {prompt_file} hints={len(hints)}")
                continue

            prompt_dir.mkdir(parents=True, exist_ok=True)
            prompt_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            stats.clips_updated += 1

    print("\n=== Summary ===")
    print(f"actions_seen={stats.actions_seen}")
    print(f"actions_missing_annotation={stats.actions_missing_annotation}")
    print(f"clips_seen={stats.clips_seen}")
    print(f"clips_updated={stats.clips_updated}")
    print(f"clips_missing_prompt={stats.clips_missing_prompt}")
    print(f"clips_skipped_existing={stats.clips_skipped_existing}")
    print(f"clips_non_numeric={stats.clips_non_numeric}")


if __name__ == "__main__":
    main()
