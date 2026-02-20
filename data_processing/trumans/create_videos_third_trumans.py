#!/usr/bin/env python3
"""Create third-person clips aligned with Trumans ego clips."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
from decord import VideoReader, cpu


@dataclass
class Stats:
    actions_seen: int = 0
    actions_missing_source: int = 0
    clips_seen: int = 0
    clips_created: int = 0
    clips_skipped_existing: int = 0
    clips_skipped_non_numeric: int = 0
    clips_skipped_short_source: int = 0


def natural_sort_key(path: Path):
    stem = path.stem
    if stem.isdigit():
        return (0, int(stem))
    return (1, stem)


def load_scene_filter(scene_filter_file: Path | None) -> set[str] | None:
    if scene_filter_file is None:
        return None
    if not scene_filter_file.exists():
        raise FileNotFoundError(f"Scene filter file not found: {scene_filter_file}")

    scenes: set[str] = set()
    for line in scene_filter_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        scenes.add(line)
    return scenes


def find_third_source(third_video_root: Path, action_name: str) -> Path | None:
    pkl_mp4 = third_video_root / f"{action_name}.pkl.mp4"
    if pkl_mp4.exists():
        return pkl_mp4
    mp4 = third_video_root / f"{action_name}.mp4"
    if mp4.exists():
        return mp4
    return None


def write_clip_from_source(
    vr: VideoReader,
    output_path: Path,
    frame_indices: list[int],
    fps: float,
) -> bool:
    total = len(vr)
    if not frame_indices:
        return False
    if max(frame_indices) >= total:
        return False

    frames = vr.get_batch(frame_indices).asnumpy()
    iio.imwrite(str(output_path), frames, fps=fps, codec="libx264")
    return True


def iter_actions(root_dir: Path, scene_filter: set[str] | None):
    for scene_dir in sorted([p for p in root_dir.iterdir() if p.is_dir()]):
        if scene_filter is not None and scene_dir.name not in scene_filter:
            continue
        for action_dir in sorted([p for p in scene_dir.iterdir() if p.is_dir()]):
            yield scene_dir, action_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create videos_third clips aligned with Trumans ego clips.")
    parser.add_argument("--root_dir", type=Path, required=True, help="Root like data/trumans/ego_render_fov90_new")
    parser.add_argument("--third_video_root", type=Path, default=Path("data/trumans/video_render"))
    parser.add_argument("--ego_videos_dirname", type=str, default="videos")
    parser.add_argument("--output_dirname", type=str, default="videos_third")
    parser.add_argument("--clip_length", type=int, default=49)
    parser.add_argument("--clip_stride", type=int, default=25)
    parser.add_argument("--frame_skip", type=int, default=3)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--scene_filter_file", type=Path, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.root_dir.exists():
        raise SystemExit(f"root_dir not found: {args.root_dir}")
    if not args.third_video_root.exists():
        raise SystemExit(f"third_video_root not found: {args.third_video_root}")

    scene_filter = load_scene_filter(args.scene_filter_file)
    stats = Stats()

    for _scene_dir, action_dir in iter_actions(args.root_dir, scene_filter):
        stats.actions_seen += 1

        ego_videos_dir = action_dir / args.ego_videos_dirname
        if not ego_videos_dir.exists():
            continue

        source_video = find_third_source(args.third_video_root, action_dir.name)
        if source_video is None:
            stats.actions_missing_source += 1
            print(f"[MISS-SOURCE] {action_dir}")
            continue

        output_dir = action_dir / args.output_dirname
        if not args.dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)

        try:
            vr = VideoReader(str(source_video), ctx=cpu(0))
        except Exception:
            print(f"[MISS-SOURCE-OPEN] {source_video}")
            stats.actions_missing_source += 1
            continue

        clip_files = sorted(ego_videos_dir.glob("*.mp4"), key=natural_sort_key)
        for clip_file in clip_files:
            stats.clips_seen += 1
            if not clip_file.stem.isdigit():
                stats.clips_skipped_non_numeric += 1
                print(f"[SKIP-NON-NUMERIC] {clip_file}")
                continue

            clip_id = int(clip_file.stem)
            output_path = output_dir / f"{clip_id:05d}.mp4"
            if args.skip_existing and output_path.exists():
                stats.clips_skipped_existing += 1
                continue

            frame_indices = [
                (clip_id * args.clip_stride + i) * args.frame_skip
                for i in range(args.clip_length)
            ]

            if args.dry_run:
                stats.clips_created += 1
                print(f"[DRY-RUN] {output_path}")
                continue

            ok = write_clip_from_source(
                vr=vr,
                output_path=output_path,
                frame_indices=frame_indices,
                fps=args.fps,
            )
            if ok:
                stats.clips_created += 1
            else:
                stats.clips_skipped_short_source += 1
                print(f"[SKIP-SHORT] {source_video} clip_id={clip_id}")

    print("\n=== Summary ===")
    print(f"actions_seen={stats.actions_seen}")
    print(f"actions_missing_source={stats.actions_missing_source}")
    print(f"clips_seen={stats.clips_seen}")
    print(f"clips_created={stats.clips_created}")
    print(f"clips_skipped_existing={stats.clips_skipped_existing}")
    print(f"clips_skipped_non_numeric={stats.clips_skipped_non_numeric}")
    print(f"clips_skipped_short_source={stats.clips_skipped_short_source}")


if __name__ == "__main__":
    main()
