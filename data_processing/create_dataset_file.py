#!/usr/bin/env python3
"""Create train/val/test dataset files for smoke or full DWM layouts."""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from natsort import natsorted
except ImportError:
    natsorted = sorted

try:
    from dataset_layout_utils import iter_taste_rob_sample_dirs, iter_trumans_action_dirs
except ImportError:
    from data_processing.dataset_layout_utils import iter_taste_rob_sample_dirs, iter_trumans_action_dirs

DEFAULT_REQUIRED_SUBDIRS = [
    "video_latents",
    "hand_video_latents",
    "static_video_latents",
    "prompt_embeds_prompts_rewrite",
]


def check_required_files(video_path: Path, processed_dir: Path, required_subdirs: List[str]) -> Tuple[bool, List[str]]:
    stem = video_path.stem
    missing = []
    for subdir in required_subdirs:
        candidate = processed_dir / subdir / f"{stem}.pt"
        if not candidate.exists():
            missing.append(subdir)
    return len(missing) == 0, missing


def find_videos_trumans(data_root: Path) -> List[Path]:
    videos = []
    for _, action_dir in iter_trumans_action_dirs(data_root):
        vid_dir = action_dir / "videos"
        if vid_dir.exists():
            videos.extend(vid_dir.glob("*.mp4"))
    return natsorted(videos)


def find_videos_taste_rob(data_root: Path) -> List[Path]:
    videos = []
    for _, sample_dir in iter_taste_rob_sample_dirs(data_root):
        vid_dir = sample_dir / "videos"
        if vid_dir.exists():
            videos.extend(vid_dir.glob("*.mp4"))
    return natsorted(videos)


def validate_and_relative(video_paths: List[Path], output_base_dir: Path, required_subdirs: List[str]) -> Tuple[List[str], List[Tuple[Path, List[str]]]]:
    valid_rel: List[str] = []
    invalid: List[Tuple[Path, List[str]]] = []
    base_resolved = output_base_dir.resolve()
    for video_path in video_paths:
        sample_root = video_path.parent.parent
        ok, missing = check_required_files(video_path, sample_root, required_subdirs)
        if ok:
            try:
                rel = video_path.resolve().relative_to(base_resolved)
            except ValueError:
                rel = video_path
            valid_rel.append(str(rel).replace("\\", "/"))
        else:
            invalid.append((video_path, missing))
    return valid_rel, invalid


def group_by_action_trumans(paths: List[str]) -> Dict[str, List[str]]:
    action_videos: Dict[str, List[str]] = {}
    for p in paths:
        parts = Path(p).parts
        if len(parts) >= 5:
            scene_name = parts[-4]
            action_name = parts[-3]
            key = f"{scene_name}_{action_name}"
            action_videos.setdefault(key, []).append(p)
    return action_videos


def create_splits_trumans(action_videos: Dict[str, List[str]], test_ratio: float, val_ratio: float, seed: int) -> Tuple[List[str], List[str], List[str]]:
    if len(action_videos) <= 1:
        flat_videos: List[str] = []
        for videos in action_videos.values():
            flat_videos.extend(videos)
        return create_splits_taste_rob(flat_videos, test_ratio, val_ratio, seed)

    random.seed(seed)
    actions = list(action_videos.keys())
    random.shuffle(actions)
    num_test_actions = max(1, int(len(actions) * test_ratio)) if actions else 0
    test_actions = set(actions[:num_test_actions])
    train_actions = actions[num_test_actions:]

    test_videos: List[str] = []
    train_videos: List[str] = []
    val_videos: List[str] = []

    for action in test_actions:
        test_videos.extend(action_videos[action])

    for action in train_actions:
        lst = list(action_videos[action])
        random.shuffle(lst)
        n_val = max(1, int(len(lst) * val_ratio)) if len(lst) > 1 else 0
        val_videos.extend(lst[:n_val])
        train_videos.extend(lst[n_val:])

    return train_videos, val_videos, test_videos


def create_splits_taste_rob(paths: List[str], test_ratio: float, val_ratio: float, seed: int) -> Tuple[List[str], List[str], List[str]]:
    random.seed(seed)
    lst = list(paths)
    random.shuffle(lst)
    n = len(lst)
    if n == 0:
        return [], [], []
    n_test = max(1, int(n * test_ratio))
    n_val = max(1, int(n * val_ratio))
    if n_test + n_val >= n:
        n_test = 1 if n >= 3 else 0
        n_val = 1 if n >= 2 else 0
    test_videos = lst[:n_test]
    val_videos = lst[n_test:n_test + n_val]
    train_videos = lst[n_test + n_val:]
    return train_videos, val_videos, test_videos


def save_splits(train: List[str], val: List[str], test: List[str], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(train) + len(val) + len(test)
    for name, lines in [("train", sorted(train)), ("val", sorted(val)), ("test", sorted(test))]:
        with open(output_dir / f"{name}.txt", "w", encoding="utf-8") as handle:
            for line in lines:
                handle.write(f"{line}\n")
    with open(output_dir / "split_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "total_videos": total,
                "train_videos": len(train),
                "val_videos": len(val),
                "test_videos": len(test),
            },
            handle,
            indent=2,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create dataset file and train/val/test splits.")
    parser.add_argument("--dataset_type", type=str, required=True, choices=["trumans", "taste_rob"])
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--output_name", type=str, default=None)
    parser.add_argument("--output_base_dir", type=Path, default=Path("."))
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--required_subdirs", nargs="+", default=None)
    parser.add_argument("--allow_missing_processed", action="store_true")
    args = parser.parse_args()

    if args.dataset_type == "trumans":
        videos = find_videos_trumans(args.data_root)
    else:
        videos = find_videos_taste_rob(args.data_root)

    required_subdirs = args.required_subdirs or DEFAULT_REQUIRED_SUBDIRS
    if args.allow_missing_processed:
        valid_rel = [str(path.resolve().relative_to(args.output_base_dir.resolve())).replace("\\", "/") if path.is_absolute() else str(path) for path in videos]
        invalid = []
    else:
        valid_rel, invalid = validate_and_relative(videos, args.output_base_dir, required_subdirs)

    if args.dataset_type == "trumans":
        train, val, test = create_splits_trumans(group_by_action_trumans(valid_rel), args.test_ratio, args.val_ratio, args.seed)
    else:
        train, val, test = create_splits_taste_rob(valid_rel, args.test_ratio, args.val_ratio, args.seed)

    output_name = args.output_name or args.dataset_type
    output_dir = args.output_dir or Path("dataset_files") / output_name
    save_splits(train, val, test, output_dir)

    if invalid:
        invalid_path = output_dir / "invalid_samples.json"
        invalid_payload = [
            {"video": str(video), "missing": missing}
            for video, missing in invalid
        ]
        invalid_path.write_text(json.dumps(invalid_payload, indent=2), encoding="utf-8")
        print(f"Wrote invalid sample report: {invalid_path}")

    print(f"Created splits under {output_dir}")


if __name__ == "__main__":
    main()
