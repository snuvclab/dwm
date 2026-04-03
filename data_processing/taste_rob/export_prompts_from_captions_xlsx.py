#!/usr/bin/env python3
"""Export TASTE-Rob captions from xlsx to prompts txt files by matching video ids."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DEFAULT_SHEET_TO_HAND = {
    "Single-Hand": "SingleHand",
    "Double-Hand": "DoubleHand",
}

# Dataset naming inconsistency: directory often uses "dinning" while xlsx uses "dining".
SCENE_ALIASES = {
    "dinning": "dining",
}


@dataclass
class Stats:
    saved: int = 0
    skipped_missing_caption: int = 0
    conflict_skipped: int = 0
    missing_video_dir: int = 0


def parse_sheet_to_hand_map(raw: str) -> dict[str, str]:
    if not raw.strip():
        return dict(DEFAULT_SHEET_TO_HAND)

    out: dict[str, str] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if ":" not in pair:
            raise ValueError(f"Invalid mapping entry '{pair}'. Expected format 'sheet:hand'.")
        sheet, hand = pair.split(":", 1)
        sheet = sheet.strip()
        hand = hand.strip()
        if not sheet or not hand:
            raise ValueError(f"Invalid mapping entry '{pair}'. Empty key/value is not allowed.")
        out[sheet] = hand
    if not out:
        raise ValueError("No valid mapping entries found in --sheet_to_hand_map.")
    return out


def normalize_scene(scene: str) -> str:
    base = scene.strip().split("_")[0].lower()
    return SCENE_ALIASES.get(base, base)


def build_caption_lookup(captions_xlsx: Path, sheet_to_hand: dict[str, str]) -> dict[tuple[str, str, int], str]:
    lookup: dict[tuple[str, str, int], str] = {}
    xls = pd.ExcelFile(captions_xlsx)
    available = set(xls.sheet_names)

    for sheet_name, hand_name in sheet_to_hand.items():
        if sheet_name not in available:
            print(f"[WARN] sheet not found: {sheet_name}")
            continue

        df = pd.read_excel(captions_xlsx, sheet_name=sheet_name)
        required = {"id", "scene", "caption"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Sheet '{sheet_name}' missing required columns: {sorted(missing)}")

        for _, row in df.iterrows():
            try:
                vid = int(row["id"])
            except (TypeError, ValueError):
                continue
            scene = normalize_scene(str(row["scene"]))
            caption = str(row["caption"]).strip()
            key = (hand_name, scene, vid)
            if key not in lookup:
                lookup[key] = caption
    return lookup


def export_prompts(
    data_root: Path,
    captions_xlsx: Path,
    prompt_dir_name: str,
    video_dir_name: str,
    sheet_to_hand_map: dict[str, str],
    dry_run: bool,
) -> Stats:
    stats = Stats()
    lookup = build_caption_lookup(captions_xlsx=captions_xlsx, sheet_to_hand=sheet_to_hand_map)

    for hand_name in sorted(set(sheet_to_hand_map.values())):
        hand_dir = data_root / hand_name
        if not hand_dir.exists():
            print(f"[WARN] hand dir not found: {hand_dir}")
            stats.missing_video_dir += 1
            continue

        for scene_dir in sorted(p for p in hand_dir.iterdir() if p.is_dir()):
            video_dir = scene_dir / video_dir_name
            if not video_dir.exists():
                continue
            prompt_dir = scene_dir / prompt_dir_name
            scene_key = normalize_scene(scene_dir.name)

            for video_path in sorted(video_dir.glob("*.mp4")):
                try:
                    vid = int(video_path.stem)
                except ValueError:
                    print(f"[WARN] non-numeric video id, skip: {video_path}")
                    continue

                caption = lookup.get((hand_name, scene_key, vid))
                if caption is None or caption == "" or caption.lower() == "nan":
                    stats.skipped_missing_caption += 1
                    print(f"[MISSING] {hand_name}/{scene_dir.name}/{vid}")
                    continue

                prompt_path = prompt_dir / f"{vid}.txt"
                if prompt_path.exists():
                    stats.conflict_skipped += 1
                    print(f"[SKIP-CONFLICT] {prompt_path}")
                    continue

                stats.saved += 1
                if dry_run:
                    print(f"[DRY-RUN] save {prompt_path}")
                    continue

                prompt_dir.mkdir(parents=True, exist_ok=True)
                prompt_path.write_text(caption + "\n", encoding="utf-8")

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export prompts_aux txt files from TASTE-Rob captions xlsx.")
    parser.add_argument("--data_root", type=Path, required=True, help="Root like data/taste_rob_fps16_480_832")
    parser.add_argument("--captions_xlsx", type=Path, required=True, help="Path to captions.xlsx")
    parser.add_argument("--prompt_dir_name", type=str, default="prompts_aux")
    parser.add_argument("--video_dir_name", type=str, default="videos")
    parser.add_argument(
        "--sheet_to_hand_map",
        type=str,
        default="Single-Hand:SingleHand,Double-Hand:DoubleHand",
        help="Comma-separated mapping, e.g. 'Single-Hand:SingleHand,Double-Hand:DoubleHand'",
    )
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    sheet_to_hand = parse_sheet_to_hand_map(args.sheet_to_hand_map)

    if not args.data_root.exists():
        raise SystemExit(f"data_root not found: {args.data_root}")
    if not args.captions_xlsx.exists():
        raise SystemExit(f"captions_xlsx not found: {args.captions_xlsx}")

    stats = export_prompts(
        data_root=args.data_root,
        captions_xlsx=args.captions_xlsx,
        prompt_dir_name=args.prompt_dir_name,
        video_dir_name=args.video_dir_name,
        sheet_to_hand_map=sheet_to_hand,
        dry_run=args.dry_run,
    )

    print("\n=== Summary ===")
    print(f"saved={stats.saved}")
    print(f"skipped_missing_caption={stats.skipped_missing_caption}")
    print(f"conflict_skipped={stats.conflict_skipped}")
    print(f"missing_video_dir={stats.missing_video_dir}")


if __name__ == "__main__":
    main()
