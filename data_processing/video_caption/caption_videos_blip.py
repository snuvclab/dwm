#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from decord import VideoReader, cpu
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

try:
    from internvl2_video_recaptioning import (
        gather_jobs_taste_rob,
        gather_jobs_trumans,
        load_scene_filter,
        output_path_for_video,
        split_items,
    )
except ImportError:
    from data_processing.video_caption.internvl2_video_recaptioning import (
        gather_jobs_taste_rob,
        gather_jobs_trumans,
        load_scene_filter,
        output_path_for_video,
        split_items,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-friendly BLIP captioner for sampled video frames.")
    parser.add_argument("--root_dir", type=Path, required=True)
    parser.add_argument("--dataset_type", choices=["trumans", "taste_rob"], required=True)
    parser.add_argument("--video_folder_name", type=str, default="videos")
    parser.add_argument("--output_folder_name", type=str, default="prompts")
    parser.add_argument("--scene_filter_file", type=Path, default=None)
    parser.add_argument("--array_index", type=int, default=None)
    parser.add_argument("--num_splits", type=int, default=1)
    parser.add_argument("--split_by", choices=["action", "video"], default="video")
    parser.add_argument("--num_sampled_frames", type=int, default=3)
    parser.add_argument("--model_name", type=str, default="Salesforce/blip-image-captioning-base")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def sample_frames(video_path: Path, num_sampled_frames: int) -> list[Image.Image]:
    vr = VideoReader(str(video_path), ctx=cpu(0))
    total = len(vr)
    if total == 0:
        raise RuntimeError(f"Empty video: {video_path}")
    if num_sampled_frames <= 1:
        indices = [total // 2]
    else:
        indices = torch.linspace(0, total - 1, steps=num_sampled_frames).round().to(torch.int64).tolist()
    frames = vr.get_batch(indices).asnumpy()
    return [Image.fromarray(frame) for frame in frames]


def normalize_caption(text: str) -> str:
    return " ".join(text.strip().strip(".").split())


def combine_captions(captions: list[str]) -> str:
    unique: list[str] = []
    seen = set()
    for caption in captions:
        norm = normalize_caption(caption)
        if not norm:
            continue
        lowered = norm.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        unique.append(norm)
    if not unique:
        return "A first-person video clip."
    if len(unique) == 1:
        return unique[0] + "."
    return " ".join(f"{caption}." for caption in unique)


def main() -> None:
    args = parse_args()
    if not args.root_dir.exists():
        raise SystemExit(f"root_dir not found: {args.root_dir}")

    scene_filter = load_scene_filter(args.scene_filter_file)
    if args.dataset_type == "trumans":
        all_jobs = gather_jobs_trumans(args.root_dir, args.video_folder_name, scene_filter)
    else:
        all_jobs = gather_jobs_taste_rob(args.root_dir, args.video_folder_name, scene_filter)

    if args.split_by == "action":
        grouped: dict[tuple[str, str], list] = {}
        for job in all_jobs:
            grouped.setdefault((job.scene, job.action), []).append(job)
        selected_keys = split_items(sorted(grouped.keys()), args.array_index, args.num_splits)
        jobs = []
        for key in selected_keys:
            jobs.extend(grouped[key])
    else:
        jobs = split_items(all_jobs, args.array_index, args.num_splits)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor = BlipProcessor.from_pretrained(args.model_name)
    model = BlipForConditionalGeneration.from_pretrained(args.model_name)
    model = model.to(device)
    model.eval()

    generated = 0
    skipped = 0
    for job in jobs:
        save_path = output_path_for_video(job.video_path, args.output_folder_name, save_format="txt")
        if args.skip_existing and save_path.exists():
            skipped += 1
            continue

        images = sample_frames(job.video_path, args.num_sampled_frames)
        captions: list[str] = []
        for image in images:
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
            captions.append(processor.decode(output_ids[0], skip_special_tokens=True))

        final_caption = combine_captions(captions)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(final_caption + "\n", encoding="utf-8")
        generated += 1
        print(f"generated {job.video_path} -> {save_path}")

    print(f"done generated={generated} skipped={skipped} total={len(jobs)}")


if __name__ == "__main__":
    main()
