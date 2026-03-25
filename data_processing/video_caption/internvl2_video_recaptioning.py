#!/usr/bin/env python3
"""
InternVL2 video recaptioning for Trumans data.

Source attribution:
- This script is based on ideas and implementation patterns from:
  `VideoX-Fun/videox_fun/video_caption/internvl2_video_recaptioning_my.py`

Refactor notes:
- Path conventions simplified for Trumans `{scene}/{action}/{videos,videos_third,prompts_aux}` layout.
- Prompt text is loaded from `--input_prompt_file` (no hardcoded prompt in runner scripts).
- Third-person mode defaults to JSON output for `prompts_aux`.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from decord import VideoReader, cpu


@dataclass
class Job:
    scene: str
    action: str
    video_path: Path


@dataclass
class Stats:
    jobs_total: int = 0
    jobs_selected: int = 0
    clips_seen: int = 0
    clips_generated: int = 0
    clips_skipped_existing: int = 0
    clips_failed: int = 0


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


def split_items(items: list, array_index: int | None, num_splits: int) -> list:
    if array_index is None:
        return items
    if array_index < 0 or array_index >= num_splits:
        raise ValueError(f"array_index out of range: {array_index} for num_splits={num_splits}")
    chunk = (len(items) + num_splits - 1) // num_splits
    start = array_index * chunk
    end = min(start + chunk, len(items))
    return items[start:end]


def read_prompt_file(input_prompt_file: Path) -> str:
    if not input_prompt_file.exists():
        raise FileNotFoundError(f"input_prompt_file not found: {input_prompt_file}")
    prompt = input_prompt_file.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"input_prompt_file is empty: {input_prompt_file}")
    return prompt


def sample_frames(video_path: Path, num_sampled_frames: int, sample_method: str) -> np.ndarray:
    vr = VideoReader(str(video_path), ctx=cpu(0))
    n = len(vr)
    if n == 0:
        raise RuntimeError(f"Empty video: {video_path}")

    if sample_method == "mid":
        mid = n // 2
        indices = [mid]
    elif sample_method == "image":
        indices = [0]
    else:
        if num_sampled_frames <= 1:
            indices = [0]
        else:
            indices = np.linspace(0, n - 1, num_sampled_frames).round().astype(int).tolist()

    frames = vr.get_batch(indices).asnumpy()
    return frames


def build_chat_prompt(tokenizer, base_prompt: str, num_frames: int) -> str:
    placeholders = "".join(f"Frame{i}: <image>\n" for i in range(1, num_frames + 1))
    messages = [{"role": "user", "content": f"{placeholders}{base_prompt}"}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_third_context_for_ego(action_dir: Path, clip_stem: str, third_prompt_dirname: str):
    context_path = action_dir / third_prompt_dirname / f"{clip_stem}.json"
    if not context_path.exists():
        return None, []
    try:
        data = json.loads(context_path.read_text(encoding="utf-8"))
    except Exception:
        return None, []
    prompt = data.get("prompt", None)
    hints = data.get("action_hints", [])
    if not isinstance(hints, list):
        hints = []
    hints = [str(x).strip() for x in hints if str(x).strip()]
    return prompt, hints


def build_ego_prompt_with_context(
    tokenizer,
    base_prompt: str,
    num_frames: int,
    third_prompt: str | None,
    action_hints: list[str],
) -> str:
    prompt_text = base_prompt.strip()
    chunks = []
    if third_prompt:
        chunks.append(third_prompt.strip())
    if action_hints:
        chunks.append(" ; ".join(action_hints))
    if chunks:
        prompt_text += (
            "\n\nReference (use for action accuracy):\n"
            "- Include the hinted action(s) as concrete interactions.\n"
            "- Rewrite naturally from first-person viewpoint.\n"
            "- Do not mention references.\n\n"
            + "\n".join(chunks).strip()
        )
    return build_chat_prompt(tokenizer, prompt_text, num_frames=num_frames)


def iter_action_dirs(root_dir: Path, scene_filter: set[str] | None):
    for scene_dir in sorted([p for p in root_dir.iterdir() if p.is_dir()]):
        if scene_filter is not None and scene_dir.name not in scene_filter:
            continue
        for action_dir in sorted([p for p in scene_dir.iterdir() if p.is_dir()]):
            yield scene_dir.name, action_dir.name, action_dir


def gather_jobs_trumans(root_dir: Path, video_folder_name: str, scene_filter: set[str] | None) -> list[Job]:
    jobs: list[Job] = []
    for scene, action, action_dir in iter_action_dirs(root_dir, scene_filter):
        video_dir = action_dir / video_folder_name
        if not video_dir.exists():
            continue
        for video_path in sorted(video_dir.glob("*.mp4"), key=natural_sort_key):
            jobs.append(Job(scene=scene, action=action, video_path=video_path))
    return jobs


def gather_jobs_taste_rob(root_dir: Path, video_folder_name: str, scene_filter: set[str] | None) -> list[Job]:
    jobs: list[Job] = []
    for split_dir in sorted([p for p in root_dir.iterdir() if p.is_dir()]):
        for scene_dir in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
            if scene_filter is not None and scene_dir.name not in scene_filter:
                continue
            video_dir = scene_dir / video_folder_name
            if not video_dir.exists():
                continue
            for video_path in sorted(video_dir.glob("*.mp4"), key=natural_sort_key):
                jobs.append(Job(scene=split_dir.name, action=scene_dir.name, video_path=video_path))
    return jobs


def output_path_for_video(
    video_path: Path,
    output_folder_name: str,
    save_format: str,
) -> Path:
    action_dir = video_path.parent.parent
    output_dir = action_dir / output_folder_name
    ext = ".json" if save_format == "json" else ".txt"
    return output_dir / f"{video_path.stem}{ext}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refactored InternVL2 recaptioning for Trumans.")
    parser.add_argument("--root_dir", type=Path, required=True)
    parser.add_argument("--video_type", type=str, choices=["third_person", "egocentric"], required=True)
    parser.add_argument("--video_folder_name", type=str, default=None)
    parser.add_argument("--output_folder_name", type=str, default=None)
    parser.add_argument("--input_prompt_file", type=Path, required=True)
    parser.add_argument("--model_path", type=str, default="OpenGVLab/InternVL2-40B-AWQ")
    parser.add_argument("--num_sampled_frames", type=int, default=16)
    parser.add_argument("--frame_sample_method", type=str, choices=["uniform", "mid", "image"], default="uniform")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--array_index", type=int, default=None)
    parser.add_argument("--num_splits", type=int, default=8)
    parser.add_argument("--split_by", type=str, choices=["action", "video"], default="action")
    parser.add_argument("--scene_filter_file", type=Path, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dataset_type", type=str, choices=["trumans", "taste_rob"], default="trumans")
    parser.add_argument("--save_format", type=str, choices=["auto", "json", "txt"], default="auto")
    parser.add_argument("--use_third_person_context", action="store_true")
    parser.add_argument("--third_prompt_dirname", type=str, default="prompts_aux")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--tensor_parallel_size", type=int, default=None)
    return parser.parse_args()


def patch_tokenizer_for_vllm(tokenizer) -> None:
    cls = tokenizer.__class__
    if not hasattr(cls, "all_special_tokens_extended"):
        cls.all_special_tokens_extended = property(lambda self: list(self.all_special_tokens))
    if not hasattr(tokenizer, "all_special_tokens_extended"):
        tokenizer.all_special_tokens_extended = list(tokenizer.all_special_tokens)


def main() -> None:
    args = parse_args()
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    if not args.root_dir.exists():
        raise SystemExit(f"root_dir not found: {args.root_dir}")

    if args.video_folder_name is None:
        args.video_folder_name = "videos_third" if args.video_type == "third_person" else "videos"
    if args.output_folder_name is None:
        args.output_folder_name = "prompts_aux" if args.video_type == "third_person" else "prompts"
    if args.save_format == "auto":
        args.save_format = "json" if args.video_type == "third_person" else "txt"

    base_prompt = read_prompt_file(args.input_prompt_file)
    scene_filter = load_scene_filter(args.scene_filter_file)
    if args.dataset_type == "trumans":
        all_jobs = gather_jobs_trumans(args.root_dir, args.video_folder_name, scene_filter)
    else:
        all_jobs = gather_jobs_taste_rob(args.root_dir, args.video_folder_name, scene_filter)

    stats = Stats(jobs_total=len(all_jobs))
    if not all_jobs:
        if args.dataset_type == "trumans":
            raise SystemExit(f"No videos found under {args.root_dir}/*/*/{args.video_folder_name}")
        raise SystemExit(f"No videos found under {args.root_dir}/*/*/{args.video_folder_name}")

    if args.split_by == "action":
        grouped: dict[tuple[str, str], list[Job]] = {}
        for j in all_jobs:
            grouped.setdefault((j.scene, j.action), []).append(j)
        action_keys = sorted(grouped.keys())
        selected_keys = split_items(action_keys, args.array_index, args.num_splits)
        jobs = []
        for key in selected_keys:
            jobs.extend(grouped[key])
    else:
        jobs = split_items(all_jobs, args.array_index, args.num_splits)

    stats.jobs_selected = len(jobs)
    print(f"[INFO] jobs_total={stats.jobs_total}, jobs_selected={stats.jobs_selected}")

    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    patch_tokenizer_for_vllm(tokenizer)
    tp = args.tensor_parallel_size or (torch.cuda.device_count() if torch.cuda.is_available() else 1)
    quantization = "awq" if "awq" in args.model_path.lower() else None
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        max_model_len=8192,
        limit_mm_per_prompt={"image": max(1, args.num_sampled_frames)},
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=tp,
        quantization=quantization,
        dtype="float16",
        mm_processor_kwargs={"max_dynamic_patch": 1},
    )

    stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
    stop_token_ids = []
    for token in stop_tokens:
        tid = tokenizer.convert_tokens_to_ids(token)
        if tid is not None:
            stop_token_ids.append(tid)
    sampling_params = SamplingParams(
        temperature=0.1,
        max_tokens=512,
        stop_token_ids=stop_token_ids,
        repetition_penalty=1.1,
        skip_special_tokens=True,
    )

    for job in jobs:
        stats.clips_seen += 1
        save_path = output_path_for_video(job.video_path, args.output_folder_name, args.save_format)
        if args.skip_existing and save_path.exists():
            stats.clips_skipped_existing += 1
            continue

        try:
            frames = sample_frames(
                video_path=job.video_path,
                num_sampled_frames=args.num_sampled_frames,
                sample_method=args.frame_sample_method,
            )
            if args.video_type == "egocentric" and args.use_third_person_context:
                third_prompt, action_hints = load_third_context_for_ego(
                    action_dir=job.video_path.parent.parent,
                    clip_stem=job.video_path.stem,
                    third_prompt_dirname=args.third_prompt_dirname,
                )
                prompt = build_ego_prompt_with_context(
                    tokenizer=tokenizer,
                    base_prompt=base_prompt,
                    num_frames=max(1, frames.shape[0]),
                    third_prompt=third_prompt,
                    action_hints=action_hints,
                )
            else:
                prompt = build_chat_prompt(
                    tokenizer=tokenizer,
                    base_prompt=base_prompt,
                    num_frames=max(1, frames.shape[0]),
                )

            outputs = llm.generate(
                [
                    {
                        "prompt": prompt,
                        "multi_modal_data": {"image": frames},
                    }
                ],
                sampling_params=sampling_params,
            )
            caption = outputs[0].outputs[0].text.strip()

            save_path.parent.mkdir(parents=True, exist_ok=True)
            if args.save_format == "json":
                payload = {"prompt": caption}
                if args.video_type == "third_person":
                    payload["action_hints"] = []
                save_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            else:
                save_path.write_text(caption + "\n", encoding="utf-8")
            stats.clips_generated += 1
            print(f"[OK] {job.video_path} -> {save_path}")
        except Exception as e:
            stats.clips_failed += 1
            print(f"[FAIL] {job.video_path}: {e}")
        finally:
            torch.cuda.empty_cache()
            gc.collect()

    print("\n=== Summary ===")
    print(f"jobs_total={stats.jobs_total}")
    print(f"jobs_selected={stats.jobs_selected}")
    print(f"clips_seen={stats.clips_seen}")
    print(f"clips_generated={stats.clips_generated}")
    print(f"clips_skipped_existing={stats.clips_skipped_existing}")
    print(f"clips_failed={stats.clips_failed}")


if __name__ == "__main__":
    main()
