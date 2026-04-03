#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import imageio.v3 as iio
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.cogvideox.diffusers_compat import disable_broken_torchao

disable_broken_torchao()

import torch
from diffusers.utils import export_to_video

from training.cogvideox.config_loader import load_experiment_config
from training.cogvideox.static_hand_utils import (
    build_lora_config,
    build_pipeline_from_config,
    load_lora_weights_into_transformer,
    load_non_lora_state_dict,
    load_prompt_text,
    read_dataset_entries,
    resolve_sample_paths,
    write_inference_metadata,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_video_frames(result: Any) -> np.ndarray:
    frames = getattr(result, "frames", result)
    if isinstance(frames, dict):
        frames = frames.get("frames", frames)
    if isinstance(frames, tuple):
        frames = frames[0]
    if isinstance(frames, list):
        frames = frames[0]

    if isinstance(frames, torch.Tensor):
        frames = frames.detach().cpu().float().numpy()

    frames = np.asarray(frames)
    if frames.ndim == 5:
        frames = frames[0]
    if frames.ndim != 4:
        raise ValueError(f"Expected video frames with 4 dimensions, got shape {frames.shape}")
    return frames


def ensure_video_fhwc_uint8(video: Any) -> np.ndarray:
    if isinstance(video, torch.Tensor):
        video = video.detach().cpu().float().numpy()

    video = np.asarray(video)
    if video.ndim == 5 and video.shape[0] == 1:
        video = video[0]
    if video.ndim == 4 and video.shape[0] in (3, 4):
        video = np.transpose(video, (1, 2, 3, 0))
    if video.ndim != 4:
        raise ValueError(f"Expected video with 4 dimensions, got shape {video.shape}")

    if np.issubdtype(video.dtype, np.floating):
        video = np.clip(video, 0.0, 1.0)
        video = (video * 255.0).round().astype(np.uint8)
    elif video.dtype != np.uint8:
        video = np.clip(video, 0, 255).astype(np.uint8)
    return video


def resize_video_fhwc(video: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    if video.shape[1] == target_h and video.shape[2] == target_w:
        return video

    frames = torch.from_numpy(video).permute(0, 3, 1, 2).float()
    frames = torch.nn.functional.interpolate(frames, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return frames.permute(0, 2, 3, 1).clamp(0, 255).round().to(torch.uint8).cpu().numpy()


def prepare_comparison_parts(parts: list[np.ndarray], target_h: int, target_w: int) -> list[np.ndarray]:
    normalized = [resize_video_fhwc(part, target_h=target_h, target_w=target_w) for part in parts]
    min_frames = min(part.shape[0] for part in normalized)
    return [part[:min_frames] for part in normalized]


def save_outputs(
    *,
    generated_frames: np.ndarray,
    output_dir: Path,
    base_name: str,
    fps: int,
    gt_video: Optional[np.ndarray] = None,
    static_video: Optional[np.ndarray] = None,
    hand_video: Optional[np.ndarray] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_path = output_dir / f"{base_name}.mp4"
    generated_named_path = output_dir / f"{base_name}_generated.mp4"

    export_to_video(generated_frames, str(generated_path), fps=fps)
    export_to_video(generated_frames, str(generated_named_path), fps=fps)

    generated_u8 = ensure_video_fhwc_uint8(generated_frames)

    if gt_video is not None:
        gt_u8 = ensure_video_fhwc_uint8(gt_video)
        comparison_parts = prepare_comparison_parts(
            [generated_u8, gt_u8],
            target_h=int(generated_u8.shape[1]),
            target_w=int(generated_u8.shape[2]),
        )
        comparison_video = np.concatenate(comparison_parts, axis=2)
        export_to_video(comparison_video, str(output_dir / f"{base_name}_comparison.mp4"), fps=fps)

    full_parts: list[np.ndarray] = []
    if static_video is not None:
        full_parts.append(ensure_video_fhwc_uint8(static_video))
    if hand_video is not None:
        full_parts.append(ensure_video_fhwc_uint8(hand_video))
    full_parts.append(generated_u8)
    if gt_video is not None:
        full_parts.append(ensure_video_fhwc_uint8(gt_video))

    if len(full_parts) > 1:
        full_parts = prepare_comparison_parts(
            full_parts,
            target_h=int(generated_u8.shape[1]),
            target_w=int(generated_u8.shape[2]),
        )
        full_comparison = np.concatenate(full_parts, axis=2)
        export_to_video(full_comparison, str(output_dir / f"{base_name}_full_comparison.mp4"), fps=fps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference for the DWM CogVideoX static-hand-concat model.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Checkpoint directory or final output directory.")
    parser.add_argument("--experiment_config", type=str, default=None, help="Experiment YAML. Auto-detected if omitted.")
    parser.add_argument("--override", type=str, nargs="*", help="Config override entries in key=value form.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory for generated videos.")
    parser.add_argument("--data_root", type=str, default=None, help="Override data_root from config.")
    parser.add_argument("--prompt_subdir", type=str, default=None, help="Override prompt_subdir from config.")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dataset_output_layout", type=str, default="flat", choices=["flat", "per_sample"])
    parser.add_argument("--chunk_id", type=int, default=None)
    parser.add_argument("--num_chunks", type=int, default=None)
    parser.add_argument("--shard_rank", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=None)
    parser.add_argument("--use_empty_prompts", action="store_true")
    parser.add_argument("--prompt", type=str, default=None, help="Explicit prompt text for single-video inference.")
    parser.add_argument("--prompt_file", type=str, default=None, help="Prompt txt file for single-video inference.")
    parser.add_argument("--static_video", type=str, default=None, help="Static condition video for single-video inference.")
    parser.add_argument("--hand_video", type=str, default=None, help="Hand condition video for single-video inference.")

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset_file", type=str, help="Dataset file with relative video paths.")
    source.add_argument("--video", type=str, help="Single video path.")
    return parser.parse_args()


def resolve_shard_info(args: argparse.Namespace) -> tuple[int, int]:
    shard_rank = args.shard_rank
    num_shards = args.num_shards

    if shard_rank is None and "LOCAL_RANK" in os.environ:
        shard_rank = int(os.environ["LOCAL_RANK"])
    if num_shards is None and "WORLD_SIZE" in os.environ:
        num_shards = int(os.environ["WORLD_SIZE"])

    if shard_rank is None:
        shard_rank = 0
    if num_shards is None:
        num_shards = 1

    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if shard_rank < 0 or shard_rank >= num_shards:
        raise ValueError(f"shard_rank must satisfy 0 <= shard_rank < num_shards, got {shard_rank} vs {num_shards}")
    return shard_rank, num_shards


def resolve_chunk_info(args: argparse.Namespace) -> tuple[int, int]:
    chunk_id = args.chunk_id
    num_chunks = args.num_chunks

    if chunk_id is None and "SLURM_ARRAY_TASK_ID" in os.environ:
        chunk_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    if num_chunks is None and "SLURM_ARRAY_TASK_COUNT" in os.environ:
        num_chunks = int(os.environ["SLURM_ARRAY_TASK_COUNT"])

    if chunk_id is None and num_chunks is None:
        return 0, 1
    if chunk_id is None or num_chunks is None:
        raise ValueError("Both --chunk_id and --num_chunks must be provided together.")
    if num_chunks < 1:
        raise ValueError(f"num_chunks must be >= 1, got {num_chunks}")
    if chunk_id < 0 or chunk_id >= num_chunks:
        raise ValueError(f"chunk_id must satisfy 0 <= chunk_id < num_chunks, got {chunk_id} vs {num_chunks}")
    return chunk_id, num_chunks


def get_partition_bounds(total: int, part_id: int, num_parts: int) -> tuple[int, int]:
    start = part_id * total // num_parts
    end = (part_id + 1) * total // num_parts
    return start, end


def resolve_execution_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank < torch.cuda.device_count():
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cuda")


def find_experiment_config(checkpoint_path: str) -> Optional[str]:
    checkpoint_dir = Path(checkpoint_path)
    search_dir = checkpoint_dir.parent if checkpoint_dir.name.startswith("checkpoint-") else checkpoint_dir
    yaml_files = sorted(list(search_dir.glob("*.yaml")) + list(search_dir.glob("*.yml")))
    if yaml_files:
        return str(yaml_files[0])
    return None


def load_config(args: argparse.Namespace) -> Dict[str, Any]:
    config_path = args.experiment_config or find_experiment_config(args.checkpoint_path)
    if config_path is None:
        raise FileNotFoundError("Could not auto-detect experiment config. Pass --experiment_config explicitly.")
    config = load_experiment_config(config_path, overrides=args.override)
    if args.data_root is not None:
        config.setdefault("data", {})["data_root"] = args.data_root
    if args.prompt_subdir is not None:
        config.setdefault("data", {})["prompt_subdir"] = args.prompt_subdir
    return config


def build_inference_pipeline(checkpoint_path: str, config: Dict[str, Any]):
    training_mode = config["training"].get("mode", "lora")
    model_path = checkpoint_path if os.path.isdir(os.path.join(checkpoint_path, "transformer")) else None
    lora_path = os.path.join(checkpoint_path, "pytorch_lora_weights.safetensors")
    lora_bin_path = os.path.join(checkpoint_path, "pytorch_lora_weights.bin")

    if training_mode == "full" and model_path is not None:
        return build_pipeline_from_config(config, pretrained_transformer_path=checkpoint_path)

    if training_mode == "lora":
        if os.path.exists(lora_path) or os.path.exists(lora_bin_path):
            lora_state_dict = pipeline_lora_state_dict(checkpoint_path)
            lora_rank = infer_lora_rank(lora_state_dict) or config["training"].get("lora_rank", 64)
            lora_training_config = copy.deepcopy(config["training"])
            lora_training_config["lora_rank"] = lora_rank
            if int(lora_training_config.get("lora_alpha", lora_rank)) != int(lora_rank):
                logger.warning(
                    "Overriding inference LoRA alpha from %s to %s to match checkpoint rank.",
                    lora_training_config.get("lora_alpha"),
                    lora_rank,
                )
                lora_training_config["lora_alpha"] = lora_rank
            pipeline = build_pipeline_from_config(config)
            pipeline.transformer.add_adapter(build_lora_config(lora_training_config))
            unexpected = load_lora_weights_into_transformer(pipeline.transformer, checkpoint_path)
            if unexpected:
                logger.warning("Unexpected LoRA keys while loading %s: %s", checkpoint_path, unexpected)
            load_non_lora_state_dict(pipeline.transformer, checkpoint_path)
            return pipeline
        if model_path is not None:
            return build_pipeline_from_config(config, pretrained_transformer_path=checkpoint_path)
        raise FileNotFoundError(f"No LoRA weights or transformer subfolder found under checkpoint: {checkpoint_path}")

    if model_path is not None:
        return build_pipeline_from_config(config, pretrained_transformer_path=checkpoint_path)
    return build_pipeline_from_config(config)


def pipeline_lora_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    from training.cogvideox.pipeline import CogVideoXFunStaticHandConcatPipeline

    return CogVideoXFunStaticHandConcatPipeline.lora_state_dict(checkpoint_path)


def infer_lora_rank(lora_state_dict: Dict[str, torch.Tensor]) -> Optional[int]:
    for key, value in lora_state_dict.items():
        if "lora_A" in key and value.ndim == 2:
            return int(value.shape[0])
        if "lora_B" in key and value.ndim == 2:
            return int(value.shape[1])
    return None


def resolve_single_video_paths(
    video_path: Path, data_config: Dict[str, Any], args: argparse.Namespace
) -> tuple[Path, Path, str, Optional[Path]]:
    prompt_subdir = data_config.get("prompt_subdir", "prompts_rewrite")
    hand_video_subdir = data_config.get("hand_video_subdir", "videos_hands")
    static_video_subdir = data_config.get("static_video_subdir", "videos_static")
    sample_root = video_path.parent.parent

    static_video = Path(args.static_video) if args.static_video else sample_root / static_video_subdir / video_path.name
    hand_video = Path(args.hand_video) if args.hand_video else sample_root / hand_video_subdir / video_path.name

    if args.prompt is not None:
        prompt = args.prompt
        prompt_path = None
    elif args.prompt_file is not None:
        prompt_path = Path(args.prompt_file)
        prompt = prompt_path.read_text(encoding="utf-8").strip()
    else:
        prompt_path = sample_root / prompt_subdir / f"{video_path.stem}.txt"
        prompt = load_prompt_text(prompt_path)
    return static_video, hand_video, prompt, prompt_path


def load_condition_video(path: Path) -> np.ndarray:
    video = iio.imread(path).astype(np.float32) / 255.0
    return video.transpose(3, 0, 1, 2)[np.newaxis, :]


def generate_one(
    pipeline,
    *,
    prompt: str,
    static_video_path: Path,
    hand_video_path: Path,
    num_frames: int,
    height: int,
    width: int,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
    pipeline_type: str,
) -> np.ndarray:
    device = pipeline._execution_device
    static_video = load_condition_video(static_video_path)
    hand_video = load_condition_video(hand_video_path)

    pipeline_args = {
        "prompt": prompt,
        "static_videos": static_video,
        "hand_videos": hand_video,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "use_dynamic_cfg": False,
        "generator": torch.Generator(device=device).manual_seed(seed),
    }
    if "fun" in pipeline_type:
        pipeline_args["mask_video"] = torch.zeros((1, 1, num_frames, height, width), dtype=torch.uint8)

    result = pipeline(**pipeline_args)
    return extract_video_frames(result)


def main() -> None:
    args = parse_args()
    config = load_config(args)
    data_config = config["data"]
    pipeline_type = config["pipeline"]["type"]
    training_config = config["training"]
    custom_settings = training_config.get("custom_settings", {})
    data_root = data_config["data_root"]
    prompt_subdir = data_config.get("prompt_subdir", "prompts_rewrite")

    output_dir = args.output_dir or os.path.join(args.checkpoint_path, "inference")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    chunk_id, num_chunks = resolve_chunk_info(args)
    shard_rank, num_shards = resolve_shard_info(args)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    weight_dtype = torch.bfloat16 if "5b" in config["model"]["base_model_name_or_path"].lower() else torch.float16
    pipeline = build_inference_pipeline(args.checkpoint_path, config)
    device = resolve_execution_device()
    pipeline.to(device=device, dtype=weight_dtype)

    if custom_settings.get("enable_slicing", False):
        pipeline.vae.enable_slicing()
    if custom_settings.get("enable_tiling", False):
        pipeline.vae.enable_tiling()

    height = args.height or data_config.get("height_buckets", 480)
    width = args.width or data_config.get("width_buckets", 720)
    num_frames = args.num_frames or custom_settings.get("max_num_frames", 49)

    metadata: Dict[str, Any] = {
        "checkpoint_path": args.checkpoint_path,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "num_frames": num_frames,
        "height": height,
        "width": width,
        "prompt_subdir": prompt_subdir,
        "seed": args.seed,
        "chunk_id": chunk_id,
        "num_chunks": num_chunks,
        "shard_rank": shard_rank,
        "num_shards": num_shards,
    }

    if args.dataset_file:
        entries = read_dataset_entries(data_root, args.dataset_file)
        if args.max_samples is not None:
            entries = entries[: args.max_samples]
        total_entries = len(entries)
        chunk_start, chunk_end = get_partition_bounds(total_entries, chunk_id, num_chunks)
        indexed_entries = list(enumerate(entries))[chunk_start:chunk_end]
        metadata.update(
            {
                "mode": "dataset_file",
                "dataset_file": args.dataset_file,
                "num_samples": total_entries,
                "chunk_start": chunk_start,
                "chunk_end": chunk_end,
                "num_chunk_samples": len(indexed_entries),
            }
        )
        if shard_rank == 0:
            write_inference_metadata(output_path, metadata)
        logger.info(
            "processing chunk %d/%d with %d samples; shard %d/%d",
            chunk_id,
            num_chunks,
            len(indexed_entries),
            shard_rank,
            num_shards,
        )
        for local_index, (global_index, relative_video_path) in enumerate(indexed_entries):
            if local_index % num_shards != shard_rank:
                continue
            sample_paths = resolve_sample_paths(
                data_root=data_root,
                relative_video_path=relative_video_path,
                prompt_subdir=prompt_subdir,
                prompt_embeds_subdir=data_config.get("prompt_embeds_subdir", "prompt_embeds_rewrite"),
                hand_video_subdir=data_config.get("hand_video_subdir", "videos_hands"),
                hand_latents_subdir=data_config.get("hand_video_latents_subdir", "hand_video_latents"),
                static_video_subdir=data_config.get("static_video_subdir", "videos_static"),
                static_latents_subdir=data_config.get("static_video_latents_subdir", "static_video_latents"),
                video_latents_subdir=data_config.get("video_latents_subdir", "video_latents"),
            )
            prompt = "" if args.use_empty_prompts else load_prompt_text(sample_paths.prompt_path)
            logger.info(
                "[chunk %d/%d shard %d/%d] [%d/%d global=%d] generating %s",
                chunk_id,
                num_chunks,
                shard_rank,
                num_shards,
                local_index + 1,
                len(indexed_entries),
                global_index,
                relative_video_path,
            )
            sample_output_dir = output_path
            if args.dataset_output_layout == "per_sample":
                sample_output_dir = output_path / Path(relative_video_path).stem
                sample_output_dir.mkdir(parents=True, exist_ok=True)
            output_video_path = sample_output_dir / f"{Path(relative_video_path).stem}.mp4"
            if args.skip_existing and output_video_path.exists():
                logger.info("[shard %d/%d] skipping existing %s", shard_rank, num_shards, output_video_path)
                continue
            sample_seed = args.seed + global_index
            sample_metadata = dict(metadata)
            sample_metadata.update(
                {
                    "relative_video_path": relative_video_path,
                    "video": str(sample_paths.video_path),
                    "static_video": str(sample_paths.static_video_path),
                    "hand_video": str(sample_paths.hand_video_path),
                    "prompt_path": str(sample_paths.prompt_path),
                    "seed": sample_seed,
                    "global_index": global_index,
                }
            )
            if shard_rank == 0 and args.dataset_output_layout == "per_sample":
                write_inference_metadata(sample_output_dir, sample_metadata)
            generated_frames = generate_one(
                pipeline,
                prompt=prompt,
                static_video_path=sample_paths.static_video_path,
                hand_video_path=sample_paths.hand_video_path,
                num_frames=num_frames,
                height=height,
                width=width,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                seed=sample_seed,
                pipeline_type=pipeline_type,
            )
            save_outputs(
                generated_frames=generated_frames,
                output_dir=sample_output_dir,
                base_name=Path(relative_video_path).stem,
                fps=args.fps,
                gt_video=iio.imread(sample_paths.video_path),
                static_video=iio.imread(sample_paths.static_video_path),
                hand_video=iio.imread(sample_paths.hand_video_path),
            )
            (sample_output_dir / f"{Path(relative_video_path).stem}.txt").write_text(prompt, encoding="utf-8")
        return

    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = (Path(data_root) / video_path).resolve()
    static_video_path, hand_video_path, prompt, prompt_path = resolve_single_video_paths(video_path, data_config, args)
    if args.use_empty_prompts:
        prompt = ""
    metadata.update(
        {
            "mode": "single_video",
            "video": str(video_path),
            "static_video": str(static_video_path),
            "hand_video": str(hand_video_path),
            "prompt_path": None if prompt_path is None else str(prompt_path),
        }
    )
    write_inference_metadata(output_path, metadata)
    generated_frames = generate_one(
        pipeline,
        prompt=prompt,
        static_video_path=static_video_path,
        hand_video_path=hand_video_path,
        num_frames=num_frames,
        height=height,
        width=width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        pipeline_type=pipeline_type,
    )
    save_outputs(
        generated_frames=generated_frames,
        output_dir=output_path,
        base_name=video_path.stem,
        fps=args.fps,
        gt_video=iio.imread(video_path),
        static_video=iio.imread(static_video_path),
        hand_video=iio.imread(hand_video_path),
    )
    (output_path / f"{video_path.stem}.txt").write_text(prompt, encoding="utf-8")


if __name__ == "__main__":
    main()
