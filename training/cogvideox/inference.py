#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

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
    load_video_clip,
    read_dataset_entries,
    resolve_sample_paths,
    write_inference_metadata,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    parser.add_argument("--use_empty_prompts", action="store_true")
    parser.add_argument("--prompt", type=str, default=None, help="Explicit prompt text for single-video inference.")
    parser.add_argument("--prompt_file", type=str, default=None, help="Prompt txt file for single-video inference.")
    parser.add_argument("--static_video", type=str, default=None, help="Static condition video for single-video inference.")
    parser.add_argument("--hand_video", type=str, default=None, help="Hand condition video for single-video inference.")

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset_file", type=str, help="Dataset file with relative video paths.")
    source.add_argument("--video", type=str, help="Single video path.")
    return parser.parse_args()


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


def resolve_single_video_paths(video_path: Path, data_config: Dict[str, Any], args: argparse.Namespace) -> tuple[Path, Path, str]:
    prompt_subdir = data_config.get("prompt_subdir", "prompts_rewrite")
    hand_video_subdir = data_config.get("hand_video_subdir", "videos_hands")
    static_video_subdir = data_config.get("static_video_subdir", "videos_static")
    sample_root = video_path.parent.parent

    static_video = Path(args.static_video) if args.static_video else sample_root / static_video_subdir / video_path.name
    hand_video = Path(args.hand_video) if args.hand_video else sample_root / hand_video_subdir / video_path.name

    if args.prompt is not None:
        prompt = args.prompt
    elif args.prompt_file is not None:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8").strip()
    else:
        prompt = load_prompt_text(sample_root / prompt_subdir / f"{video_path.stem}.txt")
    return static_video, hand_video, prompt


def generate_one(pipeline, *, prompt: str, static_video_path: Path, hand_video_path: Path, output_path: Path, num_frames: int, height: int, width: int, num_inference_steps: int, guidance_scale: float, seed: int, fps: int) -> None:
    device = pipeline._execution_device
    static_video = load_video_clip(static_video_path, max_num_frames=num_frames, height=height, width=width).unsqueeze(0)
    hand_video = load_video_clip(hand_video_path, max_num_frames=num_frames, height=height, width=width).unsqueeze(0)
    result = pipeline(
        prompt=prompt,
        static_videos=static_video,
        hand_videos=hand_video,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        use_dynamic_cfg=False,
        generator=torch.Generator(device=device).manual_seed(seed),
        output_type="np",
    ).frames[0]
    export_to_video(result, str(output_path), fps=fps)


def main() -> None:
    args = parse_args()
    config = load_config(args)
    data_config = config["data"]
    training_config = config["training"]
    custom_settings = training_config.get("custom_settings", {})
    data_root = data_config["data_root"]
    prompt_subdir = data_config.get("prompt_subdir", "prompts_rewrite")

    output_dir = args.output_dir or os.path.join(args.checkpoint_path, "inference")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    weight_dtype = torch.bfloat16 if "5b" in config["model"]["base_model_name_or_path"].lower() else torch.float16
    pipeline = build_inference_pipeline(args.checkpoint_path, config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    }

    if args.dataset_file:
        entries = read_dataset_entries(data_root, args.dataset_file)
        if args.max_samples is not None:
            entries = entries[: args.max_samples]
        metadata.update({"mode": "dataset_file", "dataset_file": args.dataset_file, "num_samples": len(entries)})
        write_inference_metadata(output_path, metadata)
        for index, relative_video_path in enumerate(entries):
            sample_paths = resolve_sample_paths(
                data_root=data_root,
                relative_video_path=relative_video_path,
                prompt_subdir=prompt_subdir,
                prompt_embeds_subdir=data_config.get("prompt_embeds_subdir", "prompt_embeds_prompts_rewrite"),
                hand_video_subdir=data_config.get("hand_video_subdir", "videos_hands"),
                hand_latents_subdir=data_config.get("hand_video_latents_subdir", "hand_video_latents"),
                static_video_subdir=data_config.get("static_video_subdir", "videos_static"),
                static_latents_subdir=data_config.get("static_video_latents_subdir", "static_video_latents"),
                video_latents_subdir=data_config.get("video_latents_subdir", "video_latents"),
            )
            prompt = "" if args.use_empty_prompts else load_prompt_text(sample_paths.prompt_path)
            logger.info("[%d/%d] generating %s", index + 1, len(entries), relative_video_path)
            output_video_path = output_path / f"{Path(relative_video_path).stem}.mp4"
            generate_one(
                pipeline,
                prompt=prompt,
                static_video_path=sample_paths.static_video_path,
                hand_video_path=sample_paths.hand_video_path,
                output_path=output_video_path,
                num_frames=num_frames,
                height=height,
                width=width,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                seed=args.seed + index,
                fps=args.fps,
            )
            (output_path / f"{Path(relative_video_path).stem}.txt").write_text(prompt, encoding="utf-8")
        return

    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = (Path(data_root) / video_path).resolve()
    static_video_path, hand_video_path, prompt = resolve_single_video_paths(video_path, data_config, args)
    if args.use_empty_prompts:
        prompt = ""
    metadata.update({"mode": "single_video", "video": str(video_path), "static_video": str(static_video_path), "hand_video": str(hand_video_path)})
    write_inference_metadata(output_path, metadata)
    output_video_path = output_path / f"{video_path.stem}.mp4"
    generate_one(
        pipeline,
        prompt=prompt,
        static_video_path=static_video_path,
        hand_video_path=hand_video_path,
        output_path=output_video_path,
        num_frames=num_frames,
        height=height,
        width=width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        fps=args.fps,
    )
    (output_path / f"{video_path.stem}.txt").write_text(prompt, encoding="utf-8")


if __name__ == "__main__":
    main()
