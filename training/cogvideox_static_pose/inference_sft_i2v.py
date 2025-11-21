#!/usr/bin/env python3
"""
Inference script for CogVideoX SFT (Supervised Fine-Tuning) checkpoints.
Loads the training configuration and checkpoint produced by
`cogvideox_text_to_video_pose_sft_unified.py` and runs qualitative inference.
"""

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple

import imageio.v3 as iio
import numpy as np
import torch
from tqdm.auto import tqdm

from training.cogvideox_static_pose.config_loader import load_experiment_config
from training.cogvideox_static_pose.cogvideox_text_to_video_pose_sft_unified import (
    setup_pipeline_from_config,
)
from training.cogvideox_static_pose.cogvideox_transformer_with_conditions import (
    CogVideoXTransformer3DModelWithConcat,
    CogVideoXTransformer3DModelWithAdaLNPose,
    CogVideoXTransformer3DModelWithAdaLNPosePerFrame,
)


SUPPORTED_PIPELINES = {
    "cogvideox_fun_static_to_video_pose_adaln",
    "cogvideox_fun_static_to_video_pose_adaln_perframe",
    "cogvideox_static_to_video_pose_concat",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference for CogVideoX SFT checkpoints")
    parser.add_argument("--config", type=str, required=True, help="Training config YAML used for this checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None, help="Directory containing SFT checkpoint files")
    parser.add_argument("--data_root", type=str, default="./data", help="Root directory containing video data")
    parser.add_argument("--video_list", type=str, required=True, help="Text file listing validation videos (relative paths)")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save generated videos (defaults to checkpoint/eval_for_qual)")
    parser.add_argument("--no_prompt", action="store_true", help="Do not use the stored prompt text")
    parser.add_argument("--device", type=str, default="cuda", help="Computation device")
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="CFG guidance scale")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of denoising steps")
    parser.add_argument("--seed", type=int, default=43, help="Random seed for sampling")
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="The video is not of a high quality, it has a low resolution. Watermark present in each frame. The background is solid. Strange body and strange trajectory. Distortion. ",
        help="Negative prompt for classifier-free guidance",
    )
    return parser.parse_args()


def _load_video_tensor(path: Path) -> torch.Tensor:
    """Load video file and convert to tensor in [-1, 1] with shape [1, C, F, H, W]."""
    frames = iio.imread(path.as_posix()).astype(np.float32) / 255.0  # [F, H, W, C]
    if frames.ndim != 4:
        raise ValueError(f"Video at {path} has invalid shape {frames.shape}")
    frames = np.transpose(frames, (3, 0, 1, 2))  # [C, F, H, W]
    tensor = torch.from_numpy(frames).unsqueeze(0)  # [1, C, F, H, W]
    tensor = tensor * 2.0 - 1.0
    return tensor


def _load_pose_params(path: Path) -> torch.Tensor:
    """Load pose parameters from checkpoint (supports dict with body_pose)."""
    data = torch.load(path, map_location="cpu")
    if isinstance(data, dict):
        for key in ("body_pose", "pose_params", "hand_motions"):
            if key in data:
                data = data[key]
                break
    if isinstance(data, torch.Tensor):
        pose = data
    else:
        pose = torch.as_tensor(data)
    if pose.ndim == 2:
        pose = pose.unsqueeze(0)
    return pose


def _load_prompt(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _find_optional_file(path: Path) -> Optional[Path]:
    return path if path.exists() else None


def _load_sft_transformer(checkpoint_dir: Path, config: dict, pipeline_type: str):
    """Load SFT transformer from checkpoint directory."""    
    if pipeline_type == "cogvideox_static_to_video_pose_concat":
        # Load concat transformer
        concat_config = config.get("pipeline", {}).get("concat", {})
        condition_channels = concat_config.get("condition_channels", 16)
        transformer = CogVideoXTransformer3DModelWithConcat.from_pretrained(
            pretrained_model_name_or_path=checkpoint_dir,
            base_model_name_or_path=config["model"]["base_model_name_or_path"],
            condition_channels=condition_channels,
        )
    else:
        raise ValueError(f"Unsupported pipeline type for SFT loading: {pipeline_type}")
    
    return transformer


def prepare_pipeline(args: argparse.Namespace):
    config = load_experiment_config(args.config, overrides=None)

    pipeline_type = config["pipeline"]["type"]
    if pipeline_type not in SUPPORTED_PIPELINES:
        raise ValueError(f"Pipeline type '{pipeline_type}' is not supported by this script.")

    if args.checkpoint is None:
        checkpoint_dir = None
    else:
        checkpoint_dir = Path(args.checkpoint)
    
        # Load SFT transformer from checkpoint
        print(f"📥 Loading SFT transformer from {checkpoint_dir / 'transformer'}")
    sft_transformer = _load_sft_transformer(checkpoint_dir, config, pipeline_type)
    
    # Setup pipeline with SFT transformer
    pipeline = setup_pipeline_from_config(config)
    
    # Replace transformer with SFT checkpoint
    pipeline.transformer = sft_transformer
    
    pipeline.to(args.device)
    pipeline.vae.to(dtype=torch.bfloat16 if "5b" in config["model"]["base_model_name_or_path"].lower() else torch.float16)
    pipeline.transformer.to(dtype=torch.bfloat16 if "5b" in config["model"]["base_model_name_or_path"].lower() else torch.float16)
    pipeline.text_encoder.to(dtype=torch.bfloat16 if "5b" in config["model"]["base_model_name_or_path"].lower() else torch.float16)

    print(f"✅ Loaded SFT checkpoint from {checkpoint_dir}")
    return pipeline, config


def run_inference(args: argparse.Namespace) -> None:
    pipeline, config = prepare_pipeline(args)
    pipeline.set_progress_bar_config(disable=True)

    data_root = Path(args.data_root)
    prompt_subdir = config["data"].get("prompt_subdir", "prompts_ego")

    video_list = Path(args.video_list)
    with video_list.open("r", encoding="utf-8") as f:
        video_entries = [line.strip() for line in f if line.strip()]

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint) / "eval_for_qual"
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    pipeline_type = config["pipeline"]["type"]
    static_video_entries = [str(Path(entry).parent.parent / "videos_static" / Path(entry).name) for entry in video_entries]
    for entry in tqdm(static_video_entries, desc="Generating videos"):
        rel_path = Path(entry)
        video_path = data_root / rel_path
        if not video_path.exists():
            print(f"⚠️ Video not found, skipping: {video_path}")
            continue

        static_video = _load_video_tensor(video_path)
        mask_video = torch.zeros((1, 1, static_video.shape[2], static_video.shape[3], static_video.shape[4]), dtype=torch.uint8)

        prompt_path = data_root / rel_path.parent.parent / prompt_subdir / f"{video_path.stem}.txt"
        prompt = "" if args.no_prompt else (_load_prompt(prompt_path) if prompt_path.exists() else "")
        
        # Prepare pipeline arguments based on pipeline type
        if pipeline_type == "cogvideox_static_to_video_pose_concat":
            # For concat pipeline, need hand_videos (video tensor)
            hand_video_path = video_path.parent.parent / "videos_hands" / video_path.name
            if not hand_video_path.exists():
                print(f"⚠️ Hand video file missing, skipping: {hand_video_path}")
                continue
            hand_video = _load_video_tensor(hand_video_path)
            
            for idx in range(3):
                name = entry.replace("/", "_").replace(".mp4", f"_{idx}.mp4")
                save_path = output_dir / name
                
                # Skip if file already exists
                if save_path.exists():
                    print(f"⏭️  Skipping {save_path.name} (already exists)")
                    continue
                
                with torch.no_grad():
                    output = pipeline(
                        prompt=prompt,
                        negative_prompt=args.negative_prompt,
                        num_frames=static_video.shape[2],
                        static_videos=static_video,
                        hand_videos=hand_video,
                        guidance_scale=args.guidance_scale,
                        num_inference_steps=args.num_inference_steps,
                    )

                generated = output.frames[0]
                if isinstance(generated, torch.Tensor):
                    generated = generated.cpu().numpy()
                if generated.ndim == 4:  # [F, H, W, C]
                    video_array = generated
                elif generated.ndim == 5:  # [B, F, H, W, C]
                    video_array = generated[0]
                else:
                    raise ValueError(f"Unexpected generated video shape: {generated.shape}")

                if video_array.dtype != np.uint8:
                    if video_array.max() <= 1.0:
                        video_array = (video_array * 255.0).clip(0, 255).astype(np.uint8)
                    else:
                        video_array = video_array.astype(np.uint8)

                iio.imwrite(save_path.as_posix(), video_array, fps=8)
        else:
            # For AdaLN variants, use pose_params
            motion_path = video_path.parent.parent / "hand_motions" / f"{video_path.stem}.pt"
            if not motion_path.exists():
                print(f"⚠️ Hand motion file missing, skipping: {motion_path}")
                continue
            pose_params = _load_pose_params(motion_path)
            
            for idx in range(3):
                name = entry.replace("/", "_").replace(".mp4", f"_{idx}.mp4")
                save_path = output_dir / name
                
                # Skip if file already exists
                if save_path.exists():
                    print(f"⏭️  Skipping {save_path.name} (already exists)")
                    continue
                
                with torch.no_grad():
                    output = pipeline(
                        prompt=prompt,
                        negative_prompt=args.negative_prompt,
                        num_frames=static_video.shape[2],
                        static_videos=static_video,
                        pose_params=pose_params.to(args.device, dtype=pipeline.transformer.dtype),
                        mask_video=mask_video,
                        guidance_scale=args.guidance_scale,
                        num_inference_steps=args.num_inference_steps,
                    )

                generated = output.frames[0]
                if isinstance(generated, torch.Tensor):
                    generated = generated.cpu().numpy()
                if generated.ndim == 4:  # [F, H, W, C]
                    video_array = generated
                elif generated.ndim == 5:  # [B, F, H, W, C]
                    video_array = generated[0]
                else:
                    raise ValueError(f"Unexpected generated video shape: {generated.shape}")

                if video_array.dtype != np.uint8:
                    if video_array.max() <= 1.0:
                        video_array = (video_array * 255.0).clip(0, 255).astype(np.uint8)
                    else:
                        video_array = video_array.astype(np.uint8)

                iio.imwrite(save_path.as_posix(), video_array, fps=8)

    print(f"Generation finished. Videos saved to: {output_dir}")


def main():
    args = parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()