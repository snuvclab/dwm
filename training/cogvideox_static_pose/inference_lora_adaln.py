#!/usr/bin/env python3
"""
Inference script for CogVideoX-Fun static-to-video AdaLN variants.
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

from diffusers.utils import convert_unet_state_dict_to_peft
from peft import LoraConfig, set_peft_model_state_dict

from training.cogvideox_static_pose.config_loader import load_experiment_config
from training.cogvideox_static_pose.cogvideox_text_to_video_pose_sft_unified import (
    setup_pipeline_from_config,
)
from training.cogvideox_static_pose.cogvideox_fun_static_to_video_pose_adaln_pipeline import (
    CogVideoXFunStaticToVideoPoseAdaLNPipeline,
    CogVideoXFunStaticToVideoPoseAdaLNPerFramePipeline,
)


SUPPORTED_PIPELINES = {
    "cogvideox_fun_static_to_video_pose_adaln",
    "cogvideox_fun_static_to_video_pose_adaln_perframe",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference for CogVideoX-Fun AdaLN LoRA checkpoints")
    parser.add_argument("--config", type=str, required=True, help="Training config YAML used for this checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Directory containing LoRA checkpoint files")
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


def _apply_projection_weights(transformer, state_dict: dict) -> None:
    proj = getattr(transformer.patch_embed, "proj", None)
    if proj is None:
        return
    weight = state_dict.get("transformer.patch_embed.proj.weight")
    bias = state_dict.get("transformer.patch_embed.proj.bias")
    if weight is not None and proj.weight.shape == weight.shape:
        proj.weight.data.copy_(weight)
    if bias is not None and proj.bias is not None and proj.bias.shape == bias.shape:
        proj.bias.data.copy_(bias)


def _apply_cond_proj_weights(transformer, state_dict: dict) -> None:
    cond_proj = getattr(transformer.patch_embed, "cond_proj", None)
    if cond_proj is None:
        return
    if "transformer.patch_embed.cond_proj.weight" in state_dict:
        cond_proj.weight.data.copy_(state_dict["transformer.patch_embed.cond_proj.weight"])
    if cond_proj.bias is not None and "transformer.patch_embed.cond_proj.bias" in state_dict:
        cond_proj.bias.data.copy_(state_dict["transformer.patch_embed.cond_proj.bias"])

    # Optional auxiliary layers
    if hasattr(transformer.patch_embed, "cond_norm") and "transformer.patch_embed.cond_norm.weight" in state_dict:
        transformer.patch_embed.cond_norm.weight.data.copy_(state_dict["transformer.patch_embed.cond_norm.weight"])
        transformer.patch_embed.cond_norm.bias.data.copy_(state_dict["transformer.patch_embed.cond_norm.bias"])

    if hasattr(transformer.patch_embed, "cond_gate") and "transformer.patch_embed.cond_gate" in state_dict:
        transformer.patch_embed.cond_gate.data.copy_(state_dict["transformer.patch_embed.cond_gate"])

    if hasattr(transformer.patch_embed, "add_conv_in") and "transformer.patch_embed.add_conv_in.weight" in state_dict:
        transformer.patch_embed.add_conv_in.weight.data.copy_(state_dict["transformer.patch_embed.add_conv_in.weight"])
        if "transformer.patch_embed.add_conv_in.bias" in state_dict and transformer.patch_embed.add_conv_in.bias is not None:
            transformer.patch_embed.add_conv_in.bias.data.copy_(state_dict["transformer.patch_embed.add_conv_in.bias"])

    if hasattr(transformer.patch_embed, "add_norm") and transformer.patch_embed.add_norm is not None:
        if "transformer.patch_embed.add_norm.weight" in state_dict:
            transformer.patch_embed.add_norm.weight.data.copy_(state_dict["transformer.patch_embed.add_norm.weight"])
        if "transformer.patch_embed.add_norm.bias" in state_dict:
            transformer.patch_embed.add_norm.bias.data.copy_(state_dict["transformer.patch_embed.add_norm.bias"])

    if hasattr(transformer.patch_embed, "add_zero_proj") and transformer.patch_embed.add_zero_proj is not None:
        if "transformer.patch_embed.add_zero_proj.weight" in state_dict:
            transformer.patch_embed.add_zero_proj.weight.data.copy_(state_dict["transformer.patch_embed.add_zero_proj.weight"])
        if "transformer.patch_embed.add_zero_proj.bias" in state_dict and transformer.patch_embed.add_zero_proj.bias is not None:
            transformer.patch_embed.add_zero_proj.bias.data.copy_(state_dict["transformer.patch_embed.add_zero_proj.bias"])


def _load_additional_weights(transformer, checkpoint_dir: Path) -> None:
    non_lora_path = checkpoint_dir / "non_lora_weights.pt"
    if non_lora_path.exists():
        state_dict = torch.load(non_lora_path, map_location="cpu")
        model_state = transformer.state_dict()
        loaded = []
        for name, tensor in state_dict.items():
            if name in model_state and model_state[name].shape == tensor.shape:
                model_state[name].copy_(tensor)
                loaded.append(name)
        if loaded:
            print(f"✅ Loaded non-LoRA weights: {len(loaded)} tensors from {non_lora_path.name}")
        else:
            print(f"⚠️ non_lora_weights.pt found but no matching parameters were loaded.")
        return

    projection_path = checkpoint_dir / "projection_layer_weights.pt"
    if projection_path.exists():
        state_dict = torch.load(projection_path, map_location="cpu")
        _apply_projection_weights(transformer, state_dict)
        print(f"✅ Loaded projection weights from {projection_path.name}")

    cond_proj_path = checkpoint_dir / "cond_proj_weights.pt"
    if cond_proj_path.exists():
        state_dict = torch.load(cond_proj_path, map_location="cpu")
        _apply_cond_proj_weights(transformer, state_dict)
        print(f"✅ Loaded adapter projection weights from {cond_proj_path.name}")


def _setup_lora_adapter(transformer, config: dict) -> None:
    training_cfg = config.get("training", {})
    lora_rank = training_cfg.get("lora_rank", 64)
    lora_alpha = training_cfg.get("lora_alpha", 64)
    lora_cfg = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        init_lora_weights=True,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    transformer.add_adapter(lora_cfg)


def _load_lora_weights(pipeline, checkpoint_dir: Path, pipeline_type: str, adapter_name: str = "default") -> None:
    """Load LoRA weights using pipeline's lora_state_dict method, matching training script logic."""
    # Use pipeline's lora_state_dict method (same as training script)
    if pipeline_type == "cogvideox_fun_static_to_video_pose_adaln":
        lora_state_dict = CogVideoXFunStaticToVideoPoseAdaLNPipeline.lora_state_dict(str(checkpoint_dir))
    elif pipeline_type == "cogvideox_fun_static_to_video_pose_adaln_perframe":
        lora_state_dict = CogVideoXFunStaticToVideoPoseAdaLNPerFramePipeline.lora_state_dict(str(checkpoint_dir))
    else:
        raise ValueError(f"Unsupported pipeline type for LoRA loading: {pipeline_type}")
    
    if not lora_state_dict:
        print("⚠️ No LoRA weights found; running with base model weights.")
        return
    
    # Remove "transformer." prefix and convert to PEFT format (same as training script)
    transformer_state_dict = {
        f'{k.replace("transformer.", "")}': v for k, v in lora_state_dict.items() if k.startswith("transformer.")
    }
    transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
    set_peft_model_state_dict(pipeline.transformer, transformer_state_dict, adapter_name=adapter_name)
    print(f"✅ Loaded LoRA weights from {checkpoint_dir}")


def prepare_pipeline(args: argparse.Namespace):
    config = load_experiment_config(args.config, overrides=None)

    pipeline_type = config["pipeline"]["type"]
    if pipeline_type not in SUPPORTED_PIPELINES:
        raise ValueError(f"Pipeline type '{pipeline_type}' is not supported by this script.")

    pipeline = setup_pipeline_from_config(config)
    pipeline.to(args.device)
    pipeline.vae.to(dtype=torch.bfloat16 if "5b" in config["model"]["base_model_name_or_path"].lower() else torch.float16)
    pipeline.transformer.to(dtype=torch.bfloat16 if "5b" in config["model"]["base_model_name_or_path"].lower() else torch.float16)
    pipeline.text_encoder.to(dtype=torch.bfloat16 if "5b" in config["model"]["base_model_name_or_path"].lower() else torch.float16)

    checkpoint_dir = Path(args.checkpoint)
    _setup_lora_adapter(pipeline.transformer, config)
    _load_lora_weights(pipeline, checkpoint_dir, pipeline_type)
    _load_additional_weights(pipeline.transformer, checkpoint_dir)

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
    static_video_entries = [str(Path(entry).parent.parent / "videos_static" / Path(entry).name) for entry in video_entries]
    for entry in tqdm(static_video_entries, desc="Generating videos"):
        rel_path = Path(entry)
        video_path = data_root / rel_path
        if not video_path.exists():
            print(f"⚠️ Video not found, skipping: {video_path}")
            continue

        static_video = _load_video_tensor(video_path)
        mask_video = torch.zeros((1, 1, static_video.shape[2], static_video.shape[3], static_video.shape[4]), dtype=torch.uint8)

        motion_path = video_path.parent.parent / "hand_motions" / f"{video_path.stem}.pt"
        if not motion_path.exists():
            print(f"⚠️ Hand motion file missing, skipping: {motion_path}")
            continue
        pose_params = _load_pose_params(motion_path)

        prompt_path = data_root / rel_path.parent.parent / prompt_subdir / f"{video_path.stem}.txt"
        prompt = "" if args.no_prompt else (_load_prompt(prompt_path) if prompt_path.exists() else "")
        for idx in range(3):
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

            name = entry.replace("/", "_").replace(".mp4", f"_{idx}.mp4")
            save_path = output_dir / name
            iio.imwrite(save_path.as_posix(), video_array, fps=8)

    print(f"Generation finished. Videos saved to: {output_dir}")


def main():
    args = parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()