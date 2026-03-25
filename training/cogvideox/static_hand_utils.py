from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import decord
import torch
import torch.nn.functional as F

from training.cogvideox.diffusers_compat import disable_broken_torchao

disable_broken_torchao()

from diffusers.utils import convert_unet_state_dict_to_peft
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file, save_file

from training.cogvideox.models import CogVideoXFunStaticHandConcatTransformer3DModel


decord.bridge.set_bridge("torch")


@dataclass(frozen=True)
class StaticHandSamplePaths:
    video_path: Path
    video_latents_path: Path
    prompt_path: Path
    prompt_embeds_path: Path
    hand_video_path: Path
    hand_latents_path: Path
    static_video_path: Path
    static_latents_path: Path


def read_dataset_entries(data_root: str, dataset_files: Sequence[str] | str) -> List[str]:
    if isinstance(dataset_files, str):
        dataset_file_list = [dataset_files]
    else:
        dataset_file_list = list(dataset_files)

    entries: List[str] = []
    for dataset_file in dataset_file_list:
        dataset_path = Path(dataset_file)
        if not dataset_path.is_absolute():
            dataset_path = Path(data_root) / dataset_file

        with dataset_path.open("r", encoding="utf-8") as handle:
            entries.extend(line.strip() for line in handle if line.strip())

    return entries


def resolve_sample_paths(
    data_root: str,
    relative_video_path: str,
    prompt_subdir: str,
    prompt_embeds_subdir: str,
    hand_video_subdir: str = "videos_hands",
    hand_latents_subdir: str = "hand_video_latents",
    static_video_subdir: str = "videos_static",
    static_latents_subdir: str = "static_video_latents",
    video_latents_subdir: str = "video_latents",
) -> StaticHandSamplePaths:
    video_path = Path(data_root) / relative_video_path
    sequence_root = video_path.parent.parent
    stem = video_path.stem

    return StaticHandSamplePaths(
        video_path=video_path,
        video_latents_path=sequence_root / video_latents_subdir / f"{stem}.pt",
        prompt_path=sequence_root / prompt_subdir / f"{stem}.txt",
        prompt_embeds_path=sequence_root / prompt_embeds_subdir / f"{stem}.pt",
        hand_video_path=sequence_root / hand_video_subdir / f"{stem}.mp4",
        hand_latents_path=sequence_root / hand_latents_subdir / f"{stem}.pt",
        static_video_path=sequence_root / static_video_subdir / f"{stem}.mp4",
        static_latents_path=sequence_root / static_latents_subdir / f"{stem}.pt",
    )


def load_prompt_text(path: Path, default: str = "") -> str:
    if not path.exists():
        return default
    return path.read_text(encoding="utf-8").strip()


def load_tensor(path: Path) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=True)


def _sample_video_frames(video_path: Path, max_num_frames: int) -> torch.Tensor:
    video_reader = decord.VideoReader(uri=video_path.as_posix())
    video_num_frames = len(video_reader)
    step = max(1, video_num_frames // max_num_frames)
    indices = list(range(0, video_num_frames, step))[:max_num_frames]
    frames = video_reader.get_batch(indices).float()
    frames = frames.permute(0, 3, 1, 2).contiguous()  # [F, C, H, W]
    return frames


def _resize_video_frames(frames: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if frames.shape[-2:] == (height, width):
        return frames
    return F.interpolate(frames, size=(height, width), mode="bilinear", align_corners=False)


def _match_video_length(frames: torch.Tensor, target_frames: int) -> torch.Tensor:
    current_frames = frames.shape[0]
    if current_frames == target_frames:
        return frames
    if current_frames > 1:
        frames = frames.permute(1, 0, 2, 3).unsqueeze(0)
        frames = F.interpolate(frames, size=(target_frames, frames.shape[-2], frames.shape[-1]), mode="trilinear", align_corners=False)
        return frames.squeeze(0).permute(1, 0, 2, 3).contiguous()

    return frames.repeat(target_frames, 1, 1, 1)


def load_video_clip(video_path: Path, max_num_frames: int, height: int, width: int) -> torch.Tensor:
    frames = _sample_video_frames(video_path, max_num_frames=max_num_frames)
    frames = _resize_video_frames(frames, height=height, width=width)
    frames = _match_video_length(frames, target_frames=max_num_frames)
    frames = frames / 127.5 - 1.0
    return frames.permute(1, 0, 2, 3).contiguous()  # [C, F, H, W]


def latent_frames_for_video_frames(max_num_frames: int) -> int:
    return max(1, math.ceil(max_num_frames / 4))


def coerce_latent_tensor(
    value: torch.Tensor,
    target_frames: int,
    height: int,
    width: int,
) -> torch.Tensor:
    squeeze_batch = False
    if value.ndim == 4:
        if value.shape[0] in {16, 32}:
            value = value.unsqueeze(0)
            squeeze_batch = True
        elif value.shape[1] in {16, 32}:
            value = value.permute(1, 0, 2, 3).unsqueeze(0)
            squeeze_batch = True
        else:
            raise ValueError(f"Unsupported 4D latent shape: {tuple(value.shape)}")
    elif value.ndim == 5:
        if value.shape[1] in {16, 32}:
            pass
        elif value.shape[2] in {16, 32}:
            value = value.permute(0, 2, 1, 3, 4)
        else:
            raise ValueError(f"Unsupported 5D latent shape: {tuple(value.shape)}")
    else:
        raise ValueError(f"Expected a 4D or 5D latent tensor, got shape {tuple(value.shape)}")

    target_height = max(1, height // 8)
    target_width = max(1, width // 8)
    _, _, frames, current_height, current_width = value.shape
    if (frames, current_height, current_width) != (target_frames, target_height, target_width):
        value = F.interpolate(
            value.float(),
            size=(target_frames, target_height, target_width),
            mode="trilinear",
            align_corners=False,
        ).to(dtype=value.dtype)

    if squeeze_batch:
        return value.squeeze(0).contiguous()
    return value.contiguous()


def coerce_video_tensor(
    video: torch.Tensor,
    target_frames: int,
    height: int,
    width: int,
) -> torch.Tensor:
    if video.ndim == 4:
        if video.shape[0] in {1, 3, 6}:
            video = video.unsqueeze(0)  # [1, C, F, H, W]
        elif video.shape[1] in {1, 3, 6}:
            video = video.permute(1, 0, 2, 3).unsqueeze(0)
        else:
            raise ValueError(f"Unsupported 4D video shape: {tuple(video.shape)}")
    elif video.ndim == 5:
        if video.shape[1] in {1, 3, 6}:
            pass
        elif video.shape[2] in {1, 3, 6}:
            video = video.permute(0, 2, 1, 3, 4)
        else:
            raise ValueError(f"Unsupported 5D video shape: {tuple(video.shape)}")
    else:
        raise ValueError(f"Expected a 4D or 5D tensor, got shape {tuple(video.shape)}")

    video = video.float()
    if video.max() > 1.0 or video.min() < -1.0:
        video = video / 127.5 - 1.0
    elif video.min() >= 0.0:
        video = video * 2.0 - 1.0

    batch, channels, frames, current_height, current_width = video.shape
    if (current_height, current_width) != (height, width):
        video = F.interpolate(video, size=(frames, height, width), mode="trilinear", align_corners=False)
    if frames != target_frames:
        video = F.interpolate(video, size=(target_frames, height, width), mode="trilinear", align_corners=False)

    return video.contiguous()


def build_lora_config(training_config: Dict[str, Any]) -> LoraConfig:
    return LoraConfig(
        r=training_config.get("lora_rank", 64),
        lora_alpha=training_config.get("lora_alpha", 64),
        init_lora_weights=True,
        target_modules=training_config.get("lora_target_modules", ["to_k", "to_q", "to_v", "to_out.0"]),
    )


def collect_non_lora_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    state_dict: Dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if param.requires_grad and "lora" not in name.lower():
            state_dict[name] = param.detach().cpu()
    return state_dict


def save_non_lora_state_dict(output_dir: str, state_dict: Dict[str, torch.Tensor]) -> None:
    if not state_dict:
        return
    save_file(state_dict, os.path.join(output_dir, "non_lora_weights.safetensors"))


def load_non_lora_state_dict(model: torch.nn.Module, input_dir: str) -> List[str]:
    state_path = os.path.join(input_dir, "non_lora_weights.safetensors")
    if not os.path.exists(state_path):
        return []

    non_lora_state_dict = load_file(state_path)
    model_state_dict = model.state_dict()
    loaded_keys: List[str] = []
    for name, value in non_lora_state_dict.items():
        if name in model_state_dict:
            model_state_dict[name].copy_(value.to(model_state_dict[name].dtype))
            loaded_keys.append(name)
    return loaded_keys


def load_lora_weights_into_transformer(transformer: torch.nn.Module, checkpoint_path: str) -> List[str]:
    from training.cogvideox.pipeline import CogVideoXFunStaticHandConcatPipeline

    lora_state_dict = CogVideoXFunStaticHandConcatPipeline.lora_state_dict(checkpoint_path)
    transformer_state_dict = {
        key.replace("transformer.", ""): value
        for key, value in lora_state_dict.items()
        if key.startswith("transformer.")
    }
    if not transformer_state_dict:
        return []

    transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
    incompatible = set_peft_model_state_dict(transformer, transformer_state_dict, adapter_name="default")
    if incompatible is None:
        return []
    return list(getattr(incompatible, "unexpected_keys", []) or [])


def write_inference_metadata(output_dir: Path, metadata: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)


def build_pipeline_from_config(
    config: Dict[str, Any],
    pretrained_transformer_path: Optional[str] = None,
    transformer: Optional[CogVideoXFunStaticHandConcatTransformer3DModel] = None,
    torch_dtype: Optional[torch.dtype] = None,
) -> "CogVideoXFunStaticHandConcatPipeline":
    from training.cogvideox.pipeline import CogVideoXFunStaticHandConcatPipeline

    model_config = config["model"]
    pipeline_config = config["pipeline"]

    if torch_dtype is None:
        base_model_path = model_config["base_model_name_or_path"]
        torch_dtype = torch.bfloat16 if "5b" in base_model_path.lower() else torch.float16

    return CogVideoXFunStaticHandConcatPipeline.from_pretrained(
        pretrained_model_name_or_path=pretrained_transformer_path,
        base_model_name_or_path=model_config["base_model_name_or_path"],
        transformer=transformer,
        condition_channels=pipeline_config.get("condition_channels", 16),
        torch_dtype=torch_dtype,
        revision=model_config.get("revision"),
        variant=model_config.get("variant"),
    )
