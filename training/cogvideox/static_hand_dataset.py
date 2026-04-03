from __future__ import annotations

from typing import Any, Dict, List, Sequence

import torch
from torch.utils.data import Dataset

from training.cogvideox.static_hand_utils import (
    coerce_latent_tensor,
    latent_frames_for_video_frames,
    load_prompt_text,
    load_tensor,
    load_video_clip,
    read_dataset_entries,
    resolve_sample_paths,
)


class StaticHandConcatDataset(Dataset):
    """Minimal dataset for the standalone CogVideoX static-hand-concat path."""

    def __init__(
        self,
        data_root: str,
        dataset_file: Sequence[str] | str,
        max_num_frames: int = 49,
        height: int = 480,
        width: int = 720,
        load_tensors: bool = True,
        prompt_subdir: str = "prompts_rewrite",
        prompt_embeds_subdir: str = "prompt_embeds_rewrite",
        hand_video_subdir: str = "videos_hands",
        hand_video_latents_subdir: str = "hand_video_latents",
        static_video_subdir: str = "videos_static",
        static_video_latents_subdir: str = "static_video_latents",
        video_latents_subdir: str = "video_latents",
    ) -> None:
        self.data_root = data_root
        self.entries = read_dataset_entries(data_root, dataset_file)
        self.max_num_frames = max_num_frames
        self.height = height
        self.width = width
        self.load_tensors = load_tensors
        self.prompt_subdir = prompt_subdir
        self.prompt_embeds_subdir = prompt_embeds_subdir
        self.hand_video_subdir = hand_video_subdir
        self.hand_video_latents_subdir = hand_video_latents_subdir
        self.static_video_subdir = static_video_subdir
        self.static_video_latents_subdir = static_video_latents_subdir
        self.video_latents_subdir = video_latents_subdir

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        relative_video_path = self.entries[index]
        sample_paths = resolve_sample_paths(
            data_root=self.data_root,
            relative_video_path=relative_video_path,
            prompt_subdir=self.prompt_subdir,
            prompt_embeds_subdir=self.prompt_embeds_subdir,
            hand_video_subdir=self.hand_video_subdir,
            hand_latents_subdir=self.hand_video_latents_subdir,
            static_video_subdir=self.static_video_subdir,
            static_latents_subdir=self.static_video_latents_subdir,
            video_latents_subdir=self.video_latents_subdir,
        )
        target_latent_frames = latent_frames_for_video_frames(self.max_num_frames)

        if self.load_tensors:
            prompt_value = load_tensor(sample_paths.prompt_embeds_path)
            video_value = coerce_latent_tensor(
                load_tensor(sample_paths.video_latents_path),
                target_frames=target_latent_frames,
                height=self.height,
                width=self.width,
            )

            try:
                hand_value = coerce_latent_tensor(
                    load_tensor(sample_paths.hand_latents_path),
                    target_frames=target_latent_frames,
                    height=self.height,
                    width=self.width,
                )
            except FileNotFoundError:
                hand_value = load_video_clip(
                    sample_paths.hand_video_path,
                    max_num_frames=self.max_num_frames,
                    height=self.height,
                    width=self.width,
                )

            try:
                static_value = coerce_latent_tensor(
                    load_tensor(sample_paths.static_latents_path),
                    target_frames=target_latent_frames,
                    height=self.height,
                    width=self.width,
                )
            except FileNotFoundError:
                static_value = load_video_clip(
                    sample_paths.static_video_path,
                    max_num_frames=self.max_num_frames,
                    height=self.height,
                    width=self.width,
                )
        else:
            prompt_value = load_prompt_text(sample_paths.prompt_path)
            video_value = load_video_clip(
                sample_paths.video_path,
                max_num_frames=self.max_num_frames,
                height=self.height,
                width=self.width,
            )
            hand_value = load_video_clip(
                sample_paths.hand_video_path,
                max_num_frames=self.max_num_frames,
                height=self.height,
                width=self.width,
            )
            static_value = load_video_clip(
                sample_paths.static_video_path,
                max_num_frames=self.max_num_frames,
                height=self.height,
                width=self.width,
            )

        return {
            "prompt": prompt_value,
            "prompt_text": load_prompt_text(sample_paths.prompt_path),
            "video": video_value,
            "hand_videos": hand_value,
            "static_videos": static_value,
            "relative_video_path": relative_video_path,
        }


def collate_static_hand_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    prompts = batch[0]["prompt"]
    if isinstance(prompts, torch.Tensor):
        collated_prompts: Any = torch.stack([item["prompt"] for item in batch])
    else:
        collated_prompts = [item["prompt"] for item in batch]

    return {
        "prompts": collated_prompts,
        "prompt_texts": [item["prompt_text"] for item in batch],
        "videos": torch.stack([item["video"] for item in batch]),
        "hand_videos": torch.stack([item["hand_videos"] for item in batch]),
        "static_videos": torch.stack([item["static_videos"] for item in batch]),
        "relative_video_paths": [item["relative_video_path"] for item in batch],
    }


__all__ = ["StaticHandConcatDataset", "collate_static_hand_batch"]
