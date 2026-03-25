#!/usr/bin/env python3
"""Encode videos and prompts using CogVideoX VAE and T5 text encoder."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.cogvideox.diffusers_compat import disable_broken_torchao

disable_broken_torchao()

import decord
import torch
from diffusers import AutoencoderKLCogVideoX
from einops import rearrange
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer

try:
    from dataset_layout_utils import iter_taste_rob_sample_dirs, iter_trumans_action_dirs
except ImportError:
    from data_processing.dataset_layout_utils import iter_taste_rob_sample_dirs, iter_trumans_action_dirs


decord.bridge.set_bridge("torch")

DTYPE_MAPPING = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

DEFAULT_MODEL_ID = "THUDM/CogVideoX-5b"


def load_cogvideox_vae(model_id: str, device: torch.device, dtype: torch.dtype):
    vae = AutoencoderKLCogVideoX.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype).eval()
    return vae.to(device=device)


def load_cogvideox_text_encoder(model_id: str, device: torch.device, dtype: torch.dtype):
    tokenizer = T5Tokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=dtype).eval()
    return tokenizer, text_encoder.to(device=device)


def sample_frame_indices(num_frames: int, max_num_frames: int) -> List[int]:
    if num_frames <= 0:
        raise ValueError(f"Video has no frames: {num_frames}")
    if max_num_frames <= 1:
        return [0]
    if num_frames == 1:
        return [0] * max_num_frames
    return torch.linspace(0, num_frames - 1, steps=max_num_frames).round().to(torch.int64).tolist()


@torch.no_grad()
def encode_video(vae, video_path: str, device: torch.device, max_num_frames: int) -> torch.Tensor:
    vr = decord.VideoReader(video_path)
    video = vr.get_batch(sample_frame_indices(len(vr), max_num_frames=max_num_frames))
    video = video.float() / 255.0
    video = video * 2.0 - 1.0
    video = rearrange(video, "f h w c -> c f h w").unsqueeze(0)
    video = video.to(device=device, dtype=vae.dtype)
    latents = vae._encode(video)
    return latents.squeeze(0).to("cpu")


@torch.no_grad()
def encode_prompt(tokenizer, text_encoder, prompt: str, device: torch.device, max_length: int = 226, pad_to_max_length: bool = True) -> torch.Tensor:
    inputs = tokenizer(
        prompt,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    embeds = text_encoder(input_ids=input_ids)[0]
    if not pad_to_max_length:
        actual_len = int(attention_mask.sum().item())
        embeds = embeds[:, :actual_len, :]
    return embeds.squeeze(0).to("cpu")


def build_jobs(data_root: Path, dataset_type: str, modalities: List[str], prompt_subdir: str = "prompts_rewrite") -> Dict[str, List[Dict[str, str]]]:
    jobs: Dict[str, List[Dict[str, str]]] = {mod: [] for mod in modalities}
    modality_mapping = {
        "videos": ("videos", "video_latents"),
        "static_videos": ("videos_static", "static_video_latents"),
        "hand_videos": ("videos_hands", "hand_video_latents"),
        "prompts": (prompt_subdir, f"prompt_embeds_{prompt_subdir}"),
    }

    if dataset_type == "trumans":
        iterator = iter_trumans_action_dirs(data_root)
    elif dataset_type == "taste_rob":
        iterator = iter_taste_rob_sample_dirs(data_root)
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    for _, sample_dir in iterator:
        for modality in modalities:
            if modality not in modality_mapping:
                continue
            src_dir_name, out_dir_name = modality_mapping[modality]
            src_dir = sample_dir / src_dir_name
            out_dir = sample_dir / out_dir_name
            if not src_dir.exists():
                continue
            pattern = "*.txt" if modality == "prompts" else "*.mp4"
            for src_file in sorted(src_dir.glob(pattern)):
                jobs[modality].append({"src": str(src_file), "out": str(out_dir / f"{src_file.stem}.pt")})
    return jobs


def shard_jobs(jobs: List[Dict[str, str]], rank: int, world_size: int) -> List[Dict[str, str]]:
    return jobs[rank::world_size]


def safe_read_first_line(txt_path: Path) -> Optional[str]:
    if not txt_path.exists():
        return None
    try:
        with txt_path.open("r", encoding="utf-8") as handle:
            return handle.readline().strip()
    except Exception:
        return None


def process_modality(modality: str, jobs: List[Dict[str, str]], vae, tokenizer, text_encoder, device: torch.device, args, rank: int, world_size: int):
    my_jobs = shard_jobs(jobs, rank, world_size)
    if args.debug:
        my_jobs = my_jobs[:1]
    if not my_jobs:
        print(f"[rank {rank}/{world_size}] No {modality} jobs assigned")
        return

    processed = 0
    skipped = 0
    errors = []
    pbar = tqdm(my_jobs, desc=f"[rank {rank}/{world_size}] {modality}", dynamic_ncols=True)
    for job in pbar:
        src_path = Path(job["src"])
        out_path = Path(job["out"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if args.skip_existing and out_path.exists():
            skipped += 1
            continue
        try:
            if modality == "prompts":
                prompt = safe_read_first_line(src_path)
                if prompt is None:
                    errors.append({"src": str(src_path), "reason": "cannot_read_prompt"})
                    continue
                embeds = encode_prompt(tokenizer, text_encoder, prompt, device, max_length=args.max_sequence_length, pad_to_max_length=args.pad_to_max_length)
                torch.save(embeds, out_path)
            else:
                latents = encode_video(vae, str(src_path), device, max_num_frames=args.max_num_frames)
                torch.save(latents, out_path)
            processed += 1
        except Exception as exc:
            errors.append({"src": str(src_path), "reason": str(exc)})
        if processed % 10 == 0:
            pbar.set_postfix(done=processed, skipped=skipped, errors=len(errors))

    print(f"[rank {rank}/{world_size}] {modality}: processed={processed} skipped={skipped} errors={len(errors)}")
    if errors:
        error_path = Path(args.error_dir) / f"encode_{args.dataset_type}_{modality}_rank{rank}.json"
        error_path.parent.mkdir(parents=True, exist_ok=True)
        error_path.write_text(json.dumps(errors, indent=2), encoding="utf-8")
        print(f"[rank {rank}/{world_size}] wrote errors to {error_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode DWM dataset modalities with CogVideoX.")
    parser.add_argument("--dataset_type", required=True, choices=["trumans", "taste_rob"])
    parser.add_argument("--data_root", type=Path, default=None)
    parser.add_argument("--modalities", nargs="+", required=True, choices=["videos", "static_videos", "hand_videos", "prompts"])
    parser.add_argument("--prompt_subdir", type=str, default="prompts_rewrite")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--dtype", type=str, default="bf16", choices=sorted(DTYPE_MAPPING))
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max_sequence_length", type=int, default=226)
    parser.add_argument("--max_num_frames", type=int, default=49)
    parser.add_argument("--pad_to_max_length", action="store_true", default=True)
    parser.add_argument("--error_dir", type=Path, default=Path("out/encode_cogvideox_errors"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root
    if data_root is None:
        if args.dataset_type == "trumans":
            data_root = Path("data_refactor/trumans")
        else:
            data_root = Path("data_refactor/taste_rob_resized")
    data_root = data_root.resolve()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = DTYPE_MAPPING[args.dtype]
    jobs = build_jobs(data_root, args.dataset_type, args.modalities, prompt_subdir=args.prompt_subdir)

    vae = None
    tokenizer = None
    text_encoder = None
    if any(mod != "prompts" for mod in args.modalities):
        vae = load_cogvideox_vae(args.model_id, device, dtype)
    if "prompts" in args.modalities:
        tokenizer, text_encoder = load_cogvideox_text_encoder(args.model_id, device, dtype)

    for modality in args.modalities:
        process_modality(modality, jobs.get(modality, []), vae, tokenizer, text_encoder, device, args, args.rank, args.world_size)


if __name__ == "__main__":
    main()
