#!/usr/bin/env python3
"""Encode videos and rewritten prompts for the DWM WAN training path."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import decord
import torch
import torch.nn.functional as F
from decord import VideoReader
from einops import rearrange
from tqdm import tqdm
from transformers import AutoTokenizer

from training.wan.models import AutoencoderKLWan, AutoencoderKLWan3_8, WanT5EncoderModel

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

DEFAULT_MODEL_PATH_2_1 = str(Path("~/.cache/modelscope/hub/models/PAI/Wan2.1-Fun-V1.1-14B-InP/").expanduser())
DEFAULT_MODEL_PATH_2_2 = str(Path("~/.cache/modelscope/hub/models/PAI/Wan2.2-Fun-14B-InP").expanduser())

DEFAULT_TEXT_ENCODER_KWARGS = {
    "vocab": 256384,
    "dim": 4096,
    "dim_attn": 4096,
    "dim_ffn": 10240,
    "num_heads": 64,
    "num_layers": 24,
    "num_buckets": 32,
    "shared_pos": False,
    "dropout": 0.0,
}

VIDEO_MODALITIES = {"videos", "static_videos", "hand_videos"}


def normalize_prompt_subdir(prompt_subdir: str) -> str:
    return prompt_subdir[len("prompts_") :] if prompt_subdir.startswith("prompts_") else prompt_subdir


def load_wan_vae(model_path: str, device: torch.device, dtype: torch.dtype, wan_version: str):
    if wan_version == "2.2":
        vae_cls = AutoencoderKLWan3_8
        vae_kwargs = {
            "vae_type": "AutoencoderKLWan3_8",
            "vae_subpath": "Wan2.2_VAE.pth",
            "temporal_compression_ratio": 4,
            "spatial_compression_ratio": 16,
        }
    else:
        vae_cls = AutoencoderKLWan
        vae_kwargs = {
            "vae_type": "AutoencoderKLWan",
            "vae_subpath": "Wan2.1_VAE.pth",
            "temporal_compression_ratio": 4,
            "spatial_compression_ratio": 8,
        }

    vae = vae_cls.from_pretrained(
        os.path.join(model_path, vae_kwargs["vae_subpath"]),
        additional_kwargs=vae_kwargs,
    ).eval()
    return vae.to(device=device, dtype=dtype)


def load_wan_text_encoder(model_path: str, device: torch.device, dtype: torch.dtype):
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_path, "google/umt5-xxl"))
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(model_path, "models_t5_umt5-xxl-enc-bf16.pth"),
        additional_kwargs=DEFAULT_TEXT_ENCODER_KWARGS,
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
    ).eval()
    return tokenizer, text_encoder.to(device=device)


@torch.no_grad()
def encode_video(vae, video_path: str, device: torch.device, align_width_to_32: bool) -> torch.Tensor:
    vr = VideoReader(video_path)
    video = vr.get_batch(list(range(len(vr)))).float() / 255.0
    video = video * 2.0 - 1.0

    if align_width_to_32:
        frames, height, width, channels = video.shape
        aligned_width = round(width / 32) * 32
        if aligned_width != width:
            video = video.permute(0, 3, 1, 2)
            video = F.interpolate(video, size=(height, aligned_width), mode="bilinear", align_corners=False)
            video = video.permute(0, 2, 3, 1)

    video = rearrange(video, "f h w c -> c f h w").unsqueeze(0)
    video = video.to(device=device, dtype=vae.model.decoder.conv1.weight.dtype)
    return vae.encode(video).latent_dist.sample().squeeze(0).cpu()


@torch.no_grad()
def encode_fun_inp_i2v_video(vae, video_path: str, device: torch.device, align_width_to_32: bool) -> torch.Tensor:
    vr = VideoReader(video_path)
    video = vr.get_batch(list(range(len(vr)))).float() / 255.0
    video = video * 2.0 - 1.0

    if align_width_to_32:
        frames, height, width, channels = video.shape
        aligned_width = round(width / 32) * 32
        if aligned_width != width:
            video = video.permute(0, 3, 1, 2)
            video = F.interpolate(video, size=(height, aligned_width), mode="bilinear", align_corners=False)
            video = video.permute(0, 2, 3, 1)

    conditioned_video = torch.zeros_like(video)
    conditioned_video[0] = video[0]
    conditioned_video = rearrange(conditioned_video, "f h w c -> c f h w").unsqueeze(0)
    conditioned_video = conditioned_video.to(device=device, dtype=vae.model.decoder.conv1.weight.dtype)
    return vae.encode(conditioned_video).latent_dist.sample().squeeze(0).cpu()


@torch.no_grad()
def encode_prompt(tokenizer, text_encoder, prompt: str, device: torch.device, max_length: int, pad_to_max_length: bool) -> torch.Tensor:
    inputs = tokenizer(prompt, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    embeds = text_encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
    if not pad_to_max_length:
        embeds = embeds[:, : int(attention_mask.sum().item()), :]
    return embeds.squeeze(0).cpu()


def build_modality_mapping(prompt_subdir: str, wan_version: str) -> Dict[str, Tuple[str, str]]:
    suffix = "_wan" if wan_version == "2.1" else "_wan_2_2"
    normalized_prompt_subdir = normalize_prompt_subdir(prompt_subdir)
    return {
        "videos": ("videos", f"video_latents{suffix}"),
        "static_videos": ("videos_static", f"static_video_latents{suffix}"),
        "hand_videos": ("videos_hands", f"hand_video_latents{suffix}"),
        "prompts": (f"prompts_{normalized_prompt_subdir}", f"prompt_embeds_{normalized_prompt_subdir}{suffix}"),
    }


def safe_read_first_line(txt_path: Path) -> Optional[str]:
    if not txt_path.exists():
        return None
    try:
        with txt_path.open("r", encoding="utf-8") as handle:
            return handle.readline().strip()
    except Exception:
        return None


def atomic_torch_save(tensor: torch.Tensor, out_path: Path) -> None:
    tmp_path = out_path.with_name(f".{out_path.name}.{os.getpid()}.tmp")
    try:
        with tmp_path.open("wb") as handle:
            torch.save(tensor, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, out_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def has_valid_output(out_path: Path) -> bool:
    if not out_path.exists():
        return False
    try:
        if out_path.stat().st_size == 0:
            out_path.unlink()
            return False
    except OSError:
        return False
    return True


def dataset_entry_to_clip_key(entry: str) -> Optional[Tuple[str, ...]]:
    parts = Path(entry).parts
    if "videos" not in parts:
        return None
    video_index = len(parts) - 1 - parts[::-1].index("videos")
    if video_index + 1 >= len(parts):
        return None
    return tuple(parts[:video_index] + (Path(parts[video_index + 1]).stem,))


def load_selected_clip_keys(dataset_files: Sequence[str]) -> List[Tuple[str, ...]]:
    selected: List[Tuple[str, ...]] = []
    for dataset_file in dataset_files:
        file_path = Path(dataset_file)
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
        for raw_line in file_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            clip_key = dataset_entry_to_clip_key(line)
            if clip_key is not None:
                selected.append(clip_key)
    return selected


def clip_key_selected(
    clip_key: Tuple[str, ...],
    selected_clip_keys: Optional[Sequence[Tuple[str, ...]]],
) -> bool:
    if not selected_clip_keys:
        return True
    for selected_key in selected_clip_keys:
        if len(selected_key) >= len(clip_key) and tuple(selected_key[-len(clip_key) :]) == clip_key:
            return True
    return False


def build_jobs(
    data_root: Path,
    dataset_type: str,
    modalities: List[str],
    prompt_subdir: str,
    wan_version: str,
    selected_clip_keys: Optional[Sequence[Tuple[str, ...]]] = None,
) -> Dict[str, List[Dict[str, str]]]:
    jobs = {modality: [] for modality in modalities}
    modality_mapping = build_modality_mapping(prompt_subdir=prompt_subdir, wan_version=wan_version)
    suffix = "_wan" if wan_version == "2.1" else "_wan_2_2"

    if dataset_type == "trumans":
        iterator = (action_dir for _, action_dir in iter_trumans_action_dirs(data_root))
    elif dataset_type == "taste_rob":
        iterator = (sample_dir for _, sample_dir in iter_taste_rob_sample_dirs(data_root))
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    for sample_root in iterator:
        rel_parts = sample_root.relative_to(data_root).parts
        for modality in modalities:
            src_dir_name, out_dir_name = modality_mapping[modality]
            src_dir = sample_root / src_dir_name
            out_dir = sample_root / out_dir_name
            if not src_dir.exists():
                continue
            pattern = "*.txt" if modality == "prompts" else "*.mp4"
            for src_file in sorted(src_dir.glob(pattern)):
                clip_key = tuple(rel_parts + (src_file.stem,))
                if not clip_key_selected(clip_key, selected_clip_keys):
                    continue
                job = {"src": str(src_file), "out": str(out_dir / f"{src_file.stem}.pt")}
                if modality == "videos":
                    job["fun_inp_out"] = str(sample_root / f"fun_inp_i2v_latents{suffix}" / f"{src_file.stem}.pt")
                jobs[modality].append(job)
    return jobs


def shard_jobs(jobs: List[Dict[str, str]], rank: int, world_size: int) -> List[Dict[str, str]]:
    return jobs[rank::world_size]


def process_modality(modality: str, jobs: List[Dict[str, str]], vae, tokenizer, text_encoder, device: torch.device, args, rank: int, world_size: int) -> None:
    my_jobs = shard_jobs(jobs, rank, world_size)
    if args.debug:
        my_jobs = my_jobs[: args.debug_limit]
    if not my_jobs:
        print(f"[rank {rank}/{world_size}] No {modality} jobs assigned")
        return

    errors = []
    progress = tqdm(my_jobs, desc=f"[rank {rank}/{world_size}] {modality}", dynamic_ncols=True)
    for job in progress:
        src_path = Path(job["src"])
        out_path = Path(job["out"])
        fun_inp_out_path = Path(job["fun_inp_out"]) if "fun_inp_out" in job else None

        out_path.parent.mkdir(parents=True, exist_ok=True)
        if fun_inp_out_path is not None:
            fun_inp_out_path.parent.mkdir(parents=True, exist_ok=True)

        out_exists = has_valid_output(out_path)
        fun_inp_exists = fun_inp_out_path is None or has_valid_output(fun_inp_out_path)
        if args.skip_existing and out_exists and fun_inp_exists:
            continue

        try:
            if modality == "prompts":
                prompt = safe_read_first_line(src_path)
                if prompt is None:
                    raise RuntimeError("cannot_read_prompt")
                embeds = encode_prompt(
                    tokenizer,
                    text_encoder,
                    prompt,
                    device,
                    args.max_sequence_length,
                    args.pad_to_max_length,
                )
                atomic_torch_save(embeds, out_path)
            else:
                align_width_to_32 = args.wan_version == "2.2"
                if not (args.skip_existing and out_exists):
                    latents = encode_video(vae, str(src_path), device, align_width_to_32)
                    atomic_torch_save(latents, out_path)
                if modality == "videos" and fun_inp_out_path is not None and not (args.skip_existing and fun_inp_exists):
                    fun_inp_i2v_latents = encode_fun_inp_i2v_video(vae, str(src_path), device, align_width_to_32)
                    atomic_torch_save(fun_inp_i2v_latents, fun_inp_out_path)
        except Exception as exc:
            errors.append({"src": str(src_path), "reason": str(exc)})

    if errors:
        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        error_path = log_dir / f"errors_{modality}_rank{rank}.jsonl"
        with error_path.open("w", encoding="utf-8") as handle:
            for err in errors:
                handle.write(json.dumps(err) + "\n")
        print(f"[rank {rank}] Error log: {error_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Encode videos/prompts with WAN VAE/T5")
    parser.add_argument("--data_root", type=str, required=True, help="Canonical dataset root, e.g. data/trumans or data/taste_rob")
    parser.add_argument("--dataset_type", type=str, required=True, choices=["trumans", "taste_rob"])
    parser.add_argument("--modalities", type=str, nargs="+", required=True, choices=["videos", "static_videos", "hand_videos", "prompts"])
    parser.add_argument("--prompt_subdir", type=str, default="prompts_rewrite", help="Prompt directory name or suffix")
    parser.add_argument("--dataset_files", type=str, nargs="*", default=None, help="Optional dataset split files containing videos/*.mp4 entries")
    parser.add_argument("--model_path", type=str, default=None, help="Path to WAN model directory")
    parser.add_argument("--wan_version", type=str, default="2.1", choices=["2.1", "2.2"])
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--pad_to_max_length", action="store_true")
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--world_size", type=int, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--log_dir", type=str, default="logs/encode_wan")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_limit", type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.prompt_subdir = normalize_prompt_subdir(args.prompt_subdir)

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    if args.model_path is None:
        args.model_path = DEFAULT_MODEL_PATH_2_2 if args.wan_version == "2.2" else DEFAULT_MODEL_PATH_2_1

    if args.rank is not None and args.world_size is not None:
        rank, world_size = args.rank, args.world_size
    elif "SLURM_ARRAY_TASK_ID" in os.environ:
        rank = int(os.environ["SLURM_ARRAY_TASK_ID"])
        world_size = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    elif "LOCAL_RANK" in os.environ:
        rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ.get("WORLD_SIZE", 1))
    else:
        rank, world_size = 0, 1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    selected_clip_keys = load_selected_clip_keys(args.dataset_files) if args.dataset_files else None
    all_jobs = build_jobs(
        data_root=data_root,
        dataset_type=args.dataset_type,
        modalities=args.modalities,
        prompt_subdir=args.prompt_subdir,
        wan_version=args.wan_version,
        selected_clip_keys=selected_clip_keys,
    )

    dtype = DTYPE_MAPPING[args.dtype]
    vae = None
    tokenizer = None
    text_encoder = None

    if VIDEO_MODALITIES & set(args.modalities):
        vae = load_wan_vae(args.model_path, device, dtype, args.wan_version)
    if "prompts" in args.modalities:
        tokenizer, text_encoder = load_wan_text_encoder(args.model_path, device, dtype)

    for modality in args.modalities:
        process_modality(modality, all_jobs.get(modality, []), vae, tokenizer, text_encoder, device, args, rank, world_size)


if __name__ == "__main__":
    main()
