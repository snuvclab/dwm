import argparse
import functools
import json
import os
import pathlib
import queue
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Union
import random

import torch
import torch.distributed as dist
from diffusers import AutoencoderKLCogVideoX
from diffusers.training_utils import set_seed
from diffusers.utils import export_to_video, get_logger
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer


import decord  # isort:skip

from dataset import BucketSampler, VideoDatasetWithResizing, VideoDatasetWithResizeAndRectangleCrop  # isort:skip


decord.bridge.set_bridge("torch")

logger = get_logger(__name__)

DTYPE_MAPPING = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

def _get_t5_prompt_embeds(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: Union[str, List[str]],
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("`text_input_ids` must be provided when the tokenizer is not specified.")

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

    return prompt_embeds, text_input_ids


def encode_prompt(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: Union[str, List[str]],
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt_embeds, text_input_ids = _get_t5_prompt_embeds(
        tokenizer,
        text_encoder,
        prompt=prompt,
        num_videos_per_prompt=num_videos_per_prompt,
        max_sequence_length=max_sequence_length,
        device=device,
        dtype=dtype,
        text_input_ids=text_input_ids,
    )
    return prompt_embeds, text_input_ids


def compute_prompt_embeddings(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompts: List[str],
    max_sequence_length: int,
    device: torch.device,
    dtype: torch.dtype,
    requires_grad: bool = False,
):
    if requires_grad:
        prompt_embeds, text_input_ids = encode_prompt(
            tokenizer,
            text_encoder,
            prompts,
            num_videos_per_prompt=1,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )
    else:
        with torch.no_grad():
            prompt_embeds, text_input_ids = encode_prompt(
                tokenizer,
                text_encoder,
                prompts,
                num_videos_per_prompt=1,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
    return prompt_embeds, text_input_ids


def main(args):
    weight_dtype = DTYPE_MAPPING[args.weight_dtype]
    

    tokenizer = T5Tokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(
        args.model_id, subfolder="text_encoder", torch_dtype=weight_dtype
    )
    text_encoder = text_encoder.to(args.device)

    # add tokens for prompt encoding
    # tokenizer.add_tokens(["<TGT>"])
    # tokenizer.add_tokens(["<MSK1>", "<MSK2>"])
    # text_encoder.resize_token_embeddings(len(tokenizer))

    for folder in tqdm(sorted(os.listdir(args.data_root))):

        try:

            output_dir = "data/sequences/processed/prompt_embeds" #+ "_".join(args.extra_tokens.split()).replace("<", "").replace(">", "") + "_random"
            os.makedirs(os.path.join(args.data_root, folder, output_dir), exist_ok=True)

            prompt_txts = sorted(os.listdir(os.path.join(args.data_root, folder, "data/sequences/processed/prompts")))
            for prompt_txt in tqdm(prompt_txts):
                with open(os.path.join(args.data_root, folder, "data/sequences/processed/prompts", prompt_txt), "r") as f:
                    prompt = f.readlines()[0]
                

                # prompts_extra_token = [args.extra_tokens + prompt]

                prompts_embeds_extra_token, text_input_ids = compute_prompt_embeddings(
                    tokenizer,
                    text_encoder,
                    [prompt],
                    args.max_sequence_length,
                    args.device,
                    weight_dtype,
                    requires_grad=False,
                )
                torch.save(prompts_embeds_extra_token[0], os.path.join(args.data_root, folder, output_dir, f"{prompt_txt[:-4]}.pt"))
        except Exception as e:
            print(f"Error processing {folder}: {e}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Prepare prompts with extra tokens")
    parser.add_argument("--data_root", type=str, required=True, help="Data root")
    parser.add_argument("--model_id", type=str, default="THUDM/CogVideoX-5b-I2V", help="Model id")
    parser.add_argument("--max_sequence_length", type=int, default=226, help="Maximum sequence length")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--weight_dtype", type=str, default="fp32", help="Weight dtype")
    args = parser.parse_args()

    main(args)