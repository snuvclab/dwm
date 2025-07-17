# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import ast
import base64
import copy
import glob
import io
import json
import logging
import math
import os
import pathlib
import pickle
import random
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import soundfile as sf
import tokenizers
import torch
import transformers
import whisper
from egogpt import conversation as conversation_lib
from egogpt.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_SPEECH_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    SPEECH_TOKEN_INDEX,
)
from egogpt.mm_utils import (
    process_anyres_image,
    process_highres_image,
    process_highres_image_crop_split,
)
from egogpt.model import *
from egogpt.train.llava_trainer import LLaVATrainer
from egogpt.utils import process_video_with_decord, process_video_with_decord_byframe
from packaging import version
from PIL import Image
from safetensors.torch import load_file as safetensor_load_file
from scipy.signal import resample
from torch.utils.data import Dataset

local_rank = None
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse(
    "0.14"
)


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    tune_speech_generator_only: bool = field(default=False)
    speech_encoder: Optional[str] = field(default=None)
    unfreeze_mm_speech_encoder: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(
        default=-1
    )  # default to the last layer
    pretrain_speech_projector: Optional[str] = field(default=None)
    speech_projector_type: Optional[str] = field(default="linear")
    speech_encoder_type: Optional[str] = field(default="whisper")
    speech_encoder_config: Optional[str] = field(
        default="models/speech_encoder/large-v3.pt"
    )
    speech_encoder_ds_rate: Optional[int] = field(default=5)
    speech_encoder_hidden_size: Optional[int] = field(default=1280)
    tune_mm_mlp_adapter: bool = field(default=False)
    tune_mm_vision_resampler: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    unfreeze_mm_vision_tower: bool = field(default=False)
    unfreeze_language_model: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(
        default=-1
    )  # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default="linear")
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default="flat")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    mm_resampler_type: Optional[str] = field(default=None)
    mm_mask_drop_mode: str = field(default="fixed")
    mm_mask_drop_skip_percentage: float = field(default=0.0)
    mm_mask_drop_ratio: float = field(default=0.25)
    mm_mask_drop_ratio_upper: Optional[float] = field(default=None)
    mm_mask_drop_ratio_lower: Optional[float] = field(default=None)
    mm_spatial_pool_stride: Optional[int] = field(default=None)
    mm_spatial_pool_mode: str = field(default="bilinear")
    mm_spatial_pool_out_channels: Optional[int] = field(default=None)
    mm_perceiver_depth: Optional[int] = field(default=3)
    mm_perceiver_latents: Optional[int] = field(default=32)
    mm_perceiver_ff_mult: Optional[float] = field(default=4)
    mm_perceiver_pretrained: Optional[str] = field(default=None)
    mm_qformer_depth: Optional[int] = field(default=3)
    mm_qformer_latents: Optional[int] = field(default=32)
    mm_qformer_pretrained: Optional[str] = field(default=None)
    rope_scaling_factor: Optional[float] = field(default=None)
    rope_scaling_type: Optional[str] = field(default=None)

    s2: Optional[bool] = field(default=False)
    s2_scales: Optional[str] = field(default="336,672,1008")

    use_pos_skipping: Optional[bool] = field(default=False)
    pos_skipping_range: Optional[int] = field(default=4096)

    mm_newline_position: Optional[str] = field(default="grid")
    delay_load: Optional[bool] = field(default=True)
    delay_load_audio: Optional[bool] = field(default=True)
    add_faster_video: Optional[bool] = field(default=False)
    faster_token_stride: Optional[int] = field(default=10)


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_aspect_ratio: str = "square"
    image_grid_pinpoints: Optional[str] = field(default=None)
    image_crop_resolution: Optional[int] = field(default=None)
    image_split_resolution: Optional[int] = field(default=None)
    video_fps: Optional[int] = field(default=1)
    frames_upbound: Optional[int] = field(default=100)
    force_sample: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={
            "help": "Compress the quantization statistics through double quantization."
        },
    )
    quant_type: str = field(
        default="nf4",
        metadata={
            "help": "Quantization data type to use. Should be one of `fp4` or `nf4`."
        },
    )
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    lora_enable: bool = field(default=False)
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    speech_projector_lr: Optional[float] = None
    gradient_checkpointing: bool = field(default=True)
    mm_speech_encoder_lr: Optional[float] = None
    diffusion_head_lr: Optional[float] = None
    group_by_varlen: bool = field(default=False)
    group_by_modality_length: bool = field(default=False)
    group_by_modality_length_auto: bool = field(default=False)
    min_lr_ratio: float = field(default=0.0)
    sample_independently: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mm_projector_lr: Optional[float] = None
    mm_vision_tower_lr: Optional[float] = None
    freeze_mm_vision_resampler: bool = field(default=False)


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(
                    f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}"
                )
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {
        k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()
    }
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {
        k: t
        for k, t in named_params
        if any(key_match in k for key_match in keys_to_match)
    }
    to_return = {
        k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()
    }
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ["speech_projector", "speech_encoder"]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ["speech_projector"]
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(["embed_tokens", "embed_in"])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(
            trainer.model.named_parameters(), keys_to_match
        )
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split("/")[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith("checkpoint-"):
                speech_projector_folder = os.path.join(
                    parent_folder, "speech_projector"
                )
                os.makedirs(speech_projector_folder, exist_ok=True)
                torch.save(
                    weight_to_save,
                    os.path.join(speech_projector_folder, f"{current_folder}.bin"),
                )
            else:
                torch.save(
                    weight_to_save, os.path.join(output_dir, f"speech_projector.bin")
                )
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(
    strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx + 2 : cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = "unknown"
        sentence["value"] = (
            BEGIN_SIGNAL + from_str + ": " + sentence["value"] + END_SIGNAL
        )
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def tokenizer_speech_token(
    prompt, tokenizer, speech_token_index=SPEECH_TOKEN_INDEX, return_tensors=None
):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<speech>")]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if (
        len(prompt_chunks) > 0
        and len(prompt_chunks[0]) > 0
        and prompt_chunks[0][0] == tokenizer.bos_token_id
    ):
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [speech_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    return input_ids


def preprocess_multimodal(sources: Sequence[str], data_args: DataArguments) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources
    # Add speech and image special tokens to the beginning of the conversation
    for source in sources:
        for sentence in source:
            if DEFAULT_SPEECH_TOKEN in sentence["value"]:
                sentence["value"] = (
                    sentence["value"].replace(DEFAULT_SPEECH_TOKEN, "").strip()
                )
                sentence["value"] = DEFAULT_SPEECH_TOKEN + "\n" + sentence["value"]
                sentence["value"] = sentence["value"].strip()
            if DEFAULT_IMAGE_TOKEN in sentence["value"]:
                sentence["value"] = (
                    sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
                )
                sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
                sentence["value"] = sentence["value"].strip()
    return sources


def preprocess_llama_2(
    sources, tokenizer: transformers.PreTrainedTokenizer, has_speech: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_speech:
        input_ids = torch.stack(
            [
                tokenizer_speech_token(prompt, tokenizer, return_tensors="pt")
                for prompt in conversations
            ],
            dim=0,
        )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_speech:
                round_len = len(tokenizer_speech_token(rou, tokenizer))
                instruction_len = len(tokenizer_speech_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_llama_3(
    sources, tokenizer: transformers.PreTrainedTokenizer, has_speech: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        assert len(source) == 2, "now only support single-turn conversation"

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_speech:
        input_ids = torch.stack(
            [
                tokenizer_speech_token(prompt, tokenizer, return_tensors="pt")
                for prompt in conversations
            ],
            dim=0,
        )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_3

    # Mask targets
    sep = "<|start_header_id|>" + conv.roles[1] + "<|end_header_id|>\n\n"
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        parts = conversation.split(sep)
        parts[0] += sep

        if has_speech:
            conversation_len = len(tokenizer_speech_token(conversation, tokenizer))
            instruction_len = len(tokenizer_speech_token(parts[0], tokenizer)) - 1
        else:
            conversation_len = len(tokenizer(conversation).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 1

        target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
        cur_len += conversation_len
        target[cur_len:] = IGNORE_INDEX

        # if cur_len < tokenizer.model_max_length:
        #     if cur_len != total_len:
        #         target[:] = IGNORE_INDEX
        #         print(
        #             f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
        #             f" (ignored)"
        #         )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources, tokenizer: transformers.PreTrainedTokenizer, has_speech: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_speech:
        input_ids = torch.stack(
            [
                tokenizer_speech_token(prompt, tokenizer, return_tensors="pt")
                for prompt in conversations
            ],
            dim=0,
        )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    if conv.sep_style == conversation_lib.SeparatorStyle.TWO:
        # Mask targets
        sep = conv.sep + conv.roles[1] + ": "
        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())

            rounds = conversation.split(conv.sep2)
            cur_len = 1
            target[:cur_len] = IGNORE_INDEX
            for i, rou in enumerate(rounds):
                if rou == "":
                    break

                parts = rou.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep

                if has_speech:
                    round_len = len(tokenizer_speech_token(rou, tokenizer))
                    instruction_len = (
                        len(tokenizer_speech_token(parts[0], tokenizer)) - 2
                    )
                else:
                    round_len = len(tokenizer(rou).input_ids)
                    instruction_len = len(tokenizer(parts[0]).input_ids) - 2

                if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                    round_len -= 1
                    instruction_len -= 1

                target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

                cur_len += round_len
            target[cur_len:] = IGNORE_INDEX

            if cur_len < tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_INDEX
                    print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                        f" (ignored)"
                    )

    elif conv.sep_style == conversation_lib.SeparatorStyle.QWEN2:
        # Mask targets
        sep = "<|im_start|>assistant\n"
        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())

            raw_rounds = conversation.split("<|im_end|>\n")
            cur_len = 0
            rounds = []
            now_str = ""
            for rou in raw_rounds:
                if len(rou) > 0:
                    rou = rou + "<|im_end|>\n"
                    if rou.startswith("<|endoftext|>"):
                        rounds[-1] = rounds[-1] + "<|endoftext|>"
                        rou = rou.replace("<|endoftext|>", "")
                        if len(rou.strip()) == 0:
                            continue
                    if "<|im_start|>assistant\n" in rou:
                        now_str += rou
                        rounds.append(now_str)
                        now_str = ""
                    else:
                        now_str += rou

            for i, rou in enumerate(rounds):
                if rou == "":
                    break

                parts = rou.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep

                if has_speech:
                    round_len = len(tokenizer_speech_token(rou, tokenizer))
                    instruction_len = (
                        len(tokenizer_speech_token(parts[0], tokenizer)) - 2
                    )
                else:
                    round_len = len(tokenizer(rou).input_ids)
                    instruction_len = len(tokenizer(parts[0]).input_ids) - 2

                try:
                    is_legacy = tokenizer.legacy
                except:
                    is_legacy = True

                if i != 0 and not is_legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                    round_len -= 1
                    instruction_len -= 1

                target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

                cur_len += round_len
            target[cur_len:] = IGNORE_INDEX

            if cur_len < tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_INDEX
                    print(
                        f"WARNING: tokenization mismatch for QWEN2: {cur_len} vs. {total_len}."
                        f" (ignored)"
                    )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_SPEECH_TOKEN in source[0]["value"]
        source[0]["value"] = DEFAULT_SPEECH_TOKEN
        conversation = (
            source[0]["value"]
            + source[1]["value"]
            + conversation_lib.default_conversation.sep
        )
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [
        tokenizer_speech_token(prompt, tokenizer, return_tensors="pt")
        for prompt in conversations
    ]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_speech_token(source[0]["value"], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess_qwen(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_speech: bool = False,
    has_image: bool = False,
    max_len=2048,
    system_message: str = "You are a helpful assistant.",
) -> Dict:
    def split_text(text, keywords):
        pattern = "(" + "|".join(map(re.escape, keywords)) + ")"
        parts = re.split(pattern, text)
        parts = [part for part in parts if part]
        return parts

    roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}

    # im_start, im_end = tokenizer.additional_special_tokens_ids

    im_start = tokenizer("<|im_start|>").input_ids[0]
    im_end = tokenizer("<|im_end|>").input_ids[0]
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []
        system = (
            [im_start]
            + _system
            + tokenizer(system_message).input_ids
            + [im_end]
            + nl_tokens
        )
        input_id += system
        target += [im_start] + [IGNORE_INDEX] * (len(system) - 3) + [im_end] + nl_tokens
        assert len(input_id) == len(target)
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            splited_sentence = split_text(sentence["value"], ["<speech>", "<image>"])
            _input_id = []
            for part in splited_sentence:
                _input_id += tokenizer(role).input_ids + nl_tokens  # add prefix
                if "<speech>" == part:
                    _input_id += [SPEECH_TOKEN_INDEX]
                elif "<image>" == part:
                    _input_id += [IMAGE_TOKEN_INDEX]
                else:
                    _input_id += tokenizer(part).input_ids
            _input_id += [im_end] + nl_tokens  # add suffix
            input_id += _input_id
            if role == "<|im_start|>user":
                _target = (
                    [im_start]
                    + [IGNORE_INDEX] * (len(_input_id) - 3)
                    + [im_end]
                    + nl_tokens
                )
            elif role == "<|im_start|>assistant":
                _target = (
                    [im_start]
                    + [IGNORE_INDEX] * len(tokenizer(role).input_ids)
                    + _input_id[len(tokenizer(role).input_ids) + 1 : -2]
                    + [im_end]
                    + nl_tokens
                )
            else:
                raise NotImplementedError
            target += _target
        assert len(input_id) == len(target)
        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_speech: bool = False,
    has_image: bool = False,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if (
        conversation_lib.default_conversation.sep_style
        == conversation_lib.SeparatorStyle.PLAIN
    ):
        return preprocess_plain(sources, tokenizer, has_image=has_image)
    if (
        conversation_lib.default_conversation.sep_style
        == conversation_lib.SeparatorStyle.LLAMA_2
    ):
        return preprocess_llama_2(
            sources, tokenizer, has_speech=has_speech, has_image=has_image
        )
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(
            sources, tokenizer, has_speech=has_speech, has_image=has_image
        )
    if (
        conversation_lib.default_conversation.sep_style
        == conversation_lib.SeparatorStyle.LLAMA_3
    ):
        return preprocess_llama_3(
            sources, tokenizer, has_speech=has_speech, has_image=has_image
        )
    if conversation_lib.default_conversation.version == "qwen":
        return preprocess_qwen(
            sources, tokenizer, has_speech=has_speech, has_image=has_image
        )
    raise NotImplementedError


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
    ):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.mel_size = 128

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            assert cur_len > 0, f"Conversation length is 0 for {sample}"
            if "image" in sample or "video" in sample or self.data_args.early_mix_text:
                length_list.append(cur_len)
            else:
                length_list.append(-cur_len)
        return length_list

    def process_audio(self, audio_file, start_frame=None, end_frame=None, fps=20):
        speech, sample_rate = sf.read(audio_file)
        if start_frame is not None and end_frame is not None:
            start_sample = start_frame * sample_rate // fps
            end_sample = end_frame * sample_rate // fps
            speech = speech[start_sample:end_sample]
        if sample_rate != 16000:
            target_length = int(len(speech) * 16000 / sample_rate)
            speech = resample(speech, target_length)
        if speech.ndim > 1:
            speech = np.mean(speech, axis=1)
        speech = whisper.pad_or_trim(speech.astype(np.float32))
        speech = whisper.log_mel_spectrogram(speech, n_mels=self.mel_size).permute(1, 0)
        speech_length = torch.LongTensor([speech.shape[0]])
        return speech, speech_length

    def process_image(self, image_file, overwrite_image_aspect_ratio=None):
        processor = self.data_args.image_processor
        # print(f"\n\nInspecting the image path, folder = {image_folder}, image={image_file}\n\n")
        try:
            image = Image.open(image_file).convert("RGB")
        except Exception as exn:
            print(f"Failed to open image {image_file}. Exception:", exn)
            raise exn

        image_size = image.size
        image_aspect_ratio = self.data_args.image_aspect_ratio
        if overwrite_image_aspect_ratio is not None:
            image_aspect_ratio = overwrite_image_aspect_ratio
        if image_aspect_ratio == "highres":
            image = process_highres_image(
                image,
                self.data_args.image_processor,
                self.data_args.image_grid_pinpoints,
            )
        elif image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
            image = process_anyres_image(
                image,
                self.data_args.image_processor,
                self.data_args.image_grid_pinpoints,
            )
        elif image_aspect_ratio == "crop_split":
            image = process_highres_image_crop_split(image, self.data_args)
        elif image_aspect_ratio == "pad":

            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result

            image = expand2square(
                image, tuple(int(x * 255) for x in processor.image_mean)
            )
            image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        else:
            image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        return image, image_size, "image"

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        while True:
            try:
                sample = self._get_item(i)
                # print("process sample",i)
                break
            except Exception as e:
                while True:
                    try:
                        i += 1
                        random_index = i % len(self.list_data_dict)
                        sample = self._get_item(random_index)
                        # print("something error, process sample",random_index)
                        break
                    except Exception as e:
                        # random_index = random.randint(0, len(self.list_data_dict) - 1)
                        continue
        return sample

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if "image" in sources[0]:
            image_file = self.list_data_dict[i]["image"]
            if type(image_file) is list:
                image = [self.process_image(f) for f in image_file]
                # Handling multi images
                # overwrite to process with simple pad
                if len(image_file) > 1:
                    image = [self.process_image(f, "pad") for f in image_file]
                    image = [[im[0], im[1], "image"] for im in image]
            else:
                image = [self.process_image(image_file)]

        if "video" or "audio" in sources[0]:
            if "video" in sources[0]:
                video_file = self.list_data_dict[i]["video"]
                # video_folder = self.data_args.video_folder
                # video_file = os.path.join(video_folder, video_file)
                if not os.path.exists(video_file):
                    print("File {} not exist!".format(video_file))
                if "start_frame" in self.list_data_dict[i]:
                    start_frame = self.list_data_dict[i]["start_frame"]
                    end_frame = self.list_data_dict[i]["end_frame"]
                    if self.list_data_dict[i].get(
                        "current_observation_frame", None
                    ):  # Customized for egoplan data
                        current_observation_frame = self.list_data_dict[i][
                            "current_observation_frame"
                        ]
                    else:
                        current_observation_frame = None
                    video = process_video_with_decord_byframe(
                        video_file,
                        start_frame,
                        end_frame,
                        self.data_args,
                        current_observation_frame,
                    )
                else:
                    (
                        video,
                        video_time,
                        frame_time,
                        num_frames,
                    ) = process_video_with_decord(video_file, self.data_args)
                processor = self.data_args.image_processor
                processed_video = processor.preprocess(video, return_tensors="pt")[
                    "pixel_values"
                ]
                image = [(processed_video, video[0].size, "video")]

            if "audio" in sources[0]:
                audio_file = self.list_data_dict[i]["audio"]
                # audio_folder = self.data_args.audio_folder
                # audio_file = os.path.join(audio_folder, audio_file)
                try:
                    if "start_frame" in self.list_data_dict[i]:
                        start_frame = self.list_data_dict[i]["start_frame"]
                        end_frame = self.list_data_dict[i]["end_frame"]
                    else:
                        start_frame = None
                        end_frame = None
                    audio, audio_length = self.process_audio(
                        audio_file, start_frame, end_frame
                    )
                except Exception as e:
                    print("audio error", e)
                    audio = [torch.zeros(3000, 128)]
                    audio_length = torch.tensor([3000])
                audio = [audio]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]), self.data_args
            )
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        has_speech = "audio" in self.list_data_dict[i]
        has_image = ("image" in self.list_data_dict[i]) or (
            "video" in self.list_data_dict[i]
        )
        data_dict = preprocess(
            sources, self.tokenizer, has_speech=has_speech, has_image=has_image
        )
        if isinstance(i, int):
            data_dict = dict(
                input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0]
            )

        if "image" or "video" in self.list_data_dict[i]:
            data_dict["image"] = image
        # audio exist in the data
        if "audio" in self.list_data_dict[i]:
            data_dict["speech"] = audio
            data_dict["speech_lengths"] = audio_length
        else:  # if no audio, add a dummy audio
            data_dict["speech"] = [torch.zeros(3000, 128)]
            data_dict["speech_lengths"] = torch.tensor([3000])
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=batch_first, padding_value=padding_value
        )
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = [
            _input_ids[: self.tokenizer.model_max_length] for _input_ids in input_ids
        ]
        labels = [_labels[: self.tokenizer.model_max_length] for _labels in labels]
        if self.tokenizer.pad_token_id is None:
            if "qwen" in self.tokenizer.name_or_path.lower():
                # print("Setting pad token to bos token for qwen model.")
                self.tokenizer.pad_token_id = 151643
            else:
                self.tokenizer.pad_token_id = (
                    self.tokenizer.eos_token_id
                )  # FIXME: this could only be triggered for llama3 model.
        input_ids = self.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = self.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        if "speech" in instances[0]:
            speeches = [instance["speech"] for instance in instances]
            speeches_lengths = [instance["speech_lengths"] for instance in instances]
            batch["speech"] = [au for audio_list in speeches for au in audio_list]

            batch["speech_lengths"] = [
                au for audio_list in speeches_lengths for au in audio_list
            ]
            batch["speech_lengths"] = torch.stack(batch["speech_lengths"])

            if all(
                x is not None and x.shape == speeches[0][0].shape
                for x in batch["speech"]
            ):
                batch["speech"] = torch.stack(batch["speech"])

        if "image" in instances[0]:
            images = [instance["image"] for instance in instances]

            batch["image_sizes"] = [im[1] for im_list in images for im in im_list]
            batch["modalities"] = [im[2] for im_list in images for im in im_list]
            images = [im[0] for im_list in images for im in im_list]

            # if all(x is not None and x.shape == images[0].shape for x in images):
            # Image: (N, P, C, H, W)
            # Video: (N, F, C, H, W)
            #     batch["images"] = torch.stack(images)
            # else:
            batch["images"] = images
        return batch


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(
        tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    if "qwen" in model_args.model_name_or_path.lower():
        model = EgoGPTQwenForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation="flash_attention_2",
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
    else:
        model = EgoGPTLlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation="flash_attention_2",
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )

    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}
        print("gradient_checkpointing", training_args.gradient_checkpointing)
        training_args.ddp_find_unused_parameters = True

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
            use_dora=True,
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)
        model.to(dtype=compute_dtype, device=training_args.device)

    if "qwen" in model_args.model_name_or_path.lower():
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[
                model_args.version
            ]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates[
                "vicuna_v1"
            ]

    model.get_model().initialize_speech_modules(
        model_args=model_args, fsdp=training_args.fsdp
    )

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args, fsdp=training_args.fsdp
        )
        # import pdb;pdb.set_trace()
        vision_tower = model.get_vision_tower()
        vision_tower.to(
            dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
            device=training_args.device,
        )
        speech_encoder = model.get_speech_encoder()
        speech_encoder.to(
            dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
            device=training_args.device,
        )

        data_args.image_processor = vision_tower.image_processor
        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        if data_args.image_grid_pinpoints is not None:
            if (
                isinstance(data_args.image_grid_pinpoints, str)
                and "x" in data_args.image_grid_pinpoints
            ):
                try:
                    patch_size = data_args.image_processor.size[0]
                except Exception as e:
                    patch_size = data_args.image_processor.size["shortest_edge"]

                assert patch_size in [
                    224,
                    336,
                    384,
                    448,
                    512,
                ], "patch_size should be in [224, 336, 384, 448, 512]"
                # Use regex to extract the range from the input string
                matches = re.findall(r"\((\d+)x(\d+)\)", data_args.image_grid_pinpoints)
                range_start = tuple(map(int, matches[0]))
                range_end = tuple(map(int, matches[-1]))
                # Generate a matrix of tuples from (range_start[0], range_start[1]) to (range_end[0], range_end[1])
                grid_pinpoints = [
                    (i, j)
                    for i in range(range_start[0], range_end[0] + 1)
                    for j in range(range_start[1], range_end[1] + 1)
                ]
                # Multiply all elements by patch_size
                data_args.image_grid_pinpoints = [
                    [dim * patch_size for dim in pair] for pair in grid_pinpoints
                ]
            elif isinstance(data_args.image_grid_pinpoints, str):
                data_args.image_grid_pinpoints = ast.literal_eval(
                    data_args.image_grid_pinpoints
                )

        model.config.image_grid_pinpoints = data_args.image_grid_pinpoints
        model.config.image_crop_resolution = data_args.image_crop_resolution
        model.config.image_split_resolution = data_args.image_split_resolution
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length
        model.config.mm_newline_position = model_args.mm_newline_position
        model.config.add_faster_video = model_args.add_faster_video
        model.config.faster_token_stride = model_args.faster_token_stride
        model.config.mm_spatial_pool_stride = model_args.mm_spatial_pool_stride

    data_args.is_multimodal = True

    model.config.tune_mm_mlp_adapter = (
        training_args.tune_mm_mlp_adapter
    ) = model_args.tune_mm_mlp_adapter
    if model_args.tune_mm_mlp_adapter:
        model.requires_grad_(False)
    if model_args.tune_mm_mlp_adapter:
        for p in model.get_model().speech_projector.parameters():
            p.requires_grad = True
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True

    model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
    if training_args.freeze_mm_mlp_adapter:
        for p in model.get_model().speech_projector.parameters():
            p.requires_grad = False
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = False

    model.config.freeze_mm_vision_resampler = training_args.freeze_mm_vision_resampler
    if training_args.freeze_mm_vision_resampler:
        for p in model.get_model().vision_resampler.parameters():
            p.requires_grad = False
    model.config.unfreeze_mm_speech_encoder = model_args.unfreeze_mm_speech_encoder
    if model_args.unfreeze_mm_speech_encoder:
        speech_encoder.requires_grad_(True)

    model.config.mm_use_im_start_end = (
        data_args.mm_use_im_start_end
    ) = model_args.mm_use_im_start_end
    model.config.mm_projector_lr = training_args.mm_projector_lr
    model.config.mm_vision_tower_lr = training_args.mm_vision_tower_lr
    model.config.speech_projector_lr = training_args.speech_projector_lr
    model.config.mm_speech_encoder_lr = training_args.mm_speech_encoder_lr
    model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
    training_args.use_im_start_end = model_args.mm_use_im_start_end

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    # test_data = data_module['train_dataset'].__getitem__(0)
    trainer = LLaVATrainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(
                non_lora_state_dict,
                os.path.join(training_args.output_dir, "non_lora_trainables.bin"),
            )
    else:
        safe_save_model_for_hf_trainer(
            trainer=trainer, output_dir=training_args.output_dir
        )


if __name__ == "__main__":
    import torch

    print("number of gpus", torch.cuda.device_count())
    train()
