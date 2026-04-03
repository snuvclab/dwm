"""Checkpoint retention helpers for WAN training scripts."""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Callable

from accelerate import DistributedType
from accelerate.checkpointing import (
    MODEL_NAME,
    OPTIMIZER_NAME,
    RNG_STATE_NAME,
    SAFE_MODEL_NAME,
    SAMPLER_NAME,
    SCALER_NAME,
    SCHEDULER_NAME,
)

ACCELERATE_STATE_GLOB_PATTERNS = (
    f"{MODEL_NAME}*",
    f"{SAFE_MODEL_NAME}*.safetensors",
    f"{OPTIMIZER_NAME}*.bin",
    f"{SCHEDULER_NAME}*.bin",
    f"{SAMPLER_NAME}*.bin",
    "dl_state_dict*.bin",
    f"{RNG_STATE_NAME}_*.pkl",
    "custom_checkpoint_*.pkl",
)
ACCELERATE_STATE_EXACT_FILES = (SCALER_NAME,)
ACCELERATE_STATE_AUXILIARY_FILES = ("latest", "zero_to_fp32.py")


def extract_checkpoint_step(checkpoint_name: str) -> int:
    """Extract the numeric step from checkpoint directory names like checkpoint-1000."""
    match = re.fullmatch(r"checkpoint-(\d+)", checkpoint_name)
    if match is None:
        raise ValueError(f"Unsupported checkpoint directory name: {checkpoint_name}")
    return int(match.group(1))


def list_checkpoint_dirs(output_dir: str) -> list[str]:
    """Return checkpoint directories sorted by training step."""
    if not os.path.exists(output_dir):
        return []

    checkpoints = []
    for entry in os.listdir(output_dir):
        entry_path = os.path.join(output_dir, entry)
        if os.path.isdir(entry_path) and re.fullmatch(r"checkpoint-(\d+)", entry):
            checkpoints.append(entry)

    checkpoints.sort(key=extract_checkpoint_step)
    return checkpoints


def iter_checkpoint_state_paths(checkpoint_dir: str, include_auxiliary: bool = False):
    """Yield accelerator-managed state paths inside a checkpoint directory."""
    checkpoint_path = Path(checkpoint_dir)
    seen = set()

    for pattern in ACCELERATE_STATE_GLOB_PATTERNS:
        for path in checkpoint_path.glob(pattern):
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            yield path

    for filename in ACCELERATE_STATE_EXACT_FILES:
        path = checkpoint_path / filename
        key = str(path)
        if path.exists() and key not in seen:
            seen.add(key)
            yield path

    if include_auxiliary:
        for filename in ACCELERATE_STATE_AUXILIARY_FILES:
            path = checkpoint_path / filename
            key = str(path)
            if path.exists() and key not in seen:
                seen.add(key)
                yield path


def checkpoint_has_full_state(checkpoint_dir: str) -> bool:
    """Return whether the checkpoint still has exact-resume accelerator state."""
    return any(iter_checkpoint_state_paths(checkpoint_dir, include_auxiliary=False))


def prune_full_state_from_checkpoint(checkpoint_dir: str) -> list[str]:
    """Remove accelerator resume artifacts while keeping evaluation weights."""
    removed_entries = []

    for path in iter_checkpoint_state_paths(checkpoint_dir, include_auxiliary=True):
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        removed_entries.append(path.name)

    removed_entries.sort()
    return removed_entries


def prune_old_checkpoint_states(output_dir: str, keep_full_state_checkpoints: int) -> list[tuple[str, list[str]]]:
    """Keep full accelerator state only for the latest checkpoints."""
    if keep_full_state_checkpoints < 0:
        raise ValueError("keep_full_state_checkpoints must be >= 0")

    checkpoints = list_checkpoint_dirs(output_dir)
    if keep_full_state_checkpoints >= len(checkpoints):
        return []

    checkpoints_to_prune = checkpoints if keep_full_state_checkpoints == 0 else checkpoints[:-keep_full_state_checkpoints]
    pruned = []

    for checkpoint_name in checkpoints_to_prune:
        checkpoint_dir = os.path.join(output_dir, checkpoint_name)
        removed_entries = prune_full_state_from_checkpoint(checkpoint_dir)
        if removed_entries:
            pruned.append((checkpoint_name, removed_entries))

    return pruned


def save_training_checkpoint(
    save_path: str,
    output_dir: str,
    accelerator,
    keep_full_state_checkpoints: int,
    save_weights_fn: Callable[[str], None],
    logger,
) -> None:
    """Save weights, optionally save accelerator state, then prune older full states."""
    save_weights_fn(save_path)

    if keep_full_state_checkpoints > 0:
        if checkpoint_has_full_state(save_path):
            logger.info("Accelerator state already exists at %s; skipping full-state save.", save_path)
        else:
            accelerator.save_state(save_path)
            logger.info("Saved accelerator state to %s", save_path)

    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        pruned_checkpoints = prune_old_checkpoint_states(output_dir, keep_full_state_checkpoints)
        for checkpoint_name, removed_entries in pruned_checkpoints:
            logger.info(
                "Pruned full-state files from %s: %s",
                checkpoint_name,
                ", ".join(removed_entries),
            )

    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        accelerator.wait_for_everyone()
