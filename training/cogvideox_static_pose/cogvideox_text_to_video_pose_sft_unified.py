# Copyright 2024 The HuggingFace Team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import logging
import math
import os
import random
import shutil
import argparse
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict
import numpy as np
import imageio.v3 as iio

import diffusers
import torch
import transformers
import wandb
from accelerate import Accelerator, DistributedType, init_empty_weights
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    set_seed,
)
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXPipeline,
    CogVideoXTransformer3DModel,
)
from training.cogvideox_static_pose.cogvideox_pose_pipeline import CogVideoXPosePipeline
from training.cogvideox_static_pose.cogvideox_pose_adapter_pipeline import CogVideoXPoseAdapterPipeline
from training.cogvideox_static_pose.config_loader import load_experiment_config
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params
from diffusers.utils import export_to_video, convert_unet_state_dict_to_peft
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from huggingface_hub import create_repo, upload_folder
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, T5EncoderModel

from args import get_args  # isort:skip
from dataset import BucketSampler, VideoDatasetWithResizing, VideoDatasetWithResizeAndRectangleCrop, VideoDatasetWithConditions, VideoDatasetWithConditionsAndResizing, VideoDatasetWithConditionsAndResizeAndRectangleCrop  # isort:skip
from text_encoder import compute_prompt_embeddings  # isort:skip
from utils import (
    get_gradient_norm,
    get_optimizer,
    prepare_rotary_positional_embeddings,
    print_memory,
    reset_memory,
    unwrap_model,
)  # isort:skip

logger = get_logger(__name__)


def setup_pipeline_from_config(config: Dict[str, Any]):
    """
    Setup pipeline based on configuration.
    
    Args:
        config: Merged configuration dictionary
        
    Returns:
        pipeline: Configured pipeline instance
        transformer: Transformer model
        scheduler: Scheduler instance
    """
    pipeline_type = config["pipeline"]["type"]
    model_config = config["model"]
    
    print(f"🔧 Setting up pipeline: {pipeline_type}")
    
    # Determine load dtype based on model size
    model_path = model_config["pretrained_model_name_or_path"]
    load_dtype = torch.bfloat16 if "5b" in model_path.lower() else torch.float16
    
    if pipeline_type == "cogvideox_pose":
        # Setup CogVideoX Pose Pipeline
        pipeline = CogVideoXPosePipeline.from_pretrained(
            base_model_name_or_path=model_path,
            torch_dtype=load_dtype,
            revision=model_config.get("revision"),
            variant=model_config.get("variant"),
        )
        
    elif pipeline_type == "cogvideox_pose_adapter":
        # Setup CogVideoX Pose Adapter Pipeline
        adapter_config = config.get("adapter", {})
        pipeline = CogVideoXPoseAdapterPipeline.from_pretrained(
            base_model_name_or_path=model_path,
            torch_dtype=load_dtype,
            revision=model_config.get("revision"),
            variant=model_config.get("variant"),
            freeze_hand_branch=adapter_config.get("freeze_hand_branch", False),
            freeze_static_branch=adapter_config.get("freeze_static_branch", False),
            adapter_norm=adapter_config.get("norm", "group"),
            adapter_groups=adapter_config.get("groups", 32),
        )
        
    else:
        raise ValueError(f"Unsupported pipeline type: {pipeline_type}")
    
    return pipeline


def setup_training_mode(transformer, config: Dict[str, Any]):
    """
    Setup training mode (full or LoRA) based on configuration.
    
    Args:
        transformer: Transformer model to configure
        config: Training configuration
        
    Returns:
        num_trainable_parameters: Number of trainable parameters
    """
    training_mode = config["training"]["mode"]
    
    print(f"🚀 Setting up training mode: {training_mode}")
    
    if training_mode == "full":
        return setup_full_training(transformer, config)
    elif training_mode == "lora":
        return setup_lora_training(transformer, config)
    else:
        raise ValueError(f"Unsupported training mode: {training_mode}")


def setup_full_training(transformer, config: Dict[str, Any]):
    """Setup full fine-tuning mode."""
    print("🔧 Setting up full fine-tuning...")
    
    # Enable gradient checkpointing if specified
    if config["training"].get("custom_settings", {}).get("gradient_checkpointing", False):
        transformer.enable_gradient_checkpointing()
        print("✅ Gradient checkpointing enabled")
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    print(f"📊 Total trainable parameters: {trainable_params:,}")
    
    return trainable_params


def setup_lora_training(transformer, config: Dict[str, Any]):
    """Setup LoRA fine-tuning mode."""
    print("🔧 Setting up LoRA fine-tuning...")
    
    # Get LoRA parameters from config
    lora_rank = config["training"].get("lora_rank", 64)
    lora_alpha = config["training"].get("lora_alpha", 32)
    freeze_projection = config["training"].get("freeze_projection", False)
    
    print(f"   LoRA Rank: {lora_rank}")
    print(f"   LoRA Alpha: {lora_alpha}")
    print(f"   Freeze Projection: {freeze_projection}")
    
    # First, freeze all parameters
    for name, param in transformer.named_parameters():
        param.requires_grad_(False)
    
    # Add LoRA to attention layers
    transformer_lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        init_lora_weights=True,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    transformer.add_adapter(transformer_lora_config)
    
    # Handle projection layer based on pipeline type
    if hasattr(transformer, 'patch_embed') and hasattr(transformer.patch_embed, 'proj'):
        if not freeze_projection:
            transformer.patch_embed.proj.requires_grad_(True)
            print("✅ Projection layer set to trainable")
        else:
            print("🔒 Projection layer frozen")
    
    # Unfreeze LoRA parameters
    lora_params = 0
    for name, param in transformer.named_parameters():
        if "lora" in name.lower():
            param.requires_grad_(True)
            lora_params += param.numel()
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    
    print(f"📊 LoRA parameters: {lora_params:,}")
    print(f"📊 Total trainable parameters: {trainable_params:,}")
    
    return trainable_params


def create_save_hooks(accelerator, transformer, config: Dict[str, Any]):
    """Create save and load hooks for the model."""
    pipeline_type = config["pipeline"]["type"]
    training_mode = config["training"]["mode"]
    
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                if isinstance(unwrap_model(accelerator, model), type(unwrap_model(accelerator, transformer))):
                    model = unwrap_model(accelerator, model)
                    
                    if training_mode == "lora":
                        # Save LoRA weights
                        transformer_lora_layers = get_peft_model_state_dict(model)
                        
                        # Save projection layer weights if applicable
                        projection_state_dict = None
                        if hasattr(model, 'patch_embed') and hasattr(model.patch_embed, 'proj'):
                            projection_state_dict = {
                                "transformer.patch_embed.proj.weight": model.patch_embed.proj.weight.data,
                                "transformer.patch_embed.proj.bias": model.patch_embed.proj.bias.data if model.patch_embed.proj.bias is not None else None,
                            }
                        
                        # Save LoRA weights
                        if pipeline_type == "cogvideox_pose":
                            CogVideoXPosePipeline.save_lora_weights(
                                output_dir,
                                transformer_lora_layers=transformer_lora_layers,
                            )
                        elif pipeline_type == "cogvideox_pose_adapter":
                            CogVideoXPoseAdapterPipeline.save_lora_weights(
                                output_dir,
                                transformer_lora_layers=transformer_lora_layers,
                            )
                        
                        # Save projection layer weights separately
                        if projection_state_dict:
                            torch.save(projection_state_dict, os.path.join(output_dir, "projection_layer_weights.pt"))
                    else:
                        # Save full model
                        model.save_pretrained(
                            os.path.join(output_dir, "transformer"), 
                            safe_serialization=True, 
                            max_shard_size="5GB"
                        )
                    
                    # Remove from weights list
                    if weights:
                        weights.pop()
    
    def load_model_hook(models, input_dir):
        transformer_ = None
        
        if not accelerator.distributed_type == DistributedType.DEEPSPEED:
            while len(models) > 0:
                model = models.pop()
                if isinstance(unwrap_model(accelerator, model), type(unwrap_model(accelerator, transformer))):
                    transformer_ = unwrap_model(accelerator, model)
        else:
            transformer_ = CogVideoXTransformer3DModel.from_pretrained(
                config["model"]["pretrained_model_name_or_path"], 
                subfolder="transformer"
            )
        
        if training_mode == "lora":
            # Load LoRA weights
            if pipeline_type == "cogvideox_pose":
                lora_state_dict = CogVideoXPosePipeline.lora_state_dict(input_dir)
            elif pipeline_type == "cogvideox_pose_adapter":
                lora_state_dict = CogVideoXPoseAdapterPipeline.lora_state_dict(input_dir)
            
            transformer_state_dict = {
                f'{k.replace("transformer.", "")}': v for k, v in lora_state_dict.items() if k.startswith("transformer.")
            }
            transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
            set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
            
            # Load projection layer weights if they exist
            projection_file = os.path.join(input_dir, "projection_layer_weights.pt")
            if os.path.exists(projection_file):
                projection_state_dict = torch.load(projection_file, map_location="cpu")
                if hasattr(transformer_, 'patch_embed') and hasattr(transformer_.patch_embed, 'proj'):
                    if "transformer.patch_embed.proj.weight" in projection_state_dict:
                        transformer_.patch_embed.proj.weight.data.copy_(projection_state_dict["transformer.patch_embed.proj.weight"])
                    if "transformer.patch_embed.proj.bias" in projection_state_dict and projection_state_dict["transformer.patch_embed.proj.bias"] is not None:
                        transformer_.patch_embed.proj.bias.data.copy_(projection_state_dict["transformer.patch_embed.proj.bias"])
        else:
            # Load full model
            load_model = CogVideoXTransformer3DModel.from_pretrained(os.path.join(input_dir, "transformer"))
            transformer_.register_to_config(**load_model.config)
            transformer_.load_state_dict(load_model.state_dict())
            del load_model
        
        # Cast training parameters if needed
        if config["training"].get("custom_settings", {}).get("mixed_precision") in ["fp16", "bf16"]:
            cast_training_params([transformer_])
    
    return save_model_hook, load_model_hook


def main():
    parser = argparse.ArgumentParser(description="Unified CogVideoX Pose Training Script")
    parser.add_argument("--experiment_config", type=str, required=True,
                       help="Path to experiment YAML config file")
    parser.add_argument("--override", type=str, nargs="*",
                       help="Override config values (key=value format)")
    args = parser.parse_args()
    
    # Load configuration
    print("🚀 Loading experiment configuration...")
    config = load_experiment_config(args.experiment_config, args.override)
    
    # Extract configuration sections
    experiment_config = config["experiment"]
    pipeline_config = config["pipeline"]
    training_config = config["training"]
    data_config = config["data"]
    model_config = config["model"]
    logging_config = config["logging"]
    
    # Setup accelerator
    logging_dir = Path(model_config["output_dir"], "logs")
    accelerator_project_config = ProjectConfiguration(
        project_dir=model_config["output_dir"], 
        logging_dir=logging_dir
    )
    
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    init_process_group_kwargs = InitProcessGroupKwargs(backend="nccl", timeout=timedelta(seconds=1800))
    
    accelerator = Accelerator(
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        mixed_precision=training_config.get("custom_settings", {}).get("mixed_precision", "no"),
        log_with=logging_config["report_to"],
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs, init_process_group_kwargs],
    )
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    
    # Set seed
    if training_config.get("seed") is not None:
        set_seed(training_config["seed"])
    
    # Setup pipeline
    pipeline = setup_pipeline_from_config(config)
    transformer = pipeline.transformer
    scheduler = pipeline.scheduler
    vae = pipeline.vae
    text_encoder = pipeline.text_encoder
    tokenizer = pipeline.tokenizer
    
    # Setup training mode
    num_trainable_parameters = setup_training_mode(transformer, config)
    
    # Create save/load hooks
    save_model_hook, load_model_hook = create_save_hooks(accelerator, transformer, config)
    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)
    
    # Setup optimizer and scheduler
    optimizer = get_optimizer(
        params_to_optimize=[{"params": list(filter(lambda p: p.requires_grad, transformer.parameters()))}],
        optimizer_name=training_config.get("optimizer", "adamw"),
        learning_rate=training_config["learning_rate"],
        weight_decay=training_config.get("weight_decay", 0.01),
    )
    
    # Setup dataset and dataloader
    dataset_init_kwargs = {
        "data_root": data_config["data_root"],
        "dataset_file": data_config["dataset_file"],
        "max_num_frames": training_config.get("custom_settings", {}).get("max_num_frames", 48),
        "load_tensors": training_config.get("custom_settings", {}).get("load_tensors", False),
        "random_flip": training_config.get("custom_settings", {}).get("random_flip", False),
        "height_buckets": data_config.get("height_buckets", 480),
        "width_buckets": data_config.get("width_buckets", 720),
        "frame_buckets": data_config.get("frame_buckets", 49),
    }
    
    train_dataset = VideoDatasetWithConditionsAndResizing(**dataset_init_kwargs)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        sampler=BucketSampler(train_dataset, batch_size=training_config["batch_size"], shuffle=True),
        collate_fn=lambda x: {
            "videos": torch.stack([item["video"] for item in x[0]]),
            "prompts": [item["prompt"] for item in x[0]],
            "hand_videos": torch.stack([item["hand_videos"] for item in x[0]]) if "hand_videos" in x[0][0] else None,
            "static_videos": torch.stack([item["static_videos"] for item in x[0]]) if "static_videos" in x[0][0] else None,
        },
        num_workers=data_config.get("dataloader_num_workers", 0),
        pin_memory=data_config.get("pin_memory", True),
    )
    
    # Setup learning rate scheduler
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_config["gradient_accumulation_steps"])
    max_train_steps = training_config["num_epochs"] * num_update_steps_per_epoch
    
    lr_scheduler = get_scheduler(
        training_config.get("lr_scheduler", "cosine"),
        optimizer=optimizer,
        num_warmup_steps=training_config.get("lr_warmup_steps", 0) * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
        num_cycles=training_config.get("lr_num_cycles", 1),
    )
    
    # Prepare everything with accelerator
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )
    
    # Initialize trackers
    if accelerator.is_main_process:
        tracker_name = logging_config.get("tracker_name", f"{pipeline_config['type']}-training")
        accelerator.init_trackers(tracker_name, config=config)
    
    # Training loop
    print(f"🚀 Starting training for {training_config['num_epochs']} epochs...")
    
    global_step = 0
    for epoch in range(training_config["num_epochs"]):
        transformer.train()
        
        progress_bar = tqdm(
            total=len(train_dataloader),
            desc=f"Epoch {epoch + 1}/{training_config['num_epochs']}",
            disable=not accelerator.is_local_main_process,
        )
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                # Training step logic here
                # (This would include the actual training computation)
                # For now, just a placeholder
                loss = torch.tensor(0.0, device=accelerator.device)
            
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                # Log metrics
                if accelerator.is_main_process:
                    accelerator.log({
                        "train_loss": loss.item(),
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                        "epoch": epoch,
                        "step": global_step,
                    })
                
                # Save checkpoint
                if global_step % training_config.get("custom_settings", {}).get("checkpointing_steps", 500) == 0:
                    save_path = os.path.join(model_config["output_dir"], f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")
                
                # Check if we've reached max_train_steps
                if global_step >= training_config["num_epochs"]:
                    break
        
        if global_step >= training_config["num_epochs"]:
            break
    
    # Save final model
    if accelerator.is_main_process:
        accelerator.save_state(os.path.join(model_config["output_dir"], "final"))
        logger.info("Training completed!")
    
    accelerator.end_training()


if __name__ == "__main__":
    main()
