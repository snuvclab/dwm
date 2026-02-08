# Copyright 2024 The HuggingFace Team.
# Modified for WAN-based training with VideoX-Fun style.
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""
WAN-based training script for DWM (Dexterous World Model).

Supports:
- LoRA training (VideoX-Fun lora_utils style)
- SFT training (full or partial fine-tuning)
- Flow Matching loss
- WanTransformer3DModelWithConcat for conditional generation
"""

import argparse
import gc
import logging
import math
import os
import time
import pickle
import random
import re
import shutil
from datetime import datetime as dt, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from safetensors.torch import load_file, save_file

import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import (
    DistributedDataParallelKwargs,
    DummyScheduler,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    set_seed,
)
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils import export_to_video
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, RandomSampler
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, T5Tokenizer
from transformers.utils import ContextManagers
import imageio.v3 as iio

# WAN models and utilities from training.wan_static_pose
from training.wan_static_pose.models import (
    AutoencoderKLWan,
    AutoencoderKLWan3_8,
    CLIPModel,
    WanT5EncoderModel,
    WanTransformer3DModel,
)
from training.wan_static_pose.models.wan_transformer3d_with_conditions import (
    WanTransformer3DModelWithConcat,
)
from training.wan_static_pose.models.wan_transformer3d_vace import (
    WanTransformer3DVace,
)
from training.wan_static_pose.pipeline.pipeline_wan_fun_inpaint import (
    WanFunInpaintPipeline,
)
from training.wan_static_pose.pipeline.pipeline_wan_fun_inpaint_hand_concat import (
    WanFunInpaintHandConcatPipeline,
)
from training.wan_static_pose.pipeline.pipeline_wan2_2_fun_inpaint_hand_concat import (
    Wan2_2FunInpaintHandConcatPipeline,
)
from training.wan_static_pose.pipeline.pipeline_wan_fun_inpaint_hand_vace import (
    WanFunInpaintHandVacePipeline,
)
from training.wan_static_pose.utils.lora_utils import (
    create_network,
    merge_lora,
    unmerge_lora,
)
from training.wan_static_pose.utils.utils import (
    filter_kwargs,
    get_image_to_video_latent,
    save_videos_grid,
)
from training.wan_static_pose.utils.discrete_sampler import DiscreteSampling
from training.wan_static_pose.config_loader import load_experiment_config

# Dataset
from dataset import (
    BucketSampler,
    VideoDatasetWithConditionsAndResizing,
)

# Wandb for logging
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

logger = get_logger(__name__)


def resize_mask(mask, latent, process_first_frame_only=True):
    """Resize mask to match latent dimensions."""
    latent_size = latent.size()
    batch_size, channels, num_frames, height, width = mask.shape

    if process_first_frame_only:
        target_size = list(latent_size[2:])
        target_size[0] = 1
        first_frame_resized = F.interpolate(
            mask[:, :, 0:1, :, :],
            size=target_size,
            mode='trilinear',
            align_corners=False
        )
        
        target_size = list(latent_size[2:])
        target_size[0] = target_size[0] - 1
        if target_size[0] != 0:
            remaining_frames_resized = F.interpolate(
                mask[:, :, 1:, :, :],
                size=target_size,
                mode='trilinear',
                align_corners=False
            )
            resized_mask = torch.cat([first_frame_resized, remaining_frames_resized], dim=2)
        else:
            resized_mask = first_frame_resized
    else:
        target_size = list(latent_size[2:])
        resized_mask = F.interpolate(
            mask,
            size=target_size,
            mode='trilinear',
            align_corners=False
        )
    return resized_mask


def get_sigmas(noise_scheduler, timesteps, n_dim=4, dtype=torch.float32, device="cuda"):
    """Get sigmas for flow matching."""
    sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def custom_mse_loss(noise_pred, target, weighting=None, threshold=50):
    """Custom MSE loss with optional weighting and threshold."""
    noise_pred = noise_pred.float()
    target = target.float()
    diff = noise_pred - target
    mse_loss = F.mse_loss(noise_pred, target, reduction='none')
    mask = (diff.abs() <= threshold).float()
    masked_loss = mse_loss * mask
    if weighting is not None:
        masked_loss = masked_loss * weighting
    final_loss = masked_loss.mean()
    return final_loss


def log_validation(
    vae,
    text_encoder,
    tokenizer,
    clip_image_encoder,
    transformer3d,
    network,
    config,
    args,
    accelerator,
    weight_dtype,
    global_step,
    load_tensors: bool = False,
    vae_path: str = None,
    vae_kwargs: dict = None,
    text_encoder_path: str = None,
    text_encoder_kwargs: dict = None,
    tokenizer_path: str = None,
):
    """Run validation during training using dataset-based validation samples."""
    try:
        logger.info("Running validation...")

        # If load_tensors=True, VAE/text_encoder were deleted during training. Reload them for validation.
        models_reloaded = False
        if load_tensors and vae is None:
            if vae_path is None:
                logger.warning("load_tensors=True but vae_path not provided. Skipping validation.")
                return
            logger.info(f"📦 Reloading models for validation...")

            # Reload VAE
            logger.info(f"   Loading VAE from {vae_path}...")
            Choosen_AutoencoderKL = {
                "AutoencoderKLWan": AutoencoderKLWan,
                "AutoencoderKLWan3_8": AutoencoderKLWan3_8
            }[vae_kwargs.get('vae_type', 'AutoencoderKLWan') if vae_kwargs else 'AutoencoderKLWan']
            vae = Choosen_AutoencoderKL.from_pretrained(
                vae_path,
                additional_kwargs=vae_kwargs or {},
            ).eval()
            vae = vae.to(accelerator.device, dtype=weight_dtype)

            # Reload text_encoder and tokenizer
            if text_encoder_path is not None and tokenizer_path is not None:
                logger.info(f"   Loading text encoder from {text_encoder_path}...")
                text_encoder = WanT5EncoderModel.from_pretrained(
                    text_encoder_path,
                    additional_kwargs=text_encoder_kwargs or {},
                    torch_dtype=weight_dtype,
                ).eval()
                text_encoder = text_encoder.to(accelerator.device)

                logger.info(f"   Loading tokenizer from {tokenizer_path}...")
                tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)

            models_reloaded = True
            logger.info("   Models reloaded successfully.")

        # Extract config values
        pipeline_config = config.get('pipeline', {})
        training_config = config.get('training', {})
        transformer_config = config.get('transformer', {})
        data_config = config.get('data', {})
        scheduler_kwargs = config.get('scheduler_kwargs', {})
        transformer_additional_kwargs = config.get('transformer_additional_kwargs', {})

        # Propagate fixed FPS (used for RoPE scaling) into transformer loading kwargs
        if pipeline_config.get("fps") is not None:
            transformer_additional_kwargs["fps"] = int(pipeline_config["fps"])
        
        # Check if WAN 2.2 (needed for transformer loading and CLIP encoder)
        is_wan2_2 = "2.2" in pipeline_config.get("type", "")
        
        # Get base model path (same as in main training)
        base_model_path = pipeline_config.get('base_model_name_or_path', args.pretrained_model_name_or_path)

        # Load validation set from config
        validation_entries = data_config.get("validation_set")
        if validation_entries is None:
            logger.warning("No validation_set specified in config. Skipping validation.")
            return
        
        data_root = data_config.get("data_root", args.train_data_dir)
        
        # Load validation video paths
        if isinstance(validation_entries, (list, tuple)):
            validation_set = []
            for entry in validation_entries:
                validation_set_path = os.path.join(data_root, entry)
                if os.path.exists(validation_set_path):
                    with open(validation_set_path, "r") as f:
                        validation_set.extend([video.strip() for video in f.readlines()])
        else:
            validation_set_path = os.path.join(data_root, validation_entries)
            if os.path.exists(validation_set_path):
                with open(validation_set_path, "r") as f:
                    validation_set = [video.strip() for video in f.readlines()]
            else:
                logger.warning(f"Validation set file not found: {validation_set_path}")
                return
        
        # Limit validation samples
        max_validation_videos = data_config.get("max_validation_videos", 2)
        
        # For multi-GPU training, multiply by number of GPUs to ensure each GPU gets different videos
        total_validation_videos = max_validation_videos * accelerator.num_processes
        
        # Select validation videos (sequential selection)
        if len(validation_set) > total_validation_videos:
            validation_set = validation_set[:total_validation_videos]
        else:
            validation_set = validation_set[:total_validation_videos]
        
        # Distribute videos across GPUs - each GPU gets different videos
        videos_per_gpu = len(validation_set) // accelerator.num_processes
        start_idx = accelerator.process_index * videos_per_gpu
        end_idx = start_idx + videos_per_gpu if accelerator.process_index < accelerator.num_processes - 1 else len(validation_set)
        
        # Each GPU gets its own subset of videos
        validation_set = validation_set[start_idx:end_idx]
        
        if not validation_set:
            logger.warning("No validation videos found. Skipping validation.")
            return
        
        logger.info(f"🎯 Multi-GPU Validation Strategy:")
        logger.info(f"   - Total GPUs: {accelerator.num_processes}")
        logger.info(f"   - Videos per GPU: {videos_per_gpu}")
        logger.info(f"   - GPU {accelerator.process_index}: Processing {len(validation_set)} validation videos (indices {start_idx}-{end_idx-1})")

        # Determine pretrain mode
        pretrain_mode = training_config.get("pretrain_mode", "t2v")
        transformer_type = transformer_config.get("class", "WanTransformer3DModelWithConcat")
        is_wan2_2 = "2.2" in pipeline_config.get("type", "")

        # Create a separate transformer for validation so merge_lora does not modify training weights in-place
        transformer_path = os.path.join(
            base_model_path,
            transformer_additional_kwargs.get("transformer_subpath", "transformer"),
        )
        if transformer_type == "WanTransformer3DModel":
            transformer3d_val = WanTransformer3DModel.from_pretrained(
                transformer_path,
                transformer_additional_kwargs=transformer_additional_kwargs,
                torch_dtype=weight_dtype,
            )
        elif transformer_type == "WanTransformer3DModelWithConcat":
            transformer3d_val = WanTransformer3DModelWithConcat.from_pretrained(
                transformer_path,
                transformer_additional_kwargs={
                    "condition_channels": getattr(args, "condition_channels", 16),
                    "is_wan2_2": is_wan2_2,
                    **transformer_additional_kwargs,
                },
                torch_dtype=weight_dtype,
            )
        elif transformer_type == "WanTransformer3DVace":
            vace_layers = transformer_config.get("vace_layers", None)
            vace_in_dim = transformer_config.get("vace_in_dim", 16)
            transformer3d_val = WanTransformer3DVace.from_pretrained(
                transformer_path,
                transformer_additional_kwargs={
                    "vace_layers": vace_layers,
                    "vace_in_dim": vace_in_dim,
                    "is_wan2_2": is_wan2_2,
                    **transformer_additional_kwargs,
                },
                torch_dtype=weight_dtype,
            )
        else:
            raise ValueError(f"Unsupported transformer type: {transformer_type}")
        # Sync weights from training transformer (base weights; LoRA will be merged later)
        training_state = accelerator.unwrap_model(transformer3d).state_dict()
        m, u = transformer3d_val.load_state_dict(training_state, strict=False)
        if u:
            logger.debug(f"Validation transformer: unexpected keys (e.g. LoRA) not loaded: {len(u)}")
        transformer3d_val = transformer3d_val.to(accelerator.device, dtype=weight_dtype)
        transformer3d_val.eval()

        scheduler = FlowMatchEulerDiscreteScheduler(
            **filter_kwargs(FlowMatchEulerDiscreteScheduler, scheduler_kwargs)
        )

        # Check pipeline type to select appropriate pipeline
        pipeline_type = pipeline_config.get("type", "")

        if is_wan2_2:
            # WAN 2.2: No CLIP encoder needed
            logger.info("🔧 Using WAN 2.2 pipeline for validation")
            pipeline = Wan2_2FunInpaintHandConcatPipeline(
                vae=accelerator.unwrap_model(vae),
                text_encoder=accelerator.unwrap_model(text_encoder),
                tokenizer=tokenizer,
                transformer=transformer3d_val,
                transformer_2=None,  # Can be loaded from config if needed
                scheduler=scheduler,
            )
        else:
            # WAN 2.1: CLIP encoder required
            clip_image_encoder_for_pipeline = clip_image_encoder
            if clip_image_encoder_for_pipeline is None:
                logger.warning("⚠️  Creating dummy CLIP encoder for pipeline compatibility")
                # Create a minimal dummy CLIP encoder
                clip_image_encoder_for_pipeline = CLIPModel()
                clip_image_encoder_for_pipeline.eval()
                clip_image_encoder_for_pipeline.requires_grad_(False)
            
            if pipeline_type == "wan2.1_fun_inp":
                logger.info("🔧 Using WAN 2.1 base pipeline for validation")
                pipeline = WanFunInpaintPipeline(
                    vae=accelerator.unwrap_model(vae),
                    text_encoder=accelerator.unwrap_model(text_encoder),
                    tokenizer=tokenizer,
                    transformer=transformer3d_val,
                    scheduler=scheduler,
                    clip_image_encoder=clip_image_encoder_for_pipeline,
                )
            elif pipeline_type == "wan2.1_fun_inp_hand_concat":
                logger.info("🔧 Using WAN 2.1 hand concat pipeline for validation")
                pipeline = WanFunInpaintHandConcatPipeline(
                    vae=accelerator.unwrap_model(vae),
                    text_encoder=accelerator.unwrap_model(text_encoder),
                    tokenizer=tokenizer,
                    transformer=transformer3d_val,
                    scheduler=scheduler,
                    clip_image_encoder=clip_image_encoder_for_pipeline,
                )
            elif pipeline_type == "wan2.1_fun_inp_hand_vace":
                logger.info("🔧 Using WAN 2.1 VACE pipeline for validation")
                pipeline = WanFunInpaintHandVacePipeline(
                    vae=accelerator.unwrap_model(vae),
                    text_encoder=accelerator.unwrap_model(text_encoder),
                    tokenizer=tokenizer,
                    transformer=transformer3d_val,
                    scheduler=scheduler,
                    clip_image_encoder=clip_image_encoder_for_pipeline,
                )
            else:
                raise ValueError(f"Unsupported WAN 2.1 pipeline type: {pipeline_type}")
        pipeline = pipeline.to(accelerator.device, dtype=weight_dtype)

        # Merge LoRA weights for validation if using LoRA training
        if args.train_mode == "lora" and network is not None:
            pipeline = merge_lora(
                pipeline, None, 1, accelerator.device,
                state_dict=accelerator.unwrap_model(network).state_dict(),
                transformer_only=True
            )

        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
        
        # Get subdirectory names from config
        prompt_subdir = data_config.get("prompt_subdir", "prompts")
        hand_video_subdir = data_config.get("hand_video_subdir", "videos_hands")
        static_video_subdir = data_config.get("static_video_subdir", "videos_static")

        # Run validation for each sample
        for i, video_path in enumerate(validation_set):
            video_path_obj = Path(video_path)
            video_name = video_path_obj.stem
            
            # Construct paths for condition videos and prompt
            full_video_path = Path(data_root) / video_path
            prompt_path = full_video_path.parent.parent / prompt_subdir / f"{video_name}.txt"
            hand_video_path = full_video_path.parent.parent / hand_video_subdir / video_path_obj.name
            static_video_path = full_video_path.parent.parent / static_video_subdir / video_path_obj.name
            
            # Check if required files exist
            if not prompt_path.exists():
                logger.warning(f"Prompt not found: {prompt_path}. Skipping.")
                continue
            if not static_video_path.exists():
                logger.warning(f"Static video not found: {static_video_path}. Skipping.")
                continue
            
            # Load prompt
            with open(prompt_path, "r") as f:
                prompt_text = f.read().strip()
            
            # Load GT video for comparison
            gt_video = iio.imread(str(full_video_path)).astype(np.float32) / 255.0
            
            # Load static video
            static_video = iio.imread(str(static_video_path)).astype(np.float32) / 255.0
            static_video = torch.from_numpy(static_video).permute(3, 0, 1, 2).unsqueeze(0)  # [1, c, f, h, w]
            
            # Load hand video if exists
            if hand_video_path.exists():
                hand_video = iio.imread(str(hand_video_path)).astype(np.float32) / 255.0
                hand_video = torch.from_numpy(hand_video).permute(3, 0, 1, 2).unsqueeze(0)  # [1, c, f, h, w]
            else:
                hand_video = None
            
            # Align width to 32 for WAN 2.2 compatibility
            if is_wan2_2:
                # Get original dimensions
                num_frames = gt_video.shape[0]
                orig_height, orig_width = gt_video.shape[1], gt_video.shape[2]
                
                # Calculate aligned width (round to nearest multiple of 32)
                aligned_width = round(orig_width / 32) * 32
                
                if aligned_width != orig_width:
                    # Resize GT video: [f, h, w, c] -> resize -> [f, h, w, c]
                    gt_video_tensor = torch.from_numpy(gt_video).permute(0, 3, 1, 2)  # [f, c, h, w]
                    gt_video_tensor = F.interpolate(
                        gt_video_tensor, 
                        size=(orig_height, aligned_width), 
                        mode='bilinear', 
                        align_corners=False
                    )
                    gt_video = gt_video_tensor.permute(0, 2, 3, 1).numpy()  # [f, h, w, c]
                    
                    # Resize static video: [1, c, f, h, w] -> [1, c, f, h, w]
                    static_video = F.interpolate(
                        static_video.squeeze(0),  # [c, f, h, w]
                        size=(orig_height, aligned_width),
                        mode='bilinear',
                        align_corners=False
                    ).unsqueeze(0)  # [1, c, f, h, w]
                    
                    # Resize hand video if exists: [1, c, f, h, w] -> [1, c, f, h, w]
                    if hand_video is not None:
                        hand_video = F.interpolate(
                            hand_video.squeeze(0),  # [c, f, h, w]
                            size=(orig_height, aligned_width),
                            mode='bilinear',
                            align_corners=False
                        ).unsqueeze(0)  # [1, c, f, h, w]
                    
                    height, width = orig_height, aligned_width
                else:
                    height, width = orig_height, orig_width
            else:
                num_frames = gt_video.shape[0]
                height, width = gt_video.shape[1], gt_video.shape[2]
            
            with torch.no_grad():
                with torch.autocast("cuda", dtype=weight_dtype):
                    
                    mask_video = torch.zeros(1, 1, num_frames, height, width)
                    
                    if transformer_type == "WanTransformer3DModel":
                        # Base pipeline uses 'video' parameter instead of 'static_video'
                        if pretrain_mode == "t2v":
                            sample = pipeline(
                                prompt_text,
                                num_frames=num_frames,
                                negative_prompt="bad detailed",
                                height=height,
                                width=width,
                                guidance_scale=6.0,
                                generator=generator,
                                video=torch.zeros_like(static_video),
                                mask_video=torch.ones_like(mask_video) * 255, 
                            ).videos
                        elif pretrain_mode == "v2v":
                            sample = pipeline(
                                prompt_text,
                                num_frames=num_frames,
                                negative_prompt="bad detailed",
                                height=height,
                                width=width,
                                guidance_scale=6.0,
                                generator=generator,
                                video=static_video.to(accelerator.device, dtype=weight_dtype),
                                mask_video=mask_video.to(accelerator.device, dtype=weight_dtype),
                            ).videos
                        elif pretrain_mode == "i2v":
                            mask_video[:, :, 1:, :, :] = 255
                            sample = pipeline(
                                prompt_text,
                                num_frames=num_frames,
                                negative_prompt="bad detailed",
                                height=height,
                                width=width,
                                guidance_scale=6.0,
                                generator=generator,
                                video=static_video.to(accelerator.device, dtype=weight_dtype),
                                mask_video=mask_video.to(accelerator.device, dtype=weight_dtype),
                            ).videos
                    elif transformer_type == "WanTransformer3DModelWithConcat":
                        # Concat pipeline uses 'static_video' and 'hand_video'
                        sample = pipeline(
                            prompt_text,
                            num_frames=num_frames,
                            negative_prompt="bad detailed",
                            height=height,
                            width=width,
                            guidance_scale=6.0,
                            generator=generator,
                            static_video=static_video.to(accelerator.device, dtype=weight_dtype),
                            mask_video=mask_video.to(accelerator.device, dtype=weight_dtype),
                            hand_video=hand_video.to(accelerator.device, dtype=weight_dtype) if hand_video is not None else None,
                        ).videos
                    elif transformer_type == "WanTransformer3DVace":
                        # VACE pipeline uses 'static_video', 'hand_video', and 'vace_context_scale'
                        vace_context_scale = training_config.get("vace_context_scale", 1.0)
                        sample = pipeline(
                            prompt_text,
                            num_frames=num_frames,
                            negative_prompt="bad detailed",
                            height=height,
                            width=width,
                            guidance_scale=6.0,
                            generator=generator,
                            static_video=static_video.to(accelerator.device, dtype=weight_dtype),
                            mask_video=mask_video.to(accelerator.device, dtype=weight_dtype),
                            hand_video=hand_video.to(accelerator.device, dtype=weight_dtype) if hand_video is not None else None,
                            vace_context_scale=vace_context_scale,
                        ).videos
                    else:
                        raise ValueError(f"Unsupported transformer type: {transformer_type}")
                    
                    # Prepare videos for comparison: static | hand | generated | gt
                    # All videos should be [1, c, f, h, w] format
                    static_tensor = static_video.cpu()  # Already [1, c, f, h, w]
                    
                    if hand_video is not None:
                        hand_tensor = hand_video.cpu()  # Already [1, c, f, h, w]
                    else:
                        # Create black video if hand_video is None
                        hand_tensor = torch.zeros_like(static_tensor)
                    
                    sample_tensor = sample.cpu()  # Already [1, c, f, h, w]
                    gt_tensor = torch.from_numpy(gt_video).permute(3, 0, 1, 2).unsqueeze(0)  # [1, c, f, h, w]
                    
                    # Ensure all videos have the same number of frames
                    num_frames_actual = min(
                        static_tensor.shape[2], 
                        hand_tensor.shape[2], 
                        sample_tensor.shape[2], 
                        gt_tensor.shape[2]
                    )
                    static_tensor = static_tensor[:, :, :num_frames_actual, :, :]
                    hand_tensor = hand_tensor[:, :, :num_frames_actual, :, :]
                    sample_tensor = sample_tensor[:, :, :num_frames_actual, :, :]
                    gt_tensor = gt_tensor[:, :, :num_frames_actual, :, :]
                    
                    # Create comparison video: static | hand | generated | gt (concatenate along width)
                    comparison = torch.cat([static_tensor, hand_tensor, sample_tensor, gt_tensor], dim=4)  # [1, c, f, h, 4*w]
                    
                    # Prepare file paths (cogvideox style)
                    phase_name = "validation"
                    save_name = video_name
                    gpu_suffix = f"_gpu{accelerator.process_index}" if accelerator.num_processes > 1 else ""
                    
                    # Ensure output directory exists
                    os.makedirs(args.output_dir, exist_ok=True)
                    
                    # Save generated video
                    generated_filename = os.path.join(args.output_dir, f"step_{global_step}_{phase_name}_generated_{save_name}{gpu_suffix}.mp4")
                    save_videos_grid(sample_tensor, generated_filename, fps=8)
                    
                    # Save comparison video (static | hand | generated | gt)
                    comparison_filename = os.path.join(args.output_dir, f"step_{global_step}_{phase_name}_comparison_{save_name}{gpu_suffix}.mp4")
                    save_videos_grid(comparison, comparison_filename, fps=8)
                    
                    logger.info(f"📹 GPU {accelerator.process_index}: Saved validation videos for {save_name}")
                    logger.info(f"   Generated: {generated_filename}")
                    logger.info(f"   Comparison (static|hand|generated|gt): {comparison_filename}")
                    
                    # Log to wandb if available (only on main process)
                    if accelerator.is_main_process:
                        for tracker in accelerator.trackers:
                            if tracker.name == "wandb" and HAS_WANDB:
                                caption = f"step_{global_step}_{phase_name}_{video_name}_comparison (static|hand|generated|gt)"
                                tracker.log({
                                    phase_name: [
                                        wandb.Video(comparison_filename, caption=caption, fps=8)
                                    ]
                                }, step=global_step)
                    
                    logger.info(f"Validation {i+1}/{len(validation_set)}: {video_name} saved")

        del pipeline
        # Clean up reloaded models if they were loaded for this validation run
        if models_reloaded:
            logger.info("📦 Cleaning up reloaded models after validation...")
            del vae
            if text_encoder is not None:
                del text_encoder
            if tokenizer is not None:
                del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    except Exception as e:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        logger.error(f"Validation error: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_model_lora(output_dir, network, weight_dtype, accelerator, transformer3d=None, trainable_parameter_patterns=None):
    """Save LoRA weights and non-LoRA trainable weights."""
    if accelerator.is_main_process:
        # Save LoRA weights
        safetensor_save_path = os.path.join(output_dir, "lora_diffusion_pytorch_model.safetensors")
        accelerator.unwrap_model(network).save_weights(safetensor_save_path, weight_dtype, None)
        logger.info(f"Saved LoRA weights to {safetensor_save_path}")
        
        # Save non-LoRA trainable weights (e.g., patch_embedding)
        if transformer3d is not None and trainable_parameter_patterns:
            non_lora_state_dict = {}
            model = accelerator.unwrap_model(transformer3d)
            for name, param in model.named_parameters():
                for pattern in trainable_parameter_patterns:
                    if re.search(pattern, name):
                        non_lora_state_dict[name] = param.data.clone()
                        break
            
            if non_lora_state_dict:
                non_lora_path = os.path.join(output_dir, "non_lora_weights.safetensors")
                save_file(non_lora_state_dict, non_lora_path)
                logger.info(f"Saved non-LoRA weights ({len(non_lora_state_dict)} params) to {non_lora_path}")


def save_model_sft(output_dir, transformer3d, accelerator):
    """Save SFT model weights."""
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(transformer3d)
        model.save_pretrained(os.path.join(output_dir, "transformer"), safe_serialization=True, max_shard_size="5GB")
        logger.info(f"Saved transformer to {output_dir}/transformer")


def save_model_vace(output_dir, transformer3d, accelerator):
    """Save VACE weights (vace_blocks + vace_patch_embedding only)."""
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(transformer3d)

        # Extract only VACE-related weights
        vace_state_dict = {}
        for name, param in model.named_parameters():
            if "vace_blocks" in name or "vace_patch_embedding" in name:
                vace_state_dict[name] = param.data.clone()

        if vace_state_dict:
            vace_path = os.path.join(output_dir, "vace_weights.safetensors")
            save_file(vace_state_dict, vace_path)
            logger.info(f"Saved VACE weights ({len(vace_state_dict)} params) to {vace_path}")
        else:
            logger.warning("No VACE weights found to save!")


def main():
    from training.wan_static_pose.args import add_all_args
    
    parser = argparse.ArgumentParser(description="WAN-based DWM Training Script")
    # Core arguments
    parser.add_argument("--experiment_config", type=str, required=True,
                        help="Path to experiment YAML config file")
    parser.add_argument("--override", type=str, nargs="*",
                        help="Override config values (key=value format)")
    parser.add_argument("--mode", type=str, choices=["debug", "slurm_test", "slurm", "batch"], default="slurm",
                        help="Training mode: debug, slurm_test, slurm, or batch")
    # # Add all other arguments from args.py
    add_all_args(parser)
    args = parser.parse_args()
    
    # ========== Load Configuration ==========
    print("🚀 Loading experiment configuration...")
    config = load_experiment_config(args.experiment_config, args.override)
    
    # Extract configuration sections
    experiment_config = config.get("experiment", {})
    training_config = config.get("training", {})
    pipeline_config = config.get("pipeline", {})
    transformer_config = config.get("transformer", {})
    data_config = config.get("data", {})

    # Print training FPS (used for RoPE scaling in WanTransformer3DModel)
    if pipeline_config.get("fps") is not None:
        print(f"🎞️ Training FPS: {int(pipeline_config['fps'])} (from pipeline_config)")
    else:
        print("🎞️ Training FPS: 16 (default)")
    
    # Model-related configs from pipeline config (merged from configs/pipelines/*.yaml)
    text_encoder_kwargs = config.get("text_encoder_kwargs", {})
    vae_kwargs = config.get("vae_kwargs", {})
    scheduler_kwargs = config.get("scheduler_kwargs", {})
    transformer_additional_kwargs = config.get("transformer_additional_kwargs", {})
    image_encoder_kwargs = config.get("image_encoder_kwargs", {})

    # Propagate fixed FPS (used for RoPE scaling) into transformer loading kwargs
    if pipeline_config.get("fps") is not None:
        transformer_additional_kwargs["fps"] = int(pipeline_config["fps"])
    
    # Override args with config values (config values take priority over command line args)
    # Use "is not None" check to ensure config values (including 0, False, empty string) override args
    if training_config.get("learning_rate") is not None:
        args.learning_rate = float(training_config["learning_rate"])
    if training_config.get("batch_size") is not None:
        args.train_batch_size = int(training_config["batch_size"])
    if training_config.get("max_train_steps") is not None:
        args.max_train_steps = int(training_config["max_train_steps"])
    if training_config.get("gradient_accumulation_steps") is not None:
        args.gradient_accumulation_steps = int(training_config["gradient_accumulation_steps"])
    # Note: args.pretrained_model_name_or_path is NOT overridden by pipeline_config
    # to allow specifying SFT checkpoint path via CLI (--pretrained_model_name_or_path)
    # pipeline_config["base_model_name_or_path"] is used for model structure definition
    if args.pretrained_model_name_or_path is None and pipeline_config.get("base_model_name_or_path") is not None:
        args.pretrained_model_name_or_path = pipeline_config["base_model_name_or_path"]
    if data_config.get("data_root") is not None:
        args.train_data_dir = data_config["data_root"]
    if data_config.get("dataset_file") is not None:
        args.train_data_meta = data_config["dataset_file"]
    # Determine transformer type
    transformer_type = transformer_config.get("class", "WanTransformer3DModelWithConcat")
    if transformer_config.get("condition_channels") is not None:
        args.condition_channels = int(transformer_config["condition_channels"])
    if training_config.get("mode") is not None:
        args.train_mode = training_config["mode"]
    if training_config.get("lora_rank") is not None:
        args.rank = int(training_config["lora_rank"])
    if training_config.get("lora_alpha") is not None:
        args.network_alpha = int(training_config["lora_alpha"])
    
    # ========== Setup Output Directory with SLURM Support ==========
    # Check for resume configuration
    resume_from_checkpoint = training_config.get("resume_from_checkpoint") or args.resume_from_checkpoint
    if resume_from_checkpoint:
        print(f"🔄 Resume mode enabled: resume_from_checkpoint = {resume_from_checkpoint}")
    
    # Setup output directory based on mode
    exp_name = experiment_config.get("name", "wan_training")
    exp_date = experiment_config.get("date", "unknown")
    
    # Extract date suffix
    if exp_date != "unknown":
        try:
            parsed_date = dt.strptime(exp_date, "%Y-%m-%d")
            date_suffix = parsed_date.strftime("%y%m%d")
        except:
            date_suffix = "unknown"
    else:
        date_suffix = "unknown"
    
    # Construct output directory based on mode
    base_output_dir = f"outputs/{date_suffix}/{exp_name}"
    if args.mode == "debug":
        args.output_dir = f"{base_output_dir}_debug"
        print(f"🔧 Debug mode: Output directory set to {args.output_dir}")
    elif args.mode == "slurm_test":
        slurm_job_id = training_config.get("slurm_job_id")
        if resume_from_checkpoint and slurm_job_id:
            print(f"🔄 Resume mode: Using SLURM Job ID from config: {slurm_job_id}")
            args.output_dir = f"{base_output_dir}_slurm_{slurm_job_id}"
        else:
            args.output_dir = f"{base_output_dir}_slurm_test"
        print(f"🧪 SLURM test mode: Output directory set to {args.output_dir}")
    elif args.mode == "slurm":
        if resume_from_checkpoint:
            slurm_job_id = training_config.get("slurm_job_id")
            if slurm_job_id:
                print(f"🔄 Resume mode: Using SLURM Job ID from config: {slurm_job_id}")
            else:
                slurm_job_id = os.environ.get("SLURM_JOB_ID", "unknown")
                print(f"🔄 Resume mode: Using current SLURM Job ID: {slurm_job_id}")
        else:
            slurm_job_id = os.environ.get("SLURM_JOB_ID", "unknown")
            print(f"🚀 New training: Using current SLURM Job ID: {slurm_job_id}")
        args.output_dir = f"{base_output_dir}_slurm_{slurm_job_id}"
        print(f"📁 SLURM mode: Output directory set to {args.output_dir}")
    elif args.mode == "batch":
        # Batch mode: add timestamp to output directory (use env var so all processes get same dir)
        timestamp = os.environ.get("BATCH_TIMESTAMP")
        if not timestamp:
            timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"{base_output_dir}_batch_{timestamp}"
        print(f"🚀 Batch mode: Output directory set to {args.output_dir}")
    
    print(f"📁 Final output directory: {args.output_dir}")
    print(f"   - Base: {base_output_dir}")
    print(f"   - Experiment: {exp_name}")
    print(f"   - Date: {exp_date} -> {date_suffix}")
    print(f"   - Mode: {args.mode}")
    
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    init_process_group_kwargs = InitProcessGroupKwargs(backend="nccl", timeout=timedelta(seconds=1800))

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=training_config.get("custom_settings", {}).get("mixed_precision", args.mixed_precision),
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs, init_process_group_kwargs],
    )
    
    # ========== Detect DeepSpeed / FSDP ==========
    deepspeed_plugin = accelerator.state.deepspeed_plugin if hasattr(accelerator.state, "deepspeed_plugin") else None
    fsdp_plugin = accelerator.state.fsdp_plugin if hasattr(accelerator.state, "fsdp_plugin") else None
    
    if deepspeed_plugin is not None:
        zero_stage = int(deepspeed_plugin.zero_stage)
        fsdp_stage = 0
        print(f"🚀 Using DeepSpeed Zero stage: {zero_stage}")
        args.use_deepspeed = True
    elif fsdp_plugin is not None:
        from torch.distributed.fsdp import ShardingStrategy
        zero_stage = 0
        if fsdp_plugin.sharding_strategy is ShardingStrategy.FULL_SHARD:
            fsdp_stage = 3
        elif fsdp_plugin.sharding_strategy is None:
            fsdp_stage = 3
        elif fsdp_plugin.sharding_strategy is ShardingStrategy.SHARD_GRAD_OP:
            fsdp_stage = 2
        else:
            fsdp_stage = 0
        print(f"🚀 Using FSDP stage: {fsdp_stage}")
        args.use_fsdp = True
    else:
        zero_stage = 0
        fsdp_stage = 0
        print("🚀 DeepSpeed/FSDP is not enabled.")

    # Create output directory and copy experiment config
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Copy experiment config YAML to output directory
        if args.experiment_config:
            config_filename = os.path.basename(args.experiment_config)
            output_config_path = os.path.join(args.output_dir, config_filename)
            shutil.copy2(args.experiment_config, output_config_path)
            print(f"📋 Copied experiment config to: {output_config_path}")

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
    if args.seed is not None:
        set_seed(args.seed)
        rng = np.random.default_rng(np.random.PCG64(args.seed + accelerator.process_index))
        torch_rng = torch.Generator(accelerator.device).manual_seed(args.seed + accelerator.process_index)
    else:
        rng = None
        torch_rng = None

    # Create output directory
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # Determine weight dtype
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # ========== Load Models ==========
    print("🔧 Loading models...")
    
    # Load scheduler
    noise_scheduler = FlowMatchEulerDiscreteScheduler(
        **filter_kwargs(FlowMatchEulerDiscreteScheduler, scheduler_kwargs)
    )

    # Load tokenizer
    tokenizer_path = os.path.join(args.pretrained_model_name_or_path, text_encoder_kwargs.get('tokenizer_subpath', 'tokenizer'))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Load text encoder
    text_encoder_path = os.path.join(args.pretrained_model_name_or_path, text_encoder_kwargs.get('text_encoder_subpath', 'text_encoder'))
    text_encoder = WanT5EncoderModel.from_pretrained(
        text_encoder_path,
        additional_kwargs=text_encoder_kwargs,
        torch_dtype=weight_dtype,
    ).eval()

    # Load VAE
    Choosen_AutoencoderKL = {
        "AutoencoderKLWan": AutoencoderKLWan,
        "AutoencoderKLWan3_8": AutoencoderKLWan3_8
    }[vae_kwargs.get('vae_type', 'AutoencoderKLWan')]
    vae_path = os.path.join(args.pretrained_model_name_or_path, vae_kwargs.get('vae_subpath', 'vae'))
    vae = Choosen_AutoencoderKL.from_pretrained(
        vae_path,
        additional_kwargs=vae_kwargs,
    ).eval()

    # Check if WAN 2.2 (needed for transformer loading and CLIP encoder)
    is_wan2_2 = "2.2" in pipeline_config.get("type", "")
    
    # Load CLIP image encoder (skip for WAN 2.2)
    if is_wan2_2:
        logger.info("⚠️  WAN 2.2 detected: Skipping CLIP image encoder loading")
        clip_image_encoder = None
    else:
        clip_image_encoder = CLIPModel.from_pretrained(
            os.path.join(args.pretrained_model_name_or_path, image_encoder_kwargs.get('image_encoder_subpath', 'image_encoder')),
        ).eval()

    # Load transformer
    if transformer_type == "WanTransformer3DModel":
        print(f"🔧 Loading WanTransformer3DModel (base model)...")
        transformer3d = WanTransformer3DModel.from_pretrained(
            os.path.join(args.pretrained_model_name_or_path, transformer_additional_kwargs.get('transformer_subpath', 'transformer')),
            transformer_additional_kwargs=transformer_additional_kwargs,
            torch_dtype=weight_dtype,
        )
    elif transformer_type == "WanTransformer3DModelWithConcat":
        print(f"🔧 Loading WanTransformer3DModelWithConcat with {args.condition_channels} condition channels...")
        transformer3d = WanTransformer3DModelWithConcat.from_pretrained(
            os.path.join(args.pretrained_model_name_or_path, transformer_additional_kwargs.get('transformer_subpath', 'transformer')),
            transformer_additional_kwargs={
                "condition_channels": args.condition_channels,
                "is_wan2_2": is_wan2_2,
                **transformer_additional_kwargs,
            },
            torch_dtype=weight_dtype,
        )
    elif transformer_type == "WanTransformer3DVace":
        # Get VACE-specific configuration
        vace_layers = transformer_config.get('vace_layers', None)  # e.g., [0, 2, 4, 6, ...] or None for default
        vace_in_dim = transformer_config.get('vace_in_dim', 16)  # Hand latent channels

        print(f"🔧 Loading WanTransformer3DVace with VACE conditioning...")
        print(f"   vace_layers: {vace_layers if vace_layers else 'default (every 2nd layer)'}")
        print(f"   vace_in_dim: {vace_in_dim}")

        transformer3d = WanTransformer3DVace.from_pretrained(
            os.path.join(args.pretrained_model_name_or_path, transformer_additional_kwargs.get('transformer_subpath', 'transformer')),
            transformer_additional_kwargs={
                "vace_layers": vace_layers,
                "vace_in_dim": vace_in_dim,
                "is_wan2_2": is_wan2_2,
                **transformer_additional_kwargs,
            },
            torch_dtype=weight_dtype,
        )
    else:
        raise ValueError(f"Unsupported transformer type: {transformer_type}")

    # Ensure VACE modules are in the correct dtype (they're not loaded from checkpoint)
    if transformer_type == "WanTransformer3DVace":
        if hasattr(transformer3d, 'vace_blocks'):
            for block in transformer3d.vace_blocks:
                block.to(weight_dtype)
            print(f"✅ Converted vace_blocks to {weight_dtype}")
        if hasattr(transformer3d, 'vace_patch_embedding'):
            transformer3d.vace_patch_embedding.to(weight_dtype)
            print(f"✅ Converted vace_patch_embedding to {weight_dtype}")

    # Load custom weights if provided
    if args.transformer_path is not None:
        print(f"Loading transformer from: {args.transformer_path}")
        if args.transformer_path.endswith("safetensors"):
            state_dict = load_file(args.transformer_path)
        else:
            state_dict = torch.load(args.transformer_path, map_location="cpu")
        state_dict = state_dict.get("state_dict", state_dict)
        m, u = transformer3d.load_state_dict(state_dict, strict=False)
        print(f"Missing keys: {len(m)}, Unexpected keys: {len(u)}")

    if args.vae_path is not None:
        print(f"Loading VAE from: {args.vae_path}")
        if args.vae_path.endswith("safetensors"):
            state_dict = load_file(args.vae_path)
        else:
            state_dict = torch.load(args.vae_path, map_location="cpu")
        state_dict = state_dict.get("state_dict", state_dict)
        m, u = vae.load_state_dict(state_dict, strict=False)
        print(f"Missing keys: {len(m)}, Unexpected keys: {len(u)}")

    # ========== Setup Training Mode ==========
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    if clip_image_encoder is not None:
        clip_image_encoder.requires_grad_(False)
    transformer3d.requires_grad_(False)

    network = None
    trainable_params = None

    # Get trainable_parameter_patterns from config (for non-LoRA trainable params like patch_embedding)
    trainable_parameter_patterns = training_config.get("trainable_parameter_patterns", [])
    
    if args.train_mode == "lora":
        print("🔧 Setting up LoRA training...")
        
        # Create LoRA network using VideoX-Fun style
        network = create_network(
            1.0,
            args.rank,
            args.network_alpha,
            text_encoder,
            transformer3d,
            neuron_dropout=None,
            skip_name=args.lora_skip_name,
        )
        network.apply_to(text_encoder, transformer3d, args.train_text_encoder, True)
        
        # Enable additional trainable parameters (e.g., patch_embedding for channel expansion)
        non_lora_trainable_params = []
        if trainable_parameter_patterns:
            print(f"📋 Trainable parameter patterns: {trainable_parameter_patterns}")
            for name, param in transformer3d.named_parameters():
                for pattern in trainable_parameter_patterns:
                    if re.search(pattern, name):
                        param.requires_grad = True
                        non_lora_trainable_params.append(param)
                        print(f"   ✓ Enabled: {name}")
                        break
        
        trainable_params = list(filter(lambda p: p.requires_grad, network.parameters()))
        trainable_params_optim = network.prepare_optimizer_params(
            args.learning_rate / 2, args.learning_rate, args.learning_rate
        )
        
        # Add non-LoRA trainable params to optimizer
        if non_lora_trainable_params:
            trainable_params_optim.append({
                'params': non_lora_trainable_params,
                'lr': args.learning_rate,
            })
            trainable_params.extend(non_lora_trainable_params)
            print(f"📊 Non-LoRA trainable parameters: {sum(p.numel() for p in non_lora_trainable_params):,}")
        
        print(f"📊 LoRA trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    elif args.train_mode == "sft":
        print("🔧 Setting up SFT training...")
        
        transformer3d.train()
        
        # Enable specific modules for training
        if args.trainable_modules:
            for name, param in transformer3d.named_parameters():
                for trainable_module_name in args.trainable_modules + args.trainable_modules_low_learning_rate:
                    if trainable_module_name in name:
                        param.requires_grad = True
                        break
        else:
            # Full fine-tuning
            transformer3d.requires_grad_(True)
        
        trainable_params = list(filter(lambda p: p.requires_grad, transformer3d.parameters()))
        
        # Setup optimizer params with different learning rates
        trainable_params_optim = [
            {'params': [], 'lr': args.learning_rate},
            {'params': [], 'lr': args.learning_rate / 2},
        ]
        in_already = []
        for name, param in transformer3d.named_parameters():
            if not param.requires_grad or name in in_already:
                continue
            
            high_lr = False
            if args.trainable_modules:
                for module_name in args.trainable_modules:
                    if module_name in name:
                        trainable_params_optim[0]['params'].append(param)
                        in_already.append(name)
                        high_lr = True
                        break
            
            if not high_lr:
                for module_name in args.trainable_modules_low_learning_rate:
                    if module_name in name:
                        trainable_params_optim[1]['params'].append(param)
                        in_already.append(name)
                        break
                else:
                    # Default to high learning rate if no match
                    if not args.trainable_modules:
                        trainable_params_optim[0]['params'].append(param)
        
        print(f"📊 SFT trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    elif args.train_mode == "vace":
        print("🔧 Setting up VACE training (backbone frozen, only vace_blocks + vace_patch_embedding trainable)...")

        # Ensure we're using WanTransformer3DVace
        if transformer_type != "WanTransformer3DVace":
            raise ValueError(f"VACE training mode requires WanTransformer3DVace, but got {transformer_type}")

        # Freeze everything first (already done above, but be explicit)
        transformer3d.requires_grad_(False)

        # Enable only VACE-specific modules
        vace_trainable_params = []
        for name, param in transformer3d.named_parameters():
            if "vace_blocks" in name or "vace_patch_embedding" in name:
                param.requires_grad = True
                vace_trainable_params.append(param)
                print(f"   ✓ Enabled: {name}")

        if not vace_trainable_params:
            raise ValueError("No VACE parameters found to train! Check if transformer has vace_blocks and vace_patch_embedding.")

        trainable_params = vace_trainable_params
        trainable_params_optim = [
            {'params': vace_trainable_params, 'lr': args.learning_rate},
        ]

        print(f"📊 VACE trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    # Enable gradient checkpointing (from args or config)
    gradient_checkpointing = args.gradient_checkpointing or training_config.get("custom_settings", {}).get("gradient_checkpointing", False)
    if gradient_checkpointing:
        transformer3d.enable_gradient_checkpointing()
        print("✅ Gradient checkpointing enabled")

    # Enable TF32
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # ========== Setup Optimizer ==========
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
        except ImportError:
            raise ImportError("Please install bitsandbytes: pip install bitsandbytes")
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        trainable_params_optim,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-08,
    )

    # ========== Setup Learning Rate Scheduler ==========
    use_deepspeed_scheduler = (
        accelerator.state.deepspeed_plugin is not None
        and "scheduler" in accelerator.state.deepspeed_plugin.deepspeed_config
    )

    if use_deepspeed_scheduler:
        lr_scheduler = DummyScheduler(
            name=training_config.get("lr_scheduler", "cosine"),
            optimizer=optimizer,
            total_num_steps=args.max_train_steps * accelerator.num_processes,
            num_warmup_steps=training_config.get("lr_warmup_steps", 0) * accelerator.num_processes,
        )
    else:
        lr_scheduler = get_scheduler(
            training_config.get("lr_scheduler", "cosine"),
            optimizer=optimizer,
            num_warmup_steps=training_config.get("lr_warmup_steps", 0) * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
            num_cycles=training_config.get("lr_num_cycles", 1),
        )

    # ========== Setup Dataset ==========
    print("🔧 Setting up dataset...")
    
    # Use is_wan2_2 already defined above (for width alignment to 32)
    align_width_to_32 = is_wan2_2
    if align_width_to_32:
        print("📐 WAN 2.2 detected: Aligning width to multiples of 32 for compatibility")
    
    # Setup dataset and dataloader
    # Read load_tensors from training.custom_settings (same as cogvideox pattern)
    load_tensors = training_config.get("custom_settings", {}).get("load_tensors", False)
    if load_tensors:
        print("📦 Using pre-encoded latents (load_tensors=True)")

    dataset_init_kwargs = {
        "data_root": data_config["data_root"],
        "dataset_file": data_config["dataset_file"],
        "max_num_frames": data_config.get("max_num_frames", 49),
        "load_tensors": load_tensors,
        "random_flip": data_config.get("random_flip", False),
        "height_buckets": data_config.get("height_buckets", 480),
        "width_buckets": data_config.get("width_buckets", 720),
        "frame_buckets": data_config.get("frame_buckets", 49),
        "prompt_subdir": data_config.get("prompt_subdir", "prompts"),
        "prompt_embeds_subdir": data_config.get("prompt_embeds_subdir", "prompt_embeds_ego_fun_rewrite_wan"),
        "hand_video_subdir": data_config.get("hand_video_subdir", "videos_hands"),
        "hand_video_latents_subdir": data_config.get("hand_video_latents_subdir", "hand_video_latents_wan"),
        "video_latents_subdir": data_config.get("video_latents_subdir", "video_latents_wan"),
        "static_video_latents_subdir": data_config.get("static_video_latents_subdir", "static_video_latents_wan"),
        "align_width_to_32": align_width_to_32,
    }
    
    train_dataset = VideoDatasetWithConditionsAndResizing(**dataset_init_kwargs)

    def collate_fn(batch):
        def stack_or_none(key: str):
            if key in batch[0] and batch[0][key] is not None:
                first_tensor = batch[0][key]
                stacked_items = []
                for item in batch:
                    value = item.get(key)
                    if value is None:
                        stacked_items.append(torch.zeros_like(first_tensor))
                    else:
                        stacked_items.append(value)
                return torch.stack(stacked_items)
            return None

        collated = {
            "videos": torch.stack([item["video"] for item in batch]),
            "prompts": (
                torch.stack([item["prompt"] for item in batch])
                if isinstance(batch[0]["prompt"], torch.Tensor)
                else [item["prompt"] for item in batch]
            ),
            "hand_videos": stack_or_none("hand_videos"),
            "static_videos": stack_or_none("static_videos"),
            "masks": stack_or_none("masks"),
        }

        return collated
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_config["batch_size"],
        shuffle=True,
        drop_last=True,  # Drop last incomplete batch to ensure consistent batch size
        collate_fn=collate_fn,
        num_workers=data_config.get("dataloader_num_workers", 0),
        pin_memory=data_config.get("pin_memory", True),
    )

    # ========== Prepare with Accelerator ==========
    use_deepspeed_or_fsdp = (fsdp_stage != 0) or (zero_stage != 0)
    
    if args.train_mode == "lora":
        if use_deepspeed_or_fsdp:
            # DeepSpeed/FSDP: Attach network to transformer so DeepSpeed can find all params
            transformer3d.network = network
            transformer3d = transformer3d.to(weight_dtype)
            transformer3d, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                transformer3d, optimizer, train_dataloader, lr_scheduler
            )
            print(f"📦 Prepared transformer3d with attached LoRA network (DeepSpeed/FSDP mode)")
        else:
            # No DeepSpeed/FSDP: Just prepare network
            network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                network, optimizer, train_dataloader, lr_scheduler
            )
    else:
        transformer3d, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            transformer3d, optimizer, train_dataloader, lr_scheduler
        )

    # Move models to device (skip transformer3d if already prepared with DeepSpeed/FSDP)
    if not load_tensors:
        # Only keep VAE on device when encoding videos during training
        vae.to(accelerator.device, dtype=weight_dtype)
    if not (args.train_mode == "lora" and use_deepspeed_or_fsdp):
        transformer3d.to(accelerator.device, dtype=weight_dtype)
    if not load_tensors:
        # Only keep text_encoder on device when encoding prompts during training
        text_encoder.to(accelerator.device)
    if clip_image_encoder is not None:
        clip_image_encoder.to(accelerator.device, dtype=weight_dtype)

    # Free up memory when using pre-encoded latents
    if load_tensors:
        logger.info("📦 load_tensors=True: Deleting VAE and text_encoder to save memory")
        del vae
        del text_encoder
        del tokenizer
        vae = None
        text_encoder = None
        tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(accelerator.device)
        logger.info("   VAE and text_encoder deleted. GPU memory freed.")

    # Recalculate training steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # ========== Initialize Trackers (WandB) ==========
    if accelerator.is_main_process:
        # Create run name: {date}_{name}_{mode}_{job_id}
        slurm_job_id = os.environ.get("SLURM_JOB_ID", "local")
        run_name = f"{date_suffix}_{exp_name}_{args.mode}_{slurm_job_id}"
        
        # Prepare tracker config
        tracker_config = {
            "experiment": {
                "name": exp_name,
                "date": exp_date if 'exp_date' in dir() else "unknown",
                "output_dir": args.output_dir,
            },
            "training": {
                "mode": args.train_mode,
                "learning_rate": args.learning_rate,
                "batch_size": args.train_batch_size,
                "max_train_steps": args.max_train_steps,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
            },
            "model": {
                "base_model_name_or_path": args.pretrained_model_name_or_path,
                "condition_channels": args.condition_channels,
            },
            "system": {
                "num_gpus": accelerator.num_processes,
                "mixed_precision": args.mixed_precision,
            }
        }
        
        # Setup wandb init kwargs
        wandb_init_kwargs = {
            "name": run_name,
        }
        if args.tracker_entity_name:
            wandb_init_kwargs["entity"] = args.tracker_entity_name
        
        # Set wandb to offline mode for debug and slurm_test modes; disabled for batch
        if args.mode in ["debug", "slurm_test"]:
            wandb_init_kwargs["mode"] = "offline"
            print(f"🔧 WandB set to offline mode for {args.mode}")
        elif args.mode == "batch":
            wandb_init_kwargs["mode"] = "disabled"
            print(f"🔧 Batch mode: WandB disabled")
        
        accelerator.init_trackers(
            project_name=args.tracker_project_name,
            config=tracker_config,
            init_kwargs={"wandb": wandb_init_kwargs}
        )
        
        print(f"🔗 Experiment initialized: {run_name}")

    # ========== Training Loop ==========
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    first_epoch = 0
    initial_global_step = 0

    # Resume from checkpoint (supports both config-based and arg-based)
    checkpoint_to_resume = resume_from_checkpoint if 'resume_from_checkpoint' in dir() and resume_from_checkpoint else args.resume_from_checkpoint
    
    if checkpoint_to_resume:
        if checkpoint_to_resume != "latest":
            # Use the provided checkpoint path directly
            checkpoint_path = checkpoint_to_resume
            
            if not os.path.exists(checkpoint_path):
                accelerator.print(f"Checkpoint '{checkpoint_path}' does not exist. Starting a new training run.")
                checkpoint_to_resume = None
                initial_global_step = 0
            else:
                accelerator.print(f"Resuming from checkpoint {checkpoint_path}")
                
                # Try to load full accelerator state first, fallback to weight-only loading if it fails
                try:
                    accelerator.load_state(checkpoint_path)
                    accelerator.print(f"Successfully loaded full accelerator state from {checkpoint_path}")
                except Exception as e:
                    accelerator.print(f"Failed to load full accelerator state: {e}")
                    accelerator.print(f"Falling back to weight-only loading...")
                    
                    if args.train_mode == "lora":
                        # Load LoRA weights
                        lora_path = os.path.join(checkpoint_path, "lora_diffusion_pytorch_model.safetensors")
                        if os.path.exists(lora_path):
                            state_dict = load_file(lora_path, device=str(accelerator.device))
                            m, u = accelerator.unwrap_model(network).load_state_dict(state_dict, strict=False)
                            print(f"Loaded LoRA: missing {len(m)}, unexpected {len(u)}")
                        
                        # Load non-LoRA trainable weights (e.g., patch_embedding)
                        non_lora_path = os.path.join(checkpoint_path, "non_lora_weights.safetensors")
                        if os.path.exists(non_lora_path):
                            non_lora_state_dict = load_file(non_lora_path, device=str(accelerator.device))
                            m, u = accelerator.unwrap_model(transformer3d).load_state_dict(non_lora_state_dict, strict=False)
                            print(f"Loaded non-LoRA weights: missing {len(m)}, unexpected {len(u)}")
                    elif args.train_mode == "vace":
                        # Load VACE weights (vace_blocks + vace_patch_embedding)
                        vace_path = os.path.join(checkpoint_path, "vace_weights.safetensors")
                        if os.path.exists(vace_path):
                            vace_state_dict = load_file(vace_path, device=str(accelerator.device))
                            m, u = accelerator.unwrap_model(transformer3d).load_state_dict(vace_state_dict, strict=False)
                            print(f"Loaded VACE weights: missing {len(m)}, unexpected {len(u)}")
                    else:
                        # Load transformer weights for SFT
                        transformer_path = os.path.join(checkpoint_path, "transformer")
                        if os.path.exists(transformer_path):
                            state_dict = WanTransformer3DModelWithConcat.from_pretrained(transformer_path).state_dict()
                            m, u = accelerator.unwrap_model(transformer3d).load_state_dict(state_dict, strict=False)
                            print(f"Loaded transformer: missing {len(m)}, unexpected {len(u)}")
                
                # Extract global step from checkpoint name
                checkpoint_name = os.path.basename(checkpoint_path.rstrip('/'))
                if checkpoint_name.startswith("checkpoint-"):
                    global_step = int(checkpoint_name.split("-")[1])
                    initial_global_step = global_step
                    first_epoch = global_step // num_update_steps_per_epoch
                else:
                    accelerator.print(f"Warning: Could not extract global step from checkpoint name '{checkpoint_name}'")
                    initial_global_step = 0
        else:
            # Get the most recent checkpoint from output_dir
            if not os.path.exists(args.output_dir):
                accelerator.print(f"Output directory '{args.output_dir}' does not exist. Starting a new training run.")
                checkpoint_to_resume = None
                initial_global_step = 0
            else:
                dirs = os.listdir(args.output_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                if len(dirs) == 0:
                    accelerator.print("No checkpoint directories found. Starting a new training run.")
                    checkpoint_to_resume = None
                    initial_global_step = 0
                else:
                    dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                    latest_checkpoint = dirs[-1]
                    checkpoint_path = os.path.join(args.output_dir, latest_checkpoint)
                    accelerator.print(f"Resuming from latest checkpoint {checkpoint_path}")
                    
                    # Try to load full accelerator state first, fallback to weight-only loading if it fails
                    try:
                        accelerator.load_state(checkpoint_path)
                        accelerator.print(f"Successfully loaded full accelerator state from {checkpoint_path}")
                    except Exception as e:
                        accelerator.print(f"Failed to load full accelerator state: {e}")
                        accelerator.print(f"Falling back to weight-only loading...")
                        
                        if args.train_mode == "lora":
                            lora_path = os.path.join(checkpoint_path, "lora_diffusion_pytorch_model.safetensors")
                            if os.path.exists(lora_path):
                                state_dict = load_file(lora_path, device=str(accelerator.device))
                                m, u = accelerator.unwrap_model(network).load_state_dict(state_dict, strict=False)
                                print(f"Loaded LoRA: missing {len(m)}, unexpected {len(u)}")
                            
                            # Load non-LoRA trainable weights (e.g., patch_embedding)
                            non_lora_path = os.path.join(checkpoint_path, "non_lora_weights.safetensors")
                            if os.path.exists(non_lora_path):
                                non_lora_state_dict = load_file(non_lora_path, device=str(accelerator.device))
                                m, u = accelerator.unwrap_model(transformer3d).load_state_dict(non_lora_state_dict, strict=False)
                                print(f"Loaded non-LoRA weights: missing {len(m)}, unexpected {len(u)}")
                        elif args.train_mode == "vace":
                            # Load VACE weights (vace_blocks + vace_patch_embedding)
                            vace_path = os.path.join(checkpoint_path, "vace_weights.safetensors")
                            if os.path.exists(vace_path):
                                vace_state_dict = load_file(vace_path, device=str(accelerator.device))
                                m, u = accelerator.unwrap_model(transformer3d).load_state_dict(vace_state_dict, strict=False)
                                print(f"Loaded VACE weights: missing {len(m)}, unexpected {len(u)}")
                        else:
                            # Load transformer weights for SFT
                            transformer_path = os.path.join(checkpoint_path, "transformer")
                            if os.path.exists(transformer_path):
                                state_dict = WanTransformer3DModelWithConcat.from_pretrained(transformer_path).state_dict()
                                m, u = accelerator.unwrap_model(transformer3d).load_state_dict(state_dict, strict=False)
                                print(f"Loaded transformer weights: missing {len(m)}, unexpected {len(u)}")
                    
                    global_step = int(latest_checkpoint.split("-")[1])
                    initial_global_step = global_step
                    first_epoch = global_step // num_update_steps_per_epoch

    # Get step intervals from config (with args as fallback)
    checkpointing_steps = training_config.get("custom_settings", {}).get("checkpointing_steps", args.checkpointing_steps)
    validation_steps = data_config.get("validation_steps", args.validation_steps)
    init_validation_steps = data_config.get("init_validation_steps", 100)
    
    logger.info(f"📊 Step intervals: checkpointing={checkpointing_steps}, validation={validation_steps}, init_validation={init_validation_steps}")

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    # Initialize DiscreteSampling for uniform timestep sampling
    idx_sampling = DiscreteSampling(args.train_sampling_steps, uniform_sampling=args.uniform_sampling)

    # Get patch_size from transformer config (used for seq_len calculation)
    patch_size = accelerator.unwrap_model(transformer3d).config.patch_size

    # VAE mini-batch encoding for memory efficiency
    vae_mini_batch = getattr(args, 'vae_mini_batch', 32)
    
    def _batch_encode_vae(pixel_values):
        """Encode pixel values to latents in mini-batches to save memory."""
        bs = vae_mini_batch
        new_latents = []
        for i in range(0, pixel_values.shape[0], bs):
            pixel_values_bs = pixel_values[i:i + bs]
            latents_bs = vae.encode(pixel_values_bs)[0].sample()
            new_latents.append(latents_bs)
        return torch.cat(new_latents, dim=0)

    # Run initial validation at step 0 (before training starts)
    # Run on all GPUs to distribute validation videos (not just main process)
    if data_config.get("validation_set") is not None:
        if accelerator.is_main_process:
            logger.info("🎬 Running initial validation at step 0 (before training starts)")
        log_validation(
            vae, text_encoder, tokenizer, clip_image_encoder,
            transformer3d, network, config, args, accelerator,
            weight_dtype, global_step=global_step,
            load_tensors=load_tensors, vae_path=vae_path, vae_kwargs=vae_kwargs,
            text_encoder_path=text_encoder_path, text_encoder_kwargs=text_encoder_kwargs,
            tokenizer_path=tokenizer_path,
        )

    requires_hand_videos = transformer_type in ["WanTransformer3DModelWithConcat", "WanTransformer3DVace"]
    pretrain_mode = training_config.get("pretrain_mode", "t2v")

    # Time-based checkpoint saving (47h45m); only when slurm and partition is not h100
    training_start_time = time.time()
    time_based_checkpoint_saved = False
    TIME_LIMIT_SECONDS = 47 * 3600 + 45 * 60  # 47 hours 45 minutes
    enable_time_limit_checkpoint = (
        args.mode == "slurm"
        and os.environ.get("SLURM_JOB_PARTITION", "") != "h100"
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [network] if args.train_mode == "lora" else [transformer3d]

            with accelerator.accumulate(models_to_accumulate):
                # Get batch data
                videos = batch["videos"].to(accelerator.device, dtype=weight_dtype)
                prompts = batch["prompts"]
                hand_videos = batch.get("hand_videos")
                static_videos = batch.get("static_videos")

                # ===== Encode videos to latents =====
                if load_tensors:
                    # Pre-encoded latents: already [B, C, F', H', W'] from dataset
                    # encode_with_wan.py already called .sample(), no need for VAE
                    latents = videos  # Already latents
                else:
                    # Raw videos: [B, F, C, H, W] -> encode to latents
                    videos = rearrange(videos, "b f c h w -> b c f h w")
                    with torch.no_grad():
                        latents = _batch_encode_vae(videos)

                # ===== Encode condition videos =====
                dropout_mask = None  # Initialize dropout_mask
                if requires_hand_videos and hand_videos is not None:
                    if load_tensors:
                        # Pre-encoded hand latents: already [B, C, F', H', W']
                        hand_latents = hand_videos.to(accelerator.device, dtype=weight_dtype)
                    else:
                        # Raw hand videos: encode to latents
                        hand_videos = hand_videos.to(accelerator.device, dtype=weight_dtype)
                        hand_videos = rearrange(hand_videos, "b f c h w -> b c f h w")
                        with torch.no_grad():
                            hand_latents = _batch_encode_vae(hand_videos)

                    # Apply hand dropout for classifier-free guidance training
                    # This helps the model learn to generate without hand information
                    hand_dropout_prob = training_config.get("hand_dropout_prob", 0.0)
                    if hand_dropout_prob > 0:
                        # Get batch size from hand_latents shape
                        hand_batch_size = hand_latents.shape[0]
                        # Sample dropout mask: each sample in batch has independent dropout probability
                        dropout_mask = torch.rand(hand_batch_size, device=accelerator.device) < hand_dropout_prob

                        if dropout_mask.any():
                            # Zero out hand latents for dropped samples
                            # hand_latents shape: [B, 16, F, H, W]
                            hand_latents[dropout_mask] = torch.zeros_like(hand_latents[dropout_mask])

                            # Log dropout statistics (only occasionally to avoid spam)
                            if global_step % 100 == 0:
                                num_dropped = dropout_mask.sum().item()
                                accelerator.log({"hand_dropout_count": num_dropped}, step=global_step)
                else:
                    hand_latents = None

                if static_videos is not None:
                    if load_tensors:
                        # Pre-encoded static latents: already [B, C, F', H', W']
                        static_latents = static_videos.to(accelerator.device, dtype=weight_dtype)
                        bs, _, latent_frames, latent_height, latent_width = static_latents.size()

                        # Create mask directly in latent space (4 channels for WAN Fun format)
                        # mask_latents shape: [B, 4, F', H', W']
                        if transformer_type == "WanTransformer3DModel":
                            if pretrain_mode == "v2v":
                                mask_latents = torch.ones(bs, 4, latent_frames, latent_height, latent_width,
                                                        device=accelerator.device, dtype=weight_dtype)
                            elif pretrain_mode == "i2v":
                                mask_latents = torch.zeros(bs, 4, latent_frames, latent_height, latent_width,
                                                        device=accelerator.device, dtype=weight_dtype)
                                mask_latents[:, :, 0:1, :, :] = 1.0
                            elif pretrain_mode == "t2v":
                                mask_latents = torch.zeros(bs, 4, latent_frames, latent_height, latent_width,
                                                        device=accelerator.device, dtype=weight_dtype)
                        else:
                            # WithConcat / Vace: full conditioning (v2v-style, all ones)
                            mask_latents = torch.ones(bs, 4, latent_frames, latent_height, latent_width,
                                                    device=accelerator.device, dtype=weight_dtype)

                        # Create inpaint-style input for WAN Fun (y argument)
                        inpaint_latents = torch.cat([mask_latents, static_latents], dim=1)  # [b, 4+16=20, f, h, w]
                    else:
                        # Raw static videos: encode to latents
                        # static_videos is [B, F, C, H, W] before rearrange
                        bs, num_frames, _, height, width = static_videos.size()

                        # Create mask in pixel space: pretrain_mode (t2v/v2v/i2v) only for WanTransformer3DModel
                        if transformer_type == "WanTransformer3DModel":
                            if pretrain_mode == "v2v":
                                mask = torch.ones(latents.shape[0], 1, num_frames, height, width,
                                                device=accelerator.device, dtype=weight_dtype)
                            elif pretrain_mode == "i2v":
                                mask = torch.zeros(latents.shape[0], 1, num_frames, height, width,
                                                device=accelerator.device, dtype=weight_dtype)
                                mask[:, :, 0:1, :, :] = 1.0
                            elif pretrain_mode == "t2v":
                                mask = torch.zeros(latents.shape[0], 1, num_frames, height, width,
                                                device=accelerator.device, dtype=weight_dtype)
                        else:
                            # WithConcat / Vace: full conditioning (v2v-style)
                            mask = torch.ones(latents.shape[0], 1, num_frames, height, width,
                                            device=accelerator.device, dtype=weight_dtype)

                        # Encode static videos to latents
                        static_videos = static_videos.to(accelerator.device, dtype=weight_dtype)
                        static_videos = rearrange(static_videos, "b f c h w -> b c f h w")
                        static_videos = static_videos * mask
                        with torch.no_grad():
                            static_latents = _batch_encode_vae(static_videos)

                        # Convert mask from pixel space to latent space
                        # Channel layout for WanTransformer3DModelWithConcat:
                        # y: inpaint_latents = mask + static_latents [b, 4+16=20, f, h, w] (WAN Fun format)
                        mask = torch.concat(
                            [
                                torch.repeat_interleave(mask[:, :, 0:1], repeats=4, dim=2),
                                mask[:, :, 1:]
                            ], dim=2
                        )
                        mask = mask.view(bs, mask.shape[2] // 4, 4, height, width)
                        mask = mask.transpose(1, 2)
                        mask_latents = resize_mask(mask, latents, process_first_frame_only=True)

                        # Create inpaint-style input for WAN Fun (y argument)
                        inpaint_latents = torch.cat([mask_latents, static_latents], dim=1)  # [b, 4+16=20, f, h, w]
                else:
                    inpaint_latents = None

                # Encode prompts
                with torch.no_grad():
                    if load_tensors:
                        # Pre-encoded embeddings from dataset
                        # prompts is a list of tensors [seq_len_i, dim] from collate (variable lengths)
                        # or a stacked tensor [B, seq_len, dim] if same lengths
                        if isinstance(prompts, torch.Tensor):
                            # If stacked tensor, convert to list for WAN transformer
                            prompt_embeds = [p.to(accelerator.device, dtype=weight_dtype) for p in prompts]
                        else:
                            # Already a list of tensors
                            prompt_embeds = [p.to(accelerator.device, dtype=weight_dtype) for p in prompts]
                    elif isinstance(prompts, list):
                        prompt_ids = tokenizer(
                            prompts,
                            padding="max_length",
                            max_length=args.tokenizer_max_length,
                            truncation=True,
                            add_special_tokens=True,
                            return_tensors="pt"
                        )
                        text_input_ids = prompt_ids.input_ids.to(latents.device)
                        prompt_attention_mask = prompt_ids.attention_mask.to(latents.device)
                        seq_lens = prompt_attention_mask.gt(0).sum(dim=1).long()
                        prompt_embeds = text_encoder(text_input_ids, attention_mask=prompt_attention_mask)[0]
                        prompt_embeds = [u[:v].to(weight_dtype) for u, v in zip(prompt_embeds, seq_lens)]
                    else:
                        prompt_embeds = prompts.to(dtype=weight_dtype)

                # Sample noise
                bsz, channel, num_frames, height, width = latents.size()
                noise = torch.randn(latents.size(), device=latents.device, generator=torch_rng, dtype=weight_dtype)

                # Sample timesteps
                if not args.uniform_sampling:
                    # Use density-based sampling (weighting scheme based)
                    u = compute_density_for_timestep_sampling(
                        weighting_scheme=args.weighting_scheme,
                        batch_size=bsz,
                        logit_mean=args.logit_mean,
                        logit_std=args.logit_std,
                        mode_scale=args.mode_scale,
                    )
                    indices = (u * noise_scheduler.config.num_train_timesteps).long()
                else:
                    # Use DiscreteSampling for uniform sampling (supports distributed training)
                    indices = idx_sampling(bsz, generator=torch_rng, device=latents.device)
                    indices = indices.long().cpu()

                timesteps = noise_scheduler.timesteps[indices.cpu()].to(device=latents.device)

                # Get sigmas and create noisy latents (Flow Matching)
                sigmas = get_sigmas(noise_scheduler, timesteps, n_dim=latents.ndim, dtype=latents.dtype, device=accelerator.device)
                noisy_latents = (1.0 - sigmas) * latents + sigmas * noise

                # Flow Matching target
                # If hand_latents were dropped out, use static_latents as target instead of full latents
                # This teaches the model to generate static-only content when hand info is missing
                if dropout_mask is not None and dropout_mask.any() and static_latents is not None:
                    # For dropped samples, use static_latents as target
                    # target shape: [B, 16, F, H, W]
                    target = noise - latents.clone()
                    # Replace target for dropped samples with static-based target
                    target[dropout_mask] = noise[dropout_mask] - static_latents[dropout_mask]
                else:
                    target = noise - latents

                # Calculate seq_len for transformer (based on patched dimensions)
                seq_len = math.ceil(
                    (height / patch_size[1]) * (width / patch_size[2]) * (num_frames / patch_size[0])
                )

                # Forward pass
                # x: noisy_latents [B, 16, F, H, W]
                # y: inpaint_latents [B, 17, F, H, W] (mask + static)
                # condition_latents: hand_latents [B, 16, F, H, W] (extra conditions, only for WithConcat)
                #
                # WanTransformer3DModelWithConcat.forward handles conversion to list format
                # and concatenation of condition_latents
                with torch.amp.autocast(device_type='cuda', dtype=weight_dtype):
                    if transformer_type == "WanTransformer3DModel":
                        # Base model: no condition_latents
                        noise_pred = transformer3d(
                            x=noisy_latents,
                            t=timesteps,
                            context=prompt_embeds,
                            seq_len=seq_len,
                            y=torch.zeros_like(inpaint_latents) if pretrain_mode == "t2v" else inpaint_latents,
                            clip_fea=None,  # Can add CLIP features later
                        )
                    elif transformer_type == "WanTransformer3DModelWithConcat":
                        # Concat model: with condition_latents
                        noise_pred = transformer3d(
                            x=noisy_latents,
                            t=timesteps,
                            context=prompt_embeds,
                            seq_len=seq_len,
                            y=inpaint_latents,
                            clip_fea=None,  # Can add CLIP features later
                            condition_latents=hand_latents,  # Extra condition channels
                        )
                    elif transformer_type == "WanTransformer3DVace":
                        # VACE model: hand condition via vace_context (list format)
                        # Convert hand_latents to list format for VACE
                        if hand_latents is not None:
                            vace_context = [hand_latents[i] for i in range(hand_latents.shape[0])]
                        else:
                            vace_context = None

                        # Get vace_context_scale from config (default 1.0)
                        vace_context_scale = training_config.get("vace_context_scale", 1.0)

                        noise_pred = transformer3d(
                            x=noisy_latents,
                            t=timesteps,
                            context=prompt_embeds,
                            seq_len=seq_len,
                            y=inpaint_latents,
                            clip_fea=None,  # Can add CLIP features later
                            vace_context=vace_context,
                            vace_context_scale=vace_context_scale,
                        )
                    else:
                        raise ValueError(f"Unsupported transformer type: {transformer_type}")

                # Compute loss with weighting
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                loss = custom_mse_loss(noise_pred.float(), target.float(), weighting.float())
                loss = loss.mean()

                # Gather losses
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backward
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    if args.train_mode == "lora":
                        accelerator.clip_grad_norm_(network.parameters(), args.max_grad_norm)
                    else:
                        accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Update progress
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                # Save checkpoint (step interval or time limit; time limit only when slurm and partition != h100)
                elapsed_time = time.time() - training_start_time
                should_save_time_based_checkpoint = (
                    enable_time_limit_checkpoint
                    and not time_based_checkpoint_saved
                    and elapsed_time >= TIME_LIMIT_SECONDS
                )
                if should_save_time_based_checkpoint or global_step % checkpointing_steps == 0:
                    if accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED:
                        if should_save_time_based_checkpoint:
                            time_based_checkpoint_saved = True
                            elapsed_hours = elapsed_time / 3600
                            logger.info(f"⏰ Time-based checkpoint trigger: {elapsed_hours:.2f} hours elapsed (limit: 47.75 hours)")
                        # Remove old checkpoints if limit is set (all processes wait)
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir) if os.path.exists(args.output_dir) else []
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                if accelerator.is_main_process:
                                    for removing_checkpoint in checkpoints[:num_to_remove]:
                                        removing_path = os.path.join(args.output_dir, removing_checkpoint)
                                        logger.info(f"Removing old checkpoint: {removing_path}")
                                        shutil.rmtree(removing_path)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        
                        # Save model weights
                        os.makedirs(save_path, exist_ok=True)
                        
                        if args.train_mode == "lora":
                            save_model_lora(save_path, network, weight_dtype, accelerator, transformer3d, trainable_parameter_patterns)
                        elif args.train_mode == "vace":
                            save_model_vace(save_path, transformer3d, accelerator)
                        else:
                            save_model_sft(save_path, transformer3d, accelerator)

                        logger.info(f"Saved checkpoint to {save_path}")

                        # Save full accelerator state
                        accelerator.save_state(save_path)
                        logger.info(f"Saved accelerator state to {save_path}")

                # Validation (runs if validation_set is defined in config)
                # Early phase: every init_validation_steps until first checkpoint; then every validation_steps (CogVideoX style)
                should_run_validation = data_config.get("validation_set") is not None and (
                    (global_step % init_validation_steps == 0 and global_step < checkpointing_steps)
                    or global_step % validation_steps == 0
                )
                if should_run_validation:
                    if accelerator.is_main_process:
                        logger.info(f"🎬 Running validation at step {global_step}")
                    log_validation(
                        vae, text_encoder, tokenizer, clip_image_encoder,
                        transformer3d, network, config, args, accelerator,
                        weight_dtype, global_step,
                        load_tensors=load_tensors, vae_path=vae_path, vae_kwargs=vae_kwargs,
                        text_encoder_path=text_encoder_path, text_encoder_kwargs=text_encoder_kwargs,
                        tokenizer_path=tokenizer_path,
                    )

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # Save final model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
        os.makedirs(save_path, exist_ok=True)

        if args.train_mode == "lora":
            save_model_lora(save_path, network, weight_dtype, accelerator, transformer3d, trainable_parameter_patterns)
        elif args.train_mode == "vace":
            save_model_vace(save_path, transformer3d, accelerator)
        else:
            save_model_sft(save_path, transformer3d, accelerator)

        logger.info(f"Training completed. Final checkpoint saved to {save_path}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
