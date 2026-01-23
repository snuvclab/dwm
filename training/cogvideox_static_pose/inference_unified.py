#!/usr/bin/env python3

import argparse
import os
import random
import json
import time
import yaml
import shutil
from typing import List, Optional, Dict, Any
import numpy as np
import PIL
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import imageio.v3 as iio
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging

from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXTransformer3DModel,
)
from cogvideox_fun_transformer_with_conditions import (
    CrossTransformer3DModel,
    CrossTransformer3DModelWithAdapter,
)
from diffusers.utils import export_to_video, convert_unet_state_dict_to_peft
from transformers import AutoTokenizer, T5EncoderModel
from peft import get_peft_model_state_dict, set_peft_model_state_dict, LoraConfig

# Import all pipeline types
from cogvideox_pose_concat_pipeline import CogVideoXPoseConcatPipeline
from cogvideox_pose_adapter_pipeline import CogVideoXPoseAdapterPipeline
from cogvideox_pose_adaln_pipeline import CogVideoXPoseAdaLNPipeline
from cogvideox_static_to_video_pose_concat_pipeline import CogVideoXStaticToVideoPipeline, CogVideoXStaticToVideoPoseConcatPipeline
from cogvideox_fun_static_to_video_pose_concat_pipeline import (
    CogVideoXFunStaticToVideoPipeline,
    CogVideoXFunStaticToVideoJointGenerationPipeline,
)

# Import video metrics
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_all(seed: int = 0) -> None:
    """Set random seeds of all components."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_experiment_config(config_path: str) -> Dict[str, Any]:
    """Load experiment configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def find_experiment_config(checkpoint_path: str) -> Optional[str]:
    """Find experiment config YAML file in checkpoint parent directory."""
    checkpoint_dir = Path(checkpoint_path)
    parent_dir = checkpoint_dir.parent
    
    # Look for YAML files in parent directory
    yaml_files = list(parent_dir.glob("*.yaml")) + list(parent_dir.glob("*.yml"))
    
    if yaml_files:
        # Return the first YAML file found
        return str(yaml_files[0])
    
    return None


def setup_lora_adapter(transformer, config: Dict[str, Any]):
    """Setup LoRA adapter for inference (similar to training setup)."""
    logger.info("🔧 Setting up LoRA adapter for inference...")
    
    # Get LoRA parameters from config
    lora_rank = config.get("training", {}).get("lora_rank", 64)
    lora_alpha = config.get("training", {}).get("lora_alpha", 64)
    
    logger.info(f"   LoRA Rank: {lora_rank}")
    logger.info(f"   LoRA Alpha: {lora_alpha}")
    
    # Add LoRA to attention layers (same config as training)
    transformer_lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        init_lora_weights=True,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    transformer.add_adapter(transformer_lora_config)
    
    logger.info("✅ LoRA adapter setup completed")


def load_checkpoint_with_config(pipeline, checkpoint_path: str, config: Dict[str, Any]):
    """Load checkpoint based on training mode from config."""
    # Allow running from base model without a checkpoint
    if checkpoint_path is None or str(checkpoint_path).strip() == "":
        logger.info("ℹ️  No checkpoint_path provided. Skipping checkpoint loading and using base model weights.")
        return
    training_mode = config.get("training", {}).get("mode", "full")
    pipeline_type = config.get("pipeline", {}).get("type", "cogvideox_pose_concat")
    
    logger.info(f"🔧 Loading checkpoint with training mode: {training_mode}")
    logger.info(f"🔧 Pipeline type: {pipeline_type}")
    
    if training_mode == "lora":
        # Load LoRA weights
        logger.info("🔧 Loading LoRA weights...")
        
        # Load LoRA state dict based on pipeline type
        if pipeline_type == "cogvideox_pose_concat":
            lora_state_dict = CogVideoXPoseConcatPipeline.lora_state_dict(checkpoint_path)
        elif pipeline_type == "cogvideox_pose_adapter":
            lora_state_dict = CogVideoXPoseAdapterPipeline.lora_state_dict(checkpoint_path)
        elif pipeline_type == "cogvideox_pose_adaln":
            lora_state_dict = CogVideoXPoseAdaLNPipeline.lora_state_dict(checkpoint_path)
        elif pipeline_type == "cogvideox_fun_static_to_video_pose_concat":
            lora_state_dict = CogVideoXFunStaticToVideoPipeline.lora_state_dict(checkpoint_path)
        elif pipeline_type == "cogvideox_fun_static_to_video_joint_generation":
            lora_state_dict = CogVideoXFunStaticToVideoJointGenerationPipeline.lora_state_dict(checkpoint_path)
        elif pipeline_type == "cogvideox_fun_static_to_video_cross":
            lora_state_dict = CogVideoXFunStaticToVideoCrossPipeline.lora_state_dict(checkpoint_path)
        elif pipeline_type == "cogvideox_fun_static_to_video_cross_pose_adapter":
            lora_state_dict = CogVideoXFunStaticToVideoCrossPipeline.lora_state_dict(checkpoint_path)
        elif pipeline_type == "cogvideox_fun_static_to_video_pose_adapter":
            lora_state_dict = CogVideoXFunStaticToVideoPipeline.lora_state_dict(checkpoint_path)
        elif pipeline_type == "cogvideox_static_to_video_pose_concat":
            lora_state_dict = CogVideoXStaticToVideoPoseConcatPipeline.lora_state_dict(checkpoint_path)
        else:
            raise ValueError(f"Unsupported pipeline type for LoRA loading: {pipeline_type}")
        
        # Convert and load LoRA weights
        transformer_state_dict = convert_unet_state_dict_to_peft(lora_state_dict)
        set_peft_model_state_dict(pipeline.transformer, transformer_state_dict, adapter_name="default")
        
        # Load non-LoRA weights (trainable parameters) if they exist
        non_lora_file = os.path.join(checkpoint_path, "non_lora_weights.pt")
        if os.path.exists(non_lora_file):
            non_lora_state_dict = torch.load(non_lora_file, map_location="cpu")
            model_state_dict = pipeline.transformer.state_dict()
            loaded_keys = []
            for name, param_data in non_lora_state_dict.items():
                if name in model_state_dict:
                    model_state_dict[name].copy_(param_data)
                    loaded_keys.append(name)
                else:
                    print(f"⚠️ {name} not found in model state dict")
            logger.info(f"✅ Loaded non-LoRA weights: {loaded_keys}")
        else:
            # Legacy support: Load projection layer weights if they exist
            if "concat" in pipeline_type:
                # Concat models: load proj weights (modify existing proj)
                projection_file = os.path.join(checkpoint_path, "projection_layer_weights.pt")
                if os.path.exists(projection_file):
                    projection_state_dict = torch.load(projection_file, map_location="cpu")
                    if hasattr(pipeline.transformer, 'patch_embed') and hasattr(pipeline.transformer.patch_embed, 'proj'):
                        loaded_keys = []
                        if "transformer.patch_embed.proj.weight" in projection_state_dict:
                            pipeline.transformer.patch_embed.proj.weight.data.copy_(projection_state_dict["transformer.patch_embed.proj.weight"])
                            loaded_keys.append("proj.weight")
                        if "transformer.patch_embed.proj.bias" in projection_state_dict and projection_state_dict["transformer.patch_embed.proj.bias"] is not None:
                            pipeline.transformer.patch_embed.proj.bias.data.copy_(projection_state_dict["transformer.patch_embed.proj.bias"])
                            loaded_keys.append("proj.bias")
                        logger.info(f"✅ Loaded concat projection weights: {loaded_keys}")
                    else:
                        logger.info("⚠️ No patch_embed.proj found for concat model")
                else:
                    logger.info("⚠️ No projection_layer_weights.pt found for concat model")
            elif "adapter" in pipeline_type:
                # Adapter models: load cond_proj weights (including cond_norm and cond_gate for v2)
                cond_proj_file = os.path.join(checkpoint_path, "cond_proj_weights.pt")
                if os.path.exists(cond_proj_file):
                    cond_proj_state_dict = torch.load(cond_proj_file, map_location="cpu")
                    if hasattr(pipeline.transformer.patch_embed, 'cond_proj'):
                        loaded_keys = []
                        if "transformer.patch_embed.cond_proj.weight" in cond_proj_state_dict:
                            pipeline.transformer.patch_embed.cond_proj.weight.data.copy_(cond_proj_state_dict["transformer.patch_embed.cond_proj.weight"])
                            loaded_keys.append("cond_proj.weight")
                        if "transformer.patch_embed.cond_proj.bias" in cond_proj_state_dict and cond_proj_state_dict["transformer.patch_embed.cond_proj.bias"] is not None:
                            pipeline.transformer.patch_embed.cond_proj.bias.data.copy_(cond_proj_state_dict["transformer.patch_embed.cond_proj.bias"])
                            loaded_keys.append("cond_proj.bias")
                        
                        # Load cond_norm weights for CogVideoXPatchEmbedWithAdapterV2
                        if hasattr(pipeline.transformer.patch_embed, 'cond_norm'):
                            if "transformer.patch_embed.cond_norm.weight" in cond_proj_state_dict:
                                pipeline.transformer.patch_embed.cond_norm.weight.data.copy_(cond_proj_state_dict["transformer.patch_embed.cond_norm.weight"])
                                loaded_keys.append("cond_norm.weight")
                            if "transformer.patch_embed.cond_norm.bias" in cond_proj_state_dict:
                                pipeline.transformer.patch_embed.cond_norm.bias.data.copy_(cond_proj_state_dict["transformer.patch_embed.cond_norm.bias"])
                                loaded_keys.append("cond_norm.bias")
                        
                        # Load cond_gate for CogVideoXPatchEmbedWithAdapterV2
                        if hasattr(pipeline.transformer.patch_embed, 'cond_gate'):
                            if "transformer.patch_embed.cond_gate" in cond_proj_state_dict:
                                pipeline.transformer.patch_embed.cond_gate.data.copy_(cond_proj_state_dict["transformer.patch_embed.cond_gate"])
                                loaded_keys.append("cond_gate")
                        
                        logger.info(f"✅ Loaded adapter projection weights: {loaded_keys}")
                    else:
                        logger.info("⚠️ No patch_embed.cond_proj found for adapter model")
                else:
                    logger.info("⚠️ No cond_proj_weights.pt found for adapter model")
        
        logger.info("✅ LoRA weights loaded successfully")
    else:
        # For full model, the pipeline was already loaded from checkpoint in build_pipeline
        logger.info("🔧 Full model weights already loaded from checkpoint in build_pipeline")
        logger.info("✅ Full model weights loaded successfully")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Unified CogVideoX Inference with Multiple Pipeline Support")

    # Pipeline configuration
    parser.add_argument(
        "--pipeline_type",
        type=str,
        required=False,
        choices=[
            "cogvideox_pose_concat",
            "cogvideox_pose_adapter",
            "cogvideox_pose_adaln",
            "cogvideox_i2v",
            "cogvideox_static_to_video",
            "cogvideox_static_to_video_pose_concat",
            "cogvideox_fun_static_to_video_pose_concat",
            "cogvideox_fun_static_to_video_joint_generation",
            "cogvideox_fun_static_to_video_cross",
            "cogvideox_fun_static_to_video_cross_pose_adapter",
            "cogvideox_fun_static_to_video_pose_adapter",
        ],
        help="Type of pipeline to use for inference (will be read from config if not provided)"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to the trained checkpoint directory (optional; if omitted, base model weights are used)"
    )
    parser.add_argument(
        "--experiment_config",
        type=str,
        default=None,
        help="Path to experiment config YAML file (will be auto-detected from checkpoint if not provided)"
    )

    # Data configuration
    parser.add_argument(
        "--dataset_file",
        type=str,
        default=None,
        help="Path to dataset file containing video paths (optional if custom paths provided)"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data",
        help="Root directory containing the dataset"
    )
    
    # Custom paths (alternative to dataset_file)
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="Custom video path (alternative to dataset_file)"
    )
    parser.add_argument(
        "--static_video_path",
        type=str,
        default=None,
        help="Custom static video path"
    )
    parser.add_argument(
        "--hand_video_path",
        type=str,
        default=None,
        help="Custom hand video path"
    )
    parser.add_argument(
        "--mask_video_path",
        type=str,
        default=None,
        help="Custom mask video path"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Custom text prompt (alternative to reading from file)"
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default=None,
        help="Custom prompt file path"
    )
    parser.add_argument(
        "--prompt_subdir",
        type=str,
        default=None,
        help="Subdirectory name for prompts (overrides config value, default: 'prompts')"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Path to save the outputs (default: {checkpoint_path}/{eval_subfolder})"
    )
    parser.add_argument(
        "--eval_subfolder",
        type=str,
        default="eval",
        help="Subfolder name under checkpoint_path for evaluation results (default: eval)"
    )

    # Inference parameters
    parser.add_argument(
        "--mode",
        type=str,
        default="s2v",
        choices=["s2v", "i2v", "t2v"],
        help="Inference mode: s2v (static-to-video, default), i2v (image-to-video), t2v (text-to-video)"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of inference steps"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=6.0,
        help="Guidance scale for classifier-free guidance"
    )
    parser.add_argument(
        "--use_dynamic_cfg",
        action="store_true",
        default=True,
        help="Use dynamic cfg"
    )
    parser.add_argument(
        "--use_same_noise",
        action="store_true",
        default=False,
        help="(Joint generation) Use identical noise for the second video (per-call override).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Height of the output video"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=720,
        help="Width of the output video"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=49,
        help="Number of frames to generate"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=8,
        choices=[8, 10, 12, 15, 24],
        help="Frames per second"
    )

    # Batch processing
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=None,
        help="Maximum number of files to process"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference (currently only 1 supported)"
    )

    # Multi-GPU support
    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
        help="Local rank for distributed training"
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=1,
        help="Number of processes for distributed training"
    )

    # Metrics and evaluation
    parser.add_argument(
        "--compute_metrics",
        action="store_true",
        default=True,
        help="Compute video quality metrics (PSNR, SSIM, LPIPS)"
    )
    parser.add_argument(
        "--save_comparison_videos",
        action="store_true",
        default=True,
        help="Save side-by-side comparison videos"
    )

    # Other options
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--use_empty_prompts",
        action="store_true",
        default=False,
        help="Use empty prompts instead of provided text prompts"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Suffix to append to output filenames (e.g., '_no_prompt', '_cfg7.5')"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples to generate per video (default: 1)",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated list of seeds to use (overrides num_samples if provided)",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="The video is not of a high quality, it has a low resolution. Watermark present in each frame. The background is solid. Strange body and strange trajectory. Distortion. ",
        help="Negative prompt for classifier-free guidance",
    )
    
    # SLURM array split support
    parser.add_argument(
        "--split_id",
        type=int,
        default=None,
        help="Split ID for SLURM array jobs (0-based, from SLURM_ARRAY_TASK_ID)"
    )
    parser.add_argument(
        "--num_splits",
        type=int,
        default=None,
        help="Total number of splits for SLURM array jobs"
    )
    
    # Skip existing files
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="Skip inference if output files already exist (default: True)"
    )
    parser.add_argument(
        "--no_skip_existing",
        action="store_false",
        dest="skip_existing",
        help="Disable skip_existing (force re-run even if files exist)"
    )

    return parser.parse_args()


def setup_distributed():
    """Setup distributed training if available."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        
        return rank, world_size, local_rank
    else:
        return 0, 1, 0


def build_pipeline(args: argparse.Namespace, config: Dict[str, Any]):
    """Build the appropriate pipeline based on pipeline_type from config."""
    
    # Get pipeline type from config if not provided in args
    pipeline_type = args.pipeline_type or config.get("pipeline", {}).get("type", "cogvideox_pose_concat")
    training_mode = config.get("training", {}).get("mode", "full")
    
    logger.info(f"🔧 Building {pipeline_type} pipeline from: {args.checkpoint_path}")
    logger.info(f"🔧 Training mode: {training_mode}")
    
    # Get base model path from config
    base_model_path = config.get("model", {}).get("base_model_name_or_path", "alibaba-pai/CogVideoX-Fun-V1.1-5b-InP")
    logger.info(f"🔧 Base model: {base_model_path}")
    
    # Determine weight dtype based on model size (same logic as training script)
    weight_dtype = torch.bfloat16 if "5b" in base_model_path.lower() else torch.float16
    
    # Get pipeline config
    pipeline_config = config.get("pipeline", {})
    
    # Determine whether to load from checkpoint or base model
    if training_mode == "lora":
        # For LoRA, always start from base model
        pretrained_model_path = None
        logger.info("🔧 LoRA mode: Starting from base model")
    else:
        # For full model, load from checkpoint
        pretrained_model_path = args.checkpoint_path
        logger.info("🔧 Full mode: Loading from checkpoint")
    
    if pipeline_type == "cogvideox_pose_concat":
        concat_config = config.get("concat", {})
        pipeline = CogVideoXPoseConcatPipeline.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_path,
            base_model_name_or_path=base_model_path,
            torch_dtype=weight_dtype,
            condition_channels=concat_config.get("condition_channels", None),
        )
    elif pipeline_type == "cogvideox_pose_adapter":
        adapter_config = config.get("adapter", {})
        pipeline = CogVideoXPoseAdapterPipeline.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_path,
            base_model_name_or_path=base_model_path,
            torch_dtype=weight_dtype,
            freeze_hand_branch=adapter_config.get("freeze_hand_branch", False),
            freeze_static_branch=adapter_config.get("freeze_static_branch", False),
            adapter_norm=adapter_config.get("norm", "group"),
            adapter_groups=adapter_config.get("groups", 32),
        )
    elif pipeline_type == "cogvideox_pose_adaln":
        adaln_config = config.get("adaln", {})
        smpl_pose_dim = adaln_config.get("smpl_pose_dim", 63)
        smpl_embed_dim = adaln_config.get("smpl_embed_dim", 512)
        pipeline = CogVideoXPoseAdaLNPipeline.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_path,
            base_model_name_or_path=base_model_path,
            torch_dtype=weight_dtype,
            smpl_pose_dim=smpl_pose_dim,
            smpl_embed_dim=smpl_embed_dim,
        )
    elif pipeline_type == "cogvideox_static_to_video":
        pipeline = CogVideoXStaticToVideoPipeline.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_path,
            base_model_name_or_path=base_model_path,
            torch_dtype=weight_dtype,
        )
    elif pipeline_type == "cogvideox_static_to_video_pose_concat":
        pipeline = CogVideoXStaticToVideoPoseConcatPipeline.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_path,
            base_model_name_or_path=base_model_path,
            torch_dtype=weight_dtype,
            use_adapter=False,
        )
    elif pipeline_type == "cogvideox_fun_static_to_video_pose_concat":
        condition_channels = pipeline_config.get("condition_channels", 16)
        pipeline = CogVideoXFunStaticToVideoPipeline.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_path,
            base_model_name_or_path=base_model_path,
            torch_dtype=weight_dtype,
            condition_channels=condition_channels,
            use_adapter=False,
        )
    elif pipeline_type == "cogvideox_fun_static_to_video_joint_generation":
        condition_channels = pipeline_config.get("condition_channels", 16)
        num_output_videos = pipeline_config.get("num_output_videos", 2)
        expand_input_channels = pipeline_config.get("expand_input_channels", False)
        use_same_noise = pipeline_config.get("use_same_noise", False)
        pipeline = CogVideoXFunStaticToVideoJointGenerationPipeline.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_path,
            base_model_name_or_path=base_model_path,
            torch_dtype=weight_dtype,
            condition_channels=condition_channels,
            num_output_videos=num_output_videos,
            expand_input_channels=expand_input_channels,
            use_same_noise=use_same_noise,
        )
    elif pipeline_type == "cogvideox_fun_static_to_video_cross":
        cross_attn_interval = pipeline_config.get("cross_attn_interval", 2)
        cross_attn_dim_head = pipeline_config.get("cross_attn_dim_head", 128)
        cross_attn_num_heads = pipeline_config.get("cross_attn_num_heads", 16)
        cross_attn_kv_dim = pipeline_config.get("cross_attn_kv_dim", None)
        pipeline = CogVideoXFunStaticToVideoCrossPipeline.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_path,
            base_model_name_or_path=base_model_path,
            torch_dtype=weight_dtype,
            is_train_cross=True,
            cross_attn_interval=cross_attn_interval,
            cross_attn_dim_head=cross_attn_dim_head,
            cross_attn_num_heads=cross_attn_num_heads,
            cross_attn_kv_dim=cross_attn_kv_dim,
        )
    elif pipeline_type == "cogvideox_fun_static_to_video_cross_pose_adapter":
        cross_attn_interval = pipeline_config.get("cross_attn_interval", 2)
        cross_attn_dim_head = pipeline_config.get("cross_attn_dim_head", 128)
        cross_attn_num_heads = pipeline_config.get("cross_attn_num_heads", 16)
        cross_attn_kv_dim = pipeline_config.get("cross_attn_kv_dim", None)
        condition_channels = pipeline_config.get("condition_channels", 16)
        pipeline = CogVideoXFunStaticToVideoCrossPipeline.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_path,
            base_model_name_or_path=base_model_path,
            torch_dtype=weight_dtype,
            is_train_cross=True,
            cross_attn_interval=cross_attn_interval,
            cross_attn_dim_head=cross_attn_dim_head,
            cross_attn_num_heads=cross_attn_num_heads,
            cross_attn_kv_dim=cross_attn_kv_dim,
            condition_channels=condition_channels,
        )
    elif pipeline_type == "cogvideox_fun_static_to_video_pose_adapter":
        condition_channels = pipeline_config.get("condition_channels", 16)
        use_adapter = pipeline_config.get("use_adapter", True)
        adapter_version = pipeline_config.get("adapter_version", "v1")
        pipeline = CogVideoXFunStaticToVideoPipeline.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_path,
            base_model_name_or_path=base_model_path,
            torch_dtype=weight_dtype,
            condition_channels=condition_channels,
            use_adapter=use_adapter,
            adapter_version=adapter_version,
        )
    else:
        raise ValueError(f"Unknown pipeline type: {pipeline_type}")
    
    # Set dtype for all components to ensure consistency (same as training script)
    pipeline.vae.to(dtype=weight_dtype)
    pipeline.text_encoder.to(dtype=weight_dtype)
    pipeline.transformer.to(dtype=weight_dtype)
    
    # Enable memory optimizations
    pipeline.vae.enable_slicing()
    pipeline.vae.enable_tiling()
    pipeline.to(device)
    
    # Setup LoRA adapter if in LoRA mode
    if training_mode == "lora":
        setup_lora_adapter(pipeline.transformer, config)
    
    # Load checkpoint based on training mode
    load_checkpoint_with_config(pipeline, args.checkpoint_path, config)
    
    logger.info(f"✅ {pipeline_type} pipeline built successfully with dtype: {weight_dtype}")
    return pipeline


def load_video_metrics():
    """Load video quality metrics."""
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(device)
    
    return psnr, ssim, lpips


def compute_video_metrics(generated_video, gt_video, psnr, ssim, lpips):
    """Compute video quality metrics between generated and ground truth videos."""
    # Ensure videos have the same number of frames
    min_frames = min(len(generated_video), len(gt_video))
    generated_video = generated_video[:min_frames]
    gt_video = gt_video[:min_frames]
    
    # Convert to tensors and normalize to [0, 1]
    if isinstance(generated_video, np.ndarray):
        generated_tensor = torch.from_numpy(generated_video).float()
    else:
        generated_tensor = generated_video.float()
    
    if isinstance(gt_video, np.ndarray):
        gt_tensor = torch.from_numpy(gt_video).float()
    else:
        gt_tensor = gt_video.float()
    
    # Move to device and add batch dimension
    generated_tensor = generated_tensor.to(device).unsqueeze(0)  # [1, T, H, W, C]
    gt_tensor = gt_tensor.to(device).unsqueeze(0)  # [1, T, H, W, C]
    
    # Reshape to [B*T, C, H, W] for metrics
    B, T, H, W, C = generated_tensor.shape
    generated_flat = generated_tensor.reshape(B*T, C, H, W) # [B*T, C, H, W]
    gt_flat = gt_tensor.view(B*T, C, H, W) # [B*T, C, H, W]
    
    # Compute metrics
    psnr_value = psnr(generated_flat, gt_flat).item()
    ssim_value = ssim(generated_flat, gt_flat).item()
    lpips_value = lpips(generated_flat, gt_flat).item()
    
    return {
        "psnr": psnr_value,
        "ssim": ssim_value,
        "lpips": lpips_value
    }


def find_files_by_path(
    data_root: Path,
    video_path: str,
    custom_paths: Optional[dict] = None,
    config: Optional[Dict[str, Any]] = None,
    prompt_subdir: Optional[str] = None,
    pipeline_type: Optional[str] = None,
) -> dict:
    """Find all related files for a given video path or use custom paths.
    
    Args:
        data_root: Root directory for data
        video_path: Relative video path (used for auto-derivation if custom_paths not provided)
        custom_paths: Optional dict with custom paths:
            - video: Custom GT video path
            - static_video: Custom static video path
            - hand_video: Custom hand video path
            - mask_video: Custom mask video path
            - prompt: Custom prompt text or prompt file path
            - human_motions: Custom human motions file path
        config: Optional config dict to read prompt_subdir and hand_video_subdir
        prompt_subdir: Optional prompt subdirectory name (overrides config value)
    
    Returns:
        dict with file paths for video, static_video, hand_video, mask_video, prompt, human_motions
    """
    # Use custom paths if provided
    if custom_paths:
        logger.info("🔧 Using custom paths instead of auto-derivation")
        files = {
            'video': custom_paths.get('video'),
            'static_video': custom_paths.get('static_video'),
            'hand_video': custom_paths.get('hand_video'),
            'mask_video': custom_paths.get('mask_video'),
            'prompt': custom_paths.get('prompt'),
            'human_motions': custom_paths.get('human_motions'),
        }
        
        # Check if files exist and print status
        for file_type, file_path in files.items():
            if file_path and Path(file_path).exists():
                logger.info(f"✅ Custom {file_type}: {file_path}")
            elif file_path:
                logger.warning(f"⚠️  Custom {file_type} not found: {file_path}")
            else:
                logger.debug(f"ℹ️  Custom {file_type} not provided")
        
        return files
    
    # Auto-derive paths from video_path (existing behavior)
    files = {}
    
    # Get base path from video_path parent directory
    # video_path: .../processed2/videos/{video_name}.mp4
    # base_path: .../processed2/
    video_path_obj = Path(video_path)
    video_name = video_path_obj.stem
    base_path = data_root / video_path_obj.parent.parent  # videos/ -> processed2/
    
    # Read prompt_subdir and hand_video_subdir from config (with defaults)
    data_config = config.get("data", {}) if config else {}
    # Use provided prompt_subdir if available, otherwise use config, otherwise default
    if prompt_subdir is None:
        prompt_subdir = data_config.get("prompt_subdir", "prompts")
    hand_video_subdir = data_config.get("hand_video_subdir", "videos_hands")
    
    # Define file paths based on pipeline type
    video_file = base_path / 'videos' / f"{video_name}.mp4"
    static_video_file = base_path / 'videos_static' / f"{video_name}.mp4"
    hand_video_file = base_path / hand_video_subdir / f"{video_name}.mp4"
    mask_video_file = None
    # Only load mask video when running I2V pipeline (or if explicitly provided via custom_paths)
    if pipeline_type == "cogvideox_i2v":
        mask_video_file = base_path / 'warped_mask_videos' / f"{video_name}.mp4"
    prompt_file = base_path / prompt_subdir / f"{video_name}.txt"
    human_motions_file = base_path / 'human_motions' / f"{video_name}.pt"
    
    files = {
        'video': str(video_file),
        'static_video': str(static_video_file),
        'hand_video': str(hand_video_file),
        'mask_video': str(mask_video_file) if mask_video_file is not None else None,
        'prompt': str(prompt_file),
        'human_motions': str(human_motions_file),
    }
    
    # Check if files exist and print status
    for file_type, file_path in files.items():
        if not file_path:
            continue
        if Path(file_path).exists():
            logger.debug(f"✅ Found {file_type}: {file_path}")
        else:
            logger.debug(f"⚠️  Missing {file_type}: {file_path}")
            files[file_type] = None
    
    return files


def get_output_filename(video_path, suffix=""):
    """Generate output filename from video path."""
    video_path_obj = Path(video_path)
    video_name = video_path_obj.stem
    
    # Extract scene and action names from path
    # video_path: .../scene_name/action_name/processed2/videos/video_name.mp4
    base_path = video_path_obj.parent.parent  # videos/ -> processed2/
    action_name = base_path.parent.name
    scene_name = base_path.parent.parent.name
    
    base_name = f"{scene_name}_{action_name}_{video_name}"
    
    # Add suffix to base_name if provided
    if suffix:
        base_name = f"{base_name}{suffix}"
    
    return f"{base_name}_generated.mp4"

def get_output_filenames_joint(video_path, suffix=""):
    """Generate output filenames for joint generation (two videos)."""
    generated_filename_base = get_output_filename(video_path, suffix)
    base_name = generated_filename_base.replace("_generated.mp4", "")
    return (f"{base_name}_generated_v1.mp4", f"{base_name}_generated_v2.mp4")


def check_output_exists(video_path, output_dir, args, suffix="", pipeline_type: Optional[str] = None):
    """Check if output files already exist in subfolder or unified output_dir.
    
    For distributed/array modes, checks both:
    1. Subfolder (rank_{rank} or split_{split_id}) - current output location
    2. Unified output_dir (args.output_dir) - already moved location
    """
    if not args.skip_existing:
        return False
    if pipeline_type == "cogvideox_fun_static_to_video_joint_generation":
        filenames = list(get_output_filenames_joint(video_path, suffix))
    else:
        filenames = [get_output_filename(video_path, suffix)]
    output_dir = Path(output_dir)
    unified_output_dir = Path(args.output_dir)
    
    # Check in subfolder (current output location)
    subfolder_files = [output_dir / fn for fn in filenames]
    if all(p.exists() for p in subfolder_files):
        logger.info(f"⏭️  Skipping {video_path}: outputs exist in {output_dir}")
        return True
    
    # Check in unified output_dir (if different from subfolder)
    # This handles the case where files have already been moved
    if unified_output_dir != output_dir:
        unified_files = [unified_output_dir / fn for fn in filenames]
        if all(p.exists() for p in unified_files):
            logger.info(f"⏭️  Skipping {video_path}: outputs exist in unified dir {unified_output_dir}")
            return True
    
    return False


def _extract_generated_videos(output) -> List[Any]:
    """Normalize pipeline output to a list of generated videos (torch.Tensor or np.ndarray)."""
    if isinstance(output, dict):
        frames = output.get("frames", None)
        if isinstance(frames, list):
            return frames
        return [frames]
    if isinstance(output, tuple):
        return list(output)
    # PipelineOutput object (diffusers) or a raw tensor/array
    frames = getattr(output, "frames", output)
    if isinstance(frames, list):
        return frames
    return [frames]


def _video_to_numpy(video: Any) -> np.ndarray:
    """Convert a video to numpy, removing batch dim if present."""
    if video is None:
        return None
    if isinstance(video, torch.Tensor):
        video = video.detach().cpu().numpy()
    # [B, F, H, W, C] -> [F, H, W, C] if B==1
    if isinstance(video, np.ndarray) and video.ndim == 5 and video.shape[0] == 1:
        video = video[0]
    return video


def run_single_inference(args, pipeline, video_path, output_dir, psnr, ssim, lpips, config, seed=None, suffix="", compute_metrics=True):
    """Run inference for a single video."""
    
    logger.info(f"🎯 Processing: {video_path}")

    # Get pipeline type from config
    pipeline_type = args.pipeline_type or config.get("pipeline", {}).get("type", "cogvideox_pose_concat")

    # Check if output already exists (skip if exists)
    if check_output_exists(video_path, output_dir, args, suffix, pipeline_type=pipeline_type):
        # Return success result without actually running inference
        return {
            "video_path": video_path,
            "success": True,
            "generation_time": 0.0,
            "metrics": {},
            "generated_shape": None,
            "skipped": True,
        }
    
    # Prepare custom paths if provided
    custom_input_mode = (
        args.video_path is not None
        or args.static_video_path is not None
        or args.hand_video_path is not None
        or args.mask_video_path is not None
        or (args.prompt_file is not None)
        or (args.prompt and args.prompt.strip())
    )
    custom_paths = None
    if custom_input_mode:
        custom_paths = {
            'video': str(args.video_path) if args.video_path else None,
            'static_video': str(args.static_video_path) if args.static_video_path else None,
            'hand_video': str(args.hand_video_path) if args.hand_video_path else None,
            'mask_video': str(args.mask_video_path) if args.mask_video_path else None,
            'prompt': str(args.prompt_file) if args.prompt_file else None,
            'human_motions': None,
        }
    
    # Find related files (auto-derive or use custom paths)
    files = find_files_by_path(
        Path(args.data_root),
        video_path,
        custom_paths=custom_paths,
        config=config,
        prompt_subdir=args.prompt_subdir,
        pipeline_type=pipeline_type,
    )

    print(files)
    # Load required data based on pipeline type
    if pipeline_type == "cogvideox_i2v":
        # I2V pipeline only needs an image (use first frame of hand video)
        if files['hand_video'] is None:
            raise ValueError(f"Hand video required for I2V pipeline: {files['hand_video']}")
        
        hand_video = iio.imread(files['hand_video']).astype(np.float32) / 255.0
        hand_video = hand_video.transpose(3, 0, 1, 2)  # [F, H, W, C] -> [C, F, H, W]
        hand_video = hand_video[np.newaxis, :]  # Add batch dimension: [C, F, H, W] -> [1, C, F, H, W]
        image = PIL.Image.fromarray((hand_video[0, :, 0, :, :].transpose(1, 2, 0) * 255).astype(np.uint8))
        
        pipeline_args = {
            "image": image,
            "prompt": "",
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "use_dynamic_cfg": args.use_dynamic_cfg,
        }
        
    elif pipeline_type == "cogvideox_pose_adaln":
        # AdaLN pipeline needs image and pose parameters
        if files['hand_video'] is None:
            raise ValueError(f"Hand video required for AdaLN pipeline: {files['hand_video']}")
        if files['human_motions'] is None:
            raise ValueError(f"Human motions required for AdaLN pipeline: {files['human_motions']}")
        
        hand_video = iio.imread(files['hand_video']).astype(np.float32) / 255.0
        hand_video = hand_video.transpose(3, 0, 1, 2)  # [F, H, W, C] -> [C, F, H, W]
        hand_video = hand_video[np.newaxis, :]  # Add batch dimension: [C, F, H, W] -> [1, C, F, H, W]
        image = PIL.Image.fromarray((hand_video[0, :, 0, :, :].transpose(1, 2, 0) * 255).astype(np.uint8))
        pose_params = torch.load(files['human_motions'], map_location='cpu')
        
        pipeline_args = {
            "image": image,
            "pose_params": pose_params,
            "prompt": "",
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "use_dynamic_cfg": args.use_dynamic_cfg,
        }
        
    elif pipeline_type == "cogvideox_static_to_video":
        # Static-to-video pipeline needs static videos
        if files['static_video'] is None:
            raise ValueError(f"Static video required for static-to-video pipeline: {files['static_video']}")
        
        static_video = iio.imread(files['static_video']).astype(np.float32) / 255.0
        static_video = static_video.transpose(3, 0, 1, 2)  # [F, H, W, C] -> [C, F, H, W]
        static_video = static_video[np.newaxis, :]  # Add batch dimension: [C, F, H, W] -> [1, C, F, H, W]
        
        pipeline_args = {
            "static_videos": static_video,
            "prompt": "",
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "use_dynamic_cfg": args.use_dynamic_cfg,
        }
        
    elif pipeline_type == "cogvideox_static_to_video_pose_concat":
        # Static-to-video pose concat pipeline needs both static and hand videos
        if files['static_video'] is None:
            raise ValueError(f"Static video required for static-to-video pose concat pipeline: {files['static_video']}")
        if files['hand_video'] is None:
            raise ValueError(f"Hand video required for static-to-video pose concat pipeline: {files['hand_video']}")
        
        static_video = iio.imread(files['static_video']).astype(np.float32) / 255.0
        static_video = static_video.transpose(3, 0, 1, 2)  # [F, H, W, C] -> [C, F, H, W]
        static_video = static_video[np.newaxis, :]  # Add batch dimension: [C, F, H, W] -> [1, C, F, H, W]
        
        hand_video = iio.imread(files['hand_video']).astype(np.float32) / 255.0
        hand_video = hand_video.transpose(3, 0, 1, 2)  # [F, H, W, C] -> [C, F, H, W]
        hand_video = hand_video[np.newaxis, :]  # Add batch dimension: [C, F, H, W] -> [1, C, F, H, W]
        
        mask_video = None
        if files['mask_video'] is not None:
            mask_video = 1 - iio.imread(files['mask_video']).astype(np.float32) / 255.0
            mask_video = mask_video.transpose(3, 0, 1, 2)  # [F, H, W, C] -> [C, F, H, W]
            mask_video = mask_video[np.newaxis, :]  # Add batch dimension: [C, F, H, W] -> [1, C, F, H, W]
        pipeline_args = {
            "static_videos": static_video,
            "hand_pose_videos": hand_video,
            "mask_video": mask_video,
            "prompt": "",
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "use_dynamic_cfg": args.use_dynamic_cfg,
        }
        
    elif pipeline_type == "cogvideox_fun_static_to_video_pose_concat":
        # Fun static-to-video pose concat pipeline needs both static and hand videos
        if files['static_video'] is None:
            raise ValueError(f"Static video required for fun static-to-video pose concat pipeline: {files['static_video']}")
        if files['hand_video'] is None:
            raise ValueError(f"Hand video required for fun static-to-video pose concat pipeline: {files['hand_video']}")
        
        static_video = iio.imread(files['static_video']).astype(np.float32) / 255.0
        static_video = static_video.transpose(3, 0, 1, 2)  # [F, H, W, C] -> [C, F, H, W]
        static_video = static_video[np.newaxis, :]  # Add batch dimension: [C, F, H, W] -> [1, C, F, H, W]
        
        hand_video = iio.imread(files['hand_video']).astype(np.float32) / 255.0
        hand_video = hand_video.transpose(3, 0, 1, 2)  # [F, H, W, C] -> [C, F, H, W]
        hand_video = hand_video[np.newaxis, :]  # Add batch dimension: [C, F, H, W] -> [1, C, F, H, W]
        
        if files['mask_video'] is not None:
            mask_video = 1 - iio.imread(files['mask_video']).astype(np.float32) / 255.0
            mask_video = mask_video.transpose(3, 0, 1, 2)  # [F, H, W, C] -> [C, F, H, W]
            mask_video = mask_video[np.newaxis, :]  # Add batch dimension: [C, F, H, W] -> [1, C, F, H, W]
        elif args.mode == "i2v":
            mask_video = np.zeros_like(static_video[:, :1])
            mask_video[:, :, 1:, :, :] = 255
        elif args.mode == "t2v":
            mask_video = np.zeros_like(static_video[:, :1])
            mask_video[:] = 255
        else:
            mask_video = None
        pipeline_args = {
            "static_videos": static_video,
            "hand_videos": hand_video,
            "prompt": "",
            "mask_video": mask_video,
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "use_dynamic_cfg": args.use_dynamic_cfg,
        }
    elif pipeline_type == "cogvideox_fun_static_to_video_joint_generation":
        # Joint generation pipeline needs both static and hand videos (same inputs as fun pose concat)
        if files['static_video'] is None:
            raise ValueError(f"Static video required for fun joint generation pipeline: {files['static_video']}")
        if files['hand_video'] is None:
            raise ValueError(f"Hand video required for fun joint generation pipeline: {files['hand_video']}")

        static_video = iio.imread(files['static_video']).astype(np.float32) / 255.0
        static_video = static_video.transpose(3, 0, 1, 2)  # [F, H, W, C] -> [C, F, H, W]
        static_video = static_video[np.newaxis, :]  # [1, C, F, H, W]

        hand_video = iio.imread(files['hand_video']).astype(np.float32) / 255.0
        hand_video = hand_video.transpose(3, 0, 1, 2)  # [F, H, W, C] -> [C, F, H, W]
        hand_video = hand_video[np.newaxis, :]  # [1, C, F, H, W]

        mask_video = None
        if files['mask_video'] is not None:
            mask_video = 1 - iio.imread(files['mask_video']).astype(np.float32) / 255.0
            mask_video = mask_video.transpose(3, 0, 1, 2)  # [F, H, W, C] -> [C, F, H, W]
            mask_video = mask_video[np.newaxis, :]  # [1, C, F, H, W]
        else:
            mask_video = None
        pipeline_args = {
            "static_videos": static_video,
            "hand_videos": hand_video,
            "prompt": "",
            "mask_video": mask_video,
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "use_dynamic_cfg": args.use_dynamic_cfg,
            "use_same_noise": args.use_same_noise,
            # JointGenerationPipeline in this repo defaults return_dict=False; set explicitly to be robust.
            "return_dict": True,
        }
        
    elif pipeline_type in ["cogvideox_fun_static_to_video_cross", "cogvideox_fun_static_to_video_cross_pose_adapter"]:
        # Fun static-to-video cross pipeline needs both static and hand videos
        if files['static_video'] is None:
            raise ValueError(f"Static video required for fun static-to-video cross pipeline: {files['static_video']}")
        if files['hand_video'] is None:
            raise ValueError(f"Hand video required for fun static-to-video cross pipeline: {files['hand_video']}")
        
        static_video = iio.imread(files['static_video']).astype(np.float32) / 255.0
        static_video = static_video.transpose(3, 0, 1, 2)  # [F, H, W, C] -> [C, F, H, W]
        static_video = static_video[np.newaxis, :]  # Add batch dimension: [C, F, H, W] -> [1, C, F, H, W]
        
        hand_video = iio.imread(files['hand_video']).astype(np.float32) / 255.0
        hand_video = hand_video.transpose(3, 0, 1, 2)  # [F, H, W, C] -> [C, F, H, W]
        hand_video = hand_video[np.newaxis, :]  # Add batch dimension: [C, F, H, W] -> [1, C, F, H, W]
        
        pipeline_args = {
            "static_videos": static_video,
            "hand_videos": hand_video,
            "prompt": "",
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "use_dynamic_cfg": args.use_dynamic_cfg,
        }
        
    elif pipeline_type == "cogvideox_fun_static_to_video_pose_adapter":
        # Fun static-to-video pose adapter pipeline needs both static and hand videos
        if files['static_video'] is None:
            raise ValueError(f"Static video required for fun static-to-video pose adapter pipeline: {files['static_video']}")
        if files['hand_video'] is None:
            raise ValueError(f"Hand video required for fun static-to-video pose adapter pipeline: {files['hand_video']}")
        
        static_video = iio.imread(files['static_video']).astype(np.float32) / 255.0
        static_video = static_video.transpose(3, 0, 1, 2)  # [F, H, W, C] -> [C, F, H, W]
        static_video = static_video[np.newaxis, :]  # Add batch dimension: [C, F, H, W] -> [1, C, F, H, W]
        
        hand_video = iio.imread(files['hand_video']).astype(np.float32) / 255.0
        hand_video = hand_video.transpose(3, 0, 1, 2)  # [F, H, W, C] -> [C, F, H, W]
        hand_video = hand_video[np.newaxis, :]  # Add batch dimension: [C, F, H, W] -> [1, C, F, H, W]
        
        pipeline_args = {
            "static_videos": static_video,
            "hand_videos": hand_video,
            "prompt": "",
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "use_dynamic_cfg": args.use_dynamic_cfg,
        }
        
    else:
        # Concat/Adapter pipelines need hand and static videos
        if files['hand_video'] is None:
            raise ValueError(f"Hand video required: {files['hand_video']}")
        if files['static_video'] is None:
            raise ValueError(f"Static video required: {files['static_video']}")
        
        hand_video = iio.imread(files['hand_video']).astype(np.float32) / 255.0
        hand_video = hand_video.transpose(3, 0, 1, 2)  # [F, H, W, C] -> [C, F, H, W]
        hand_video = hand_video[np.newaxis, :]  # Add batch dimension: [C, F, H, W] -> [1, C, F, H, W]
        
        static_video = iio.imread(files['static_video']).astype(np.float32) / 255.0
        static_video = static_video.transpose(3, 0, 1, 2)  # [F, H, W, C] -> [C, F, H, W]
        static_video = static_video[np.newaxis, :]  # Add batch dimension: [C, F, H, W] -> [1, C, F, H, W]
        
        pipeline_args = {
            "hand_videos": hand_video,
            "static_videos": static_video,
            "prompt": "",
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "use_dynamic_cfg": args.use_dynamic_cfg,
        }
    
    # Load prompt if available
    if not args.use_empty_prompts:
        if custom_input_mode and args.prompt and args.prompt.strip():
            pipeline_args["prompt"] = args.prompt.strip()
        elif files['prompt'] is not None:
            prompt_entry = files['prompt']
            prompt_path_obj = Path(prompt_entry)
            if prompt_path_obj.exists() and prompt_path_obj.is_file():
                with open(prompt_path_obj, 'r') as f:
                    pipeline_args["prompt"] = f.read().strip()
            else:
                pipeline_args["prompt"] = prompt_entry
    
    # Add negative prompt to all pipeline args
    pipeline_args["negative_prompt"] = args.negative_prompt
    
    # Generate video
    generator = torch.Generator(device=device).manual_seed(seed)
    pipeline_args["generator"] = generator
    pipeline_args["output_type"] = "np"
    
    start_time = time.time()
    output = pipeline(**pipeline_args)
    generation_time = time.time() - start_time

    generated_videos = _extract_generated_videos(output)
    generated_videos = [_video_to_numpy(v) for v in generated_videos if v is not None]

    if pipeline_type == "cogvideox_fun_static_to_video_joint_generation":
        if len(generated_videos) < 2:
            raise ValueError(f"Joint generation expected 2 videos, got {len(generated_videos)}")
        generated_video = generated_videos[0]
        logger.info(
            f"✅ Generated joint videos with shapes: v1={generated_videos[0].shape}, v2={generated_videos[1].shape} in {generation_time:.2f}s"
        )
    else:
        generated_video = generated_videos[0]
        logger.info(f"✅ Generated video with shape: {generated_video.shape} in {generation_time:.2f}s")
    
    # Load ground truth video for metrics
    gt_video = None
    if files['video'] is not None:
        gt_video = iio.imread(files['video']).astype(np.float32) / 255.0
        logger.info(f"✅ Loaded ground truth video with shape: {gt_video.shape}")
    
    # Compute metrics
    metrics = {}
    if compute_metrics and gt_video is not None:
        if pipeline_type == "cogvideox_fun_static_to_video_joint_generation":
            metrics_v1 = compute_video_metrics(generated_videos[0], gt_video, psnr, ssim, lpips)
            metrics_v2 = compute_video_metrics(generated_videos[1], gt_video, psnr, ssim, lpips)
            metrics_avg = {
                "psnr": float((metrics_v1["psnr"] + metrics_v2["psnr"]) / 2.0),
                "ssim": float((metrics_v1["ssim"] + metrics_v2["ssim"]) / 2.0),
                "lpips": float((metrics_v1["lpips"] + metrics_v2["lpips"]) / 2.0),
            }
            metrics = {
                **metrics_avg,  # keep top-level keys for overall summary compatibility
                "v1": metrics_v1,
                "v2": metrics_v2,
                "avg": metrics_avg,
            }
            logger.info(
                f"📊 Metrics(avg) - PSNR: {metrics_avg['psnr']:.3f}, SSIM: {metrics_avg['ssim']:.3f}, LPIPS: {metrics_avg['lpips']:.3f}"
            )
        else:
            metrics = compute_video_metrics(generated_video, gt_video, psnr, ssim, lpips)
            logger.info(f"📊 Metrics - PSNR: {metrics['psnr']:.3f}, SSIM: {metrics['ssim']:.3f}, LPIPS: {metrics['lpips']:.3f}")
    
    # Save outputs
    save_inference_outputs(
        args,
        video_path,
        generated_video,
        gt_video,
        files,
        metrics,
        generation_time,
        output_dir,
        config,
        seed=seed,
        suffix=suffix,
        generated_videos=generated_videos if pipeline_type == "cogvideox_fun_static_to_video_joint_generation" else None,
    )
    
    return {
        "video_path": video_path,
        "success": True,
        "generation_time": generation_time,
        "metrics": metrics,
        "generated_shape": generated_video.shape,
        "generated_shapes": (
            {"v1": list(generated_videos[0].shape), "v2": list(generated_videos[1].shape)}
            if pipeline_type == "cogvideox_fun_static_to_video_joint_generation"
            else None
        ),
    }
    
    # except Exception as e:
    #     logger.error(f"❌ Failed to process {video_path}: {e}")
    #     return {
    #         "video_path": video_path,
    #         "success": False,
    #         "error": str(e),
    #     }


def save_inference_outputs(
    args,
    video_path,
    generated_video,
    gt_video,
    files,
    metrics,
    generation_time,
    output_dir,
    config,
    seed=None,
    suffix="",
    generated_videos: Optional[List[np.ndarray]] = None,
):
    """Save inference outputs including videos and metrics."""
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get base name from video path (reuse get_output_filename logic)
    generated_filename_base = get_output_filename(video_path, suffix)
    base_name = generated_filename_base.replace("_generated.mp4", "")
    
    is_joint = generated_videos is not None and len(generated_videos) >= 2
    if is_joint:
        v1, v2 = generated_videos[0], generated_videos[1]
        # Ensure videos have the same number of frames
        frame_counts = [len(v1), len(v2), args.num_frames]
        if gt_video is not None:
            frame_counts.append(len(gt_video))
        num_frames = min(frame_counts)
        v1 = v1[:num_frames]
        v2 = v2[:num_frames]
        if gt_video is not None:
            gt_video = gt_video[:num_frames]

        # Save generated videos
        fn_v1, fn_v2 = get_output_filenames_joint(video_path, suffix)
        generated_filename_v1 = output_dir / fn_v1
        generated_filename_v2 = output_dir / fn_v2
        export_to_video(v1, str(generated_filename_v1), fps=args.fps)
        export_to_video(v2, str(generated_filename_v2), fps=args.fps)
        logger.info(f"✅ Saved generated video v1: {generated_filename_v1}")
        logger.info(f"✅ Saved generated video v2: {generated_filename_v2}")

        # Save comparison video if ground truth is available
        if gt_video is not None and args.save_comparison_videos:
            # Optionally include static/hand in comparison if available
            video_components = []
            static_path = files.get("static_video")
            hand_path = files.get("hand_video")
            static_video = None
            hand_video = None
            try:
                if static_path is not None and Path(static_path).exists():
                    static_video = iio.imread(static_path).astype(np.float32) / 255.0
                if hand_path is not None and Path(hand_path).exists():
                    hand_video = iio.imread(hand_path).astype(np.float32) / 255.0
            except Exception as e:
                logger.warning(f"⚠️  Failed to load static/hand videos for comparison: {e}")

            # Recompute num_frames if static/hand are shorter
            frame_counts_for_comparison = [num_frames]
            if static_video is not None:
                frame_counts_for_comparison.append(len(static_video))
            if hand_video is not None:
                frame_counts_for_comparison.append(len(hand_video))
            num_frames_cmp = min(frame_counts_for_comparison) if frame_counts_for_comparison else num_frames
            v1 = v1[:num_frames_cmp]
            v2 = v2[:num_frames_cmp]
            gt_video = gt_video[:num_frames_cmp]
            if static_video is not None:
                static_video = static_video[:num_frames_cmp]
            if hand_video is not None:
                hand_video = hand_video[:num_frames_cmp]

            if static_video is not None:
                video_components.append(static_video)
            if hand_video is not None:
                # Handle split_hands: 6 channels -> RGB visualization
                if hand_video.ndim == 4 and hand_video.shape[3] == 6:
                    hand_left = hand_video[:, :, :, 0:3]
                    hand_right = hand_video[:, :, :, 3:6]
                    hand_left_gray = np.mean(hand_left, axis=3, keepdims=True)
                    hand_right_gray = np.mean(hand_right, axis=3, keepdims=True)
                    hand_b = np.zeros_like(hand_left_gray)
                    hand_video_rgb = np.concatenate([hand_left_gray, hand_right_gray, hand_b], axis=3)
                    video_components.append(hand_video_rgb)
                else:
                    video_components.append(hand_video)

            video_components.extend([v1, v2, gt_video])
            comparison_video = np.concatenate(video_components, axis=2)
            comparison_filename = output_dir / f"{base_name}_comparison.mp4"
            export_to_video(comparison_video, str(comparison_filename), fps=args.fps)
            logger.info(f"✅ Saved comparison video: {comparison_filename}")
    else:
        # Single-video behavior (backward compatible)
        # Ensure videos have the same number of frames
        num_frames = min(len(generated_video), args.num_frames)
        if isinstance(generated_video, torch.Tensor):
            generated_video = generated_video.detach().cpu().numpy()
        generated_video = generated_video[:num_frames]

        # Save generated video
        generated_filename = output_dir / generated_filename_base
        export_to_video(generated_video, str(generated_filename), fps=args.fps)
        logger.info(f"✅ Saved generated video: {generated_filename}")
    
    # Save comparison video if ground truth is available
    if (not is_joint) and gt_video is not None and args.save_comparison_videos:
        gt_video = gt_video[:num_frames]
        
        # Create side-by-side comparison
        comparison_video = np.concatenate([generated_video, gt_video], axis=2)
        comparison_filename = output_dir / f"{base_name}_comparison.mp4"
        export_to_video(comparison_video, str(comparison_filename), fps=args.fps)
        logger.info(f"✅ Saved comparison video: {comparison_filename}")
    
    # Get pipeline type from config
    pipeline_type = args.pipeline_type or config.get("pipeline", {}).get("type", "cogvideox_pose_concat")
    
    # Save metrics and metadata
    result_data = {
        "video_path": video_path,
        "pipeline_type": pipeline_type,
        "checkpoint_path": args.checkpoint_path,
        "generation_time": generation_time,
        "generated_shape": list(generated_video.shape) if not is_joint else None,
        "generated_shapes": (
            {"v1": list(generated_videos[0].shape), "v2": list(generated_videos[1].shape)} if is_joint else None
        ),
        "metrics": metrics,
        "inference_params": {
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "use_dynamic_cfg": args.use_dynamic_cfg,
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "fps": args.fps,
        }
    }
    
    result_filename = output_dir / f"{base_name}_result.json"
    with open(result_filename, 'w') as f:
        json.dump(result_data, f, indent=2)
    logger.info(f"✅ Saved result metadata: {result_filename}")
    
    # Also add suffix info to result_data if provided
    if suffix:
        result_data["filename_suffix"] = suffix
    if seed is not None:
        result_data["seed"] = seed


def run_batch_inference(args, pipeline, video_paths, output_dir, psnr, ssim, lpips, config):
    """Run batch inference on multiple videos."""
    
    logger.info(f"🔄 Starting batch inference on {len(video_paths)} videos")
    
    results = []
    successful = 0
    failed = 0

    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    else:
        seeds = [args.seed + i for i in range(args.num_samples)]

    for video_path in video_paths:
        logger.info(f"📽️ Processing video: {video_path}")

        for idx, current_seed in enumerate(seeds):
            sample_suffix = args.suffix or ""
            if len(seeds) > 1:
                sample_suffix += f"_seed{current_seed}"

            result = run_single_inference(
                args,
                pipeline,
                video_path,
                output_dir,
                psnr,
                ssim,
                lpips,
                config,
                current_seed,
                sample_suffix,
                compute_metrics=args.compute_metrics and (idx == 0),
            )
            results.append(result)
            if result["success"]:
                successful += 1
                if result.get("skipped", False):
                    logger.info(f"⏭️  Skipped {video_path} (output already exists)")
            else:
                failed += 1
    
    # Get pipeline type from config
    pipeline_type = args.pipeline_type or config.get("pipeline", {}).get("type", "cogvideox_pose_concat")
    
    # Save batch summary
    summary = {
        "total_videos": len(video_paths),
        "successful": successful,
        "failed": failed,
        "success_rate": successful / len(video_paths) if video_paths else 0,
        "pipeline_type": pipeline_type,
        "checkpoint_path": args.checkpoint_path,
        "results": results,
    }
    
    summary_filename = output_dir / "batch_summary.json"
    with open(summary_filename, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\n🎯 Batch Inference Summary")
    logger.info(f"✅ Successful: {successful}")
    logger.info(f"❌ Failed: {failed}")
    logger.info(f"📊 Success Rate: {successful/len(video_paths)*100:.1f}%")
    logger.info(f"📁 Output directory: {output_dir}")
    logger.info(f"📄 Summary saved: {summary_filename}")
    
    return results


def main():
    """Main inference function."""
    
    # Parse arguments first to check for array split mode
    args = parse_args()
    
    # Auto-detect SLURM_ARRAY_TASK_ID if not provided
    if args.split_id is None and "SLURM_ARRAY_TASK_ID" in os.environ:
        try:
            args.split_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
            logger.info(f"🔧 Auto-detected SLURM_ARRAY_TASK_ID: {args.split_id}")
        except ValueError:
            pass
    
    # Check for SLURM array split mode
    # If split_id is provided, disable distributed mode (array split uses single GPU per job)
    if args.split_id is not None:
        if args.num_splits is None:
            raise ValueError("--num_splits must be provided when --split_id is specified")
        # Array split mode: force single GPU
        rank, world_size, local_rank = 0, 1, 0
        logger.info(f"📊 Array split mode: split_id={args.split_id}, num_splits={args.num_splits}")
    else:
        # Setup distributed training if available (normal mode)
        rank, world_size, local_rank = setup_distributed()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Set device
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set random seed
    seed_all(args.seed)
    
    # Load experiment config (all ranks need this for pipeline building)
    if args.experiment_config is None:
        # Try to find config in checkpoint parent directory
        config_path = find_experiment_config(args.checkpoint_path)
        if config_path is None:
            raise ValueError(f"Could not find experiment config YAML file. Please provide --experiment_config or ensure a YAML file exists in {Path(args.checkpoint_path).parent}")
        if rank == 0:
            logger.info(f"🔧 Auto-detected experiment config: {config_path}")
    else:
        config_path = args.experiment_config
        if rank == 0:
            logger.info(f"🔧 Using provided experiment config: {config_path}")
    
    # Load config (all ranks need this)
    config = load_experiment_config(config_path)
    if rank == 0:
        logger.info(f"✅ Loaded experiment config with pipeline type: {config.get('pipeline', {}).get('type', 'unknown')}")
    
    # Set default output directory if not specified
    if args.output_dir is None:
        args.output_dir = str(Path(args.checkpoint_path) / args.eval_subfolder)
        if rank == 0:
            logger.info(f"🔧 Auto-generated output directory: {args.output_dir}")
    
    # Load dataset file or use custom paths (only on rank 0, then split)
    if rank == 0 or world_size == 1:
        custom_input_mode = (
            args.video_path is not None
            or args.static_video_path is not None
            or args.hand_video_path is not None
            or args.mask_video_path is not None
            or (args.prompt_file is not None)
            or (args.prompt and args.prompt.strip())
        )

        # Load dataset file or use custom paths
        if args.dataset_file is not None:
            # Use dataset file
            dataset_file = Path(args.dataset_file)
            if not dataset_file.exists():
                raise ValueError(f"Dataset file does not exist: {dataset_file}")
            with dataset_file.open('r') as f:
                video_paths = [line.strip() for line in f if line.strip()]
            if args.max_batch_size:
                video_paths = video_paths[:args.max_batch_size]
            logger.info(f"📋 Processing {len(video_paths)} video paths from dataset file")
            
            # Apply array split if split_id is provided
            if args.split_id is not None and args.num_splits is not None:
                total = len(video_paths)
                videos_per_split = total // args.num_splits
                remainder = total % args.num_splits
                
                start_idx = args.split_id * videos_per_split + min(args.split_id, remainder)
                end_idx = start_idx + videos_per_split + (1 if args.split_id < remainder else 0)
                
                video_paths = video_paths[start_idx:end_idx]
                logger.info(f"📊 Split {args.split_id}/{args.num_splits}: Processing {len(video_paths)} videos (indices {start_idx}-{end_idx-1} of {total})")
        elif args.video_path is not None:
            video_paths = [args.video_path]
            logger.info(f"📋 Processing single custom video path: {args.video_path}")
        elif custom_input_mode:
            fallback_path = (
                args.static_video_path
                or args.hand_video_path
                or args.mask_video_path
                or "custom_input"
            )
            video_paths = [fallback_path]
            logger.info("📋 Processing custom inputs without dataset-derived video path")
        else:
            raise ValueError("Either --dataset_file, --video_path, or other custom paths (e.g., --static_video_path) must be provided")
    else:
        video_paths = []  # Will be received from rank 0
    
    # Split video_paths across GPUs for multi-GPU inference
    if world_size > 1:
        # Use GPU for broadcasting (nccl backend requires GPU tensors)
        broadcast_device = torch.device(f"cuda:{rank}")
        
        # Broadcast video_paths from rank 0 to all ranks
        if rank == 0:
            # Convert to list of strings for broadcasting
            video_paths_str = json.dumps(video_paths)
            video_paths_tensor = torch.tensor([ord(c) for c in video_paths_str], dtype=torch.int32, device=broadcast_device)
            video_paths_len = torch.tensor([len(video_paths_str)], dtype=torch.int32, device=broadcast_device)
        else:
            video_paths_tensor = torch.zeros(1, dtype=torch.int32, device=broadcast_device)
            video_paths_len = torch.zeros(1, dtype=torch.int32, device=broadcast_device)
        
        # Broadcast length first
        dist.broadcast(video_paths_len, src=0)
        if rank != 0:
            video_paths_tensor = torch.zeros(video_paths_len.item(), dtype=torch.int32, device=broadcast_device)
        
        # Broadcast video_paths
        dist.broadcast(video_paths_tensor, src=0)
        
        if rank != 0:
            video_paths_str = ''.join([chr(c.item()) for c in video_paths_tensor.cpu()])
            video_paths = json.loads(video_paths_str)
        else:
            # Move to CPU for processing
            video_paths_tensor = video_paths_tensor.cpu()
        
        # Split video_paths across ranks
        total_videos = len(video_paths)
        videos_per_rank = total_videos // world_size
        remainder = total_videos % world_size
        
        start_idx = rank * videos_per_rank + min(rank, remainder)
        end_idx = start_idx + videos_per_rank + (1 if rank < remainder else 0)
        
        my_video_paths = video_paths[start_idx:end_idx]
        
        logger.info(f"🎯 Rank {rank}/{world_size}: Processing {len(my_video_paths)} videos (indices {start_idx}-{end_idx-1} of {total_videos})")
    else:
        my_video_paths = video_paths
    
    # Get pipeline type from config if not provided
    pipeline_type = args.pipeline_type or config.get("pipeline", {}).get("type", "cogvideox_pose_concat")

    if rank == 0:
        logger.info(f"🎯 Pipeline: {pipeline_type}")
        logger.info(f"📁 Checkpoint: {args.checkpoint_path}")
        logger.info(f"📁 Output: {args.output_dir}")
    
    # Build pipeline (all ranks need this)
    pipeline = build_pipeline(args, config)
    
    # Load metrics (all ranks need this for metrics computation)
    psnr, ssim, lpips = load_video_metrics() if args.compute_metrics else (None, None, None)
    
    # Run inference on this rank's data
    if len(my_video_paths) > 0:
        # Create output directory based on mode
        if args.split_id is not None:
            # Array split mode: use split_{split_id} directory
            rank_output_dir = Path(args.output_dir) / f"split_{args.split_id}"
        elif world_size > 1:
            # Distributed mode: use rank_{rank} directory
            rank_output_dir = Path(args.output_dir) / f"rank_{rank}"
        else:
            # Single GPU mode: use output_dir directly
            rank_output_dir = Path(args.output_dir)
        rank_output_dir.mkdir(parents=True, exist_ok=True)
        
        results = run_batch_inference(args, pipeline, my_video_paths, rank_output_dir, psnr, ssim, lpips, config)
    else:
        results = []
        if args.split_id is not None:
            logger.info(f"⚠️  Split {args.split_id}: No videos assigned, skipping inference")
        else:
            logger.info(f"⚠️  Rank {rank}: No videos assigned, skipping inference")
    
    # Gather results from all ranks and merge (only on rank 0)
    if world_size > 1:
        # Wait for all ranks to finish
        dist.barrier()
        
        if rank == 0:
            logger.info(f"🔄 Merging results from {world_size} ranks...")
            
            # Collect all results
            all_results = []
            all_summaries = []
            
            for r in range(world_size):
                rank_dir = Path(args.output_dir) / f"rank_{r}"
                if rank_dir.exists():
                    # Load batch summary from each rank
                    summary_file = rank_dir / "batch_summary.json"
                    if summary_file.exists():
                        with summary_file.open('r') as f:
                            summary = json.load(f)
                        all_summaries.append(summary)
                        all_results.extend(summary.get("results", []))
            
            # Create merged summary
            merged_summary = {
                "total_videos": sum(s.get("total_videos", 0) for s in all_summaries),
                "successful": sum(s.get("successful", 0) for s in all_summaries),
                "failed": sum(s.get("failed", 0) for s in all_summaries),
                "success_rate": 0,
                "num_ranks": world_size,
                "rank_summaries": all_summaries,
                "all_results": all_results,
                "pipeline_type": pipeline_type,
                "checkpoint_path": args.checkpoint_path,
            }
            
            if merged_summary["total_videos"] > 0:
                merged_summary["success_rate"] = merged_summary["successful"] / merged_summary["total_videos"]
            
            # Save merged summary
            merged_dir = Path(args.output_dir) / "merged"
            merged_dir.mkdir(parents=True, exist_ok=True)
            merged_summary_file = merged_dir / "merged_summary.json"
            with merged_summary_file.open('w') as f:
                json.dump(merged_summary, f, indent=2)
            
            logger.info(f"✅ Merged summary saved: {merged_summary_file}")
            logger.info(f"📊 Total videos: {merged_summary['total_videos']}")
            logger.info(f"✅ Successful: {merged_summary['successful']}")
            logger.info(f"❌ Failed: {merged_summary['failed']}")
            logger.info(f"📊 Success rate: {merged_summary['success_rate']*100:.1f}%")
            
            # Use merged results for metrics computation
            results = all_results
        else:
            results = []
    else:
        # Single GPU: results already computed above
        pass
    
    # Compute overall metrics (only on rank 0)
    if rank == 0 and args.compute_metrics:
            successful_results = [r for r in results if r["success"] and r.get("metrics", {}).get("psnr") is not None]
            if successful_results:
                avg_psnr = np.mean([r["metrics"]["psnr"] for r in successful_results])
                avg_ssim = np.mean([r["metrics"]["ssim"] for r in successful_results])
                avg_lpips = np.mean([r["metrics"]["lpips"] for r in successful_results])
                
                # Calculate standard deviations
                std_psnr = np.std([r["metrics"]["psnr"] for r in successful_results])
                std_ssim = np.std([r["metrics"]["ssim"] for r in successful_results])
                std_lpips = np.std([r["metrics"]["lpips"] for r in successful_results])
                
                # Get pipeline type from config
                pipeline_type = args.pipeline_type or config.get("pipeline", {}).get("type", "cogvideox_pose_concat")
                
                # Create metrics summary
                total_videos = len(results) if results else 0
                metrics_summary = {
                    "total_videos": total_videos,
                    "successful_videos": len(successful_results),
                    "failed_videos": total_videos - len(successful_results),
                    "success_rate": len(successful_results) / total_videos if total_videos > 0 else 0,
                    "average_metrics": {
                        "psnr": {
                            "mean": float(avg_psnr),
                            "std": float(std_psnr)
                        },
                        "ssim": {
                            "mean": float(avg_ssim),
                            "std": float(std_ssim)
                        },
                        "lpips": {
                            "mean": float(avg_lpips),
                            "std": float(std_lpips)
                        }
                    },
                    "pipeline_type": pipeline_type,
                    "checkpoint_path": args.checkpoint_path,
                    "inference_params": {
                        "num_inference_steps": args.num_inference_steps,
                        "guidance_scale": args.guidance_scale,
                        "use_dynamic_cfg": args.use_dynamic_cfg,
                        "height": args.height,
                        "width": args.width,
                        "num_frames": args.num_frames,
                        "fps": args.fps,
                        "seed": args.seed
                    }
                }
                
                # Save metrics summary to file
                metrics_summary_file = Path(args.output_dir) / "metrics_summary.json"
                with open(metrics_summary_file, 'w') as f:
                    json.dump(metrics_summary, f, indent=2)
                
                # Print to terminal
                print("\n" + "="*80)
                print("🎯 INFERENCE RESULTS SUMMARY")
                print("="*80)
                total_videos = len(results) if results else 0
                print(f"📁 Checkpoint: {args.checkpoint_path}")
                print(f"🎯 Pipeline: {pipeline_type}")
                print(f"📊 Total Videos: {total_videos}")
                print(f"✅ Successful: {len(successful_results)}")
                print(f"❌ Failed: {total_videos - len(successful_results)}")
                print(f"📈 Success Rate: {len(successful_results) / total_videos * 100:.1f}%" if total_videos > 0 else "📈 Success Rate: N/A")
                print("\n📊 AVERAGE METRICS:")
                print(f"   PSNR:  {avg_psnr:.3f} ± {std_psnr:.3f}")
                print(f"   SSIM:  {avg_ssim:.3f} ± {std_ssim:.3f}")
                print(f"   LPIPS: {avg_lpips:.3f} ± {std_lpips:.3f}")
                print(f"\n📄 Metrics summary saved to: {metrics_summary_file}")
                print("="*80)
                
                logger.info(f"\n📊 Overall Metrics (n={len(successful_results)}):")
                logger.info(f"   PSNR: {avg_psnr:.3f} ± {std_psnr:.3f}")
                logger.info(f"   SSIM: {avg_ssim:.3f} ± {std_ssim:.3f}")
                logger.info(f"   LPIPS: {avg_lpips:.3f} ± {std_lpips:.3f}")
            else:
                total_videos = len(results) if results else 0
                print("\n" + "="*80)
                print("⚠️  NO SUCCESSFUL INFERENCE RESULTS")
                print("="*80)
                print(f"📊 Total Videos: {total_videos}")
                print(f"❌ All failed or no metrics computed")
                print("="*80)
    
    # Move results to output_dir (for both array split and distributed modes)
    if rank == 0:  # Only rank 0 moves results
        output_dir = Path(args.output_dir)
        
        if args.split_id is not None:
            # Array split mode: move split_{split_id}/* to output_dir/*
            split_dir = output_dir / f"split_{args.split_id}"
            if split_dir.exists() and split_dir != output_dir:
                logger.info(f"📦 Moving results from {split_dir} to {output_dir}...")
                for item in split_dir.iterdir():
                    if item.is_file():
                        # Move files directly to output_dir
                        dest = output_dir / item.name
                        if dest.exists():
                            # If file exists, append split_id to name
                            dest = output_dir / f"{item.stem}_split{args.split_id}{item.suffix}"
                        shutil.move(str(item), str(dest))
                    elif item.is_dir():
                        # Move directories to output_dir
                        dest = output_dir / item.name
                        if dest.exists():
                            # If directory exists, merge contents
                            for subitem in item.iterdir():
                                subdest = dest / subitem.name
                                if subitem.is_file():
                                    shutil.move(str(subitem), str(subdest))
                                elif subitem.is_dir():
                                    shutil.copytree(str(subitem), str(subdest), dirs_exist_ok=True)
                                    shutil.rmtree(str(subitem))
                            shutil.rmtree(str(item))
                        else:
                            shutil.move(str(item), str(dest))
                # Remove empty split directory
                try:
                    split_dir.rmdir()
                except OSError:
                    pass  # Directory not empty or doesn't exist
                logger.info(f"✅ Results moved to {output_dir}")
        elif world_size > 1:
            # Distributed mode: results are already merged in merged/ directory
            # Move merged/* to output_dir/*
            merged_dir = output_dir / "merged"
            if merged_dir.exists():
                logger.info(f"📦 Moving merged results from {merged_dir} to {output_dir}...")
                for item in merged_dir.iterdir():
                    dest = output_dir / item.name
                    if item.is_file():
                        shutil.move(str(item), str(dest))
                    elif item.is_dir():
                        if dest.exists():
                            # Merge directories
                            for subitem in item.iterdir():
                                subdest = dest / subitem.name
                                if subitem.is_file():
                                    shutil.move(str(subitem), str(subdest))
                                elif subitem.is_dir():
                                    shutil.copytree(str(subitem), str(subdest), dirs_exist_ok=True)
                                    shutil.rmtree(str(subitem))
                            shutil.rmtree(str(item))
                        else:
                            shutil.move(str(item), str(dest))
                # Remove empty merged directory
                try:
                    merged_dir.rmdir()
                except OSError:
                    pass
                logger.info(f"✅ Merged results moved to {output_dir}")
            
            # Also move rank directories' video files to output_dir
            for r in range(world_size):
                rank_dir = output_dir / f"rank_{r}"
                if rank_dir.exists():
                    logger.info(f"📦 Moving videos from {rank_dir} to {output_dir}...")
                    for item in rank_dir.iterdir():
                        if item.suffix in ['.mp4', '.json']:
                            dest = output_dir / item.name
                            if dest.exists():
                                # Append rank number if file exists
                                dest = output_dir / f"{item.stem}_rank{r}{item.suffix}"
                            shutil.move(str(item), str(dest))
                    # Remove rank directory if empty
                    try:
                        rank_dir.rmdir()
                    except OSError:
                        pass
            logger.info(f"✅ All results moved to {output_dir}")
    
    # Cleanup distributed training
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
