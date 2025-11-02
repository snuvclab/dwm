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
from cogvideox_fun_static_to_video_pose_concat_pipeline import CogVideoXFunStaticToVideoPipeline, CogVideoXFunStaticToVideoCrossPipeline

# Import video metrics
try:
    from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    METRICS_AVAILABLE = True
except ImportError:
    print("⚠️  torchmetrics not available. Video metrics will be disabled.")
    METRICS_AVAILABLE = False

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
    lora_alpha = config.get("training", {}).get("lora_alpha", 32)
    
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
        choices=["cogvideox_pose_concat", "cogvideox_pose_adapter", "cogvideox_pose_adaln", "cogvideox_i2v", "cogvideox_static_to_video", "cogvideox_static_to_video_pose_concat", "cogvideox_fun_static_to_video_pose_concat", "cogvideox_fun_static_to_video_cross", "cogvideox_fun_static_to_video_cross_pose_adapter", "cogvideox_fun_static_to_video_pose_adapter"],
        help="Type of pipeline to use for inference (will be read from config if not provided)"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the trained checkpoint directory"
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
        required=True,
        help="Path to dataset file containing video paths"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data",
        help="Root directory containing the dataset"
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
    base_model_path = config.get("model", {}).get("base_model_name_or_path", "THUDM/CogVideoX-5b")
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
    if not METRICS_AVAILABLE:
        return None, None, None
    
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(device)
    
    return psnr, ssim, lpips


def compute_video_metrics(generated_video, gt_video, psnr, ssim, lpips):
    """Compute video quality metrics between generated and ground truth videos."""
    if not METRICS_AVAILABLE or any(metric is None for metric in [psnr, ssim, lpips]):
        return {"psnr": None, "ssim": None, "lpips": None}
    
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


def find_files_by_path(data_root: Path, video_path: str) -> dict:
    """Find all related files for a given video path."""
    files = {}
    
    # Get base path from video_path parent directory
    # video_path: .../processed2/videos/{video_name}.mp4
    # base_path: .../processed2/
    video_path_obj = Path(video_path)
    video_name = video_path_obj.stem
    base_path = data_root / video_path_obj.parent.parent  # videos/ -> processed2/
    
    # Define file paths based on pipeline type
    video_file = base_path / 'videos' / f"{video_name}.mp4"
    static_video_file = base_path / 'videos_static' / f"{video_name}.mp4"
    hand_video_file = base_path / 'videos_hands' / f"{video_name}.mp4"
    prompt_file = base_path / 'prompts' / f"{video_name}.txt"
    human_motions_file = base_path / 'human_motions' / f"{video_name}.pt"
    
    files = {
        'video': str(video_file),
        'static_video': str(static_video_file),
        'hand_video': str(hand_video_file),
        'prompt': str(prompt_file),
        'human_motions': str(human_motions_file),
    }
    
    # Check if files exist and print status
    for file_type, file_path in files.items():
        if Path(file_path).exists():
            logger.debug(f"✅ Found {file_type}: {file_path}")
        else:
            logger.debug(f"⚠️  Missing {file_type}: {file_path}")
            files[file_type] = None
    
    return files


def run_single_inference(args, pipeline, video_path, output_dir, psnr, ssim, lpips, config):
    """Run inference for a single video."""
    
    logger.info(f"🎯 Processing: {video_path}")
    
    # Get pipeline type from config
    pipeline_type = args.pipeline_type or config.get("pipeline", {}).get("type", "cogvideox_pose_concat")
    
    # Find related files
    files = find_files_by_path(Path(args.data_root), video_path)
    
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
        
        pipeline_args = {
            "static_videos": static_video,
            "hand_pose_videos": hand_video,
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
    if files['prompt'] is not None and not args.use_empty_prompts:
        with open(files['prompt'], 'r') as f:
            prompt_text = f.read().strip()
        pipeline_args["prompt"] = prompt_text
    
    # Generate video
    generator = torch.Generator(device=device).manual_seed(args.seed)
    pipeline_args["generator"] = generator
    pipeline_args["output_type"] = "np"
    
    start_time = time.time()
    output = pipeline(**pipeline_args)
    generation_time = time.time() - start_time
    
    generated_video = output.frames[0]
    logger.info(f"✅ Generated video with shape: {generated_video.shape} in {generation_time:.2f}s")
    
    # Load ground truth video for metrics
    gt_video = None
    if files['video'] is not None:
        gt_video = iio.imread(files['video']).astype(np.float32) / 255.0
        logger.info(f"✅ Loaded ground truth video with shape: {gt_video.shape}")
    
    # Compute metrics
    metrics = {}
    if args.compute_metrics and gt_video is not None:
        metrics = compute_video_metrics(generated_video, gt_video, psnr, ssim, lpips)
        logger.info(f"📊 Metrics - PSNR: {metrics['psnr']:.3f}, SSIM: {metrics['ssim']:.3f}, LPIPS: {metrics['lpips']:.3f}")
    
    # Save outputs
    save_inference_outputs(args, video_path, generated_video, gt_video, files, metrics, generation_time, output_dir, config)
    
    return {
        "video_path": video_path,
        "success": True,
        "generation_time": generation_time,
        "metrics": metrics,
        "generated_shape": generated_video.shape,
    }
    
    # except Exception as e:
    #     logger.error(f"❌ Failed to process {video_path}: {e}")
    #     return {
    #         "video_path": video_path,
    #         "success": False,
    #         "error": str(e),
    #     }


def save_inference_outputs(args, video_path, generated_video, gt_video, files, metrics, generation_time, output_dir, config):
    """Save inference outputs including videos and metrics."""
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create filename from video path
    video_path_obj = Path(video_path)
    video_name = video_path_obj.stem
    
    # Extract scene and action names from path
    # video_path: .../scene_name/action_name/processed2/videos/video_name.mp4
    base_path = video_path_obj.parent.parent  # videos/ -> processed2/
    action_name = base_path.parent.name
    scene_name = base_path.parent.parent.name
    
    base_name = f"{scene_name}_{action_name}_{video_name}"
    
    # Ensure videos have the same number of frames
    num_frames = min(len(generated_video), args.num_frames)
    generated_video = generated_video[:num_frames].numpy()
    
    # Save generated video
    generated_filename = output_dir / f"{base_name}_generated.mp4"
    export_to_video(generated_video, str(generated_filename), fps=args.fps)
    logger.info(f"✅ Saved generated video: {generated_filename}")
    
    # Save comparison video if ground truth is available
    if gt_video is not None and args.save_comparison_videos:
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
        "generated_shape": list(generated_video.shape),
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


def run_batch_inference(args, pipeline, video_paths, output_dir, psnr, ssim, lpips, config):
    """Run batch inference on multiple videos."""
    
    logger.info(f"🔄 Starting batch inference on {len(video_paths)} videos")
    
    results = []
    successful = 0
    failed = 0
    
    for i, video_path in enumerate(video_paths):
        logger.info(f"\n📊 Progress: {i+1}/{len(video_paths)}")
        
        result = run_single_inference(args, pipeline, video_path, output_dir, psnr, ssim, lpips, config)
        results.append(result)
        
        if result["success"]:
            successful += 1
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
    
    # Setup distributed training if available
    rank, world_size, local_rank = setup_distributed()
    
    # Parse arguments
    args = parse_args()
    
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
    
    # Only process on rank 0 or if not distributed
    if rank == 0 or world_size == 1:
        
        # Load experiment config
        if args.experiment_config is None:
            # Try to find config in checkpoint parent directory
            config_path = find_experiment_config(args.checkpoint_path)
            if config_path is None:
                raise ValueError(f"Could not find experiment config YAML file. Please provide --experiment_config or ensure a YAML file exists in {Path(args.checkpoint_path).parent}")
            logger.info(f"🔧 Auto-detected experiment config: {config_path}")
        else:
            config_path = args.experiment_config
            logger.info(f"🔧 Using provided experiment config: {config_path}")
        
        # Load config
        config = load_experiment_config(config_path)
        logger.info(f"✅ Loaded experiment config with pipeline type: {config.get('pipeline', {}).get('type', 'unknown')}")
        
        # Set default output directory if not specified
        if args.output_dir is None:
            args.output_dir = str(Path(args.checkpoint_path) / args.eval_subfolder)
            logger.info(f"🔧 Auto-generated output directory: {args.output_dir}")
        
        # Load dataset file
        dataset_file = Path(args.dataset_file)
        if not dataset_file.exists():
            raise ValueError(f"Dataset file does not exist: {dataset_file}")
        
        with dataset_file.open('r') as f:
            video_paths = [line.strip() for line in f if line.strip()]
        
        # Limit batch size if specified
        if args.max_batch_size:
            video_paths = video_paths[:args.max_batch_size]
        
        # Get pipeline type from config if not provided
        pipeline_type = args.pipeline_type or config.get("pipeline", {}).get("type", "cogvideox_pose_concat")
                

        logger.info(f"📋 Processing {len(video_paths)} video paths")
        logger.info(f"🎯 Pipeline: {pipeline_type}")
        logger.info(f"📁 Checkpoint: {args.checkpoint_path}")
        logger.info(f"📁 Output: {args.output_dir}")
        
        # Build pipeline
        pipeline = build_pipeline(args, config)
        
        # Load metrics
        psnr, ssim, lpips = load_video_metrics() if args.compute_metrics else (None, None, None)
        
        # Run inference
        results = run_batch_inference(args, pipeline, video_paths, Path(args.output_dir), psnr, ssim, lpips, config)
        
        # Compute overall metrics
        if args.compute_metrics and METRICS_AVAILABLE:
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
                metrics_summary = {
                    "total_videos": len(video_paths),
                    "successful_videos": len(successful_results),
                    "failed_videos": len(video_paths) - len(successful_results),
                    "success_rate": len(successful_results) / len(video_paths) if video_paths else 0,
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
                print(f"📁 Checkpoint: {args.checkpoint_path}")
                print(f"🎯 Pipeline: {pipeline_type}")
                print(f"📊 Total Videos: {len(video_paths)}")
                print(f"✅ Successful: {len(successful_results)}")
                print(f"❌ Failed: {len(video_paths) - len(successful_results)}")
                print(f"📈 Success Rate: {len(successful_results) / len(video_paths) * 100:.1f}%")
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
                print("\n" + "="*80)
                print("⚠️  NO SUCCESSFUL INFERENCE RESULTS")
                print("="*80)
                print(f"📊 Total Videos: {len(video_paths)}")
                print(f"❌ All failed or no metrics computed")
                print("="*80)
    
    # Cleanup distributed training
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
