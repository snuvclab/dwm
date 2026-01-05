"""
Argument parser for WAN-based training.
"""
import argparse
import os


def _get_model_args(parser: argparse.ArgumentParser) -> None:
    """Model path arguments."""
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained WAN model or model identifier.",
    )
    parser.add_argument(
        "--transformer_path",
        type=str,
        default=None,
        help="Path to custom transformer checkpoint (safetensors or pt).",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default=None,
        help="Path to custom VAE checkpoint.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directory for cached models.",
    )


# Note: experiment_config, override, mode are defined in train_dwm_wan.py main()


def _get_training_mode_args(parser: argparse.ArgumentParser) -> None:
    """Training mode and LoRA/SFT specific arguments."""
    parser.add_argument(
        "--train_mode",
        type=str,
        default="lora",
        choices=["lora", "sft"],
        help="Training mode: lora (LoRA fine-tuning) or sft (full/partial fine-tuning).",
    )
    # LoRA parameters
    parser.add_argument("--rank", type=int, default=64, help="LoRA rank.")
    parser.add_argument("--network_alpha", type=int, default=64, help="LoRA alpha (scaling = alpha/rank).")
    parser.add_argument("--train_text_encoder", action="store_true", help="Train text encoder with LoRA.")
    parser.add_argument("--lora_skip_name", type=str, default=None, help="Skip LoRA for modules containing this name.")
    # SFT parameters
    parser.add_argument(
        "--trainable_modules",
        nargs="+",
        default=None,
        help="List of trainable module names for SFT mode (e.g., patch_embedding blocks).",
    )
    parser.add_argument(
        "--trainable_modules_low_learning_rate",
        nargs="+",
        default=[],
        help="Modules with lower learning rate (lr/2).",
    )


def _get_condition_args(parser: argparse.ArgumentParser) -> None:
    """Condition-related arguments for WanTransformer3DModelWithConcat."""
    parser.add_argument(
        "--condition_channels",
        type=int,
        default=16,
        help="Number of condition channels (e.g., 16 for hand latents).",
    )


def _get_dataset_args(parser: argparse.ArgumentParser) -> None:
    """Dataset arguments."""
    parser.add_argument("--data_root", type=str, default=None, help="Root directory for training data.")
    parser.add_argument("--dataset_file", type=str, default=None, help="Path to dataset file (txt/csv).")
    parser.add_argument("--video_column", type=str, default="video", help="Column name for video paths.")
    parser.add_argument("--hand_video_column", type=str, default="hand_video", help="Column name for hand video paths.")
    parser.add_argument("--static_video_column", type=str, default="static_video", help="Column name for static video paths.")
    parser.add_argument("--caption_column", type=str, default="text", help="Column name for captions.")
    parser.add_argument("--height_buckets", nargs="+", type=int, default=[480])
    parser.add_argument("--width_buckets", nargs="+", type=int, default=[720])
    parser.add_argument("--frame_buckets", nargs="+", type=int, default=[49])
    parser.add_argument("--max_num_frames", type=int, default=49, help="Max number of video frames.")
    parser.add_argument("--load_tensors", action="store_true", help="Load pre-encoded tensors instead of raw videos.")
    parser.add_argument("--random_flip", type=float, default=None, help="Random horizontal flip probability.")
    parser.add_argument("--dataloader_num_workers", type=int, default=0, help="Number of dataloader workers.")
    parser.add_argument("--pin_memory", action="store_true", help="Pin memory in dataloader.")


def _get_training_args(parser: argparse.ArgumentParser) -> None:
    """Core training arguments."""
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--output_dir", type=str, default="outputs/wan_training", help="Output directory.")
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size per device.")
    parser.add_argument("--num_train_epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Max training steps (overrides epochs).")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--lr_scheduler", type=str, default="constant",
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
                        help="Learning rate scheduler type.")
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="LR warmup steps.")
    parser.add_argument("--lr_num_cycles", type=int, default=1, help="Number of cycles for cosine_with_restarts.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping.")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"], help="Mixed precision mode.")
    parser.add_argument("--allow_tf32", action="store_true", help="Allow TF32 on Ampere GPUs.")
    parser.add_argument("--enable_slicing", action="store_true", help="Enable VAE slicing.")
    parser.add_argument("--enable_tiling", action="store_true", help="Enable VAE tiling.")


def _get_sampling_args(parser: argparse.ArgumentParser) -> None:
    """Flow matching / diffusion sampling arguments."""
    parser.add_argument("--train_sampling_steps", type=int, default=1000, help="Training sampling steps.")
    parser.add_argument("--uniform_sampling", action="store_true", default=True,
                        help="Whether to use uniform sampling. Default: True.")
    parser.add_argument("--weighting_scheme", type=str, default="none",
                        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
                        help="Loss weighting scheme.")
    parser.add_argument("--logit_mean", type=float, default=0.0, help="Logit normal mean.")
    parser.add_argument("--logit_std", type=float, default=1.0, help="Logit normal std.")
    parser.add_argument("--mode_scale", type=float, default=1.29, help="Mode scale.")


def _get_checkpoint_args(parser: argparse.ArgumentParser) -> None:
    """Checkpointing arguments."""
    parser.add_argument("--checkpointing_steps", type=int, default=500, help="Save checkpoint every N steps.")
    parser.add_argument("--checkpoints_total_limit", type=int, default=None, help="Max checkpoints to keep.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Resume from checkpoint path or 'latest'.")
    parser.add_argument("--save_state", action="store_true", help="Save full accelerator state (for exact resume).")


def _get_validation_args(parser: argparse.ArgumentParser) -> None:
    """Validation arguments."""
    parser.add_argument("--validation_set", type=str, default=None, help="Path to validation set file.")
    parser.add_argument("--validation_prompt", type=str, default=None, help="Validation prompt(s), separated by ---.")
    parser.add_argument("--validation_prompt_separator", type=str, default=":::", help="Separator for validation prompts.")
    parser.add_argument("--validation_steps", type=int, default=2000, help="Run validation every N steps.")
    parser.add_argument("--validation_epochs", type=int, default=None, help="Run validation every N epochs.")
    parser.add_argument("--num_validation_videos", type=int, default=1, help="Number of validation videos per prompt.")
    parser.add_argument("--max_validation_videos", type=int, default=4, help="Max validation videos per run.")
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="Guidance scale for validation.")


def _get_optimizer_args(parser: argparse.ArgumentParser) -> None:
    """Optimizer arguments."""
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw", "prodigy", "came"],
                        help="Optimizer type.")
    parser.add_argument("--use_8bit_adam", action="store_true", help="Use 8-bit Adam from bitsandbytes.")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta1.")
    parser.add_argument("--beta2", type=float, default=0.95, help="Adam beta2.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--epsilon", type=float, default=1e-8, help="Adam epsilon.")


def _get_logging_args(parser: argparse.ArgumentParser) -> None:
    """Logging and tracking arguments."""
    parser.add_argument("--logging_dir", type=str, default="logs", help="Logging directory.")
    parser.add_argument("--report_to", type=str, default="wandb", choices=["tensorboard", "wandb", "all"],
                        help="Reporting destination.")
    parser.add_argument("--tracker_project_name", type=str, default="wan-training", help="Wandb project name.")
    parser.add_argument("--tracker_entity_name", type=str, default=None, help="Wandb entity/team name.")


def _get_misc_args(parser: argparse.ArgumentParser) -> None:
    """Miscellaneous arguments."""
    parser.add_argument("--vae_mini_batch", type=int, default=32, help="VAE encoding mini-batch size.")
    parser.add_argument("--tokenizer_max_length", type=int, default=512, help="Max tokenizer length.")
    parser.add_argument("--nccl_timeout", type=int, default=600, help="NCCL timeout in seconds.")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training.")


def add_all_args(parser: argparse.ArgumentParser) -> None:
    """Add all WAN training arguments to an existing parser."""
    _get_model_args(parser)
    _get_training_mode_args(parser)
    _get_condition_args(parser)
    _get_dataset_args(parser)
    _get_training_args(parser)
    _get_sampling_args(parser)
    _get_checkpoint_args(parser)
    _get_validation_args(parser)
    _get_optimizer_args(parser)
    _get_logging_args(parser)
    _get_misc_args(parser)


def get_args() -> argparse.Namespace:
    """Parse and return all arguments for WAN training."""
    parser = argparse.ArgumentParser(description="WAN-based training script for DWM (Dexterous World Model)")
    add_all_args(parser)
    args = parser.parse_args()

    # Handle LOCAL_RANK from environment
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Set default for uniform_sampling (default: True)
    # Note: action="store_true" sets it to False if flag is not provided, so we override here
    import sys
    if '--uniform_sampling' not in sys.argv:
        args.uniform_sampling = True

    return args
