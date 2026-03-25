from __future__ import annotations

import argparse
import gc
import logging
import math
import os
import re
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.cogvideox.diffusers_compat import disable_broken_torchao

disable_broken_torchao()

import diffusers
import torch
import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs, ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params
from diffusers.utils import export_to_video
from peft import get_peft_model_state_dict
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

try:
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
except Exception:
    class DiagonalGaussianDistribution:
        """Minimal fallback used when diffusers' VAE module import is broken by optional quantization deps."""

        def __init__(self, parameters: torch.Tensor, deterministic: bool = False) -> None:
            self.parameters = parameters
            self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
            self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
            self.deterministic = deterministic
            self.std = torch.exp(0.5 * self.logvar)
            self.var = torch.exp(self.logvar)
            if deterministic:
                self.std = torch.zeros_like(self.mean)
                self.var = torch.zeros_like(self.mean)

        def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
            if self.deterministic:
                return self.mean
            noise = torch.randn(
                self.mean.shape,
                generator=generator,
                device=self.mean.device,
                dtype=self.mean.dtype,
            )
            return self.mean + self.std * noise

        def mode(self) -> torch.Tensor:
            return self.mean

from training.cogvideox.config_loader import load_experiment_config
from training.cogvideox.models import CogVideoXFunStaticHandConcatTransformer3DModel
from training.cogvideox.pipeline import CogVideoXFunStaticHandConcatPipeline
from training.cogvideox.static_hand_dataset import StaticHandConcatDataset, collate_static_hand_batch
from training.cogvideox.static_hand_utils import (
    build_lora_config,
    build_pipeline_from_config,
    collect_non_lora_state_dict,
    load_lora_weights_into_transformer,
    load_non_lora_state_dict,
    load_prompt_text,
    load_video_clip,
    read_dataset_entries,
    resolve_sample_paths,
    save_non_lora_state_dict,
)

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone DWM trainer for CogVideoX static-hand-concat.")
    parser.add_argument("--experiment_config", type=str, required=True, help="Path to the experiment YAML.")
    parser.add_argument("--override", type=str, nargs="*", help="Config override entries in key=value form.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["debug", "slurm_test", "slurm", "batch"],
        default="slurm",
        help="Run mode used for output directory naming.",
    )
    parser.add_argument("--test_dataloader", action="store_true", help="Inspect a few training batches and exit.")
    parser.add_argument(
        "--test_dataloader_samples",
        type=int,
        default=4,
        help="Number of batches to inspect in --test_dataloader mode.",
    )
    parser.add_argument(
        "--save_initial_checkpoints",
        action="store_true",
        help="Write checkpoint-0 before the first optimizer step.",
    )
    parser.add_argument(
        "--print_config_only",
        action="store_true",
        help="Load and validate the config, then print the resolved pipeline info and exit.",
    )
    return parser.parse_args()


def resolve_output_dir(experiment_config: Dict[str, Any], training_config: Dict[str, Any], mode: str) -> str:
    exp_name = experiment_config.get("name", "unknown_experiment")
    exp_date = experiment_config.get("date", "unknown_date")
    try:
        date_suffix = datetime.strptime(exp_date, "%Y-%m-%d").strftime("%y%m%d")
    except ValueError:
        date_suffix = "unknown"

    base_output_dir = f"outputs/{date_suffix}/{exp_name}"
    if mode == "debug":
        return f"{base_output_dir}_debug"
    if mode == "slurm_test":
        slurm_job_id = training_config.get("slurm_job_id")
        if training_config.get("resume_from_checkpoint") and slurm_job_id:
            return f"{base_output_dir}_slurm_{slurm_job_id}"
        return f"{base_output_dir}_slurm_test"
    if mode == "slurm":
        if training_config.get("resume_from_checkpoint"):
            slurm_job_id = training_config.get("slurm_job_id") or os.environ.get("SLURM_JOB_ID", "unknown")
        else:
            slurm_job_id = os.environ.get("SLURM_JOB_ID", "unknown")
        return f"{base_output_dir}_slurm_{slurm_job_id}"
    if mode == "batch":
        timestamp = os.environ.get("BATCH_TIMESTAMP") or datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{base_output_dir}_batch_{timestamp}"
    raise ValueError(f"Unsupported mode: {mode}")


def copy_experiment_config(experiment_config_path: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    shutil.copy2(experiment_config_path, os.path.join(output_dir, os.path.basename(experiment_config_path)))


def normalize_log_with(log_with: Any) -> Any:
    if log_with is None:
        return None
    if isinstance(log_with, str):
        lowered = log_with.strip().lower()
        if lowered in {"", "none", "null", "off", "false", "disabled"}:
            return None
        return log_with
    if isinstance(log_with, (list, tuple)):
        normalized = []
        for item in log_with:
            if item is None:
                continue
            lowered = str(item).strip().lower()
            if lowered in {"", "none", "null", "off", "false", "disabled"}:
                continue
            normalized.append(item)
        return normalized or None
    return log_with


def run_dataloader_test(dataset: StaticHandConcatDataset, dataloader: DataLoader, max_batches: int) -> None:
    print(f"dataset size: {len(dataset)}")
    print(f"dataloader size: {len(dataloader)}")
    for index, batch in enumerate(dataloader):
        if index >= max_batches:
            break
        print(f"\n--- batch {index + 1} ---")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: shape={tuple(value.shape)} dtype={value.dtype}")
            else:
                print(f"{key}: {type(value).__name__}")


def maybe_save_checkpoint(accelerator: Accelerator, output_dir: str, global_step: int) -> None:
    save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
    if os.path.exists(save_path):
        logger.warning("Checkpoint already exists, skipping save: %s", save_path)
        return
    accelerator.save_state(save_path)
    logger.info("Saved checkpoint to %s", save_path)


def get_weight_dtype(accelerator: Accelerator) -> torch.dtype:
    if accelerator.state.deepspeed_plugin is not None:
        ds_config = accelerator.state.deepspeed_plugin.deepspeed_config
        if ds_config.get("fp16", {}).get("enabled", False):
            return torch.float16
        if ds_config.get("bf16", {}).get("enabled", False):
            return torch.bfloat16
    if accelerator.mixed_precision == "fp16":
        return torch.float16
    if accelerator.mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def get_t5_prompt_embeds(
    prompt: str | List[str],
    tokenizer,
    text_encoder,
    device: torch.device,
    dtype: torch.dtype,
    max_sequence_length: int = 226,
) -> torch.Tensor:
    prompt_list = [prompt] if isinstance(prompt, str) else prompt
    text_inputs = tokenizer(
        prompt_list,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    prompt_embeds = text_encoder(text_inputs.input_ids.to(device))[0]
    return prompt_embeds.to(device=device, dtype=dtype)


def apply_regex_parameter_patterns(
    transformer: torch.nn.Module,
    trainable_patterns: Iterable[str],
    frozen_patterns: Iterable[str],
) -> None:
    for pattern in frozen_patterns:
        regex = re.compile(pattern)
        for name, param in transformer.named_parameters():
            if regex.match(name):
                param.requires_grad_(False)

    for pattern in trainable_patterns:
        regex = re.compile(pattern)
        for name, param in transformer.named_parameters():
            if regex.match(name):
                param.requires_grad_(True)


def setup_training_mode(
    transformer: torch.nn.Module,
    training_config: Dict[str, Any],
) -> Tuple[int, Optional[Any]]:
    training_mode = training_config.get("mode", "lora")
    gradient_checkpointing = training_config.get("custom_settings", {}).get("gradient_checkpointing", False)

    if training_mode == "full":
        for param in transformer.parameters():
            param.requires_grad_(True)
        if gradient_checkpointing:
            transformer.enable_gradient_checkpointing()
        trainable_params = sum(param.numel() for param in transformer.parameters() if param.requires_grad)
        return trainable_params, None

    if training_mode != "lora":
        raise ValueError(f"Unsupported training mode: {training_mode}")

    for param in transformer.parameters():
        param.requires_grad_(False)

    if gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    lora_config = build_lora_config(training_config)
    transformer.add_adapter(lora_config)
    apply_regex_parameter_patterns(
        transformer,
        training_config.get("trainable_parameter_patterns", []),
        training_config.get("frozen_parameter_patterns", []),
    )
    trainable_params = sum(param.numel() for param in transformer.parameters() if param.requires_grad)
    return trainable_params, lora_config


def create_state_hooks(
    accelerator: Accelerator,
    transformer: torch.nn.Module,
    config: Dict[str, Any],
    lora_config: Optional[Any],
) -> Tuple[Any, Any]:
    training_mode = config["training"].get("mode", "lora")

    def save_model_hook(models, weights, output_dir):
        if not accelerator.is_main_process:
            return

        if training_mode == "full":
            for model in models:
                unwrapped = accelerator.unwrap_model(model)
                if isinstance(unwrapped, CogVideoXFunStaticHandConcatTransformer3DModel):
                    unwrapped.save_pretrained(os.path.join(output_dir, "transformer"))
                if weights:
                    weights.pop()
            return

        transformer_lora_layers = None
        non_lora_state_dict: Dict[str, torch.Tensor] = {}
        for model in models:
            unwrapped = accelerator.unwrap_model(model)
            if isinstance(unwrapped, CogVideoXFunStaticHandConcatTransformer3DModel):
                transformer_lora_layers = get_peft_model_state_dict(unwrapped)
                non_lora_state_dict = collect_non_lora_state_dict(unwrapped)
            if weights:
                weights.pop()

        CogVideoXFunStaticHandConcatPipeline.save_lora_weights(
            output_dir,
            transformer_lora_layers=transformer_lora_layers,
        )
        save_non_lora_state_dict(output_dir, non_lora_state_dict)

    def load_model_hook(models, input_dir):
        while len(models) > 0:
            models.pop()

        unwrapped = accelerator.unwrap_model(transformer)
        if training_mode == "full":
            loaded_transformer = CogVideoXFunStaticHandConcatTransformer3DModel.from_pretrained(
                input_dir,
                subfolder="transformer",
            )
            unwrapped.load_state_dict(loaded_transformer.state_dict())
            return

        if lora_config is None:
            raise ValueError("LoRA config is required for LoRA checkpoint loading.")

        unexpected_keys = load_lora_weights_into_transformer(unwrapped, input_dir)
        if unexpected_keys:
            logger.warning("Unexpected LoRA keys while loading %s: %s", input_dir, unexpected_keys)
        load_non_lora_state_dict(unwrapped, input_dir)
        if accelerator.mixed_precision == "fp16":
            cast_training_params([unwrapped], dtype=torch.float32)

    return save_model_hook, load_model_hook


def build_optimizer(
    transformer: torch.nn.Module,
    training_config: Dict[str, Any],
) -> torch.optim.Optimizer:
    optimizer_name = training_config.get("optimizer", "adamw").lower()
    if optimizer_name != "adamw":
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    params_to_optimize = [param for param in transformer.parameters() if param.requires_grad]
    return torch.optim.AdamW(
        params_to_optimize,
        lr=float(training_config["learning_rate"]),
        betas=(float(training_config.get("beta1", 0.9)), float(training_config.get("beta2", 0.999))),
        weight_decay=float(training_config.get("weight_decay", 0.0)),
    )


def encode_or_sample_video(
    value: torch.Tensor,
    vae,
    vae_scaling_factor: float,
    device: torch.device,
    weight_dtype: torch.dtype,
) -> torch.Tensor:
    value = value.to(device=device)
    if value.ndim == 4:
        value = value.unsqueeze(0)

    channels = value.shape[1]
    if channels <= 6:
        value = value.to(dtype=vae.dtype)
        with torch.no_grad():
            encoded = vae.encode(value)
        latent_dist = encoded.latent_dist if hasattr(encoded, "latent_dist") else encoded[0]
        latents = latent_dist.sample() * vae_scaling_factor
    else:
        latent_dist = DiagonalGaussianDistribution(value.to(dtype=weight_dtype))
        latents = latent_dist.sample() * vae_scaling_factor

    return latents.permute(0, 2, 1, 3, 4).to(device=device, dtype=weight_dtype).contiguous()


def run_validation(
    *,
    config: Dict[str, Any],
    accelerator: Accelerator,
    transformer: torch.nn.Module,
    weight_dtype: torch.dtype,
    output_dir: str,
    step: int,
    max_videos: int,
) -> None:
    if not accelerator.is_main_process:
        return

    data_config = config["data"]
    validation_set = data_config.get("validation_set")
    if not validation_set or max_videos <= 0:
        return

    pipeline = build_pipeline_from_config(
        config,
        transformer=accelerator.unwrap_model(transformer),
        torch_dtype=weight_dtype,
    )
    pipeline.to(device=accelerator.device, dtype=weight_dtype)

    custom_settings = config["training"].get("custom_settings", {})
    if custom_settings.get("enable_slicing", False):
        pipeline.vae.enable_slicing()
    if custom_settings.get("enable_tiling", False):
        pipeline.vae.enable_tiling()

    entries = read_dataset_entries(data_config["data_root"], validation_set)[:max_videos]
    save_dir = Path(output_dir) / "validation" / f"step-{step:06d}"
    save_dir.mkdir(parents=True, exist_ok=True)

    for relative_video_path in entries:
        sample_paths = resolve_sample_paths(
            data_root=data_config["data_root"],
            relative_video_path=relative_video_path,
            prompt_subdir=data_config.get("prompt_subdir", "prompts_ego_fun_rewrite"),
            prompt_embeds_subdir=data_config.get("prompt_embeds_subdir", "prompt_embeds_ego_fun_rewrite"),
            hand_video_subdir=data_config.get("hand_video_subdir", "videos_hands"),
            hand_latents_subdir=data_config.get("hand_video_latents_subdir", "hand_video_latents"),
            static_video_subdir=data_config.get("static_video_subdir", "videos_static"),
            static_latents_subdir=data_config.get("static_video_latents_subdir", "static_video_latents"),
            video_latents_subdir=data_config.get("video_latents_subdir", "video_latents"),
        )

        prompt = load_prompt_text(sample_paths.prompt_path)
        static_video = load_video_clip(
            sample_paths.static_video_path,
            max_num_frames=custom_settings.get("max_num_frames", 49),
            height=data_config.get("height_buckets", 480),
            width=data_config.get("width_buckets", 720),
        ).unsqueeze(0)
        hand_video = load_video_clip(
            sample_paths.hand_video_path,
            max_num_frames=custom_settings.get("max_num_frames", 49),
            height=data_config.get("height_buckets", 480),
            width=data_config.get("width_buckets", 720),
        ).unsqueeze(0)

        generated = pipeline(
            prompt=prompt,
            static_videos=static_video,
            hand_videos=hand_video,
            height=data_config.get("height_buckets", 480),
            width=data_config.get("width_buckets", 720),
            num_frames=custom_settings.get("max_num_frames", 49),
            num_inference_steps=data_config.get("validation_num_inference_steps", 50),
            guidance_scale=data_config.get("validation_guidance_scale", 6.0),
            use_dynamic_cfg=data_config.get("validation_use_dynamic_cfg", False),
            output_type="np",
        ).frames[0]

        stem = Path(relative_video_path).stem
        export_to_video(generated, str(save_dir / f"{stem}.mp4"), fps=data_config.get("validation_fps", 8))
        (save_dir / f"{stem}.txt").write_text(prompt, encoding="utf-8")

    del pipeline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    os.chdir(REPO_ROOT)

    config = load_experiment_config(args.experiment_config, args.override)
    print("Resolved pipeline type:", config["pipeline"]["type"])
    print("Pipeline class:", config["pipeline"]["class"])
    print("Transformer class:", config["pipeline"]["transformer_class"])
    if args.print_config_only:
        return

    experiment_config = config["experiment"]
    training_config = config["training"]
    data_config = config["data"]
    logging_config = config["logging"]
    log_with = normalize_log_with(logging_config.get("report_to"))

    output_dir = resolve_output_dir(experiment_config, training_config, args.mode)
    config["experiment"]["output_dir"] = output_dir
    copy_experiment_config(args.experiment_config, output_dir)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    init_process_group_kwargs = InitProcessGroupKwargs(backend="nccl", timeout=timedelta(seconds=1800))
    accelerator_project_config = ProjectConfiguration(project_dir=output_dir, logging_dir=Path(output_dir, "logs"))

    accelerator = Accelerator(
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        mixed_precision=training_config.get("custom_settings", {}).get("mixed_precision", "no"),
        log_with=log_with,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs, init_process_group_kwargs],
    )

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

    if training_config.get("seed") is not None:
        set_seed(training_config["seed"])

    pipeline = build_pipeline_from_config(config)
    transformer = pipeline.transformer
    scheduler = pipeline.scheduler
    vae = pipeline.vae
    text_encoder = pipeline.text_encoder
    tokenizer = pipeline.tokenizer

    num_trainable_parameters, lora_config = setup_training_mode(transformer, training_config)
    save_model_hook, load_model_hook = create_state_hooks(accelerator, transformer, config, lora_config)
    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    optimizer = build_optimizer(transformer, training_config)

    train_dataset = StaticHandConcatDataset(
        data_root=data_config["data_root"],
        dataset_file=data_config["dataset_file"],
        max_num_frames=training_config.get("custom_settings", {}).get("max_num_frames", 49),
        height=data_config.get("height_buckets", 480),
        width=data_config.get("width_buckets", 720),
        load_tensors=training_config.get("custom_settings", {}).get("load_tensors", True),
        prompt_subdir=data_config.get("prompt_subdir", "prompts_ego_fun_rewrite"),
        prompt_embeds_subdir=data_config.get("prompt_embeds_subdir", "prompt_embeds_ego_fun_rewrite"),
        hand_video_subdir=data_config.get("hand_video_subdir", "videos_hands"),
        hand_video_latents_subdir=data_config.get("hand_video_latents_subdir", "hand_video_latents"),
        static_video_subdir=data_config.get("static_video_subdir", "videos_static"),
        static_video_latents_subdir=data_config.get("static_video_latents_subdir", "static_video_latents"),
        video_latents_subdir=data_config.get("video_latents_subdir", "video_latents"),
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_config["batch_size"],
        shuffle=True,
        drop_last=True,
        collate_fn=collate_static_hand_batch,
        num_workers=data_config.get("dataloader_num_workers", 0),
        pin_memory=data_config.get("pin_memory", True),
    )

    if args.test_dataloader:
        run_dataloader_test(train_dataset, train_dataloader, args.test_dataloader_samples)
        return

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_config["gradient_accumulation_steps"])
    max_train_steps = training_config["max_train_steps"]
    num_epochs = max(1, math.ceil(max_train_steps / max(1, num_update_steps_per_epoch)))

    use_deepspeed_scheduler = (
        accelerator.state.deepspeed_plugin is not None
        and "scheduler" in accelerator.state.deepspeed_plugin.deepspeed_config
    )
    if use_deepspeed_scheduler:
        from accelerate.utils import DummyScheduler

        lr_scheduler = DummyScheduler(
            name=training_config.get("lr_scheduler", "cosine"),
            optimizer=optimizer,
            total_num_steps=max_train_steps * accelerator.num_processes,
            num_warmup_steps=training_config.get("lr_warmup_steps", 0) * accelerator.num_processes,
        )
    else:
        lr_scheduler = get_scheduler(
            training_config.get("lr_scheduler", "cosine"),
            optimizer=optimizer,
            num_warmup_steps=training_config.get("lr_warmup_steps", 0) * accelerator.num_processes,
            num_training_steps=max_train_steps * accelerator.num_processes,
            num_cycles=training_config.get("lr_num_cycles", 1),
        )

    weight_dtype = get_weight_dtype(accelerator)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    if accelerator.mixed_precision == "fp16":
        cast_training_params([transformer], dtype=torch.float32)

    if training_config.get("custom_settings", {}).get("allow_tf32", False) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer,
        optimizer,
        train_dataloader,
        lr_scheduler,
    )

    if log_with is not None and (accelerator.distributed_type == DistributedType.DEEPSPEED or accelerator.is_main_process):
        exp_name = experiment_config.get("name", "unknown_experiment")
        exp_date = experiment_config.get("date", "unknown_date")
        try:
            date_suffix = datetime.strptime(exp_date, "%Y-%m-%d").strftime("%y%m%d")
        except ValueError:
            date_suffix = "unknown"
        slurm_job_id = os.environ.get("SLURM_JOB_ID", "local")
        run_name = f"{date_suffix}_{exp_name}_{args.mode}_{slurm_job_id}"

        tracker_config = {
            "experiment": {
                "name": exp_name,
                "date": exp_date,
                "description": experiment_config.get("description", ""),
                "output_dir": output_dir,
            },
            "pipeline": config["pipeline"],
            "training": {
                "mode": training_config.get("mode"),
                "learning_rate": training_config.get("learning_rate"),
                "batch_size": training_config.get("batch_size"),
                "max_train_steps": training_config.get("max_train_steps"),
                "num_trainable_parameters": num_trainable_parameters,
            },
            "model": config["model"],
        }
        wandb_kwargs = {"name": run_name, "entity": logging_config.get("entity_name", "vclab_2024")}
        if args.mode in {"debug", "slurm_test"}:
            wandb_kwargs["mode"] = "offline"
        elif args.mode == "batch":
            wandb_kwargs["mode"] = "disabled"

        accelerator.init_trackers(
            project_name=logging_config.get("project_name", "world_model"),
            config=tracker_config,
            init_kwargs={"wandb": wandb_kwargs},
        )

    accelerator.print("***** Running training *****")
    accelerator.print(f"  Num trainable parameters = {num_trainable_parameters}")
    accelerator.print(f"  Num examples = {len(train_dataset)}")
    accelerator.print(f"  Num batches each epoch = {len(train_dataloader)}")
    accelerator.print(f"  Num epochs = {num_epochs}")
    accelerator.print(f"  Max train steps = {max_train_steps}")
    accelerator.print(f"  Instantaneous batch size per device = {training_config['batch_size']}")

    prompt_dropout_prob = training_config.get("prompt_dropout_prob", 0.0)
    hand_dropout_prob = training_config.get("hand_dropout_prob", 0.0)
    load_tensors = training_config.get("custom_settings", {}).get("load_tensors", True)
    max_text_seq_length = getattr(transformer.config, "max_text_seq_length", 226)

    global_step = 0
    first_epoch = 0
    if training_config.get("resume_from_checkpoint"):
        checkpoint_path = training_config["resume_from_checkpoint"]
        if checkpoint_path == "latest":
            checkpoint_dirs = [
                directory
                for directory in os.listdir(output_dir)
                if directory.startswith("checkpoint-") and directory.split("-")[-1].isdigit()
            ]
            checkpoint_path = None
            if checkpoint_dirs:
                checkpoint_dirs = sorted(checkpoint_dirs, key=lambda name: int(name.split("-")[1]))
                checkpoint_path = os.path.join(output_dir, checkpoint_dirs[-1])
        if checkpoint_path and os.path.exists(checkpoint_path):
            accelerator.print(f"Resuming from checkpoint {checkpoint_path}")
            accelerator.load_state(checkpoint_path)
            checkpoint_name = os.path.basename(checkpoint_path.rstrip("/"))
            if checkpoint_name.startswith("checkpoint-"):
                global_step = int(checkpoint_name.split("-")[1])
                first_epoch = global_step // max(1, num_update_steps_per_epoch)

    if args.save_initial_checkpoints and (accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED):
        maybe_save_checkpoint(accelerator, output_dir, global_step)
        accelerator.wait_for_everyone()

    progress_bar = tqdm(
        range(global_step, max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    vae_scaling_factor = pipeline.vae.config.scaling_factor
    alphas_cumprod = scheduler.alphas_cumprod.to(accelerator.device, dtype=torch.float32)
    empty_prompt_embed = None
    if prompt_dropout_prob > 0:
        empty_prompt_embed = get_t5_prompt_embeds(
            "",
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            device=accelerator.device,
            dtype=weight_dtype,
            max_sequence_length=max_text_seq_length,
        )

    if data_config.get("validation_set"):
        run_validation(
            config=config,
            accelerator=accelerator,
            transformer=transformer,
            weight_dtype=weight_dtype,
            output_dir=output_dir,
            step=global_step,
            max_videos=min(data_config.get("max_validation_videos", 4), data_config.get("num_validation_videos", 2)),
        )

    for epoch in range(first_epoch, num_epochs):
        transformer.train()
        for batch in train_dataloader:
            logs: Dict[str, Any] = {}
            with accelerator.accumulate(transformer):
                videos = batch["videos"].to(accelerator.device, non_blocking=True)
                hand_videos = batch["hand_videos"].to(accelerator.device, non_blocking=True)
                static_videos = batch["static_videos"].to(accelerator.device, non_blocking=True)
                prompts = batch["prompts"]

                videos_latents = encode_or_sample_video(
                    videos,
                    vae=pipeline.vae,
                    vae_scaling_factor=vae_scaling_factor,
                    device=accelerator.device,
                    weight_dtype=weight_dtype,
                )
                hand_videos_latents = encode_or_sample_video(
                    hand_videos,
                    vae=pipeline.vae,
                    vae_scaling_factor=vae_scaling_factor,
                    device=accelerator.device,
                    weight_dtype=weight_dtype,
                )
                static_videos_latents = encode_or_sample_video(
                    static_videos,
                    vae=pipeline.vae,
                    vae_scaling_factor=vae_scaling_factor,
                    device=accelerator.device,
                    weight_dtype=weight_dtype,
                )

                if load_tensors:
                    prompt_embeds = prompts.to(device=accelerator.device, dtype=weight_dtype)
                else:
                    prompt_embeds = get_t5_prompt_embeds(
                        list(prompts),
                        tokenizer=tokenizer,
                        text_encoder=text_encoder,
                        device=accelerator.device,
                        dtype=weight_dtype,
                        max_sequence_length=max_text_seq_length,
                    )

                if prompt_dropout_prob > 0 and empty_prompt_embed is not None:
                    dropout_mask = torch.rand(prompt_embeds.shape[0], device=accelerator.device) < prompt_dropout_prob
                    if dropout_mask.any():
                        prompt_embeds[dropout_mask] = empty_prompt_embed.to(
                            device=prompt_embeds.device,
                            dtype=prompt_embeds.dtype,
                        )

                if hand_dropout_prob > 0:
                    dropout_mask = torch.rand(hand_videos_latents.shape[0], device=accelerator.device) < hand_dropout_prob
                    if dropout_mask.any():
                        hand_videos_latents[dropout_mask] = 0

                noise = torch.randn_like(videos_latents)
                batch_size, num_frames, _, latent_height, latent_width = videos_latents.shape
                timesteps = torch.randint(
                    0,
                    scheduler.config.num_train_timesteps,
                    (batch_size,),
                    dtype=torch.int64,
                    device=accelerator.device,
                )

                image_rotary_emb = (
                    pipeline._prepare_rotary_positional_embeddings(
                        latent_height * pipeline.vae_scale_factor_spatial,
                        latent_width * pipeline.vae_scale_factor_spatial,
                        num_frames,
                        accelerator.device,
                    )
                    if transformer.config.use_rotary_positional_embeddings
                    else None
                )

                noisy_model_input = scheduler.add_noise(videos_latents, noise, timesteps)
                mask_input = torch.ones_like(noisy_model_input[:, :, :1]) * vae_scaling_factor
                transformer_input = torch.cat(
                    [noisy_model_input, mask_input, static_videos_latents, hand_videos_latents],
                    dim=2,
                )

                model_output = transformer(
                    hidden_states=transformer_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timesteps,
                    image_rotary_emb=image_rotary_emb,
                    return_dict=False,
                )[0]
                model_pred = scheduler.get_velocity(model_output, noisy_model_input, timesteps)

                weights = 1 / (1 - alphas_cumprod[timesteps])
                while len(weights.shape) < len(model_pred.shape):
                    weights = weights.unsqueeze(-1)
                loss = torch.mean((weights * (model_pred - videos_latents) ** 2).reshape(batch_size, -1), dim=1).mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients and accelerator.distributed_type != DistributedType.DEEPSPEED:
                    accelerator.clip_grad_norm_(transformer.parameters(), training_config.get("max_grad_norm", 1.0))

                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                last_lr = lr_scheduler.get_last_lr()[0]
                logs.update({"loss": loss.detach().item(), "lr": last_lr, "epoch": epoch, "step": global_step})
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                checkpointing_steps = training_config.get("custom_settings", {}).get("checkpointing_steps", 500)
                if checkpointing_steps and global_step % checkpointing_steps == 0:
                    if accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED:
                        maybe_save_checkpoint(accelerator, output_dir, global_step)
                    accelerator.wait_for_everyone()

                validation_steps = data_config.get("validation_steps", 0)
                if validation_steps and data_config.get("validation_set") and global_step % validation_steps == 0:
                    run_validation(
                        config=config,
                        accelerator=accelerator,
                        transformer=transformer,
                        weight_dtype=weight_dtype,
                        output_dir=output_dir,
                        step=global_step,
                        max_videos=data_config.get("num_validation_videos", 2),
                    )

                if global_step >= max_train_steps:
                    break

        if global_step >= max_train_steps:
            break

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        trained_transformer = accelerator.unwrap_model(transformer)
        save_dtype = (
            torch.float16
            if training_config.get("custom_settings", {}).get("mixed_precision") == "fp16"
            else torch.bfloat16
            if training_config.get("custom_settings", {}).get("mixed_precision") == "bf16"
            else torch.float32
        )
        trained_transformer = trained_transformer.to(save_dtype)
        final_pipeline = build_pipeline_from_config(
            config,
            transformer=trained_transformer,
            torch_dtype=save_dtype,
        )
        final_pipeline.save_pretrained(output_dir)
        if training_config.get("mode", "lora") == "lora":
            CogVideoXFunStaticHandConcatPipeline.save_lora_weights(
                output_dir,
                transformer_lora_layers=get_peft_model_state_dict(trained_transformer),
            )
            save_non_lora_state_dict(output_dir, collect_non_lora_state_dict(trained_transformer))

        if data_config.get("validation_set"):
            run_validation(
                config=config,
                accelerator=accelerator,
                transformer=trained_transformer,
                weight_dtype=weight_dtype,
                output_dir=output_dir,
                step=global_step,
                max_videos=min(data_config.get("max_validation_videos", 4), 4),
            )

    accelerator.end_training()


if __name__ == "__main__":
    main()
