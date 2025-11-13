import inspect
import math
import types
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL
import torch
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDDIMScheduler,
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
)
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import BaseOutput, logging
from transformers import T5EncoderModel, T5Tokenizer

from cogvideox_fun_static_to_video_pose_concat_pipeline import (
    CogVideoXFunPipelineOutput,
    CogVideoXFunStaticToVideoPipeline,
)
from cogvideox_fun_transformer_with_conditions import (
    CogVideoXFunTransformer3DModelWithAdaLNPose,
    CogVideoXFunTransformer3DModelWithAdaLNPosePerFrame,
)


logger = logging.get_logger(__name__)


@dataclass
class CogVideoXFunStaticToVideoPoseAdaLNPipelineOutput(BaseOutput):
    frames: Union[List[PIL.Image.Image], np.ndarray]


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one.")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"Scheduler {scheduler.__class__} does not support custom timesteps."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"Scheduler {scheduler.__class__} does not support custom sigmas."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class CogVideoXFunStaticToVideoPoseAdaLNPipeline(CogVideoXFunStaticToVideoPipeline):
    """Static-to-video pipeline with AdaLN pose conditioning (global embedding)."""

    def __init__(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        vae: AutoencoderKLCogVideoX,
        transformer: CogVideoXFunTransformer3DModelWithAdaLNPose,
        scheduler: Union[CogVideoXDDIMScheduler, CogVideoXDPMScheduler],
    ):
        super().__init__(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
        )

        if not isinstance(
            transformer,
            (CogVideoXFunTransformer3DModelWithAdaLNPose, CogVideoXFunTransformer3DModelWithAdaLNPosePerFrame),
        ):
            raise ValueError(
                "Transformer must be an instance of CogVideoXFunTransformer3DModelWithAdaLNPose or CogVideoXFunTransformer3DModelWithAdaLNPosePerFrame"
            )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path=None,
        base_model_name_or_path="alibaba-pai/CogVideoX-Fun-V1.1-5b-InP",
        transformer=None,
        pose_dim: int = 102,
        pose_embed_dim: int = 512,
        *args,
        **kwargs,
    ):
        if base_model_name_or_path is not None:
            logger.info(
                "🔧 Creating Fun static-to-video AdaLN pipeline from base model: %s",
                base_model_name_or_path,
            )
            original_pipeline = CogVideoXImageToVideoPipeline.from_pretrained(
                base_model_name_or_path, *args, **kwargs
            )

        if transformer is None:
            logger.info("🔧 Creating/loading Fun AdaLN transformer")
            transformer = CogVideoXFunTransformer3DModelWithAdaLNPose.from_pretrained(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                base_model_name_or_path=base_model_name_or_path,
                torch_dtype=kwargs.get("torch_dtype", torch.bfloat16),
                pose_dim=pose_dim,
                pose_embed_dim=pose_embed_dim,
            )
        else:
            transformer = transformer

        pipeline = cls(
            tokenizer=original_pipeline.tokenizer,
            text_encoder=original_pipeline.text_encoder,
            vae=original_pipeline.vae,
            transformer=transformer,
            scheduler=original_pipeline.scheduler,
        )

        return pipeline

    def preprocess_pose_params(
        self,
        pose_params: Optional[Union[torch.Tensor, np.ndarray]] = None,
        pose_params_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
        num_frames: int = 49,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if pose_params is not None:
            if isinstance(pose_params, np.ndarray):
                pose_params = torch.from_numpy(pose_params)
            pose_params = pose_params.to(device=self._execution_device, dtype=self.transformer.dtype)

            if pose_params.dim() != 3 or pose_params.shape[-1] != self.transformer.pose_dim:
                raise ValueError(
                    "pose_params must have shape (batch, frames, 63)"
                )

        if pose_params_mask is not None:
            if isinstance(pose_params_mask, np.ndarray):
                pose_params_mask = torch.from_numpy(pose_params_mask)
            pose_params_mask = pose_params_mask.to(
                device=self._execution_device, dtype=torch.bool
            )
            if pose_params is not None and pose_params_mask.shape[0] != pose_params.shape[0]:
                raise ValueError(
                    "pose_params_mask batch size must match pose_params"
                )

        return pose_params, pose_params_mask

    @torch.no_grad()
    def __call__(
        self,
        static_videos: Optional[Union[torch.Tensor, np.ndarray, List[PIL.Image.Image]]] = None,
        image: Optional[Union[torch.Tensor, np.ndarray, PIL.Image.Image]] = None,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: int = 49,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6.0,
        use_dynamic_cfg: bool = False,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pose_params: Optional[Union[torch.Tensor, np.ndarray]] = None,
        pose_params_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
        mask_video: Optional[Union[torch.Tensor, np.ndarray]] = None,
        output_type: str = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Union[Callable[[int, int, Dict], None]]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 226,
    ) -> Union[CogVideoXFunStaticToVideoPoseAdaLNPipelineOutput, Tuple]:
        pose_params, pose_params_mask = self.preprocess_pose_params(
            pose_params, pose_params_mask, num_frames=num_frames
        )

        height = height or self.transformer.config.sample_height * self.vae_scale_factor_spatial
        width = width or self.transformer.config.sample_width * self.vae_scale_factor_spatial
        num_frames = num_frames or self.transformer.config.sample_frames

        orig_forward = self.transformer.forward

        def forward_with_pose(
            inner_self,
            hidden_states,
            encoder_hidden_states,
            timestep,
            *inner_args,
            **inner_kwargs,
        ):
            if pose_params is not None:
                pose_to_use = pose_params
                mask_to_use = pose_params_mask

                if hidden_states.shape[0] != pose_to_use.shape[0]:
                    repeat_factor = hidden_states.shape[0] // pose_to_use.shape[0]
                    pose_to_use = pose_to_use.repeat_interleave(repeat_factor, dim=0)
                    if mask_to_use is not None:
                        mask_to_use = mask_to_use.repeat_interleave(repeat_factor, dim=0)

                inner_kwargs["pose_params"] = pose_to_use
                if mask_to_use is not None:
                    inner_kwargs["pose_params_mask"] = mask_to_use

            return orig_forward(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                *inner_args,
                **inner_kwargs,
            )

        self.transformer.forward = types.MethodType(forward_with_pose, self.transformer)

        try:
            base_output = super().__call__(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                image=image,
                static_videos=static_videos,
                mask_video=mask_video,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                timesteps=timesteps,
                guidance_scale=guidance_scale,
                use_dynamic_cfg=use_dynamic_cfg,
                num_videos_per_prompt=num_videos_per_prompt,
                eta=eta,
                generator=generator,
                latents=latents,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                output_type=output_type,
                return_dict=return_dict,
                attention_kwargs=attention_kwargs,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                max_sequence_length=max_sequence_length,
            )
        finally:
            self.transformer.forward = orig_forward

        if not return_dict:
            return base_output

        frames: Union[torch.Tensor, np.ndarray, List[PIL.Image.Image]]
        if isinstance(base_output, CogVideoXFunPipelineOutput):
            if hasattr(base_output, "frames"):
                frames = base_output.frames
            elif hasattr(base_output, "videos"):
                frames = base_output.videos
            else:
                frames = base_output
        elif isinstance(base_output, tuple):
            frames = base_output[0]
        else:
            frames = base_output

        return CogVideoXFunStaticToVideoPoseAdaLNPipelineOutput(frames=frames)

    @classmethod
    def lora_state_dict(cls, input_dir: str) -> Dict[str, torch.Tensor]:
        """Load LoRA state dict from a directory."""
        lora_state_dict = {}
        
        # Try to load from safetensors first
        import os
        from safetensors.torch import load_file
        
        safetensors_path = os.path.join(input_dir, "pytorch_lora_weights.safetensors")
        if os.path.exists(safetensors_path):
            lora_state_dict = load_file(safetensors_path)
        else:
            # Try to load from pytorch format
            pytorch_path = os.path.join(input_dir, "pytorch_lora_weights.bin")
            if os.path.exists(pytorch_path):
                lora_state_dict = torch.load(pytorch_path, map_location="cpu")
        
        return lora_state_dict


class CogVideoXFunStaticToVideoPoseAdaLNPerFramePipeline(
    CogVideoXFunStaticToVideoPoseAdaLNPipeline
):
    """Static-to-video pipeline with per-frame AdaLN pose conditioning."""

    def __init__(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        vae: AutoencoderKLCogVideoX,
        transformer: CogVideoXFunTransformer3DModelWithAdaLNPosePerFrame,
        scheduler: Union[CogVideoXDDIMScheduler, CogVideoXDPMScheduler],
    ):
        super().__init__(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
        )

        if not isinstance(transformer, CogVideoXFunTransformer3DModelWithAdaLNPosePerFrame):
            raise ValueError(
                "Transformer must be an instance of CogVideoXFunTransformer3DModelWithAdaLNPosePerFrame"
            )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path=None,
        base_model_name_or_path="alibaba-pai/CogVideoX-Fun-V1.1-5b-InP",
        transformer=None,
        pose_dim: int = 102,
        pose_embed_dim: int = 512,
        *args,
        **kwargs,
    ):
        if base_model_name_or_path is not None:
            logger.info(
                "🔧 Creating Fun static-to-video AdaLN per-frame pipeline from base model: %s",
                base_model_name_or_path,
            )
            base_pipeline_kwargs = dict(kwargs)
            base_pipeline_kwargs.pop("pretrained_model_name_or_path", None)
            base_pipeline_kwargs.pop("base_model_name_or_path", None)
            original_pipeline = CogVideoXFunStaticToVideoPipeline.from_pretrained(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                base_model_name_or_path=base_model_name_or_path,
                *args,
                **base_pipeline_kwargs,
            )

        if transformer is None:
            logger.info("🔧 Creating/loading Fun AdaLN per-frame transformer")
            transformer = CogVideoXFunTransformer3DModelWithAdaLNPosePerFrame.from_pretrained(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                base_model_name_or_path=base_model_name_or_path,
                torch_dtype=kwargs.get("torch_dtype", torch.bfloat16),
                pose_dim=pose_dim,
                pose_embed_dim=pose_embed_dim,
            )
        else:
            transformer = transformer

        pipeline = cls(
            tokenizer=original_pipeline.tokenizer,
            text_encoder=original_pipeline.text_encoder,
            vae=original_pipeline.vae,
            transformer=transformer,
            scheduler=original_pipeline.scheduler,
        )

        return pipeline

    @classmethod
    def lora_state_dict(cls, input_dir: str) -> Dict[str, torch.Tensor]:
        """Load LoRA state dict from a directory."""
        # Use parent class method
        return CogVideoXFunStaticToVideoPoseAdaLNPipeline.lora_state_dict(input_dir)

