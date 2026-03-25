from __future__ import annotations

import inspect
import math
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch

from training.cogvideox.diffusers_compat import disable_broken_torchao

disable_broken_torchao()

from diffusers import CogVideoXDPMScheduler, CogVideoXImageToVideoPipeline
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.pipelines.cogvideo.pipeline_output import CogVideoXPipelineOutput
from diffusers.utils.torch_utils import randn_tensor
from safetensors.torch import load_file

from training.cogvideox.models import CogVideoXFunStaticHandConcatTransformer3DModel
from training.cogvideox.static_hand_utils import coerce_video_tensor


def retrieve_timesteps(
    scheduler,
    num_inference_steps: int,
    device: torch.device,
    timesteps: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, int]:
    if timesteps is None:
        scheduler.set_timesteps(num_inference_steps, device=device)
        return scheduler.timesteps, num_inference_steps

    accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
    if not accepts_timesteps:
        raise ValueError(f"Scheduler {scheduler.__class__.__name__} does not support custom timesteps.")
    scheduler.set_timesteps(timesteps=timesteps, device=device)
    return scheduler.timesteps, len(scheduler.timesteps)


class CogVideoXFunStaticHandConcatPipeline(CogVideoXImageToVideoPipeline):
    """Standalone inference pipeline for the DWM static-scene plus hand-latent concat model."""

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[str] = None,
        base_model_name_or_path: Optional[str] = None,
        transformer: Optional[CogVideoXFunStaticHandConcatTransformer3DModel] = None,
        condition_channels: int = 16,
        *args,
        **kwargs,
    ) -> "CogVideoXFunStaticHandConcatPipeline":
        if base_model_name_or_path is None:
            raise ValueError("base_model_name_or_path must be provided.")

        base_pipeline = super().from_pretrained(base_model_name_or_path, *args, **kwargs)
        if transformer is None:
            transformer = CogVideoXFunStaticHandConcatTransformer3DModel.from_pretrained(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                base_model_name_or_path=base_model_name_or_path,
                subfolder="transformer",
                condition_channels=condition_channels,
                torch_dtype=kwargs.get("torch_dtype"),
                revision=kwargs.get("revision"),
                variant=kwargs.get("variant"),
            )

        return cls(
            tokenizer=base_pipeline.tokenizer,
            text_encoder=base_pipeline.text_encoder,
            vae=base_pipeline.vae,
            transformer=transformer,
            scheduler=base_pipeline.scheduler,
        )

    def save_pretrained(self, save_directory: str, **kwargs) -> None:
        os.makedirs(save_directory, exist_ok=True)
        self.tokenizer.save_pretrained(os.path.join(save_directory, "tokenizer"))
        self.text_encoder.save_pretrained(os.path.join(save_directory, "text_encoder"))
        self.vae.save_pretrained(os.path.join(save_directory, "vae"))
        self.transformer.save_pretrained(os.path.join(save_directory, "transformer"))
        self.scheduler.save_pretrained(os.path.join(save_directory, "scheduler"))

    @classmethod
    def lora_state_dict(cls, input_dir: str) -> Dict[str, torch.Tensor]:
        safetensor_path = os.path.join(input_dir, "pytorch_lora_weights.safetensors")
        if os.path.exists(safetensor_path):
            return load_file(safetensor_path)

        bin_path = os.path.join(input_dir, "pytorch_lora_weights.bin")
        if os.path.exists(bin_path):
            return torch.load(bin_path, map_location="cpu")

        return {}

    @staticmethod
    def _sample_latents(encoder_output: Any) -> torch.Tensor:
        if hasattr(encoder_output, "latent_dist"):
            return encoder_output.latent_dist.sample()
        if isinstance(encoder_output, tuple) and hasattr(encoder_output[0], "sample"):
            return encoder_output[0].sample()
        if hasattr(encoder_output, "sample"):
            return encoder_output.sample()
        raise TypeError(f"Unsupported encoder output type: {type(encoder_output)}")

    def _encode_condition_video(
        self,
        video: torch.Tensor,
        *,
        num_frames: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        video = coerce_video_tensor(video, target_frames=num_frames, height=height, width=width)
        video = video.to(device=device, dtype=self.vae.dtype)

        latents = []
        with torch.no_grad():
            for index in range(video.shape[0]):
                encoded = self.vae.encode(video[index : index + 1])
                latents.append(self._sample_latents(encoded))

        video_latents = torch.cat(latents, dim=0) * self.vae.config.scaling_factor
        return video_latents.permute(0, 2, 1, 3, 4).to(device=device, dtype=dtype).contiguous()

    def _prepare_noise_latents(
        self,
        *,
        batch_size: int,
        num_channels_latents: int,
        num_frames: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]],
        latents: Optional[torch.Tensor],
    ) -> torch.Tensor:
        shape = (
            batch_size,
            num_frames,
            num_channels_latents,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        return latents * self.scheduler.init_noise_sigma

    def check_inputs(
        self,
        prompt: Optional[Union[str, List[str]]],
        static_videos: Optional[torch.Tensor],
        hand_videos: Optional[torch.Tensor],
        height: int,
        width: int,
        prompt_embeds: Optional[torch.Tensor],
    ) -> None:
        if static_videos is None:
            raise ValueError("static_videos must be provided.")
        if hand_videos is None:
            raise ValueError("hand_videos must be provided.")
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"height and width must be divisible by 8, got {height} and {width}.")
        if prompt is None and prompt_embeds is None:
            raise ValueError("Either prompt or prompt_embeds must be provided.")

    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        static_videos: Optional[torch.Tensor] = None,
        hand_videos: Optional[torch.Tensor] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: int = 49,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6.0,
        use_dynamic_cfg: bool = False,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: str = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict[str, Any]], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 226,
    ) -> Union[CogVideoXPipelineOutput, Tuple]:
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        height = height or self.transformer.config.sample_height * self.vae_scale_factor_spatial
        width = width or self.transformer.config.sample_width * self.vae_scale_factor_spatial
        self.check_inputs(prompt, static_videos, hand_videos, height, width, prompt_embeds)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            num_videos_per_prompt=1,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps=num_inference_steps,
            device=device,
            timesteps=timesteps,
        )
        self._guidance_scale = guidance_scale
        self._num_timesteps = len(timesteps)
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_channels = self.vae.config.latent_channels
        latents = self._prepare_noise_latents(
            batch_size=batch_size,
            num_channels_latents=latent_channels,
            num_frames=latent_frames,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
            latents=latents,
        )
        static_latents = self._encode_condition_video(
            static_videos,
            num_frames=num_frames,
            height=height,
            width=width,
            device=device,
            dtype=prompt_embeds.dtype,
        )
        hand_latents = self._encode_condition_video(
            hand_videos,
            num_frames=num_frames,
            height=height,
            width=width,
            device=device,
            dtype=prompt_embeds.dtype,
        )
        mask_latents = torch.ones_like(latents[:, :, :1]) * self.vae.config.scaling_factor

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta=0.0)
        image_rotary_emb = (
            self._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
            if self.transformer.config.use_rotary_positional_embeddings
            else None
        )

        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        old_pred_original_sample = None

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for index, timestep_value in enumerate(timesteps):
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep_value)

                static_input = torch.cat([static_latents] * 2) if do_classifier_free_guidance else static_latents
                hand_input = torch.cat([hand_latents] * 2) if do_classifier_free_guidance else hand_latents
                mask_input = torch.cat([mask_latents] * 2) if do_classifier_free_guidance else mask_latents

                transformer_input = torch.cat(
                    [latent_model_input, mask_input, static_input, hand_input],
                    dim=2,
                )
                timestep_batch = timestep_value.expand(transformer_input.shape[0])

                noise_pred = self.transformer(
                    hidden_states=transformer_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep_batch,
                    image_rotary_emb=image_rotary_emb,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0].float()

                if use_dynamic_cfg:
                    self._guidance_scale = 1 + guidance_scale * (
                        (1 - math.cos(math.pi * ((num_inference_steps - timestep_value.item()) / num_inference_steps) ** 5.0)) / 2
                    )

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if not isinstance(self.scheduler, CogVideoXDPMScheduler):
                    latents = self.scheduler.step(
                        noise_pred,
                        timestep_value,
                        latents,
                        **extra_step_kwargs,
                        return_dict=False,
                    )[0]
                else:
                    latents, old_pred_original_sample = self.scheduler.step(
                        noise_pred,
                        old_pred_original_sample,
                        timestep_value,
                        timesteps[index - 1] if index > 0 else None,
                        latents,
                        **extra_step_kwargs,
                        return_dict=False,
                    )
                latents = latents.to(prompt_embeds.dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    local_vars = {
                        "latents": latents,
                        "prompt_embeds": prompt_embeds,
                        "negative_prompt_embeds": negative_prompt_embeds,
                    }
                    for name in callback_on_step_end_tensor_inputs:
                        if name in local_vars:
                            callback_kwargs[name] = local_vars[name]
                    callback_outputs = callback_on_step_end(self, index, timestep_value, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if index == len(timesteps) - 1 or ((index + 1) > num_warmup_steps and (index + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if output_type == "latent":
            video = latents
        else:
            video = self.decode_latents(latents)
            video = self.video_processor.postprocess_video(video=video, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)
        return CogVideoXPipelineOutput(frames=video)


__all__ = ["CogVideoXFunStaticHandConcatPipeline"]
