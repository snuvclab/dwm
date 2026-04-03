"""
WanFunInpaintHandVacePipeline: WAN Fun Inpaint pipeline with VACE-style hand conditioning.

This pipeline extends WanFunInpaintPipeline to support hand video conditioning
through the VACE (Video Anything Controllable Editor) mechanism.

Key differences from WanFunInpaintHandConcatPipeline:
    - Uses vace_context instead of condition_latents
    - Hand condition is processed through parallel VACE blocks
    - Hints are injected via skip connections (not channel concatenation)

Architecture:
    - Hand latents are passed as vace_context to the transformer
    - Transformer processes them through vace_blocks to generate hints
    - Hints are added to main block outputs with vace_context_scale
"""

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.utils import logging
from einops import rearrange
from PIL import Image

from ..models import AutoTokenizer, CLIPModel, WanT5EncoderModel, AutoencoderKLWan
from ..models.wan_transformer3d_vace import WanTransformer3DVace
from ..utils.fm_solvers import FlowDPMSolverMultistepScheduler, get_sampling_sigmas
from ..utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .pipeline_wan_fun_inpaint import (
    WanFunInpaintPipeline,
    WanPipelineOutput,
    resize_mask,
    retrieve_timesteps,
)

logger = logging.get_logger(__name__)


class WanFunInpaintHandVacePipeline(WanFunInpaintPipeline):
    """
    WAN Fun Inpaint Pipeline with VACE-style Hand Video Conditioning.

    This pipeline uses the VACE mechanism for conditioning:
    - Hand video is encoded to latents and passed as vace_context
    - VACE blocks process the condition in parallel with main blocks
    - Hints from VACE blocks are injected via skip connections

    Args:
        tokenizer: T5 tokenizer
        text_encoder: WAN T5 text encoder
        vae: WAN VAE
        transformer: WanTransformer3DVace (with VACE blocks)
        clip_image_encoder: CLIP image encoder
        scheduler: Flow matching scheduler

    Input mapping:
        - static_video -> init video for inpainting
        - hand_video -> vace_context (processed by VACE blocks)
        - mask_video -> inpaint mask
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: WanT5EncoderModel,
        vae: AutoencoderKLWan,
        transformer: WanTransformer3DVace,
        clip_image_encoder: CLIPModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
    ):
        super().__init__(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            transformer=transformer,
            clip_image_encoder=clip_image_encoder,
            scheduler=scheduler,
        )

    def prepare_vace_context(
        self,
        hand_video: torch.FloatTensor,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[torch.Generator] = None,
    ) -> List[torch.Tensor]:
        """
        Prepare VACE context from hand video.

        Encodes hand video to latents and returns as list format
        expected by VACE transformer.

        Args:
            hand_video: Hand video tensor [B, C, F, H, W]
            height: Target height
            width: Target width
            dtype: Target dtype
            device: Target device
            generator: Random generator

        Returns:
            vace_context: List of [C, F', H', W'] tensors (one per batch)
        """
        batch_size = hand_video.shape[0]
        video_length = hand_video.shape[2]

        # Preprocess hand video
        hand_video_processed = self.image_processor.preprocess(
            rearrange(hand_video, "b c f h w -> (b f) c h w"),
            height=height,
            width=width
        )
        hand_video_processed = hand_video_processed.to(dtype=torch.float32)
        hand_video_processed = rearrange(
            hand_video_processed, "(b f) c h w -> b c f h w", f=video_length
        )
        hand_video_processed = hand_video_processed.to(device=device, dtype=dtype)

        # Encode to latents
        hand_latents = self.vae.encode(hand_video_processed).latent_dist.sample(generator)

        # Convert to list format expected by VACE transformer
        # Each element is [C, F', H', W']
        vace_context = [hand_latents[i] for i in range(batch_size)]

        return vace_context

    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 480,
        width: int = 720,
        static_video: Union[torch.FloatTensor] = None,
        mask_video: Union[torch.FloatTensor] = None,
        hand_video: Union[torch.FloatTensor] = None,
        num_frames: int = 49,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "numpy",
        return_dict: bool = False,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        clip_image: Image.Image = None,
        max_sequence_length: int = 512,
        comfyui_progressbar: bool = False,
        shift: float = 3.0,
        # VACE-specific parameter
        vace_context_scale: float = 1.0,
    ) -> Union[WanPipelineOutput, Tuple]:
        """
        Generate video with VACE-style hand conditioning.

        Args:
            prompt: Text prompt for generation
            negative_prompt: Negative text prompt
            height: Video height
            width: Video width
            static_video: Static/background video for inpainting [B, C, F, H, W]
            mask_video: Inpainting mask video [B, C, F, H, W]
            hand_video: Hand condition video [B, C, F, H, W]
            num_frames: Number of frames to generate
            num_inference_steps: Denoising steps
            timesteps: Custom timesteps
            guidance_scale: Classifier-free guidance scale
            num_videos_per_prompt: Videos per prompt
            eta: DDIM eta parameter
            generator: Random generator
            latents: Initial latents
            prompt_embeds: Pre-computed prompt embeddings
            negative_prompt_embeds: Pre-computed negative prompt embeddings
            output_type: Output format ("numpy", "latent", etc.)
            return_dict: Return as dict
            callback_on_step_end: Callback function
            attention_kwargs: Additional attention kwargs
            callback_on_step_end_tensor_inputs: Tensor inputs for callback
            clip_image: CLIP image for conditioning
            max_sequence_length: Max text sequence length
            comfyui_progressbar: Use ComfyUI progress bar
            shift: Scheduler shift parameter
            vace_context_scale: Scale for VACE hint injection (default 1.0)

        Returns:
            WanPipelineOutput with generated videos
        """
        # 0. Default values
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            callback_on_step_end_tensor_inputs,
            prompt_embeds,
            negative_prompt_embeds,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        weight_dtype = self.transformer.dtype

        do_classifier_free_guidance = self.guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        if do_classifier_free_guidance:
            in_prompt_embeds = negative_prompt_embeds + prompt_embeds
        else:
            in_prompt_embeds = prompt_embeds

        # 4. Prepare timesteps
        if isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler):
            timesteps, num_inference_steps = retrieve_timesteps(
                self.scheduler, num_inference_steps, device, timesteps, mu=1
            )
        elif isinstance(self.scheduler, FlowUniPCMultistepScheduler):
            self.scheduler.set_timesteps(num_inference_steps, device=device, shift=shift)
            timesteps = self.scheduler.timesteps
        elif isinstance(self.scheduler, FlowDPMSolverMultistepScheduler):
            sampling_sigmas = get_sampling_sigmas(num_inference_steps, shift)
            timesteps, _ = retrieve_timesteps(
                self.scheduler,
                device=device,
                sigmas=sampling_sigmas
            )
        else:
            timesteps, num_inference_steps = retrieve_timesteps(
                self.scheduler, num_inference_steps, device, timesteps
            )
        self._num_timesteps = len(timesteps)

        if comfyui_progressbar:
            from comfy.utils import ProgressBar
            pbar = ProgressBar(num_inference_steps + 2)

        # 5. Prepare latents
        if static_video is not None:
            video_length = static_video.shape[2]
            init_video = self.image_processor.preprocess(
                rearrange(static_video, "b c f h w -> (b f) c h w"),
                height=height,
                width=width
            )
            init_video = init_video.to(dtype=torch.float32)
            init_video = rearrange(init_video, "(b f) c h w -> b c f h w", f=video_length)
        else:
            init_video = None

        latent_channels = self.vae.config.latent_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            latent_channels,
            num_frames,
            height,
            width,
            weight_dtype,
            device,
            generator,
            latents,
        )

        if comfyui_progressbar:
            pbar.update(1)

        # Prepare mask latent variables
        if init_video is not None:
            if mask_video is not None and (mask_video == 255).all():
                mask_latents = torch.tile(
                    torch.zeros_like(latents)[:, :1].to(device, weight_dtype), [1, 4, 1, 1, 1]
                )
                masked_video_latents = torch.zeros_like(latents).to(device, weight_dtype)
            elif mask_video is not None:
                bs, _, video_length_mask, h_mask, w_mask = static_video.size()
                mask_condition = self.mask_processor.preprocess(
                    rearrange(mask_video, "b c f h w -> (b f) c h w"),
                    height=height,
                    width=width
                )
                mask_condition = mask_condition.to(dtype=torch.float32)
                mask_condition = rearrange(
                    mask_condition, "(b f) c h w -> b c f h w", f=video_length_mask
                )

                masked_video = init_video * (torch.tile(mask_condition, [1, 3, 1, 1, 1]) < 0.5)
                _, masked_video_latents = self.prepare_mask_latents(
                    None,
                    masked_video,
                    batch_size,
                    height,
                    width,
                    weight_dtype,
                    device,
                    generator,
                    do_classifier_free_guidance,
                    noise_aug_strength=None,
                )

                mask_condition = torch.concat(
                    [
                        torch.repeat_interleave(mask_condition[:, :, 0:1], repeats=4, dim=2),
                        mask_condition[:, :, 1:]
                    ], dim=2
                )
                mask_condition = mask_condition.view(
                    bs, mask_condition.shape[2] // 4, 4, height, width
                )
                mask_condition = mask_condition.transpose(1, 2)
                mask_latents = resize_mask(1 - mask_condition, masked_video_latents, True).to(
                    device, weight_dtype
                )
            else:
                mask_latents = torch.zeros_like(latents)[:, :1].to(device, weight_dtype)
                masked_video_latents = torch.zeros_like(latents).to(device, weight_dtype)
        else:
            mask_latents = torch.zeros(
                batch_size, 1, latents.shape[2], latents.shape[3], latents.shape[4],
                device=device, dtype=weight_dtype
            )
            masked_video_latents = torch.zeros_like(latents).to(device, weight_dtype)

        # 6. Prepare VACE context from hand video
        if hand_video is not None:
            vace_context = self.prepare_vace_context(
                hand_video=hand_video,
                height=height,
                width=width,
                dtype=weight_dtype,
                device=device,
                generator=generator,
            )
        else:
            vace_context = None

        # Prepare CLIP context
        if clip_image is not None:
            clip_image_tensor = TF.to_tensor(clip_image).sub_(0.5).div_(0.5).to(
                device, weight_dtype
            )
            clip_context = self._encode_clip_context(clip_image_tensor, weight_dtype)
        else:
            clip_image_placeholder = Image.new("RGB", (512, 512), color=(0, 0, 0))
            clip_image_tensor = TF.to_tensor(clip_image_placeholder).sub_(0.5).div_(0.5).to(
                device, weight_dtype
            )
            clip_context = self._encode_clip_context(clip_image_tensor, weight_dtype)
            clip_context = torch.zeros_like(clip_context)

        if comfyui_progressbar:
            pbar.update(1)

        # 7. Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        target_shape = (
            self.vae.latent_channels,
            (num_frames - 1) // self.vae.temporal_compression_ratio + 1,
            width // self.vae.spatial_compression_ratio,
            height // self.vae.spatial_compression_ratio
        )
        seq_len = math.ceil(
            (target_shape[2] * target_shape[3]) /
            (self.transformer.config.patch_size[1] * self.transformer.config.patch_size[2]) *
            target_shape[1]
        )

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self.transformer.num_inference_steps = num_inference_steps
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                self.transformer.current_steps = i

                if self.interrupt:
                    continue

                # Prepare model input
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                if hasattr(self.scheduler, "scale_model_input"):
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # Prepare y (mask + static latents)
                mask_input = (
                    torch.cat([mask_latents] * 2) if do_classifier_free_guidance else mask_latents
                )
                masked_video_latents_input = (
                    torch.cat([masked_video_latents] * 2)
                    if do_classifier_free_guidance
                    else masked_video_latents
                )
                y = torch.cat([mask_input, masked_video_latents_input], dim=1).to(
                    device, weight_dtype
                )

                # Prepare VACE context for classifier-free guidance
                if vace_context is not None:
                    if do_classifier_free_guidance:
                        # Duplicate vace_context for unconditional + conditional
                        vace_context_input = vace_context + vace_context
                    else:
                        vace_context_input = vace_context
                else:
                    vace_context_input = None

                # Prepare CLIP context
                clip_context_input = (
                    torch.cat([clip_context] * 2) if do_classifier_free_guidance else clip_context
                )

                # Broadcast timestep
                timestep = t.expand(latent_model_input.shape[0])

                # Predict noise with VACE conditioning
                with torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=device):
                    noise_pred = self.transformer(
                        x=latent_model_input,
                        context=in_prompt_embeds,
                        t=timestep,
                        seq_len=seq_len,
                        y=y,
                        clip_fea=clip_context_input,
                        vace_context=vace_context_input,
                        vace_context_scale=vace_context_scale,
                    )

                # Handle list output from VACE transformer
                if isinstance(noise_pred, list):
                    noise_pred = torch.stack(noise_pred)

                # Perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # Compute previous noisy sample
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]

                # Callback
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop(
                        "negative_prompt_embeds", negative_prompt_embeds
                    )

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                if comfyui_progressbar:
                    pbar.update(1)

        # 9. Decode latents
        if output_type == "numpy":
            video = self.decode_latents(latents)
        elif output_type != "latent":
            video = self.decode_latents(latents)
            video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        else:
            video = latents

        # Offload models
        self.maybe_free_model_hooks()

        if not return_dict:
            video = torch.from_numpy(video) if isinstance(video, np.ndarray) else video

        return WanPipelineOutput(videos=video)
