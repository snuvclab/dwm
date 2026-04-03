import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import WanLoraLoaderMixin
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import BaseOutput, logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from einops import rearrange
from PIL import Image
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel

from ..models.wan_transformer3d_i2v_with_conditions import WanI2VTransformer3DModelWithConcat
from ..utils.fm_solvers import FlowDPMSolverMultistepScheduler, get_sampling_sigmas
from ..utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .pipeline_wan_fun_inpaint import WanFunInpaintPipeline, retrieve_timesteps

logger = logging.get_logger(__name__)


@dataclass
class WanPipelineOutput(BaseOutput):
    videos: torch.Tensor


def retrieve_latents(encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    if hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    if hasattr(encoder_output, "latents"):
        return encoder_output.latents
    raise AttributeError("Could not access latents of provided encoder_output")


class WanI2VDiffusersHandConcatPipeline(DiffusionPipeline, WanLoraLoaderMixin):
    """
    Diffusers-based WAN i2v inpaint pipeline with additional hand-latent concatenation.

    This pipeline mirrors `WanFunInpaintHandConcatPipeline` behavior but is wired with
    diffusers-native component types so LoRA loading (`load_lora_weights`) works naturally.
    """

    _optional_components = []
    model_cpu_offload_seq = "text_encoder->image_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    # Reuse stable utility methods/properties from the existing WAN pipeline implementation.
    _get_t5_prompt_embeds = WanFunInpaintPipeline._get_t5_prompt_embeds
    encode_prompt = WanFunInpaintPipeline.encode_prompt
    decode_latents = WanFunInpaintPipeline.decode_latents
    prepare_extra_step_kwargs = WanFunInpaintPipeline.prepare_extra_step_kwargs
    check_inputs = WanFunInpaintPipeline.check_inputs
    guidance_scale = WanFunInpaintPipeline.guidance_scale
    num_timesteps = WanFunInpaintPipeline.num_timesteps
    attention_kwargs = WanFunInpaintPipeline.attention_kwargs
    interrupt = WanFunInpaintPipeline.interrupt

    def _tensor_stats(self, name: str, t: torch.Tensor) -> str:
        t = t.detach()
        return (
            f"{name}: shape={tuple(t.shape)}, dtype={t.dtype}, "
            f"mean={float(t.mean()):.6f}, std={float(t.std()):.6f}, "
            f"min={float(t.min()):.6f}, max={float(t.max()):.6f}"
        )

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        image_encoder: CLIPVisionModel,
        image_processor: CLIPImageProcessor,
        transformer: WanI2VTransformer3DModelWithConcat,
        vae: AutoencoderKLWan,
        scheduler: FlowMatchEulerDiscreteScheduler,
    ):
        super().__init__()
        # Register only the expected diffusers-native components.
        self.register_modules(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            image_encoder=image_encoder,
            image_processor=image_processor,
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
        )

        # Keep the same attribute access pattern used by existing WAN pipelines.
        if not hasattr(self.vae, "spatial_compression_ratio"):
            self.vae.spatial_compression_ratio = int(getattr(self.vae.config, "spatial_compression_ratio", 8))
        if not hasattr(self.vae, "temporal_compression_ratio"):
            temporal_ratio = getattr(self.vae.config, "temporal_compression_ratio", None)
            if temporal_ratio is None:
                temporal_ratio = getattr(self.vae.config, "time_compression_ratio", 4)
            self.vae.temporal_compression_ratio = int(temporal_ratio)
        if not hasattr(self.vae, "latent_channels"):
            latent_channels = getattr(self.vae.config, "latent_channels", None)
            if latent_channels is None:
                latent_channels = getattr(self.vae.config, "z_dim", 16)
            self.vae.latent_channels = int(latent_channels)

        self.video_processor = VideoProcessor(vae_scale_factor=self.vae.spatial_compression_ratio)
        self.vae_image_processor = VaeImageProcessor(vae_scale_factor=self.vae.spatial_compression_ratio)
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae.spatial_compression_ratio,
            do_normalize=False,
            do_binarize=True,
            do_convert_grayscale=True,
        )

    def prepare_i2v_latents(
        self,
        image: torch.FloatTensor,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        num_frames: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]],
        latents: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        num_latent_frames = (num_frames - 1) // self.vae.temporal_compression_ratio + 1
        latent_height = height // self.vae.spatial_compression_ratio
        latent_width = width // self.vae.spatial_compression_ratio

        shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"List of generators has length {len(generator)}, but effective batch size is {batch_size}."
            )
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )

        image = image.unsqueeze(2)
        video_condition = torch.cat(
            [image, image.new_zeros(image.shape[0], image.shape[1], num_frames - 1, height, width)], dim=2
        ).to(device=device, dtype=self.vae.dtype)

        if isinstance(generator, list):
            latent_condition = [retrieve_latents(self.vae.encode(video_condition), sample_mode="argmax") for _ in generator]
            latent_condition = torch.cat(latent_condition)
        else:
            latent_condition = retrieve_latents(self.vae.encode(video_condition), sample_mode="argmax")
            latent_condition = latent_condition.repeat(batch_size, 1, 1, 1, 1)

        latent_condition = latent_condition.to(dtype)
        latent_condition = (latent_condition - latents_mean) * latents_std

        mask_lat_size = torch.ones(batch_size, 1, num_frames, latent_height, latent_width, device=latents.device, dtype=latents.dtype)
        mask_lat_size[:, :, list(range(1, num_frames))] = 0
        first_frame_mask = mask_lat_size[:, :, 0:1]
        first_frame_mask = torch.repeat_interleave(first_frame_mask, dim=2, repeats=self.vae.temporal_compression_ratio)
        mask_lat_size = torch.concat([first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)
        mask_lat_size = mask_lat_size.view(batch_size, -1, self.vae.temporal_compression_ratio, latent_height, latent_width)
        mask_lat_size = mask_lat_size.transpose(1, 2).to(latent_condition.device)

        condition = torch.concat([mask_lat_size, latent_condition], dim=1)
        return latents, condition

    def _encode_clip_context_batch(self, clip_images: torch.Tensor, weight_dtype: torch.dtype) -> torch.Tensor:
        if clip_images.ndim == 3:
            clip_images = clip_images.unsqueeze(0)
        elif clip_images.ndim != 4:
            raise ValueError(f"Expected clip_images with shape [B, C, H, W], got {tuple(clip_images.shape)}")

        clip_images = clip_images.to(device=self._execution_device, dtype=torch.float32)
        if clip_images.min() < 0 or clip_images.max() > 1:
            clip_images = clip_images.mul(0.5).add(0.5)
        clip_images = clip_images.clamp(0, 1)

        image_inputs = self.image_processor(images=clip_images, return_tensors="pt").to(self._execution_device)
        outputs = self.image_encoder(**image_inputs, output_hidden_states=True)
        return outputs.hidden_states[-2].to(dtype=weight_dtype)

    @staticmethod
    def _move_prompt_context(
        prompt_context: Optional[Union[torch.Tensor, List[torch.Tensor]]],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[Union[torch.Tensor, List[torch.Tensor]]]:
        if prompt_context is None:
            return None
        if isinstance(prompt_context, list):
            return [context.to(device=device, dtype=dtype) for context in prompt_context]
        return prompt_context.to(device=device, dtype=dtype)

    def prepare_hand_latents(
        self,
        hand_video: torch.FloatTensor,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[torch.Generator] = None,
    ) -> torch.FloatTensor:
        video_length = hand_video.shape[2]

        hand_video_processed = self.vae_image_processor.preprocess(
            rearrange(hand_video, "b c f h w -> (b f) c h w"),
            height=height,
            width=width,
        )
        hand_video_processed = hand_video_processed.to(dtype=torch.float32)
        hand_video_processed = rearrange(hand_video_processed, "(b f) c h w -> b c f h w", f=video_length)
        hand_video_processed = hand_video_processed.to(device=device, dtype=self.vae.dtype)
        hand_latents = retrieve_latents(self.vae.encode(hand_video_processed), sample_mode="argmax").to(dtype)

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1).to(hand_latents.device, hand_latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            hand_latents.device, hand_latents.dtype
        )
        hand_latents = (hand_latents - latents_mean) * latents_std
        return hand_latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 480,
        width: int = 720,
        static_video: Union[torch.FloatTensor, None] = None,
        mask_video: Union[torch.FloatTensor, None] = None,
        hand_video: Union[torch.FloatTensor, None] = None,
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
        callback_on_step_end: Optional[Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        clip_image: Optional[Image.Image] = None,
        max_sequence_length: int = 512,
        comfyui_progressbar: bool = False,
        shift: float = 3.0,
    ) -> Union[WanPipelineOutput, Tuple]:
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

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

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        weight_dtype = self.transformer.dtype
        do_classifier_free_guidance = self.guidance_scale > 1.0

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
        prompt_embeds = self._move_prompt_context(prompt_embeds, device=device, dtype=weight_dtype)
        negative_prompt_embeds = self._move_prompt_context(negative_prompt_embeds, device=device, dtype=weight_dtype)

        if isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler):
            timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps, mu=1)
        elif isinstance(self.scheduler, FlowUniPCMultistepScheduler):
            self.scheduler.set_timesteps(num_inference_steps, device=device, shift=shift)
            timesteps = self.scheduler.timesteps
        elif isinstance(self.scheduler, FlowDPMSolverMultistepScheduler):
            sampling_sigmas = get_sampling_sigmas(num_inference_steps, shift)
            timesteps, num_inference_steps = retrieve_timesteps(
                self.scheduler, num_inference_steps, device, timesteps, sigmas=sampling_sigmas
            )
        else:
            timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)

        if static_video is not None:
            image_for_i2v = static_video[:, :, 0]
            image_for_i2v = F.interpolate(
                image_for_i2v.to(dtype=torch.float32), size=(height, width), mode="bilinear", align_corners=False
            ).to(device)
        elif clip_image is not None:
            image_for_i2v = self.video_processor.preprocess(clip_image, height=height, width=width).to(device, dtype=torch.float32)
        else:
            image_for_i2v = self.video_processor.preprocess(
                Image.new("RGB", (width, height), color=(0, 0, 0)), height=height, width=width
            ).to(device, dtype=torch.float32)

        latents, condition = self.prepare_i2v_latents(
            image_for_i2v,
            batch_size=batch_size * num_videos_per_prompt,
            num_channels_latents=self.vae.config.z_dim,
            height=height,
            width=width,
            num_frames=num_frames,
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=latents,
        )
        debug_enabled = not getattr(self, "_debug_stats_printed_once", False)
        if debug_enabled:
            logger.info("[I2V-DEBUG] prepare_i2v_latents done")
            logger.info("[I2V-DEBUG] %s", self._tensor_stats("latents", latents))
            logger.info("[I2V-DEBUG] %s", self._tensor_stats("condition", condition))

        if hand_video is not None:
            hand_latents = self.prepare_hand_latents(hand_video, height, width, weight_dtype, device, generator)
        else:
            hand_latents = torch.zeros_like(latents).to(device, weight_dtype)

        if clip_image is not None:
            clip_image_tensor = TF.to_tensor(clip_image).to(device=device, dtype=torch.float32)
            clip_sources = clip_image_tensor.unsqueeze(0).repeat(latents.shape[0], 1, 1, 1)
            clip_context = self._encode_clip_context_batch(clip_sources, weight_dtype)
        elif static_video is not None:
            clip_sources = image_for_i2v.to(device=device, dtype=torch.float32)
            if clip_sources.shape[0] != latents.shape[0]:
                clip_sources = clip_sources[:1].repeat(latents.shape[0], 1, 1, 1)
            clip_context = self._encode_clip_context_batch(clip_sources, weight_dtype)
        else:
            clip_image_placeholder = TF.to_tensor(Image.new("RGB", (512, 512), color=(0, 0, 0))).to(
                device=device, dtype=torch.float32
            )
            clip_context = torch.zeros_like(self._encode_clip_context_batch(clip_image_placeholder.unsqueeze(0), weight_dtype))

        if debug_enabled:
            logger.info("[I2V-DEBUG] %s", self._tensor_stats("clip_context", clip_context))
            logger.info("[I2V-DEBUG] %s", self._tensor_stats("hand_latents", hand_latents))

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        target_shape = (self.vae.latent_channels, latents.shape[2], latents.shape[3], latents.shape[4])
        seq_len = math.ceil(
            (target_shape[2] * target_shape[3]) / (self.transformer.config.patch_size[1] * self.transformer.config.patch_size[2]) * target_shape[1]
        )

        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self.transformer.num_inference_steps = num_inference_steps
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                self.transformer.current_steps = i
                if self.interrupt:
                    continue

                latent_model_input = torch.cat([latents, condition], dim=1)
                if hasattr(self.scheduler, "scale_model_input"):
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                latent_model_input = latent_model_input.to(device=device, dtype=weight_dtype)

                hand_latents_input = hand_latents.to(device, weight_dtype)
                clip_context_input = clip_context.to(device, weight_dtype)
                timestep = t.expand(latents.shape[0])

                if debug_enabled and i == 0:
                    logger.info("[I2V-DEBUG] denoise step 0 inputs")
                    logger.info("[I2V-DEBUG] %s", self._tensor_stats("hidden_states(latents+condition)", latent_model_input))
                    logger.info("[I2V-DEBUG] %s", self._tensor_stats("clip_context_input", clip_context_input))
                    logger.info("[I2V-DEBUG] %s", self._tensor_stats("hand_latents_input", hand_latents_input))

                with torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=device):
                    with self.transformer.cache_context("cond"):
                        noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=prompt_embeds,
                            encoder_hidden_states_image=clip_context_input,
                            condition_latents=hand_latents_input,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )

                if debug_enabled and i == 0:
                    logger.info("[I2V-DEBUG] %s", self._tensor_stats("noise_pred(step0)", noise_pred))

                if do_classifier_free_guidance:
                    noise_pred_text = noise_pred
                    with torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=device):
                        with self.transformer.cache_context("uncond"):
                            noise_pred_uncond = self.transformer(
                                hidden_states=latent_model_input,
                                timestep=timestep,
                                encoder_hidden_states=negative_prompt_embeds,
                                encoder_hidden_states_image=clip_context_input,
                                condition_latents=hand_latents_input,
                                attention_kwargs=attention_kwargs,
                                return_dict=False,
                            )
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {k: locals()[k] for k in callback_on_step_end_tensor_inputs}
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        # Match diffusers Wan i2v decode path: denormalize latent statistics before VAE decode.
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean

        if output_type == "numpy":
            if debug_enabled:
                logger.info("[I2V-DEBUG] %s", self._tensor_stats("latents(final)", latents))
                self._debug_stats_printed_once = True
            video = self.decode_latents(latents)
        elif output_type != "latent":
            if debug_enabled:
                logger.info("[I2V-DEBUG] %s", self._tensor_stats("latents(final)", latents))
                self._debug_stats_printed_once = True
            video = self.decode_latents(latents)
            video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        else:
            if debug_enabled:
                logger.info("[I2V-DEBUG] %s", self._tensor_stats("latents(final)", latents))
                self._debug_stats_printed_once = True
            video = latents

        self.maybe_free_model_hooks()
        if not return_dict:
            video = torch.from_numpy(video) if isinstance(video, np.ndarray) else video
        return WanPipelineOutput(videos=video)
