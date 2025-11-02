# Copyright 2024 The CogVideoX team, Tsinghua University & ZhipuAI and The HuggingFace Team.
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

import inspect
import math
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL
import PIL.Image
import torch
import torch.nn.functional as F
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.embeddings import get_1d_rotary_pos_embed
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import CogVideoXDDIMScheduler, CogVideoXDPMScheduler
from diffusers.utils import BaseOutput, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from einops import rearrange

from diffusers import CogVideoXImageToVideoPipeline

# Import our custom VideoX-Fun transformer
from cogvideox_fun_transformer_with_conditions import (
    CogVideoXFunTransformer3DModel, 
    CogVideoXFunTransformer3DModelWithConcat, 
    CogVideoXFunTransformer3DModelWithAdapter,
    CrossTransformer3DModelWithAdapter,
    CogVideoXFunTransformer3DModelWithCondToken
)

# Import models from diffusers (same as cogvideox_static_to_video_pose_concat_pipeline.py)
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDDIMScheduler,
    CogVideoXDPMScheduler,
    CogVideoXTransformer3DModel,
)
from transformers import T5EncoderModel, T5Tokenizer

# Import utility functions from diffusers
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps

# Similar to diffusers.pipelines.hunyuandit.pipeline_hunyuandit.get_resize_crop_region_for_grid
def get_resize_crop_region_for_grid(src, tgt_width, tgt_height):
    tw = tgt_width
    th = tgt_height
    h, w = src
    r = h / w
    if r > (th / tw):
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))

    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))

    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)

logger = logging.get_logger(__name__)

def crop(img, start_h, start_w, crop_h, crop_w):
    img_src = np.zeros((crop_h, crop_w, *img.shape[2:]), dtype=img.dtype)
    hsize, wsize = crop_h, crop_w
    dh, dw, sh, sw = start_h, start_w, 0, 0
    if dh < 0:
        sh = -dh
        hsize += dh
        dh = 0
    if dh + hsize > img.shape[0]:
        hsize = img.shape[0] - dh
    if dw < 0:
        sw = -dw
        wsize += dw
        dw = 0
    if dw + wsize > img.shape[1]:
        wsize = img.shape[1] - dw
    img_src[sh : sh + hsize, sw : sw + wsize] = img[dh : dh + hsize, dw : dw + wsize]
    return img_src

def imcrop_center(img_list, crop_p_h, crop_p_w):
    new_img = []
    for i, _img in enumerate(img_list):
        if crop_p_h / crop_p_w > _img.shape[0] / _img.shape[1]:  # crop left and right
            start_h = int(0)
            start_w = int((_img.shape[1] - _img.shape[0] / crop_p_h * crop_p_w) / 2)
            crop_size = (_img.shape[0], int(_img.shape[0] / crop_p_h * crop_p_w))
        else:
            start_h = int((_img.shape[0] - _img.shape[1] / crop_p_w * crop_p_h) / 2)
            start_w = int(0)
            crop_size = (int(_img.shape[1] / crop_p_w * crop_p_h), _img.shape[1])

        _img_src = crop(_img, start_h, start_w, crop_size[0], crop_size[1])
        new_img.append(_img_src)

    return new_img  # pylint: disable=invalid-name

@dataclass
class CogVideoXFunPipelineOutput(BaseOutput):
    """
    Output class for CogVideoX Fun pipeline.

    Args:
        frames (`torch.Tensor`):
            The generated video frames.
    """

    frames: torch.Tensor


# Removed CogVideoXFunStaticToVideoPoseConcatPipelineOutput - using CogVideoXFunPipelineOutput instead
EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        pass
        ```
"""




def resize_mask(mask, latent, process_first_frame_only=True):
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


def add_noise_to_reference_video(image, ratio=None):
    if ratio is None:
        sigma = torch.normal(mean=-3.0, std=0.5, size=(image.shape[0],)).to(image.device)
        sigma = torch.exp(sigma).to(image.dtype)
    else:
        sigma = torch.ones((image.shape[0],)).to(image.device, image.dtype) * ratio
    
    image_noise = torch.randn_like(image) * sigma[:, None, None, None, None]
    image_noise = torch.where(image==-1, torch.zeros_like(image), image_noise)
    image = image + image_noise
    return image


class CogVideoXFunInpaintPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-video generation using CogVideoX.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
        text_encoder ([`T5EncoderModel`]):
            Frozen text-encoder. CogVideoX_Fun uses
            [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel); specifically the
            [t5-v1_1-xxl](https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/t5-v1_1-xxl) variant.
        tokenizer (`T5Tokenizer`):
            Tokenizer of class
            [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer).
        transformer ([`CogVideoXTransformer3DModel`]):
            A text conditioned `CogVideoXTransformer3DModel` to denoise the encoded video latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded video latents.
    """

    _optional_components = []
    model_cpu_offload_seq = "text_encoder->transformer->vae"

    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
    ]

    def __init__(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        vae: AutoencoderKLCogVideoX,
        transformer: CogVideoXTransformer3DModel,
        scheduler: Union[CogVideoXDDIMScheduler, CogVideoXDPMScheduler],
    ):
        super().__init__()

        self.register_modules(
            tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, transformer=transformer, scheduler=scheduler
        )
        self.vae_scale_factor_spatial = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
        )
        self.vae_scale_factor_temporal = (
            self.vae.config.temporal_compression_ratio if hasattr(self, "vae") and self.vae is not None else 4
        )

        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_normalize=False, do_binarize=True, do_convert_grayscale=True
        )

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder(text_input_ids.to(device))[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
        """
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds

    def prepare_mask_latents(
        self, mask, masked_image, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance, noise_aug_strength
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision

        if mask is not None:
            mask = mask.to(device=device, dtype=self.vae.dtype)
            bs = 1
            new_mask = []
            for i in range(0, mask.shape[0], bs):
                mask_bs = mask[i : i + bs]
                mask_bs = self.vae.encode(mask_bs)[0]
                mask_bs = mask_bs.mode()
                new_mask.append(mask_bs)
            mask = torch.cat(new_mask, dim = 0)
            mask = mask * self.vae.config.scaling_factor

        if masked_image is not None:
            if self.transformer.config.get("add_noise_in_inpaint_model", False):
                masked_image = add_noise_to_reference_video(masked_image, ratio=noise_aug_strength)
            masked_image = masked_image.to(device=device, dtype=self.vae.dtype)
            bs = 1
            new_mask_pixel_values = []
            for i in range(0, masked_image.shape[0], bs):
                mask_pixel_values_bs = masked_image[i : i + bs]
                mask_pixel_values_bs = self.vae.encode(mask_pixel_values_bs)[0]
                mask_pixel_values_bs = mask_pixel_values_bs.mode()
                new_mask_pixel_values.append(mask_pixel_values_bs)
            masked_image_latents = torch.cat(new_mask_pixel_values, dim = 0)
            masked_image_latents = masked_image_latents * self.vae.config.scaling_factor
        else:
            masked_image_latents = None

        return mask, masked_image_latents

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents.permute(0, 2, 1, 3, 4)  # [batch_size, num_channels, num_frames, height, width]
        latents = 1 / self.vae.config.scaling_factor * latents

        frames = self.vae.decode(latents).sample
        frames = (frames / 2 + 0.5).clamp(0, 1)
        # Return PyTorch tensor instead of numpy array for video_processor compatibility
        frames = frames.cpu().float()
        return frames

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    # Copied from diffusers.pipelines.latte.pipeline_latte.LattePipeline.check_inputs
    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt,
        callback_on_step_end_tensor_inputs,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    def fuse_qkv_projections(self) -> None:
        r"""Enables fused QKV projections."""
        self.fusing_transformer = True
        self.transformer.fuse_qkv_projections()

    def unfuse_qkv_projections(self) -> None:
        r"""Disable QKV projection fusion if enabled."""
        if not self.fusing_transformer:
            logger.warning("The Transformer was not initially fused for QKV projections. Doing nothing.")
        else:
            self.transformer.unfuse_qkv_projections()
            self.fusing_transformer = False

    def _prepare_rotary_positional_embeddings(
        self,
        height: int,
        width: int,
        num_frames: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        grid_height = height // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)
        grid_width = width // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)

        p = self.transformer.config.patch_size
        p_t = self.transformer.config.patch_size_t

        base_size_width = self.transformer.config.sample_width // p
        base_size_height = self.transformer.config.sample_height // p

        if p_t is None:
            # CogVideoX 1.0
            grid_crops_coords = get_resize_crop_region_for_grid(
                (grid_height, grid_width), base_size_width, base_size_height
            )
            freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
                embed_dim=self.transformer.config.attention_head_dim,
                crops_coords=grid_crops_coords,
                grid_size=(grid_height, grid_width),
                temporal_size=num_frames,
            )
        else:
            # CogVideoX 1.5
            base_num_frames = (num_frames + p_t - 1) // p_t

            freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
                embed_dim=self.transformer.config.attention_head_dim,
                crops_coords=None,
                grid_size=(grid_height, grid_width),
                temporal_size=base_num_frames,
                grid_type="slice",
                max_size=(base_size_height, base_size_width),
            )

        freqs_cos = freqs_cos.to(device=device)
        freqs_sin = freqs_sin.to(device=device)
        return freqs_cos, freqs_sin

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def interrupt(self):
        return self._interrupt

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.StableDiffusionImg2ImgPipeline.get_timesteps
    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]

        return timesteps, num_inference_steps - t_start

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def interrupt(self):
        return self._interrupt

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.StableDiffusionImg2ImgPipeline.get_timesteps
    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]

        return timesteps, num_inference_steps - t_start

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 480,
        width: int = 720,
        video: Union[torch.FloatTensor] = None,
        mask_video: Union[torch.FloatTensor] = None,
        masked_video_latents: Union[torch.FloatTensor] = None,
        num_frames: int = 49,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6,
        use_dynamic_cfg: bool = False,
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
        max_sequence_length: int = 226,
        strength: float = 1,
        noise_aug_strength: float = 0.0563,
        comfyui_progressbar: bool = False,
    ) -> Union[CogVideoXFunPipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_frames (`int`, defaults to `48`):
                Number of frames to generate. Must be divisible by self.vae_scale_factor_temporal. Generated video will
                contain 1 extra frame because CogVideoX_Fun is conditioned with (num_seconds * fps + 1) frames where
                num_seconds is 6 and fps is 4. However, since videos can be saved at any fps, the only condition that
                needs to be satisfied is that of divisibility mentioned above.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of videos to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
                of a plain tuple.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int`, defaults to `226`):
                Maximum sequence length in encoded prompt. Must be consistent with
                `self.transformer.config.max_text_seq_length` otherwise may lead to poor results.

        Examples:

        Returns:
            [`~pipelines.cogvideo.pipeline_cogvideox.CogVideoXFunPipelineOutput`] or `tuple`:
            [`~pipelines.cogvideo.pipeline_cogvideox.CogVideoXFunPipelineOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is a list with the generated images.
        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        height = height or self.transformer.config.sample_height * self.vae_scale_factor_spatial
        width = width or self.transformer.config.sample_width * self.vae_scale_factor_spatial
        num_frames = num_frames or self.transformer.config.sample_frames

        num_videos_per_prompt = 1

        # 1. Check inputs. Raise error if not correct
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

        # 2. Default call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            negative_prompt,
            do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        # 4. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(
            num_inference_steps=num_inference_steps, strength=strength, device=device
        )
        self._num_timesteps = len(timesteps)
        if comfyui_progressbar:
            from comfy.utils import ProgressBar
            pbar = ProgressBar(num_inference_steps + 2)
        # at which timestep to set the initial noise (n.b. 50% if strength is 0.5)
        latent_timestep = timesteps[:1].repeat(batch_size * num_videos_per_prompt)
        # create a boolean to check if the strength is set to 1. if so then initialise the latents with pure noise
        is_strength_max = strength == 1.0

        # 5. Prepare latents.
        if video is not None:
            video_length = video.shape[2]
            init_video = self.image_processor.preprocess(rearrange(video, "b c f h w -> (b f) c h w"), height=height, width=width) 
            init_video = init_video.to(dtype=torch.float32)
            init_video = rearrange(init_video, "(b f) c h w -> b c f h w", f=video_length)
        else:
            init_video = None

        # Magvae needs the number of frames to be 4n + 1.
        local_latent_length = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        # For CogVideoX 1.5, the latent frames should be clipped to make it divisible by patch_size_t
        patch_size_t = self.transformer.config.patch_size_t
        additional_frames = 0
        if patch_size_t is not None and local_latent_length % patch_size_t != 0:
            additional_frames = local_latent_length % patch_size_t
            num_frames -= additional_frames * self.vae_scale_factor_temporal
        if num_frames <= 0:
            num_frames = 1
        if video_length > num_frames:
            logger.warning("The length of condition video is not right, the latent frames should be clipped to make it divisible by patch_size_t. ")
            video_length = num_frames
            video = video[:, :, :video_length]
            init_video = init_video[:, :, :video_length]
            mask_video = mask_video[:, :, :video_length]

        num_channels_latents = self.vae.config.latent_channels
        num_channels_transformer = self.transformer.config.in_channels
        return_image_latents = num_channels_transformer == num_channels_latents

        latents_outputs = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            video_length,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
            video=init_video,
            timestep=latent_timestep,
            is_strength_max=is_strength_max,
            return_noise=True,
            return_video_latents=return_image_latents,
        )
        if return_image_latents:
            latents, noise, image_latents = latents_outputs
        else:
            latents, noise = latents_outputs
        if comfyui_progressbar:
            pbar.update(1)
        if mask_video is not None:
            if (mask_video == 255).all():
                mask_latents = torch.zeros_like(latents)[:, :, :1].to(latents.device, latents.dtype)
                masked_video_latents = torch.zeros_like(latents).to(latents.device, latents.dtype)

                mask_input = torch.cat([mask_latents] * 2) if do_classifier_free_guidance else mask_latents
                masked_video_latents_input = (
                    torch.cat([masked_video_latents] * 2) if do_classifier_free_guidance else masked_video_latents
                )
                inpaint_latents = torch.cat([mask_input, masked_video_latents_input], dim=2).to(latents.dtype)
            else:
                # Prepare mask latent variables
                video_length = video.shape[2]
                mask_condition = self.mask_processor.preprocess(rearrange(mask_video, "b c f h w -> (b f) c h w"), height=height, width=width) 
                mask_condition = mask_condition.to(dtype=torch.float32)
                mask_condition = rearrange(mask_condition, "(b f) c h w -> b c f h w", f=video_length)

                if num_channels_transformer != num_channels_latents:
                    mask_condition_tile = torch.tile(mask_condition, [1, 3, 1, 1, 1])
                    if masked_video_latents is None:
                        masked_video = init_video * (mask_condition_tile < 0.5) + torch.ones_like(init_video) * (mask_condition_tile > 0.5) * -1
                    else:
                        masked_video = masked_video_latents

                    _, masked_video_latents = self.prepare_mask_latents(
                        None,
                        masked_video,
                        batch_size,
                        height,
                        width,
                        prompt_embeds.dtype,
                        device,
                        generator,
                        do_classifier_free_guidance,
                        noise_aug_strength=noise_aug_strength,
                    )
                    mask_latents = resize_mask(1 - mask_condition, masked_video_latents)
                    mask_latents = mask_latents.to(masked_video_latents.device) * self.vae.config.scaling_factor

                    mask = torch.tile(mask_condition, [1, num_channels_latents, 1, 1, 1])
                    mask = F.interpolate(mask, size=latents.size()[-3:], mode='trilinear', align_corners=True).to(latents.device, latents.dtype)
                    
                    mask_input = torch.cat([mask_latents] * 2) if do_classifier_free_guidance else mask_latents
                    masked_video_latents_input = (
                        torch.cat([masked_video_latents] * 2) if do_classifier_free_guidance else masked_video_latents
                    )

                    mask = rearrange(mask, "b c f h w -> b f c h w")
                    mask_input = rearrange(mask_input, "b c f h w -> b f c h w")
                    masked_video_latents_input = rearrange(masked_video_latents_input, "b c f h w -> b f c h w")

                    inpaint_latents = torch.cat([mask_input, masked_video_latents_input], dim=2).to(latents.dtype)
                else:
                    mask = torch.tile(mask_condition, [1, num_channels_latents, 1, 1, 1])
                    mask = F.interpolate(mask, size=latents.size()[-3:], mode='trilinear', align_corners=True).to(latents.device, latents.dtype)
                    mask = rearrange(mask, "b c f h w -> b f c h w")
                    
                    inpaint_latents = None
        else:
            if num_channels_transformer != num_channels_latents:
                mask = torch.zeros_like(latents).to(latents.device, latents.dtype)
                masked_video_latents = torch.zeros_like(latents).to(latents.device, latents.dtype)

                mask_input = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
                masked_video_latents_input = (
                    torch.cat([masked_video_latents] * 2) if do_classifier_free_guidance else masked_video_latents
                )
                inpaint_latents = torch.cat([mask_input, masked_video_latents_input], dim=1).to(latents.dtype)
            else:
                mask = torch.zeros_like(init_video[:, :1])
                mask = torch.tile(mask, [1, num_channels_latents, 1, 1, 1])
                mask = F.interpolate(mask, size=latents.size()[-3:], mode='trilinear', align_corners=True).to(latents.device, latents.dtype)
                mask = rearrange(mask, "b c f h w -> b f c h w")

                inpaint_latents = None
        if comfyui_progressbar:
            pbar.update(1)

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        
        # 7. Create rotary embeds if required
        image_rotary_emb = (
            self._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
            if self.transformer.config.use_rotary_positional_embeddings
            else None
        )
        
        # Initialize adapter_control_for_transformer (used by all pipelines)
        if not hasattr(self, 'add_control_adapter') or not self.add_control_adapter:
            adapter_control_for_transformer = None

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            # for DPM-solver++
            old_pred_original_sample = None
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                # predict noise model_output
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    image_rotary_emb=image_rotary_emb,
                    return_dict=False,
                    inpaint_latents=inpaint_latents,
                )[0]
                noise_pred = noise_pred.float()

                # perform guidance
                if use_dynamic_cfg:
                    self._guidance_scale = 1 + guidance_scale * (
                        (1 - math.cos(math.pi * ((num_inference_steps - t.item()) / num_inference_steps) ** 5.0)) / 2
                    )
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                if not isinstance(self.scheduler, CogVideoXDPMScheduler):
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                else:
                    latents, old_pred_original_sample = self.scheduler.step(
                        noise_pred,
                        old_pred_original_sample,
                        t,
                        timesteps[i - 1] if i > 0 else None,
                        latents,
                        **extra_step_kwargs,
                        return_dict=False,
                    )
                latents = latents.to(prompt_embeds.dtype)

                # call the callback, if provided
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                if comfyui_progressbar:
                    pbar.update(1)

        if output_type == "numpy":
            video = self.decode_latents(latents)
        elif not output_type == "latent":
            video = self.decode_latents(latents)
            video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            video = torch.from_numpy(video)

        return CogVideoXFunPipelineOutput(videos=video)


class CogVideoXFunStaticToVideoPipeline(CogVideoXFunInpaintPipeline):
    r"""
    Pipeline for text-to-video generation using CogVideoX with static and hand pose conditions.

    This model inherits from [`CogVideoXFunInpaintPipeline`]. It extends the base VideoX-Fun pipeline
    to support static video and hand pose conditioning.

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
        text_encoder ([`T5EncoderModel`]):
            Frozen text-encoder. CogVideoX_Fun uses
            [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel); specifically the
            [t5-v1_1-xxl](https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/t5-v1_1-xxl) variant.
        tokenizer (`T5Tokenizer`):
            Tokenizer of class
            [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer).
        transformer ([`CogVideoXTransformer3DModel`]):
            A text conditioned `CogVideoXTransformer3DModel` to denoise the encoded video latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded video latents.
        split_hands (bool):
            Whether to split hand videos into left and right hands for separate processing.
    """

    def __init__(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        vae: AutoencoderKLCogVideoX,
        transformer: CogVideoXTransformer3DModel,
        scheduler: Union[CogVideoXDDIMScheduler, CogVideoXDPMScheduler],
        split_hands: bool = False,
        compress_smpl_pos_map_temporal: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 12,
        adapter_control_type: str = "smpl_pos_map",
    ):
        super().__init__(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
        )
        self.split_hands = split_hands
        self.compress_smpl_pos_map_temporal = compress_smpl_pos_map_temporal
        self.add_control_adapter = add_control_adapter
        self.in_dim_control_adapter = in_dim_control_adapter
        self.adapter_control_type = adapter_control_type

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path=None, 
                        base_model_name_or_path="alibaba-pai/CogVideoX-Fun-V1.1-5b-InP", 
                        transformer=None, 
                        condition_channels=None, 
                        use_adapter=False,
                        adapter_version="v1",
                        use_zero_proj=False,
                        split_hands=False,
                        compress_smpl_pos_map_temporal=False,
                        add_control_adapter=False,
                        in_dim_control_adapter=12,
                        adapter_control_type="smpl_pos_map",
                        *args, **kwargs):
        """
        Load a CogVideoXFunStaticToVideoPipeline from a saved directory or base model.
        
        This method loads all components (tokenizer, text_encoder, vae, transformer, scheduler)
        from the specified directory, or uses provided components if specified.
        
        Args:
            pretrained_model_name_or_path: Path to saved pipeline or base model
            base_model_name_or_path: Optional base model path for creating pose-conditioned pipeline
            transformer: Optional transformer to use (for validation with trained transformer)
            condition_channels: Number of condition channels (0 or None for base model, >0 for concat model)
            use_adapter: Whether to use adapter-based transformer
            adapter_version: Version of adapter to use ("v1" or "v2")
            use_zero_proj: Whether to use zero projection for adapter
        """
        # Check if this is a base model path (for creating pose-conditioned pipeline)
        if base_model_name_or_path is not None:
            print(f"🔧 Loading pipeline components (tokenizer, text_encoder, vae, scheduler) from base model: {base_model_name_or_path}")
            # Load the original CogVideoX-Fun pipeline from base model to get non-transformer components
            original_pipeline = super().from_pretrained(base_model_name_or_path, *args, **kwargs)
            
        if transformer is None:
            # Determine condition_channels
            if condition_channels is None:
                condition_channels = 0  # Default to base model
            
            # Create or load appropriate transformer based on condition_channels and use_adapter
            if condition_channels > 0:
                if use_adapter:
                    print(f"🔧 Creating/loading VideoX-Fun adapter transformer with {condition_channels} condition channels (adapter_version={adapter_version})")
                    transformer = CogVideoXFunTransformer3DModelWithAdapter.from_pretrained(
                        pretrained_model_name_or_path=pretrained_model_name_or_path,
                        base_model_name_or_path=base_model_name_or_path,
                        subfolder="transformer",
                        condition_channels=condition_channels,
                        adapter_version=adapter_version,
                        use_zero_proj=use_zero_proj,
                        torch_dtype=kwargs.get("torch_dtype", torch.bfloat16),
                        revision=kwargs.get("revision", None),
                        variant=kwargs.get("variant", None),
                    )
                else:
                    print(f"🔧 Creating/loading VideoX-Fun concat transformer with {condition_channels} condition channels")
                    transformer = CogVideoXFunTransformer3DModelWithConcat.from_pretrained(
                        pretrained_model_name_or_path=pretrained_model_name_or_path,
                        base_model_name_or_path=base_model_name_or_path,
                        subfolder="transformer",
                        condition_channels=condition_channels,
                        add_control_adapter=add_control_adapter,
                        in_dim_control_adapter=in_dim_control_adapter,
                        torch_dtype=kwargs.get("torch_dtype", torch.bfloat16),
                        revision=kwargs.get("revision", None),
                        variant=kwargs.get("variant", None),
                    )
            elif add_control_adapter:
                # Control adapter with no concat
                print(f"🔧 Creating/loading VideoX-Fun transformer with control adapter (in_dim={in_dim_control_adapter})")
                transformer = CogVideoXFunTransformer3DModelWithConcat.from_pretrained(
                    pretrained_model_name_or_path=pretrained_model_name_or_path,
                    base_model_name_or_path=base_model_name_or_path,
                    subfolder="transformer",
                    condition_channels=0,  # No concat
                    add_control_adapter=True,
                    in_dim_control_adapter=in_dim_control_adapter,
                    torch_dtype=kwargs.get("torch_dtype", torch.bfloat16),
                    revision=kwargs.get("revision", None),
                    variant=kwargs.get("variant", None),
                )
            else:
                print(f"🔧 Loading base VideoX-Fun transformer")
                transformer = CogVideoXFunTransformer3DModel.from_pretrained(
                    pretrained_model_name_or_path=pretrained_model_name_or_path,
                    base_model_name_or_path=base_model_name_or_path,
                    subfolder="transformer",
                    torch_dtype=kwargs.get("torch_dtype", torch.bfloat16),
                    revision=kwargs.get("revision", None),
                    variant=kwargs.get("variant", None),
                )
        elif pretrained_model_name_or_path is not None:
            load_dtype = torch.bfloat16 if "5b" in base_model_name_or_path.lower() else torch.float16
            # Override with provided torch_dtype if specified
            if "torch_dtype" in kwargs:
                load_dtype = kwargs["torch_dtype"]
            print(f"🔧 Loading VideoX-Fun pipeline from {pretrained_model_name_or_path}")
            transformer = CogVideoXFunTransformer3DModel.from_pretrained(
                pretrained_model_name_or_path,
                subfolder="transformer",
                torch_dtype=load_dtype,
                revision=kwargs.get("revision", None),
                variant=kwargs.get("variant", None),
            )

        # Create our custom pipeline with the same components
        pipeline = cls(
            tokenizer=original_pipeline.tokenizer,
            text_encoder=original_pipeline.text_encoder,
            vae=original_pipeline.vae,
            transformer=transformer,
            scheduler=original_pipeline.scheduler,
            split_hands=split_hands,
            compress_smpl_pos_map_temporal=compress_smpl_pos_map_temporal,
            add_control_adapter=add_control_adapter,
            in_dim_control_adapter=in_dim_control_adapter,
            adapter_control_type=adapter_control_type,
        )
        
        return pipeline

    def save_pretrained(self, save_directory: str, **kwargs):
        """
        Save the pipeline to a directory.
        
        This saves all components (tokenizer, text_encoder, vae, transformer, scheduler)
        to the specified directory.
        """
        os.makedirs(save_directory, exist_ok=True)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(os.path.join(save_directory, "tokenizer"))
        
        # Save text encoder
        self.text_encoder.save_pretrained(os.path.join(save_directory, "text_encoder"))
        
        # Save VAE
        self.vae.save_pretrained(os.path.join(save_directory, "vae"))
        
        # Save transformer
        self.transformer.save_pretrained(os.path.join(save_directory, "transformer"))
        
        # Save scheduler
        self.scheduler.save_pretrained(os.path.join(save_directory, "scheduler"))
        
        # Save pipeline config
        pipeline_config = {
            "pipeline_class": "CogVideoXFunStaticToVideoPipeline",
            "version": "1.0.0",
        }
        with open(os.path.join(save_directory, "pipeline_config.json"), "w") as f:
            import json
            json.dump(pipeline_config, f)


    def prepare_latents(
        self,
        batch_size: int = 1,
        num_channels_latents: int = 16,
        height: int = 60,
        width: int = 90,
        num_frames: int = 13,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,  # For I2V fallback
        static_videos: Optional[torch.Tensor] = None,
        hand_videos: Optional[torch.Tensor] = None,
        smpl_pos_map: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        is_strength_max: bool = True,
        return_noise: bool = False,
    ):
        """Prepare latents for the pipeline with static conditions."""
        # Calculate latent dimensions
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial
        num_channels_latents = self.vae.config.latent_channels

        shape = (batch_size, num_frames, num_channels_latents, latent_height, latent_width)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            # if strength is 1. then initialise the latents to noise, else initial to image + noise
            if static_videos is not None and not is_strength_max:
                # Use static_videos as the reference for strength-based initialization
                static_videos = static_videos.to(device=device, dtype=self.vae.dtype)
                bs = 1
                new_static_videos = []
                for i in range(0, static_videos.shape[0], bs):
                    static_videos_bs = static_videos[i : i + bs]
                    static_videos_bs = self.vae.encode(static_videos_bs)[0]
                    static_videos_bs = static_videos_bs.sample()
                    new_static_videos.append(static_videos_bs)
                video_latents = torch.cat(new_static_videos, dim=0)
                video_latents = video_latents * self.vae.config.scaling_factor
                video_latents = video_latents.repeat(batch_size // video_latents.shape[0], 1, 1, 1, 1)
                video_latents = video_latents.to(device=device, dtype=dtype)
                video_latents = rearrange(video_latents, "b c f h w -> b f c h w")
                latents = self.scheduler.add_noise(video_latents, noise, timestep)
            else:
                latents = noise
            # if pure noise then scale the initial latents by the Scheduler's init sigma
            latents = latents * self.scheduler.init_noise_sigma if is_strength_max else latents
        else:
            noise = latents.to(device)
            latents = noise * self.scheduler.init_noise_sigma

        # Prepare condition latents
        static_videos_latents = None
        hand_videos_latents = None
        smpl_pos_map_latents = None
        
        if static_videos is not None:
            # Process static videos with VAE encoding
            static_videos = static_videos.to(device=device, dtype=self.vae.dtype)
            bs = 1
            with torch.no_grad():
                new_static_videos = []
                for i in range(0, static_videos.shape[0], bs):
                    static_videos_bs = static_videos[i : i + bs]
                    static_videos_bs = self.vae.encode(static_videos_bs)[0]
                    static_videos_bs = static_videos_bs.sample()
                    new_static_videos.append(static_videos_bs)
                
                static_videos_latents = torch.cat(new_static_videos, dim=0)
                static_videos_latents = static_videos_latents * self.vae.config.scaling_factor
                static_videos_latents = static_videos_latents.repeat(batch_size // static_videos_latents.shape[0], 1, 1, 1, 1)
                static_videos_latents = static_videos_latents.to(device=device, dtype=dtype)
                static_videos_latents = rearrange(static_videos_latents, "b c f h w -> b f c h w")

        if hand_videos is not None:
            # Process hand videos with VAE encoding
            hand_videos = hand_videos.to(device=device, dtype=self.vae.dtype)
            
            with torch.no_grad():
                if self.split_hands:
                    # Split hands mode: split into left and right, encode separately, then concat
                    batch_size_hand, channels, frames, height, width = hand_videos.shape
                    half_channels = channels // 2
                    
                    # Split into left and right hand videos
                    hand_left = hand_videos[:, :half_channels, :, :, :]  # [B, C/2, F, H, W]
                    hand_right = hand_videos[:, half_channels:, :, :, :]  # [B, C/2, F, H, W]
                    
                    # Encode left and right separately
                    bs = 1
                    new_hand_left = []
                    new_hand_right = []
                    
                    for i in range(0, hand_left.shape[0], bs):
                        hand_left_bs = hand_left[i : i + bs]
                        hand_right_bs = hand_right[i : i + bs]
                        
                        # Encode left hand
                        hand_left_bs = self.vae.encode(hand_left_bs)[0]
                        hand_left_bs = hand_left_bs.sample()
                        new_hand_left.append(hand_left_bs)
                        
                        # Encode right hand
                        hand_right_bs = self.vae.encode(hand_right_bs)[0]
                        hand_right_bs = hand_right_bs.sample()
                        new_hand_right.append(hand_right_bs)
                    
                    hand_left_latents = torch.cat(new_hand_left, dim=0)
                    hand_right_latents = torch.cat(new_hand_right, dim=0)
                    
                    # Concatenate along channel dimension
                    hand_videos_latents = torch.cat([hand_left_latents, hand_right_latents], dim=1)
                else:
                    # Regular mode: encode normally
                    bs = 1
                    new_hand_videos = []
                    for i in range(0, hand_videos.shape[0], bs):
                        hand_videos_bs = hand_videos[i : i + bs]
                        hand_videos_bs = self.vae.encode(hand_videos_bs)[0]
                        hand_videos_bs = hand_videos_bs.sample()
                        new_hand_videos.append(hand_videos_bs)
                    hand_videos_latents = torch.cat(new_hand_videos, dim=0)
            
            hand_videos_latents = hand_videos_latents * self.vae.config.scaling_factor
            hand_videos_latents = hand_videos_latents.repeat(batch_size // hand_videos_latents.shape[0], 1, 1, 1, 1)
            hand_videos_latents = hand_videos_latents.to(device=device, dtype=dtype)
            hand_videos_latents = rearrange(hand_videos_latents, "b c f h w -> b f c h w")

        if smpl_pos_map is not None:
            # Process SMPL pos map
            if self.compress_smpl_pos_map_temporal:
                # Already compressed: [B, F, C, H, W] where C = 12 (3 * 4)
                smpl_pos_map_latents = smpl_pos_map.to(device=device, dtype=dtype)
            else:
                # Need to VAE encode: [B, C, F, H, W]
                smpl_pos_map = smpl_pos_map.to(device=device, dtype=self.vae.dtype)
                bs = 1
                with torch.no_grad():
                    new_smpl_pos_maps = []
                    for i in range(0, smpl_pos_map.shape[0], bs):
                        smpl_pos_map_bs = smpl_pos_map[i : i + bs]
                        smpl_pos_map_bs = self.vae.encode(smpl_pos_map_bs)[0]
                        smpl_pos_map_bs = smpl_pos_map_bs.sample()
                        new_smpl_pos_maps.append(smpl_pos_map_bs)
                    
                    smpl_pos_map_latents = torch.cat(new_smpl_pos_maps, dim=0)
                    smpl_pos_map_latents = smpl_pos_map_latents * self.vae.config.scaling_factor
                    smpl_pos_map_latents = smpl_pos_map_latents.repeat(batch_size // smpl_pos_map_latents.shape[0], 1, 1, 1, 1)
                    smpl_pos_map_latents = smpl_pos_map_latents.to(device=device, dtype=dtype)
                    smpl_pos_map_latents = rearrange(smpl_pos_map_latents, "b c f h w -> b f c h w")

        # Prepare outputs
        outputs = (latents,)
        
        if return_noise:
            outputs += (noise,)
            
        # Always return static, hand, and smpl pos map video latents (can be None)
        outputs += (static_videos_latents, hand_videos_latents, smpl_pos_map_latents)
            
        return outputs

    def preprocess_hand_conditions(
        self,
        hand_videos,
        height,
        width,
        num_frames,
    ):
        """Preprocess hand condition videos.
        
        Args:
            hand_videos: Input video tensor of shape [B, C, F, H, W]
            height: Target height
            width: Target width  
            num_frames: Target number of frames
            
        Returns:
            Preprocessed video tensor of shape [B, C, F, H, W]
        """
        # Convert to torch tensor if needed
        if isinstance(hand_videos, np.ndarray):
            hand_videos = torch.from_numpy(hand_videos)
        
        # Ensure input is [B, C, F, H, W]
        if hand_videos.ndim == 4:  # [C, F, H, W] -> [1, C, F, H, W]
            hand_videos = hand_videos.unsqueeze(0)
        elif hand_videos.ndim == 5:  # Already [B, C, F, H, W]
            pass
        else:
            raise ValueError(f"Expected 4D or 5D tensor, got {hand_videos.ndim}D tensor")
        
        # Input is already [B, C, F, H, W], no need to permute
        
        # Ensure correct number of frames
        if hand_videos.shape[2] != num_frames:
            hand_videos = F.interpolate(
                hand_videos,  # [B, C, F, H, W]
                size=(num_frames, height, width),
                mode='trilinear',
                align_corners=False
            )
        
        return hand_videos

    def preprocess_smpl_pos_map(
        self,
        smpl_pos_map,
        height,
        width,
        num_frames,
    ):
        """Preprocess SMPL pos map videos.
        
        Args:
            smpl_pos_map: Input video tensor 
                - If compress_smpl_pos_map_temporal=True: [B, T', C*4, H/8, W/8] (already compressed)
                - If compress_smpl_pos_map_temporal=False: [B, C, F, H, W] (raw video)
            height: Target height
            width: Target width  
            num_frames: Target number of frames
            
        Returns:
            Preprocessed video tensor
        """
        # Convert to torch tensor if needed
        if isinstance(smpl_pos_map, np.ndarray):
            smpl_pos_map = torch.from_numpy(smpl_pos_map)
        
        if self.compress_smpl_pos_map_temporal:
            # Already compressed format: [B, T', C*4, H/8, W/8]
            # Just ensure correct dtype and device - no interpolation needed
            return smpl_pos_map
        else:
            # Raw video format: [B, C, F, H, W]
            if smpl_pos_map.ndim == 4:  # [C, F, H, W] -> [1, C, F, H, W]
                smpl_pos_map = smpl_pos_map.unsqueeze(0)
            elif smpl_pos_map.ndim != 5:
                raise ValueError(f"Expected 4D or 5D tensor, got {smpl_pos_map.ndim}D tensor")
            
            # Ensure correct number of frames
            if smpl_pos_map.shape[2] != num_frames:
                smpl_pos_map = F.interpolate(
                    smpl_pos_map,
                    size=(num_frames, height, width),
                    mode='trilinear',
                    align_corners=False
                )
            
            return smpl_pos_map

    def preprocess_adapter_control(
        self,
        adapter_control,
        height,
        width,
        num_frames,
    ):
        """Preprocess adapter control signals (e.g., SMPL pos map for adapter).
        
        Args:
            adapter_control: Input control tensor [B, T', C, H, W] (already compressed)
            height: Target height (not used, adapter handles resolution)
            width: Target width (not used, adapter handles resolution)
            num_frames: Target number of frames (not used, already compressed temporally)
            
        Returns:
            Preprocessed control tensor (same as input for compressed format)
        """
        # Convert to torch tensor if needed
        if isinstance(adapter_control, np.ndarray):
            adapter_control = torch.from_numpy(adapter_control)
        
        # For adapter control, we expect compressed format: [B, T', C, H, W]
        # The adapter will handle spatial processing through PixelUnshuffle
        # No preprocessing needed - just ensure correct format
        return adapter_control

    def preprocess_raymap(
        self,
        raymap,
        height,
        width,
        num_frames,
    ):
        """Preprocess raymap data.
        
        Args:
            raymap: Input raymap tensor [B, F, 6, H/8, W/8] (uncompressed)
            height: Target height
            width: Target width  
            num_frames: Target number of frames
            
        Returns:
            Preprocessed and temporally compressed raymap tensor [B, T', 24, H/8, W/8]
            where T' = F / vae_scale_factor_temporal, 24 = 6 * 4
        """
        # Convert to torch tensor if needed
        if isinstance(raymap, np.ndarray):
            raymap = torch.from_numpy(raymap)
        
        # Raymap comes in as [B, F, 6, h, w]
        # Need to compress temporally: [B, F, 6, h, w] -> [B, T', 24, h, w]
        # where T' = F / vae_scale_factor_temporal (4)
        
        # If frames not divisible by vae_scale_factor_temporal, pad by repeating first frames
        if raymap.shape[1] % self.vae_scale_factor_temporal != 0:
            # Calculate how many frames to pad
            padding_frames = self.vae_scale_factor_temporal - (raymap.shape[1] % self.vae_scale_factor_temporal)
            # Repeat first frames to pad
            raymap = torch.cat([raymap[:, :padding_frames], raymap], dim=1)
        
        # Temporally compress: rearrange (n=4 frames) * (c=6 channels) = 24 channels
        # b (n t) c h w -> b t (n c) h w
        camera_conditions = rearrange(
            raymap,
            "b (n t) c h w -> b t (n c) h w",
            n=self.vae_scale_factor_temporal,
        )
        
        return camera_conditions

    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt=None,
        callback_on_step_end_tensor_inputs=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        """Check inputs for the static-to-video pipeline."""
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not isinstance(callback_on_step_end_tensor_inputs, list):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be a list but is {type(callback_on_step_end_tensor_inputs)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " either forward `prompt` or `prompt_embeds`."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to either forward `negative_prompt` or"
                " `negative_prompt_embeds`."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    def preprocess_static_conditions(
        self,
        static_videos,  
        height,
        width,
        num_frames,
    ):
        """Preprocess static condition videos.
        
        Args:
            static_videos: Input video tensor of shape [B, C, F, H, W]
            height: Target height
            width: Target width  
            num_frames: Target number of frames
            
        Returns:
            Preprocessed video tensor of shape [B, C, F, H, W]
        """
        # Convert to torch tensor if needed
        if isinstance(static_videos, np.ndarray):
            static_videos = torch.from_numpy(static_videos)
        
        # Ensure input is [B, C, F, H, W]
        if static_videos.ndim == 4:  # [C, F, H, W] -> [1, C, F, H, W]
            static_videos = static_videos.unsqueeze(0)
        elif static_videos.ndim == 5:  # Already [B, C, F, H, W]
            pass
        else:
            raise ValueError(f"Expected 4D or 5D tensor, got {static_videos.ndim}D tensor")
        
        # Input is already [B, C, F, H, W], no need to permute
        
        # Ensure correct number of frames
        if static_videos.shape[2] != num_frames:
            static_videos = F.interpolate(
                static_videos,  # [B, C, F, H, W]
                size=(num_frames, height, width),
                mode='trilinear',
                align_corners=False
            )
        
        return static_videos

    def _preprocess_images(self, images, height, width):
        """Preprocess images to the required format."""
        if isinstance(images, torch.Tensor):
            images = images.cpu().numpy()
        
        if images.dtype == np.uint8:
            images = images.astype(np.float32) / 255.0
            
        if images.ndim == 3:
            images = [images]
            
        images = imcrop_center(images, height, width)
        images = self.video_processor.preprocess(images, height, width)
        return images

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 480,
        width: int = 720,
        image: Optional[Union[PIL.Image.Image, np.ndarray, torch.FloatTensor]] = None,
        video: Union[torch.FloatTensor] = None,
        mask_video: Union[torch.FloatTensor] = None,
        masked_video_latents: Union[torch.FloatTensor] = None,
        static_videos: Optional[Union[torch.FloatTensor, np.ndarray, List[PIL.Image.Image]]] = None,
        hand_videos: Optional[Union[torch.FloatTensor, np.ndarray, List[PIL.Image.Image]]] = None,
        smpl_pos_map: Optional[Union[torch.FloatTensor, np.ndarray]] = None,
        raymap: Optional[Union[torch.FloatTensor, np.ndarray]] = None,
        num_frames: int = 49,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6,
        use_dynamic_cfg: bool = False,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "np",
        return_dict: bool = False,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 226,
        strength: float = 1,
        noise_aug_strength: float = 0.0563,
        comfyui_progressbar: bool = False,
    ) -> Union[CogVideoXFunPipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_frames (`int`, defaults to `48`):
                Number of frames to generate. Must be divisible by self.vae_scale_factor_temporal. Generated video will
                contain 1 extra frame because CogVideoX_Fun is conditioned with (num_seconds * fps + 1) frames where
                num_seconds is 6 and fps is 4. However, since videos can be saved at any fps, the only condition that
                needs to be satisfied is that of divisibility mentioned above.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of videos to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
                of a plain tuple.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int`, defaults to `226`):
                Maximum sequence length in encoded prompt. Must be consistent with
                `self.transformer.config.max_text_seq_length` otherwise may lead to poor results.

        Examples:

        Returns:
            [`~pipelines.cogvideo.pipeline_cogvideox.CogVideoXFunPipelineOutput`] or `tuple`:
            [`~pipelines.cogvideo.pipeline_cogvideox.CogVideoXFunPipelineOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is a list with the generated images.
        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        height = height or self.transformer.config.sample_height * self.vae_scale_factor_spatial
        width = width or self.transformer.config.sample_width * self.vae_scale_factor_spatial
        num_frames = num_frames or self.transformer.config.sample_frames

        num_videos_per_prompt = 1

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt=prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        # 2. Default call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            negative_prompt,
            do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        # 4. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(
            num_inference_steps=num_inference_steps, strength=strength, device=device
        )
        self._num_timesteps = len(timesteps)
        if comfyui_progressbar:
            try:
                from comfy.utils import ProgressBar
                pbar = ProgressBar(num_inference_steps + 2)
            except ImportError:
                # Fallback if comfy.utils is not available
                pbar = None
        # at which timestep to set the initial noise (n.b. 50% if strength is 0.5)
        latent_timestep = timesteps[:1].repeat(batch_size * num_videos_per_prompt)
        # create a boolean to check if the strength is set to 1. if so then initialise the latents with pure noise
        is_strength_max = strength == 1.0

        # 5. Prepare latents.
        # Convert image to static_videos if provided (for I2V mode)
        if image is not None and static_videos is None:
            # Process image and convert to single-frame static_videos
            if isinstance(image, PIL.Image.Image):
                image_tensor = self.video_processor.preprocess(image, height, width, resize_mode="crop")
            elif isinstance(image, np.ndarray):
                image_tensor = self._preprocess_image(image, height, width)
            else:  # torch.Tensor
                image_tensor = image
            
            # Convert to [B, C, 1, H, W] format for static_videos
            if image_tensor.ndim == 3:  # [C, H, W]
                image_tensor = image_tensor.unsqueeze(0).unsqueeze(2)  # [1, C, 1, H, W]
            elif image_tensor.ndim == 4:  # [B, C, H, W]
                image_tensor = image_tensor.unsqueeze(2)  # [B, C, 1, H, W]
            
            # Use image as first frame, rest will be zero-padded in prepare_latents
            static_videos = image_tensor
        
        if static_videos is not None:
            # Use static_videos as the condition video
            static_videos = self.preprocess_static_conditions(static_videos, height, width, num_frames)
            video_length = static_videos.shape[2]
            init_video = self.image_processor.preprocess(rearrange(static_videos, "b c f h w -> (b f) c h w"), height=height, width=width) 
            init_video = init_video.to(dtype=torch.float32)
            init_video = rearrange(init_video, "(b f) c h w -> b c f h w", f=video_length)
        elif video is not None:
            video_length = video.shape[2]
            init_video = self.image_processor.preprocess(rearrange(video, "b c f h w -> (b f) c h w"), height=height, width=width) 
            init_video = init_video.to(dtype=torch.float32)
            init_video = rearrange(init_video, "(b f) c h w -> b c f h w", f=video_length)
        else:
            init_video = None

        # Magvae needs the number of frames to be 4n + 1.
        local_latent_length = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        # For CogVideoX 1.5, the latent frames should be clipped to make it divisible by patch_size_t
        patch_size_t = self.transformer.config.patch_size_t
        additional_frames = 0
        if patch_size_t is not None and local_latent_length % patch_size_t != 0:
            additional_frames = local_latent_length % patch_size_t
            num_frames -= additional_frames * self.vae_scale_factor_temporal
        if num_frames <= 0:
            num_frames = 1
        if video_length > num_frames:
            logger.warning("The length of condition video is not right, the latent frames should be clipped to make it divisible by patch_size_t. ")
            video_length = num_frames
            video = video[:, :, :video_length]
            init_video = init_video[:, :, :video_length]
            mask_video = mask_video[:, :, :video_length]

        # Get channel information for later use
        num_channels_latents = self.vae.config.latent_channels
        num_channels_transformer = self.transformer.config.in_channels

        # Preprocess hand_videos if provided
        processed_hand_videos = None
        if hand_videos is not None:
            processed_hand_videos = self.preprocess_hand_conditions(hand_videos, height, width, num_frames)

        # Preprocess smpl_pos_map if provided
        processed_smpl_pos_map = None
        if smpl_pos_map is not None:
            processed_smpl_pos_map = self.preprocess_smpl_pos_map(smpl_pos_map, height, width, num_frames)
        
        # Preprocess raymap if provided
        processed_raymap = None
        if raymap is not None:
            processed_raymap = self.preprocess_raymap(raymap, height, width, num_frames)
        
        # Determine adapter control based on adapter_control_type
        processed_adapter_control = None
        if self.add_control_adapter:
            if self.adapter_control_type == "smpl_pos_map" and processed_smpl_pos_map is not None:
                # Use SMPL pos map as adapter control
                processed_adapter_control = self.preprocess_adapter_control(processed_smpl_pos_map, height, width, num_frames)
                # Don't use smpl_pos_map for concat since it's used for adapter
                processed_smpl_pos_map = None
            elif self.adapter_control_type == "hand_videos" and processed_hand_videos is not None:
                # Use hand videos as adapter control
                processed_adapter_control = self.preprocess_adapter_control(processed_hand_videos, height, width, num_frames)
                # Don't use hand_videos for concat since it's used for adapter
                processed_hand_videos = None
            # Can add more control types here in the future

        latents_outputs = self.prepare_latents(
            batch_size=batch_size * num_videos_per_prompt,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            num_frames=local_latent_length,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
            latents=latents,
            static_videos=init_video,
            hand_videos=processed_hand_videos,
            smpl_pos_map=processed_smpl_pos_map,
            timestep=latent_timestep,
            is_strength_max=is_strength_max,
            return_noise=True,
        )
        # Always get static, hand, and smpl pos map video latents
        latents, noise, static_videos_latents, hand_videos_latents, smpl_pos_map_latents = latents_outputs
        if comfyui_progressbar and pbar is not None:
            pbar.update(1)
        
        # Concatenate condition latents (hand_videos + smpl_pos_map + raymap)
        control_latents = None
        control_latents_list = []
        
        if hand_videos_latents is not None:
            control_latents_list.append(hand_videos_latents)
        
        if smpl_pos_map_latents is not None:
            control_latents_list.append(smpl_pos_map_latents)
        
        # Add raymap if provided (already in compressed format [B, T', 24, H/8, W/8])
        if processed_raymap is not None:
            # Raymap is already compressed and at correct resolution
            # Just ensure it's on the correct device and dtype
            raymap_for_concat = processed_raymap.to(device=device, dtype=latents.dtype)
            control_latents_list.append(raymap_for_concat)
        
        if control_latents_list:
            # Concatenate along channel dimension: [B, F, C1, H, W] + [B, F, C2, H, W] -> [B, F, C1+C2, H, W]
            control_latents = torch.cat(control_latents_list, dim=2)
            control_latents = (
                torch.cat([control_latents] * 2) if do_classifier_free_guidance else control_latents
            )
        
        # Determine adapter_control based on adapter_control_type
        adapter_control_for_transformer = None
        if self.add_control_adapter:
            if self.adapter_control_type == "smpl_pos_map" and processed_smpl_pos_map is not None:
                # Use SMPL pos map as adapter control
                # Convert [B, F, C, H, W] -> [B, C, F, H, W] for adapter
                adapter_control_for_transformer = processed_smpl_pos_map.permute(0, 2, 1, 3, 4)
                # Don't include smpl_pos_map in control_latents concat since it's used for adapter
                if smpl_pos_map_latents is not None:
                    control_latents_list = [item for item in control_latents_list if item is not smpl_pos_map_latents]
            elif self.adapter_control_type == "hand_videos" and processed_hand_videos is not None:
                # Use hand videos as adapter control
                # Convert [B, F, C, H, W] -> [B, C, F, H, W] for adapter
                adapter_control_for_transformer = processed_hand_videos.permute(0, 2, 1, 3, 4)
                # Don't include hand_videos in control_latents concat since it's used for adapter
                if hand_videos_latents is not None:
                    control_latents_list = [item for item in control_latents_list if item is not hand_videos_latents]
            
            # Apply classifier-free guidance if needed
            if adapter_control_for_transformer is not None:
                adapter_control_for_transformer = (
                    torch.cat([adapter_control_for_transformer] * 2) if do_classifier_free_guidance else adapter_control_for_transformer
                )
        
        # Re-concatenate control_latents after potentially removing adapter control
        if control_latents_list:
            control_latents = torch.cat(control_latents_list, dim=2)
            control_latents = (
                torch.cat([control_latents] * 2) if do_classifier_free_guidance else control_latents
            )
        else:
            control_latents = None
        if mask_video is not None:
            if (mask_video == 255).all():  # redraw all frames
                mask_latents = torch.zeros_like(latents)[:, :, :1].to(latents.device, latents.dtype)
                masked_video_latents = torch.zeros_like(latents).to(latents.device, latents.dtype)

                mask_input = torch.cat([mask_latents] * 2) if do_classifier_free_guidance else mask_latents
                masked_video_latents_input = (
                    torch.cat([masked_video_latents] * 2) if do_classifier_free_guidance else masked_video_latents
                )
                inpaint_latents = torch.cat([mask_input, masked_video_latents_input], dim=2).to(latents.dtype)
            else:
                # Prepare mask latent variables
                mask_condition = self.mask_processor.preprocess(rearrange(mask_video, "b c f h w -> (b f) c h w"), height=height, width=width) 
                mask_condition = mask_condition.to(dtype=torch.float32)
                mask_condition = rearrange(mask_condition, "(b f) c h w -> b c f h w", f=video_length)

                if num_channels_transformer != num_channels_latents:
                    mask_condition_tile = torch.tile(mask_condition, [1, 3, 1, 1, 1])
                    if masked_video_latents is None:
                        masked_video = init_video * (mask_condition_tile < 0.5) + torch.ones_like(init_video) * (mask_condition_tile > 0.5) * -1
                    else:
                        masked_video = masked_video_latents

                    _, masked_video_latents = self.prepare_mask_latents(
                        None,
                        masked_video,
                        batch_size,
                        height,
                        width,
                        prompt_embeds.dtype,
                        device,
                        generator,
                        do_classifier_free_guidance,
                        noise_aug_strength=noise_aug_strength,
                    )
                    mask_latents = resize_mask(1 - mask_condition, masked_video_latents)
                    mask_latents = mask_latents.to(masked_video_latents.device) * self.vae.config.scaling_factor

                    mask = torch.tile(mask_condition, [1, num_channels_latents, 1, 1, 1])
                    mask = F.interpolate(mask, size=latents.size()[-3:], mode='trilinear', align_corners=True).to(latents.device, latents.dtype)
                    
                    mask_input = torch.cat([mask_latents] * 2) if do_classifier_free_guidance else mask_latents
                    masked_video_latents_input = (
                        torch.cat([masked_video_latents] * 2) if do_classifier_free_guidance else masked_video_latents
                    )

                    mask = rearrange(mask, "b c f h w -> b f c h w")
                    mask_input = rearrange(mask_input, "b c f h w -> b f c h w")
                    masked_video_latents_input = rearrange(masked_video_latents_input, "b c f h w -> b f c h w")
                    inpaint_latents = torch.cat([mask_input, masked_video_latents_input], dim=2).to(latents.dtype)
                else:
                    mask = torch.tile(mask_condition, [1, num_channels_latents, 1, 1, 1])
                    mask = F.interpolate(mask, size=latents.size()[-3:], mode='trilinear', align_corners=True).to(latents.device, latents.dtype)
                    mask = rearrange(mask, "b c f h w -> b f c h w")
                    
                    inpaint_latents = None
        else:
            if num_channels_transformer != num_channels_latents:
                # Create 1-channel mask
                mask = torch.zeros_like(latents[:, :, :1]).to(latents.device, latents.dtype)  # [B, 1, F, H, W]
                
                if static_videos_latents is not None:
                    # Use static_videos_latents as masked_video_latents
                    masked_video_latents = static_videos_latents.to(latents.device, latents.dtype)
                else:
                    # No static videos, use zeros
                    masked_video_latents = torch.zeros_like(latents).to(latents.device, latents.dtype)
                masked_video_latents = rearrange(masked_video_latents, "b f c h w -> b c f h w")

                # Create mask_latents using resize_mask function (similar to mask_video is not None case)
                mask_condition = self.mask_processor.preprocess(rearrange(mask, "b f c h w -> (b f) c h w"), height=height, width=width) 
                mask_condition = mask_condition.to(dtype=torch.float32)
                mask_condition = rearrange(mask_condition, "(b f) c h w -> b c f h w", f=local_latent_length)
                mask_latents = resize_mask(1 - mask_condition, masked_video_latents)
                mask_latents = mask_latents.to(masked_video_latents.device) * self.vae.config.scaling_factor

                # Handle classifier-free guidance
                if do_classifier_free_guidance:
                    mask_input = torch.cat([mask_latents, mask_latents], dim=0)  # [2B, 1, F, H, W]
                    masked_video_latents_input = torch.cat([masked_video_latents, masked_video_latents], dim=0)  # [2B, C, F, H, W]
                else:
                    mask_input = mask_latents
                    masked_video_latents_input = masked_video_latents
                
                # Concatenate mask (1 channel) + masked_video_latents (C channels) = [B, C+1, F, H, W]
                mask_input = rearrange(mask_input, "b c f h w -> b f c h w")
                masked_video_latents_input = rearrange(masked_video_latents_input, "b c f h w -> b f c h w")
                inpaint_latents = torch.cat([mask_input, masked_video_latents_input], dim=2).to(latents.dtype)
            else:
                # Standard case without additional channels
                mask = torch.zeros_like(init_video[:, :1])
                mask = torch.tile(mask, [1, num_channels_latents, 1, 1, 1])
                mask = F.interpolate(mask, size=latents.size()[-3:], mode='trilinear', align_corners=True).to(latents.device, latents.dtype)
                mask = rearrange(mask, "b c f h w -> b f c h w")

                inpaint_latents = None
        if comfyui_progressbar and pbar is not None:
            pbar.update(1)

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Create rotary embeds if required
        image_rotary_emb = (
            self._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
            if self.transformer.config.use_rotary_positional_embeddings
            else None
        )
        
        # Initialize adapter_control_for_transformer (used by all pipelines)
        if not hasattr(self, 'add_control_adapter') or not self.add_control_adapter:
            adapter_control_for_transformer = None

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            # for DPM-solver++
            old_pred_original_sample = None
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                # predict noise model_output
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    image_rotary_emb=image_rotary_emb,
                    return_dict=False,
                    inpaint_latents=inpaint_latents,
                    control_latents=control_latents,
                    adapter_control=adapter_control_for_transformer,
                )[0]
                noise_pred = noise_pred.float()

                # perform guidance
                if use_dynamic_cfg:
                    self._guidance_scale = 1 + guidance_scale * (
                        (1 - math.cos(math.pi * ((num_inference_steps - t.item()) / num_inference_steps) ** 5.0)) / 2
                    )
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                if not isinstance(self.scheduler, CogVideoXDPMScheduler):
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                else:
                    latents, old_pred_original_sample = self.scheduler.step(
                        noise_pred,
                        old_pred_original_sample,
                        t,
                        timesteps[i - 1] if i > 0 else None,
                        latents,
                        **extra_step_kwargs,
                        return_dict=False,
                    )
                latents = latents.to(prompt_embeds.dtype)

                # call the callback, if provided
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                if comfyui_progressbar:
                    pbar.update(1)

        if not output_type == "latent":
            video = self.decode_latents(latents)
            video = rearrange(video, "b c f h w -> b f h w c")
            # video = self.video_processor.postprocess_video(video=video, output_type=output_type)
            # video = rearrange(video, "b f h w c -> b c f h w")
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        # if not return_dict:
        #     if isinstance(video, np.ndarray):
        #         video = torch.from_numpy(video)
        return CogVideoXFunPipelineOutput(frames=video)

    @classmethod
    def save_lora_weights(
        cls,
        save_directory: str,
        transformer_lora_layers: Optional[Dict[str, torch.Tensor]] = None,
        is_main_process: bool = True,
        weight_name: str = "pytorch_lora_weights.safetensors",
        save_function: Optional[Callable] = None,
        safe_serialization: bool = True,
    ):
        """Save LoRA weights to a directory."""
        if transformer_lora_layers is not None:
            # Save LoRA weights
            os.makedirs(save_directory, exist_ok=True)
            
            if save_function is not None:
                save_function(transformer_lora_layers, os.path.join(save_directory, weight_name))
            elif safe_serialization:
                from safetensors.torch import save_file
                save_file(transformer_lora_layers, os.path.join(save_directory, weight_name))
            else:
                torch.save(transformer_lora_layers, os.path.join(save_directory, weight_name))

    @classmethod
    def lora_state_dict(cls, input_dir: str) -> Dict[str, torch.Tensor]:
        """Load LoRA state dict from a directory."""
        lora_state_dict = {}
        
        # Try to load from safetensors first
        safetensors_path = os.path.join(input_dir, "pytorch_lora_weights.safetensors")
        if os.path.exists(safetensors_path):
            from safetensors.torch import load_file
            lora_state_dict = load_file(safetensors_path)
        else:
            # Try to load from pytorch format
            pytorch_path = os.path.join(input_dir, "pytorch_lora_weights.bin")
            if os.path.exists(pytorch_path):
                lora_state_dict = torch.load(pytorch_path, map_location="cpu")
        
        return lora_state_dict

class CogVideoXFunStaticToVideoPoseTokenPipeline(CogVideoXFunInpaintPipeline):
    r"""
    Pipeline for text-to-video generation using CogVideoX with static and hand pose conditions.

    This model inherits from [`CogVideoXFunInpaintPipeline`]. It extends the base VideoX-Fun pipeline
    to support static video and hand pose conditioning.

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
        text_encoder ([`T5EncoderModel`]):
            Frozen text-encoder. CogVideoX_Fun uses
            [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel); specifically the
            [t5-v1_1-xxl](https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/t5-v1_1-xxl) variant.
        tokenizer (`T5Tokenizer`):
            Tokenizer of class
            [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer).
        transformer ([`CogVideoXTransformer3DModel`]):
            A text conditioned `CogVideoXTransformer3DModel` to denoise the encoded video latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded video latents.
    """

    def __init__(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        vae: AutoencoderKLCogVideoX,
        transformer: CogVideoXTransformer3DModel,
        scheduler: Union[CogVideoXDDIMScheduler, CogVideoXDPMScheduler],
    ):
        super().__init__(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path=None, 
                        base_model_name_or_path="alibaba-pai/CogVideoX-Fun-V1.1-5b-InP", 
                        transformer=None, 
                        condition_channels: Optional[int] = 16,
                        *args, **kwargs):
        """
        Load a CogVideoXFunStaticToVideoPipeline from a saved directory or base model.
        
        This method loads all components (tokenizer, text_encoder, vae, transformer, scheduler)
        from the specified directory, or uses provided components if specified.
        
        Args:
            pretrained_model_name_or_path: Path to saved pipeline or base model
            base_model_name_or_path: Optional base model path for creating pose-conditioned pipeline
            transformer: Optional transformer to use (for validation with trained transformer)
            condition_channels: Number of condition channels (0 or None for base model, >0 for concat model)
            use_adapter: Whether to use adapter-based transformer
            adapter_version: Version of adapter to use ("v1" or "v2")
        """
        # Check if this is a base model path (for creating pose-conditioned pipeline)
        if base_model_name_or_path is not None:
            print(f"🔧 Loading pipeline components (tokenizer, text_encoder, vae, scheduler) from base model: {base_model_name_or_path}")
            # Load the original CogVideoX-Fun pipeline from base model to get non-transformer components
            original_pipeline = super().from_pretrained(base_model_name_or_path, *args, **kwargs)
            
        if transformer is None:
            print(f"🔧 Loading base VideoX-Fun transformer")
            transformer = CogVideoXFunTransformer3DModelWithCondToken.from_pretrained(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                base_model_name_or_path=base_model_name_or_path,
                subfolder="transformer",
                condition_channels=condition_channels,
                torch_dtype=kwargs.get("torch_dtype", torch.bfloat16),
                revision=kwargs.get("revision", None),
                variant=kwargs.get("variant", None),
            )
        elif pretrained_model_name_or_path is not None:
            load_dtype = torch.bfloat16 if "5b" in base_model_name_or_path.lower() else torch.float16
            # Override with provided torch_dtype if specified
            if "torch_dtype" in kwargs:
                load_dtype = kwargs["torch_dtype"]
            print(f"🔧 Loading VideoX-Fun pipeline from {pretrained_model_name_or_path}")
            transformer = CogVideoXFunTransformer3DModelWithCondToken.from_pretrained(
                pretrained_model_name_or_path,
                subfolder="transformer",
                condition_channels=condition_channels,
                torch_dtype=load_dtype,
                revision=kwargs.get("revision", None),
                variant=kwargs.get("variant", None),
            )

        # Create our custom pipeline with the same components
        pipeline = cls(
            tokenizer=original_pipeline.tokenizer,
            text_encoder=original_pipeline.text_encoder,
            vae=original_pipeline.vae,
            transformer=transformer,
            scheduler=original_pipeline.scheduler,
        )
        
        return pipeline

    def save_pretrained(self, save_directory: str, **kwargs):
        """
        Save the pipeline to a directory.
        
        This saves all components (tokenizer, text_encoder, vae, transformer, scheduler)
        to the specified directory.
        """
        os.makedirs(save_directory, exist_ok=True)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(os.path.join(save_directory, "tokenizer"))
        
        # Save text encoder
        self.text_encoder.save_pretrained(os.path.join(save_directory, "text_encoder"))
        
        # Save VAE
        self.vae.save_pretrained(os.path.join(save_directory, "vae"))
        
        # Save transformer
        self.transformer.save_pretrained(os.path.join(save_directory, "transformer"))
        
        # Save scheduler
        self.scheduler.save_pretrained(os.path.join(save_directory, "scheduler"))
        
        # Save pipeline config
        pipeline_config = {
            "pipeline_class": "CogVideoXFunStaticToVideoPoseTokenPipeline",
            "version": "1.0.0",
        }
        with open(os.path.join(save_directory, "pipeline_config.json"), "w") as f:
            import json
            json.dump(pipeline_config, f)


    def prepare_latents(
        self,
        batch_size: int = 1,
        num_channels_latents: int = 16,
        height: int = 60,
        width: int = 90,
        num_frames: int = 13,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,  # For I2V fallback
        static_videos: Optional[torch.Tensor] = None,
        hand_videos: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        is_strength_max: bool = True,
        return_noise: bool = False,
    ):
        """Prepare latents for the pipeline with static conditions."""
        # Calculate latent dimensions
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial
        num_channels_latents = self.vae.config.latent_channels

        shape = (batch_size, num_frames, num_channels_latents, latent_height, latent_width)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            # if strength is 1. then initialise the latents to noise, else initial to image + noise
            if static_videos is not None and not is_strength_max:
                # Use static_videos as the reference for strength-based initialization
                static_videos = static_videos.to(device=device, dtype=self.vae.dtype)
                bs = 1
                new_static_videos = []
                for i in range(0, static_videos.shape[0], bs):
                    static_videos_bs = static_videos[i : i + bs]
                    static_videos_bs = self.vae.encode(static_videos_bs)[0]
                    static_videos_bs = static_videos_bs.sample()
                    new_static_videos.append(static_videos_bs)
                video_latents = torch.cat(new_static_videos, dim=0)
                video_latents = video_latents * self.vae.config.scaling_factor
                video_latents = video_latents.repeat(batch_size // video_latents.shape[0], 1, 1, 1, 1)
                video_latents = video_latents.to(device=device, dtype=dtype)
                video_latents = rearrange(video_latents, "b c f h w -> b f c h w")
                latents = self.scheduler.add_noise(video_latents, noise, timestep)
            else:
                latents = noise
            # if pure noise then scale the initial latents by the Scheduler's init sigma
            latents = latents * self.scheduler.init_noise_sigma if is_strength_max else latents
        else:
            noise = latents.to(device)
            latents = noise * self.scheduler.init_noise_sigma

        # Prepare condition latents
        static_videos_latents = None
        hand_videos_latents = None
        smpl_pos_map_latents = None
        
        if static_videos is not None:
            # Process static videos with VAE encoding
            static_videos = static_videos.to(device=device, dtype=self.vae.dtype)
            bs = 1
            new_static_videos = []
            for i in range(0, static_videos.shape[0], bs):
                static_videos_bs = static_videos[i : i + bs]
                static_videos_bs = self.vae.encode(static_videos_bs)[0]
                static_videos_bs = static_videos_bs.sample()
                new_static_videos.append(static_videos_bs)
            static_videos_latents = torch.cat(new_static_videos, dim=0)
            static_videos_latents = static_videos_latents * self.vae.config.scaling_factor
            static_videos_latents = static_videos_latents.repeat(batch_size // static_videos_latents.shape[0], 1, 1, 1, 1)
            static_videos_latents = static_videos_latents.to(device=device, dtype=dtype)
            static_videos_latents = rearrange(static_videos_latents, "b c f h w -> b f c h w")

        if hand_videos is not None:
            # Process hand videos with VAE encoding
            hand_videos = hand_videos.to(device=device, dtype=self.vae.dtype)
            bs = 1
            new_hand_videos = []
            for i in range(0, hand_videos.shape[0], bs):
                hand_videos_bs = hand_videos[i : i + bs]
                hand_videos_bs = self.vae.encode(hand_videos_bs)[0]
                hand_videos_bs = hand_videos_bs.sample()
                new_hand_videos.append(hand_videos_bs)
            hand_videos_latents = torch.cat(new_hand_videos, dim=0)
            hand_videos_latents = hand_videos_latents * self.vae.config.scaling_factor
            hand_videos_latents = hand_videos_latents.repeat(batch_size // hand_videos_latents.shape[0], 1, 1, 1, 1)
            hand_videos_latents = hand_videos_latents.to(device=device, dtype=dtype)
            hand_videos_latents = rearrange(hand_videos_latents, "b c f h w -> b f c h w")

        # Prepare outputs
        outputs = (latents,)
        
        if return_noise:
            outputs += (noise,)
            
        # Always return both static and hand video latents (can be None)
        outputs += (static_videos_latents, hand_videos_latents)
            
        return outputs

    def preprocess_hand_conditions(
        self,
        hand_videos,
        height,
        width,
        num_frames,
    ):
        """Preprocess hand condition videos.
        
        Args:
            hand_videos: Input video tensor of shape [B, C, F, H, W]
            height: Target height
            width: Target width  
            num_frames: Target number of frames
            
        Returns:
            Preprocessed video tensor of shape [B, C, F, H, W]
        """
        # Convert to torch tensor if needed
        if isinstance(hand_videos, np.ndarray):
            hand_videos = torch.from_numpy(hand_videos)
        
        # Ensure input is [B, C, F, H, W]
        if hand_videos.ndim == 4:  # [C, F, H, W] -> [1, C, F, H, W]
            hand_videos = hand_videos.unsqueeze(0)
        elif hand_videos.ndim == 5:  # Already [B, C, F, H, W]
            pass
        else:
            raise ValueError(f"Expected 4D or 5D tensor, got {hand_videos.ndim}D tensor")
        
        # Input is already [B, C, F, H, W], no need to permute
        
        # Ensure correct number of frames
        if hand_videos.shape[2] != num_frames:
            hand_videos = F.interpolate(
                hand_videos,  # [B, C, F, H, W]
                size=(num_frames, height, width),
                mode='trilinear',
                align_corners=False
            )
        
        return hand_videos

    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt=None,
        callback_on_step_end_tensor_inputs=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        """Check inputs for the static-to-video pipeline."""
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not isinstance(callback_on_step_end_tensor_inputs, list):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be a list but is {type(callback_on_step_end_tensor_inputs)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " either forward `prompt` or `prompt_embeds`."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to either forward `negative_prompt` or"
                " `negative_prompt_embeds`."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    def preprocess_static_conditions(
        self,
        static_videos,  
        height,
        width,
        num_frames,
    ):
        """Preprocess static condition videos.
        
        Args:
            static_videos: Input video tensor of shape [B, C, F, H, W]
            height: Target height
            width: Target width  
            num_frames: Target number of frames
            
        Returns:
            Preprocessed video tensor of shape [B, C, F, H, W]
        """
        # Convert to torch tensor if needed
        if isinstance(static_videos, np.ndarray):
            static_videos = torch.from_numpy(static_videos)
        
        # Ensure input is [B, C, F, H, W]
        if static_videos.ndim == 4:  # [C, F, H, W] -> [1, C, F, H, W]
            static_videos = static_videos.unsqueeze(0)
        elif static_videos.ndim == 5:  # Already [B, C, F, H, W]
            pass
        else:
            raise ValueError(f"Expected 4D or 5D tensor, got {static_videos.ndim}D tensor")
        
        # Input is already [B, C, F, H, W], no need to permute
        
        # Ensure correct number of frames
        if static_videos.shape[2] != num_frames:
            static_videos = F.interpolate(
                static_videos,  # [B, C, F, H, W]
                size=(num_frames, height, width),
                mode='trilinear',
                align_corners=False
            )
        
        return static_videos

    def _preprocess_images(self, images, height, width):
        """Preprocess images to the required format."""
        if isinstance(images, torch.Tensor):
            images = images.cpu().numpy()
        
        if images.dtype == np.uint8:
            images = images.astype(np.float32) / 255.0
            
        if images.ndim == 3:
            images = [images]
            
        images = imcrop_center(images, height, width)
        images = self.video_processor.preprocess(images, height, width)
        return images

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 480,
        width: int = 720,
        video: Union[torch.FloatTensor] = None,
        mask_video: Union[torch.FloatTensor] = None,
        masked_video_latents: Union[torch.FloatTensor] = None,
        static_videos: Optional[Union[torch.FloatTensor, np.ndarray, List[PIL.Image.Image]]] = None,
        hand_videos: Optional[Union[torch.FloatTensor, np.ndarray, List[PIL.Image.Image]]] = None,
        num_frames: int = 49,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6,
        use_dynamic_cfg: bool = False,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "np",
        return_dict: bool = False,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 226,
        strength: float = 1,
        noise_aug_strength: float = 0.0563,
        comfyui_progressbar: bool = False,
    ) -> Union[CogVideoXFunPipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_frames (`int`, defaults to `48`):
                Number of frames to generate. Must be divisible by self.vae_scale_factor_temporal. Generated video will
                contain 1 extra frame because CogVideoX_Fun is conditioned with (num_seconds * fps + 1) frames where
                num_seconds is 6 and fps is 4. However, since videos can be saved at any fps, the only condition that
                needs to be satisfied is that of divisibility mentioned above.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of videos to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
                of a plain tuple.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int`, defaults to `226`):
                Maximum sequence length in encoded prompt. Must be consistent with
                `self.transformer.config.max_text_seq_length` otherwise may lead to poor results.

        Examples:

        Returns:
            [`~pipelines.cogvideo.pipeline_cogvideox.CogVideoXFunPipelineOutput`] or `tuple`:
            [`~pipelines.cogvideo.pipeline_cogvideox.CogVideoXFunPipelineOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is a list with the generated images.
        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        height = height or self.transformer.config.sample_height * self.vae_scale_factor_spatial
        width = width or self.transformer.config.sample_width * self.vae_scale_factor_spatial
        num_frames = num_frames or self.transformer.config.sample_frames

        num_videos_per_prompt = 1

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt=prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        # 2. Default call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            negative_prompt,
            do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        # 4. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(
            num_inference_steps=num_inference_steps, strength=strength, device=device
        )
        self._num_timesteps = len(timesteps)
        if comfyui_progressbar:
            try:
                from comfy.utils import ProgressBar
                pbar = ProgressBar(num_inference_steps + 2)
            except ImportError:
                # Fallback if comfy.utils is not available
                pbar = None
        # at which timestep to set the initial noise (n.b. 50% if strength is 0.5)
        latent_timestep = timesteps[:1].repeat(batch_size * num_videos_per_prompt)
        # create a boolean to check if the strength is set to 1. if so then initialise the latents with pure noise
        is_strength_max = strength == 1.0

        # 5. Prepare latents.
        if static_videos is not None:
            # Use static_videos as the condition video
            static_videos = self.preprocess_static_conditions(static_videos, height, width, num_frames)
            video_length = static_videos.shape[2]
            init_video = self.image_processor.preprocess(rearrange(static_videos, "b c f h w -> (b f) c h w"), height=height, width=width) 
            init_video = init_video.to(dtype=torch.float32)
            init_video = rearrange(init_video, "(b f) c h w -> b c f h w", f=video_length)
        elif video is not None:
            video_length = video.shape[2]
            init_video = self.image_processor.preprocess(rearrange(video, "b c f h w -> (b f) c h w"), height=height, width=width) 
            init_video = init_video.to(dtype=torch.float32)
            init_video = rearrange(init_video, "(b f) c h w -> b c f h w", f=video_length)
        else:
            init_video = None

        # Magvae needs the number of frames to be 4n + 1.
        local_latent_length = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        # For CogVideoX 1.5, the latent frames should be clipped to make it divisible by patch_size_t
        patch_size_t = self.transformer.config.patch_size_t
        additional_frames = 0
        if patch_size_t is not None and local_latent_length % patch_size_t != 0:
            additional_frames = local_latent_length % patch_size_t
            num_frames -= additional_frames * self.vae_scale_factor_temporal
        if num_frames <= 0:
            num_frames = 1
        if video_length > num_frames:
            logger.warning("The length of condition video is not right, the latent frames should be clipped to make it divisible by patch_size_t. ")
            video_length = num_frames
            video = video[:, :, :video_length]
            init_video = init_video[:, :, :video_length]
            mask_video = mask_video[:, :, :video_length]

        # Get channel information for later use
        num_channels_latents = self.vae.config.latent_channels
        num_channels_transformer = self.transformer.config.in_channels

        # Preprocess hand_videos if provided
        processed_hand_videos = None
        if hand_videos is not None:
            processed_hand_videos = self.preprocess_hand_conditions(hand_videos, height, width, num_frames)

        latents_outputs = self.prepare_latents(
            batch_size=batch_size * num_videos_per_prompt,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            num_frames=local_latent_length,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
            latents=latents,
            static_videos=init_video,
            hand_videos=processed_hand_videos,
            timestep=latent_timestep,
            is_strength_max=is_strength_max,
            return_noise=True,
        )
        # Always get both static and hand video latents
        latents, noise, static_videos_latents, hand_videos_latents = latents_outputs
        if comfyui_progressbar and pbar is not None:
            pbar.update(1)
        
        # Use hand_videos_latents from prepare_latents if available
        control_latents = None
        if hand_videos_latents is not None:
            control_latents = hand_videos_latents
            control_latents = (
                torch.cat([control_latents] * 2) if do_classifier_free_guidance else control_latents
            )
        if mask_video is not None:
            if (mask_video == 255).all():  # redraw all frames
                mask_latents = torch.zeros_like(latents)[:, :, :1].to(latents.device, latents.dtype)
                masked_video_latents = torch.zeros_like(latents).to(latents.device, latents.dtype)

                mask_input = torch.cat([mask_latents] * 2) if do_classifier_free_guidance else mask_latents
                masked_video_latents_input = (
                    torch.cat([masked_video_latents] * 2) if do_classifier_free_guidance else masked_video_latents
                )
                inpaint_latents = torch.cat([mask_input, masked_video_latents_input], dim=2).to(latents.dtype)
            else:
                # Prepare mask latent variables
                mask_condition = self.mask_processor.preprocess(rearrange(mask_video, "b c f h w -> (b f) h w c"), height=height, width=width) 
                mask_condition = mask_condition.to(dtype=torch.float32)
                mask_condition = rearrange(mask_condition, "(b f) c h w -> b c f h w", f=video_length)

                if num_channels_transformer != num_channels_latents:
                    mask_condition_tile = torch.tile(mask_condition, [1, 3, 1, 1, 1])
                    if masked_video_latents is None:
                        masked_video = init_video * (mask_condition_tile < 0.5) + torch.ones_like(init_video) * (mask_condition_tile > 0.5) * -1
                    else:
                        masked_video = masked_video_latents

                    _, masked_video_latents = self.prepare_mask_latents(
                        None,
                        masked_video,
                        batch_size,
                        height,
                        width,
                        prompt_embeds.dtype,
                        device,
                        generator,
                        do_classifier_free_guidance,
                        noise_aug_strength=noise_aug_strength,
                    )
                    mask_latents = resize_mask(1 - mask_condition, masked_video_latents)
                    mask_latents = mask_latents.to(masked_video_latents.device) * self.vae.config.scaling_factor

                    mask = torch.tile(mask_condition, [1, num_channels_latents, 1, 1, 1])
                    mask = F.interpolate(mask, size=latents.size()[-3:], mode='trilinear', align_corners=True).to(latents.device, latents.dtype)
                    
                    mask_input = torch.cat([mask_latents] * 2) if do_classifier_free_guidance else mask_latents
                    masked_video_latents_input = (
                        torch.cat([masked_video_latents] * 2) if do_classifier_free_guidance else masked_video_latents
                    )

                    mask = rearrange(mask, "b c f h w -> b f c h w")
                    mask_input = rearrange(mask_input, "b c f h w -> b f c h w")
                    masked_video_latents_input = rearrange(masked_video_latents_input, "b c f h w -> b f c h w")

                    inpaint_latents = torch.cat([mask_input, masked_video_latents_input], dim=2).to(latents.dtype)
                else:
                    mask = torch.tile(mask_condition, [1, num_channels_latents, 1, 1, 1])
                    mask = F.interpolate(mask, size=latents.size()[-3:], mode='trilinear', align_corners=True).to(latents.device, latents.dtype)
                    mask = rearrange(mask, "b c f h w -> b f c h w")
                    
                    inpaint_latents = None
        else:
            if num_channels_transformer != num_channels_latents:
                # Create 1-channel mask
                mask = torch.zeros_like(latents[:, :, :1]).to(latents.device, latents.dtype)  # [B, 1, F, H, W]
                
                if static_videos_latents is not None:
                    # Use static_videos_latents as masked_video_latents
                    masked_video_latents = static_videos_latents.to(latents.device, latents.dtype)
                else:
                    # No static videos, use zeros
                    masked_video_latents = torch.zeros_like(latents).to(latents.device, latents.dtype)
                masked_video_latents = rearrange(masked_video_latents, "b f c h w -> b c f h w")

                # Create mask_latents using resize_mask function (similar to mask_video is not None case)
                mask_condition = self.mask_processor.preprocess(rearrange(mask, "b f c h w -> (b f) c h w"), height=height, width=width) 
                mask_condition = mask_condition.to(dtype=torch.float32)
                mask_condition = rearrange(mask_condition, "(b f) c h w -> b c f h w", f=local_latent_length)
                mask_latents = resize_mask(1 - mask_condition, masked_video_latents)
                mask_latents = mask_latents.to(masked_video_latents.device) * self.vae.config.scaling_factor

                # Handle classifier-free guidance
                if do_classifier_free_guidance:
                    mask_input = torch.cat([mask_latents, mask_latents], dim=0)  # [2B, 1, F, H, W]
                    masked_video_latents_input = torch.cat([masked_video_latents, masked_video_latents], dim=0)  # [2B, C, F, H, W]
                else:
                    mask_input = mask_latents
                    masked_video_latents_input = masked_video_latents
                
                # Concatenate mask (1 channel) + masked_video_latents (C channels) = [B, C+1, F, H, W]
                mask_input = rearrange(mask_input, "b c f h w -> b f c h w")
                masked_video_latents_input = rearrange(masked_video_latents_input, "b c f h w -> b f c h w")
                inpaint_latents = torch.cat([mask_input, masked_video_latents_input], dim=2).to(latents.dtype)
            else:
                # Standard case without additional channels
                mask = torch.zeros_like(init_video[:, :1])
                mask = torch.tile(mask, [1, num_channels_latents, 1, 1, 1])
                mask = F.interpolate(mask, size=latents.size()[-3:], mode='trilinear', align_corners=True).to(latents.device, latents.dtype)
                mask = rearrange(mask, "b c f h w -> b f c h w")

                inpaint_latents = None
        if comfyui_progressbar and pbar is not None:
            pbar.update(1)

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Create rotary embeds if required
        image_rotary_emb = (
            self._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
            if self.transformer.config.use_rotary_positional_embeddings
            else None
        )
        
        # Initialize adapter_control_for_transformer (used by all pipelines)
        if not hasattr(self, 'add_control_adapter') or not self.add_control_adapter:
            adapter_control_for_transformer = None

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            # for DPM-solver++
            old_pred_original_sample = None
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                # predict noise model_output
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    image_rotary_emb=image_rotary_emb,
                    return_dict=False,
                    inpaint_latents=inpaint_latents,
                    control_latents=control_latents,
                    adapter_control=adapter_control_for_transformer,
                )[0]
                noise_pred = noise_pred.float()

                # perform guidance
                if use_dynamic_cfg:
                    self._guidance_scale = 1 + guidance_scale * (
                        (1 - math.cos(math.pi * ((num_inference_steps - t.item()) / num_inference_steps) ** 5.0)) / 2
                    )
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                if not isinstance(self.scheduler, CogVideoXDPMScheduler):
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                else:
                    latents, old_pred_original_sample = self.scheduler.step(
                        noise_pred,
                        old_pred_original_sample,
                        t,
                        timesteps[i - 1] if i > 0 else None,
                        latents,
                        **extra_step_kwargs,
                        return_dict=False,
                    )
                latents = latents.to(prompt_embeds.dtype)

                # call the callback, if provided
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                if comfyui_progressbar:
                    pbar.update(1)

        if not output_type == "latent":
            video = self.decode_latents(latents)
            video = rearrange(video, "b c f h w -> b f h w c")
            # video = self.video_processor.postprocess_video(video=video, output_type=output_type)
            # video = rearrange(video, "b f h w c -> b c f h w")
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        # if not return_dict:
        #     if isinstance(video, np.ndarray):
        #         video = torch.from_numpy(video)
        return CogVideoXFunPipelineOutput(frames=video)

    @classmethod
    def save_lora_weights(
        cls,
        save_directory: str,
        transformer_lora_layers: Optional[Dict[str, torch.Tensor]] = None,
        is_main_process: bool = True,
        weight_name: str = "pytorch_lora_weights.safetensors",
        save_function: Optional[Callable] = None,
        safe_serialization: bool = True,
    ):
        """Save LoRA weights to a directory."""
        if transformer_lora_layers is not None:
            # Save LoRA weights
            os.makedirs(save_directory, exist_ok=True)
            
            if save_function is not None:
                save_function(transformer_lora_layers, os.path.join(save_directory, weight_name))
            elif safe_serialization:
                from safetensors.torch import save_file
                save_file(transformer_lora_layers, os.path.join(save_directory, weight_name))
            else:
                torch.save(transformer_lora_layers, os.path.join(save_directory, weight_name))

    @classmethod
    def lora_state_dict(cls, input_dir: str) -> Dict[str, torch.Tensor]:
        """Load LoRA state dict from a directory."""
        lora_state_dict = {}
        
        # Try to load from safetensors first
        safetensors_path = os.path.join(input_dir, "pytorch_lora_weights.safetensors")
        if os.path.exists(safetensors_path):
            from safetensors.torch import load_file
            lora_state_dict = load_file(safetensors_path)
        else:
            # Try to load from pytorch format
            pytorch_path = os.path.join(input_dir, "pytorch_lora_weights.bin")
            if os.path.exists(pytorch_path):
                lora_state_dict = torch.load(pytorch_path, map_location="cpu")
        
        return lora_state_dict

class CogVideoXI2VStaticTokenPoseAdapterPipeline(CogVideoXImageToVideoPipeline):
    r"""
    Pipeline for text-to-video generation using CogVideoX with static and hand pose conditions.

    This model inherits from [`CogVideoXFunInpaintPipeline`]. It extends the base VideoX-Fun pipeline
    to support static video and hand pose conditioning.

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
        text_encoder ([`T5EncoderModel`]):
            Frozen text-encoder. CogVideoX_Fun uses
            [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel); specifically the
            [t5-v1_1-xxl](https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/t5-v1_1-xxl) variant.
        tokenizer (`T5Tokenizer`):
            Tokenizer of class
            [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer).
        transformer ([`CogVideoXTransformer3DModel`]):
            A text conditioned `CogVideoXTransformer3DModel` to denoise the encoded video latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded video latents.
    """

    def __init__(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        vae: AutoencoderKLCogVideoX,
        transformer: CogVideoXTransformer3DModel,
        scheduler: Union[CogVideoXDDIMScheduler, CogVideoXDPMScheduler],
    ):
        super().__init__(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path=None, 
                        base_model_name_or_path="THUDM/CogVideoX-5b-I2V", 
                        transformer=None, 
                        condition_channels=None, 
                        *args, **kwargs):
        """
        Load a CogVideoXFunStaticToVideoPipeline from a saved directory or base model.
        
        This method loads all components (tokenizer, text_encoder, vae, transformer, scheduler)
        from the specified directory, or uses provided components if specified.
        
        Args:
            pretrained_model_name_or_path: Path to saved pipeline or base model
            base_model_name_or_path: Optional base model path for creating pose-conditioned pipeline
            transformer: Optional transformer to use (for validation with trained transformer)
            condition_channels: Number of condition channels (0 or None for base model, >0 for concat model)
            use_adapter: Whether to use adapter-based transformer
            adapter_version: Version of adapter to use ("v1" or "v2")
        """
        # Check if this is a base model path (for creating pose-conditioned pipeline)
        if base_model_name_or_path is not None:
            print(f"🔧 Creating CogVideoX pipeline from base model: {base_model_name_or_path}")
            # Load the original CogVideoX pipeline from base model
            original_pipeline = super().from_pretrained(base_model_name_or_path, *args, **kwargs)
            
        if transformer is None:
            # Determine condition_channels
            if condition_channels is None:
                condition_channels = 0  # Default to base model
            
            # Create or load appropriate transformer based on condition_channels
            if condition_channels > 0:
                print(f"🔧 Creating/loading CogVideoX cross transformer with {condition_channels} condition channels ")
                transformer = CogVideoXTransformer3DModelWithAdapter.from_pretrained(
                    pretrained_model_name_or_path=pretrained_model_name_or_path,
                    base_model_name_or_path=base_model_name_or_path,
                    subfolder="transformer",
                    condition_channels=condition_channels,
                    torch_dtype=kwargs.get("torch_dtype", torch.bfloat16),
                    revision=kwargs.get("revision", None),
                    variant=kwargs.get("variant", None),
                )
            else:
                print(f"🔧 Loading base CogVideoX transformer")
                transformer = CogVideoXTransformer3DModel.from_pretrained(
                    pretrained_model_name_or_path=pretrained_model_name_or_path,
                    base_model_name_or_path=base_model_name_or_path,
                    subfolder="transformer",
                    torch_dtype=kwargs.get("torch_dtype", torch.bfloat16),
                    revision=kwargs.get("revision", None),
                    variant=kwargs.get("variant", None),
                )
        elif pretrained_model_name_or_path is not None:
            load_dtype = torch.bfloat16 if "5b" in base_model_name_or_path.lower() else torch.float16
            # Override with provided torch_dtype if specified
            if "torch_dtype" in kwargs:
                load_dtype = kwargs["torch_dtype"]
            print(f"🔧 Loading CogVideoX pipeline from {pretrained_model_name_or_path}")
            transformer = CogVideoXTransformer3DModel.from_pretrained(
                pretrained_model_name_or_path,
                subfolder="transformer",
                torch_dtype=load_dtype,
                revision=kwargs.get("revision", None),
                variant=kwargs.get("variant", None),
            )

        # Create our custom pipeline with the same components
        pipeline = cls(
            tokenizer=original_pipeline.tokenizer,
            text_encoder=original_pipeline.text_encoder,
            vae=original_pipeline.vae,
            transformer=transformer,
            scheduler=original_pipeline.scheduler,
            split_hands=split_hands,
        )
        
        return pipeline

    def save_pretrained(self, save_directory: str, **kwargs):
        """
        Save the pipeline to a directory.
        
        This saves all components (tokenizer, text_encoder, vae, transformer, scheduler)
        to the specified directory.
        """
        os.makedirs(save_directory, exist_ok=True)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(os.path.join(save_directory, "tokenizer"))
        
        # Save text encoder
        self.text_encoder.save_pretrained(os.path.join(save_directory, "text_encoder"))
        
        # Save VAE
        self.vae.save_pretrained(os.path.join(save_directory, "vae"))
        
        # Save transformer
        self.transformer.save_pretrained(os.path.join(save_directory, "transformer"))
        
        # Save scheduler
        self.scheduler.save_pretrained(os.path.join(save_directory, "scheduler"))
        
        # Save pipeline config
        pipeline_config = {
            "pipeline_class": "CogVideoXFunStaticToVideoPoseTokenPipeline",
            "version": "1.0.0",
        }
        with open(os.path.join(save_directory, "pipeline_config.json"), "w") as f:
            import json
            json.dump(pipeline_config, f)


    def prepare_latents(
        self,
        batch_size: int = 1,
        num_channels_latents: int = 16,
        height: int = 60,
        width: int = 90,
        num_frames: int = 13,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,  # For I2V fallback
        static_videos: Optional[torch.Tensor] = None,
        hand_videos: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        is_strength_max: bool = True,
        return_noise: bool = False,
    ):
        """Prepare latents for the pipeline with static conditions."""
        # Calculate latent dimensions
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial
        num_channels_latents = self.vae.config.latent_channels

        shape = (batch_size, num_frames, num_channels_latents, latent_height, latent_width)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            # if strength is 1. then initialise the latents to noise, else initial to image + noise
            if static_videos is not None and not is_strength_max:
                # Use static_videos as the reference for strength-based initialization
                static_videos = static_videos.to(device=device, dtype=self.vae.dtype)
                bs = 1
                new_static_videos = []
                for i in range(0, static_videos.shape[0], bs):
                    static_videos_bs = static_videos[i : i + bs]
                    static_videos_bs = self.vae.encode(static_videos_bs)[0]
                    static_videos_bs = static_videos_bs.sample()
                    new_static_videos.append(static_videos_bs)
                video_latents = torch.cat(new_static_videos, dim=0)
                video_latents = video_latents * self.vae.config.scaling_factor
                video_latents = video_latents.repeat(batch_size // video_latents.shape[0], 1, 1, 1, 1)
                video_latents = video_latents.to(device=device, dtype=dtype)
                video_latents = rearrange(video_latents, "b c f h w -> b f c h w")
                latents = self.scheduler.add_noise(video_latents, noise, timestep)
            else:
                latents = noise
            # if pure noise then scale the initial latents by the Scheduler's init sigma
            latents = latents * self.scheduler.init_noise_sigma if is_strength_max else latents
        else:
            noise = latents.to(device)
            latents = noise * self.scheduler.init_noise_sigma

        # Prepare condition latents
        static_videos_latents = None
        hand_videos_latents = None
        smpl_pos_map_latents = None
        
        if static_videos is not None:
            # Process static videos with VAE encoding
            static_videos = static_videos.to(device=device, dtype=self.vae.dtype)
            bs = 1
            new_static_videos = []
            for i in range(0, static_videos.shape[0], bs):
                static_videos_bs = static_videos[i : i + bs]
                static_videos_bs = self.vae.encode(static_videos_bs)[0]
                static_videos_bs = static_videos_bs.sample()
                new_static_videos.append(static_videos_bs)
            static_videos_latents = torch.cat(new_static_videos, dim=0)
            static_videos_latents = static_videos_latents * self.vae.config.scaling_factor
            static_videos_latents = static_videos_latents.repeat(batch_size // static_videos_latents.shape[0], 1, 1, 1, 1)
            static_videos_latents = static_videos_latents.to(device=device, dtype=dtype)
            static_videos_latents = rearrange(static_videos_latents, "b c f h w -> b f c h w")

        if hand_videos is not None:
            # Process hand videos with VAE encoding
            hand_videos = hand_videos.to(device=device, dtype=self.vae.dtype)
            bs = 1
            new_hand_videos = []
            for i in range(0, hand_videos.shape[0], bs):
                hand_videos_bs = hand_videos[i : i + bs]
                hand_videos_bs = self.vae.encode(hand_videos_bs)[0]
                hand_videos_bs = hand_videos_bs.sample()
                new_hand_videos.append(hand_videos_bs)
            hand_videos_latents = torch.cat(new_hand_videos, dim=0)
            hand_videos_latents = hand_videos_latents * self.vae.config.scaling_factor
            hand_videos_latents = hand_videos_latents.repeat(batch_size // hand_videos_latents.shape[0], 1, 1, 1, 1)
            hand_videos_latents = hand_videos_latents.to(device=device, dtype=dtype)
            hand_videos_latents = rearrange(hand_videos_latents, "b c f h w -> b f c h w")

        # Prepare outputs
        outputs = (latents,)
        
        if return_noise:
            outputs += (noise,)
            
        # Always return both static and hand video latents (can be None)
        outputs += (static_videos_latents, hand_videos_latents)
            
        return outputs

    def preprocess_hand_conditions(
        self,
        hand_videos,
        height,
        width,
        num_frames,
    ):
        """Preprocess hand condition videos.
        
        Args:
            hand_videos: Input video tensor of shape [B, C, F, H, W]
            height: Target height
            width: Target width  
            num_frames: Target number of frames
            
        Returns:
            Preprocessed video tensor of shape [B, C, F, H, W]
        """
        # Convert to torch tensor if needed
        if isinstance(hand_videos, np.ndarray):
            hand_videos = torch.from_numpy(hand_videos)
        
        # Ensure input is [B, C, F, H, W]
        if hand_videos.ndim == 4:  # [C, F, H, W] -> [1, C, F, H, W]
            hand_videos = hand_videos.unsqueeze(0)
        elif hand_videos.ndim == 5:  # Already [B, C, F, H, W]
            pass
        else:
            raise ValueError(f"Expected 4D or 5D tensor, got {hand_videos.ndim}D tensor")
        
        # Input is already [B, C, F, H, W], no need to permute
        
        # Ensure correct number of frames
        if hand_videos.shape[2] != num_frames:
            hand_videos = F.interpolate(
                hand_videos,  # [B, C, F, H, W]
                size=(num_frames, height, width),
                mode='trilinear',
                align_corners=False
            )
        
        return hand_videos

    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt=None,
        callback_on_step_end_tensor_inputs=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        """Check inputs for the static-to-video pipeline."""
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not isinstance(callback_on_step_end_tensor_inputs, list):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be a list but is {type(callback_on_step_end_tensor_inputs)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " either forward `prompt` or `prompt_embeds`."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to either forward `negative_prompt` or"
                " `negative_prompt_embeds`."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    def preprocess_static_conditions(
        self,
        static_videos,  
        height,
        width,
        num_frames,
    ):
        """Preprocess static condition videos.
        
        Args:
            static_videos: Input video tensor of shape [B, C, F, H, W]
            height: Target height
            width: Target width  
            num_frames: Target number of frames
            
        Returns:
            Preprocessed video tensor of shape [B, C, F, H, W]
        """
        # Convert to torch tensor if needed
        if isinstance(static_videos, np.ndarray):
            static_videos = torch.from_numpy(static_videos)
        
        # Ensure input is [B, C, F, H, W]
        if static_videos.ndim == 4:  # [C, F, H, W] -> [1, C, F, H, W]
            static_videos = static_videos.unsqueeze(0)
        elif static_videos.ndim == 5:  # Already [B, C, F, H, W]
            pass
        else:
            raise ValueError(f"Expected 4D or 5D tensor, got {static_videos.ndim}D tensor")
        
        # Input is already [B, C, F, H, W], no need to permute
        
        # Ensure correct number of frames
        if static_videos.shape[2] != num_frames:
            static_videos = F.interpolate(
                static_videos,  # [B, C, F, H, W]
                size=(num_frames, height, width),
                mode='trilinear',
                align_corners=False
            )
        
        return static_videos

    def _preprocess_images(self, images, height, width):
        """Preprocess images to the required format."""
        if isinstance(images, torch.Tensor):
            images = images.cpu().numpy()
        
        if images.dtype == np.uint8:
            images = images.astype(np.float32) / 255.0
            
        if images.ndim == 3:
            images = [images]
            
        images = imcrop_center(images, height, width)
        images = self.video_processor.preprocess(images, height, width)
        return images

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 480,
        width: int = 720,
        video: Union[torch.FloatTensor] = None,
        mask_video: Union[torch.FloatTensor] = None,
        masked_video_latents: Union[torch.FloatTensor] = None,
        static_videos: Optional[Union[torch.FloatTensor, np.ndarray, List[PIL.Image.Image]]] = None,
        hand_videos: Optional[Union[torch.FloatTensor, np.ndarray, List[PIL.Image.Image]]] = None,
        num_frames: int = 49,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6,
        use_dynamic_cfg: bool = False,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "np",
        return_dict: bool = False,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 226,
        strength: float = 1,
        noise_aug_strength: float = 0.0563,
        comfyui_progressbar: bool = False,
    ) -> Union[CogVideoXFunPipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_frames (`int`, defaults to `48`):
                Number of frames to generate. Must be divisible by self.vae_scale_factor_temporal. Generated video will
                contain 1 extra frame because CogVideoX_Fun is conditioned with (num_seconds * fps + 1) frames where
                num_seconds is 6 and fps is 4. However, since videos can be saved at any fps, the only condition that
                needs to be satisfied is that of divisibility mentioned above.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of videos to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
                of a plain tuple.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int`, defaults to `226`):
                Maximum sequence length in encoded prompt. Must be consistent with
                `self.transformer.config.max_text_seq_length` otherwise may lead to poor results.

        Examples:

        Returns:
            [`~pipelines.cogvideo.pipeline_cogvideox.CogVideoXFunPipelineOutput`] or `tuple`:
            [`~pipelines.cogvideo.pipeline_cogvideox.CogVideoXFunPipelineOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is a list with the generated images.
        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        height = height or self.transformer.config.sample_height * self.vae_scale_factor_spatial
        width = width or self.transformer.config.sample_width * self.vae_scale_factor_spatial
        num_frames = num_frames or self.transformer.config.sample_frames

        num_videos_per_prompt = 1

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt=prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        # 2. Default call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            negative_prompt,
            do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        # 4. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(
            num_inference_steps=num_inference_steps, strength=strength, device=device
        )
        self._num_timesteps = len(timesteps)
        if comfyui_progressbar:
            try:
                from comfy.utils import ProgressBar
                pbar = ProgressBar(num_inference_steps + 2)
            except ImportError:
                # Fallback if comfy.utils is not available
                pbar = None
        # at which timestep to set the initial noise (n.b. 50% if strength is 0.5)
        latent_timestep = timesteps[:1].repeat(batch_size * num_videos_per_prompt)
        # create a boolean to check if the strength is set to 1. if so then initialise the latents with pure noise
        is_strength_max = strength == 1.0

        # 5. Prepare latents.
        if static_videos is not None:
            # Use static_videos as the condition video
            static_videos = self.preprocess_static_conditions(static_videos, height, width, num_frames)
            video_length = static_videos.shape[2]
            init_video = self.image_processor.preprocess(rearrange(static_videos, "b c f h w -> (b f) c h w"), height=height, width=width) 
            init_video = init_video.to(dtype=torch.float32)
            init_video = rearrange(init_video, "(b f) c h w -> b c f h w", f=video_length)
        elif video is not None:
            video_length = video.shape[2]
            init_video = self.image_processor.preprocess(rearrange(video, "b c f h w -> (b f) c h w"), height=height, width=width) 
            init_video = init_video.to(dtype=torch.float32)
            init_video = rearrange(init_video, "(b f) c h w -> b c f h w", f=video_length)
        else:
            init_video = None

        # Magvae needs the number of frames to be 4n + 1.
        local_latent_length = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        # For CogVideoX 1.5, the latent frames should be clipped to make it divisible by patch_size_t
        patch_size_t = self.transformer.config.patch_size_t
        additional_frames = 0
        if patch_size_t is not None and local_latent_length % patch_size_t != 0:
            additional_frames = local_latent_length % patch_size_t
            num_frames -= additional_frames * self.vae_scale_factor_temporal
        if num_frames <= 0:
            num_frames = 1
        if video_length > num_frames:
            logger.warning("The length of condition video is not right, the latent frames should be clipped to make it divisible by patch_size_t. ")
            video_length = num_frames
            video = video[:, :, :video_length]
            init_video = init_video[:, :, :video_length]
            mask_video = mask_video[:, :, :video_length]

        # Get channel information for later use
        num_channels_latents = self.vae.config.latent_channels
        num_channels_transformer = self.transformer.config.in_channels

        # Preprocess hand_videos if provided
        processed_hand_videos = None
        if hand_videos is not None:
            processed_hand_videos = self.preprocess_hand_conditions(hand_videos, height, width, num_frames)

        latents_outputs = self.prepare_latents(
            batch_size=batch_size * num_videos_per_prompt,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            num_frames=local_latent_length,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
            latents=latents,
            static_videos=init_video,
            hand_videos=processed_hand_videos,
            timestep=latent_timestep,
            is_strength_max=is_strength_max,
            return_noise=True,
        )
        # Always get both static and hand video latents
        latents, noise, static_videos_latents, hand_videos_latents = latents_outputs
        if comfyui_progressbar and pbar is not None:
            pbar.update(1)
        
        # Use hand_videos_latents from prepare_latents if available
        control_latents = None
        if hand_videos_latents is not None:
            control_latents = hand_videos_latents
            control_latents = (
                torch.cat([control_latents] * 2) if do_classifier_free_guidance else control_latents
            )
        if mask_video is not None:
            if (mask_video == 255).all():  # redraw all frames
                mask_latents = torch.zeros_like(latents)[:, :, :1].to(latents.device, latents.dtype)
                masked_video_latents = torch.zeros_like(latents).to(latents.device, latents.dtype)

                mask_input = torch.cat([mask_latents] * 2) if do_classifier_free_guidance else mask_latents
                masked_video_latents_input = (
                    torch.cat([masked_video_latents] * 2) if do_classifier_free_guidance else masked_video_latents
                )
                inpaint_latents = torch.cat([mask_input, masked_video_latents_input], dim=2).to(latents.dtype)
            else:
                # Prepare mask latent variables
                mask_condition = self.mask_processor.preprocess(rearrange(mask_video, "b c f h w -> (b f) h w c"), height=height, width=width) 
                mask_condition = mask_condition.to(dtype=torch.float32)
                mask_condition = rearrange(mask_condition, "(b f) c h w -> b c f h w", f=video_length)

                if num_channels_transformer != num_channels_latents:
                    mask_condition_tile = torch.tile(mask_condition, [1, 3, 1, 1, 1])
                    if masked_video_latents is None:
                        masked_video = init_video * (mask_condition_tile < 0.5) + torch.ones_like(init_video) * (mask_condition_tile > 0.5) * -1
                    else:
                        masked_video = masked_video_latents

                    _, masked_video_latents = self.prepare_mask_latents(
                        None,
                        masked_video,
                        batch_size,
                        height,
                        width,
                        prompt_embeds.dtype,
                        device,
                        generator,
                        do_classifier_free_guidance,
                        noise_aug_strength=noise_aug_strength,
                    )
                    mask_latents = resize_mask(1 - mask_condition, masked_video_latents)
                    mask_latents = mask_latents.to(masked_video_latents.device) * self.vae.config.scaling_factor

                    mask = torch.tile(mask_condition, [1, num_channels_latents, 1, 1, 1])
                    mask = F.interpolate(mask, size=latents.size()[-3:], mode='trilinear', align_corners=True).to(latents.device, latents.dtype)
                    
                    mask_input = torch.cat([mask_latents] * 2) if do_classifier_free_guidance else mask_latents
                    masked_video_latents_input = (
                        torch.cat([masked_video_latents] * 2) if do_classifier_free_guidance else masked_video_latents
                    )

                    mask = rearrange(mask, "b c f h w -> b f c h w")
                    mask_input = rearrange(mask_input, "b c f h w -> b f c h w")
                    masked_video_latents_input = rearrange(masked_video_latents_input, "b c f h w -> b f c h w")

                    inpaint_latents = torch.cat([mask_input, masked_video_latents_input], dim=2).to(latents.dtype)
                else:
                    mask = torch.tile(mask_condition, [1, num_channels_latents, 1, 1, 1])
                    mask = F.interpolate(mask, size=latents.size()[-3:], mode='trilinear', align_corners=True).to(latents.device, latents.dtype)
                    mask = rearrange(mask, "b c f h w -> b f c h w")
                    
                    inpaint_latents = None
        else:
            if num_channels_transformer != num_channels_latents:
                # Create 1-channel mask
                mask = torch.zeros_like(latents[:, :, :1]).to(latents.device, latents.dtype)  # [B, 1, F, H, W]
                
                if static_videos_latents is not None:
                    # Use static_videos_latents as masked_video_latents
                    masked_video_latents = static_videos_latents.to(latents.device, latents.dtype)
                else:
                    # No static videos, use zeros
                    masked_video_latents = torch.zeros_like(latents).to(latents.device, latents.dtype)
                masked_video_latents = rearrange(masked_video_latents, "b f c h w -> b c f h w")

                # Create mask_latents using resize_mask function (similar to mask_video is not None case)
                mask_condition = self.mask_processor.preprocess(rearrange(mask, "b f c h w -> (b f) c h w"), height=height, width=width) 
                mask_condition = mask_condition.to(dtype=torch.float32)
                mask_condition = rearrange(mask_condition, "(b f) c h w -> b c f h w", f=local_latent_length)
                mask_latents = resize_mask(1 - mask_condition, masked_video_latents)
                mask_latents = mask_latents.to(masked_video_latents.device) * self.vae.config.scaling_factor

                # Handle classifier-free guidance
                if do_classifier_free_guidance:
                    mask_input = torch.cat([mask_latents, mask_latents], dim=0)  # [2B, 1, F, H, W]
                    masked_video_latents_input = torch.cat([masked_video_latents, masked_video_latents], dim=0)  # [2B, C, F, H, W]
                else:
                    mask_input = mask_latents
                    masked_video_latents_input = masked_video_latents
                
                # Concatenate mask (1 channel) + masked_video_latents (C channels) = [B, C+1, F, H, W]
                mask_input = rearrange(mask_input, "b c f h w -> b f c h w")
                masked_video_latents_input = rearrange(masked_video_latents_input, "b c f h w -> b f c h w")
                inpaint_latents = torch.cat([mask_input, masked_video_latents_input], dim=2).to(latents.dtype)
            else:
                # Standard case without additional channels
                mask = torch.zeros_like(init_video[:, :1])
                mask = torch.tile(mask, [1, num_channels_latents, 1, 1, 1])
                mask = F.interpolate(mask, size=latents.size()[-3:], mode='trilinear', align_corners=True).to(latents.device, latents.dtype)
                mask = rearrange(mask, "b c f h w -> b f c h w")

                inpaint_latents = None
        if comfyui_progressbar and pbar is not None:
            pbar.update(1)

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Create rotary embeds if required
        image_rotary_emb = (
            self._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
            if self.transformer.config.use_rotary_positional_embeddings
            else None
        )
        
        # Initialize adapter_control_for_transformer (used by all pipelines)
        if not hasattr(self, 'add_control_adapter') or not self.add_control_adapter:
            adapter_control_for_transformer = None

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            # for DPM-solver++
            old_pred_original_sample = None
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                # predict noise model_output
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    image_rotary_emb=image_rotary_emb,
                    return_dict=False,
                    inpaint_latents=inpaint_latents,
                    control_latents=control_latents,
                    adapter_control=adapter_control_for_transformer,
                )[0]
                noise_pred = noise_pred.float()

                # perform guidance
                if use_dynamic_cfg:
                    self._guidance_scale = 1 + guidance_scale * (
                        (1 - math.cos(math.pi * ((num_inference_steps - t.item()) / num_inference_steps) ** 5.0)) / 2
                    )
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                if not isinstance(self.scheduler, CogVideoXDPMScheduler):
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                else:
                    latents, old_pred_original_sample = self.scheduler.step(
                        noise_pred,
                        old_pred_original_sample,
                        t,
                        timesteps[i - 1] if i > 0 else None,
                        latents,
                        **extra_step_kwargs,
                        return_dict=False,
                    )
                latents = latents.to(prompt_embeds.dtype)

                # call the callback, if provided
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                if comfyui_progressbar:
                    pbar.update(1)

        if not output_type == "latent":
            video = self.decode_latents(latents)
            video = rearrange(video, "b c f h w -> b f h w c")
            # video = self.video_processor.postprocess_video(video=video, output_type=output_type)
            # video = rearrange(video, "b f h w c -> b c f h w")
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        # if not return_dict:
        #     if isinstance(video, np.ndarray):
        #         video = torch.from_numpy(video)
        return CogVideoXFunPipelineOutput(frames=video)

    @classmethod
    def save_lora_weights(
        cls,
        save_directory: str,
        transformer_lora_layers: Optional[Dict[str, torch.Tensor]] = None,
        is_main_process: bool = True,
        weight_name: str = "pytorch_lora_weights.safetensors",
        save_function: Optional[Callable] = None,
        safe_serialization: bool = True,
    ):
        """Save LoRA weights to a directory."""
        if transformer_lora_layers is not None:
            # Save LoRA weights
            os.makedirs(save_directory, exist_ok=True)
            
            if save_function is not None:
                save_function(transformer_lora_layers, os.path.join(save_directory, weight_name))
            elif safe_serialization:
                from safetensors.torch import save_file
                save_file(transformer_lora_layers, os.path.join(save_directory, weight_name))
            else:
                torch.save(transformer_lora_layers, os.path.join(save_directory, weight_name))

    @classmethod
    def lora_state_dict(cls, input_dir: str) -> Dict[str, torch.Tensor]:
        """Load LoRA state dict from a directory."""
        lora_state_dict = {}
        
        # Try to load from safetensors first
        safetensors_path = os.path.join(input_dir, "pytorch_lora_weights.safetensors")
        if os.path.exists(safetensors_path):
            from safetensors.torch import load_file
            lora_state_dict = load_file(safetensors_path)
        else:
            # Try to load from pytorch format
            pytorch_path = os.path.join(input_dir, "pytorch_lora_weights.bin")
            if os.path.exists(pytorch_path):
                lora_state_dict = torch.load(pytorch_path, map_location="cpu")
        
        return lora_state_dict


class CogVideoXFunStaticToVideoCrossPipeline(CogVideoXFunInpaintPipeline):
    r"""
    Pipeline for text-to-video generation using CogVideoX with static and hand pose conditions.

    This model inherits from [`CogVideoXFunInpaintPipeline`]. It extends the base VideoX-Fun pipeline
    to support static video and hand pose conditioning.

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
        text_encoder ([`T5EncoderModel`]):
            Frozen text-encoder. CogVideoX_Fun uses
            [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel); specifically the
            [t5-v1_1-xxl](https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/t5-v1_1-xxl) variant.
        tokenizer (`T5Tokenizer`):
            Tokenizer of class
            [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer).
        transformer ([`CogVideoXTransformer3DModel`]):
            A text conditioned `CogVideoXTransformer3DModel` to denoise the encoded video latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded video latents.
    """

    def __init__(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        vae: AutoencoderKLCogVideoX,
        transformer: CogVideoXTransformer3DModel,
        scheduler: Union[CogVideoXDDIMScheduler, CogVideoXDPMScheduler],
    ):
        super().__init__(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path=None, 
                        base_model_name_or_path="alibaba-pai/CogVideoX-Fun-V1.1-5b-InP", 
                        transformer=None, 
                        condition_channels=None, 
                        use_adapter=False,
                        adapter_version="v1",
                        is_train_cross: bool = True,
                        cross_attn_interval: int = 2,
                        cross_attn_dim_head: int = 128,
                        cross_attn_num_heads: int = 16,
                        cross_attn_kv_dim: int = None,
                        *args, **kwargs):
        """
        Load a CogVideoXFunStaticToVideoCrossPipeline from a saved directory or base model.
        
        This method loads all components (tokenizer, text_encoder, vae, transformer, scheduler)
        from the specified directory, or uses provided components if specified.
        
        Args:
            pretrained_model_name_or_path: Path to saved pipeline or base model
            base_model_name_or_path: Optional base model path for creating pose-conditioned pipeline
            transformer: Optional transformer to use (for validation with trained transformer)
            condition_channels: Number of condition channels (0 or None for base model, >0 for concat model)
            use_adapter: Whether to use adapter-based transformer
            adapter_version: Version of adapter to use ("v1" or "v2")
            is_train_cross: Whether to train cross attention
            cross_attn_interval: Interval for cross attention
            cross_attn_dim_head: Dimension of head for cross attention
            cross_attn_num_heads: Number of heads for cross attention
            cross_attn_kv_dim: Dimension of key/value for cross attention
        """
        # Check if this is a base model path (for creating pose-conditioned pipeline)
        if base_model_name_or_path is not None:
            print(f"🔧 Loading pipeline components (tokenizer, text_encoder, vae, scheduler) from base model: {base_model_name_or_path}")
            # Load the original CogVideoX-Fun pipeline from base model to get non-transformer components
            original_pipeline = super().from_pretrained(base_model_name_or_path, *args, **kwargs)
            
        if transformer is None:
            # Determine condition_channels
            if condition_channels is None:
                condition_channels = 0  # Default to base model
            
            # Create or load appropriate transformer based on condition_channels
            if condition_channels > 0:
                print(f"🔧 Creating/loading VideoX-Fun cross transformer with {condition_channels} condition channels (adapter_version={adapter_version})")
                transformer = CrossTransformer3DModelWithAdapter.from_pretrained(
                    pretrained_model_name_or_path=pretrained_model_name_or_path,
                    base_model_name_or_path=base_model_name_or_path,
                    subfolder="transformer",
                    condition_channels=condition_channels,
                    adapter_version=adapter_version,
                    is_train_cross=is_train_cross,
                    cross_attn_interval=cross_attn_interval,
                    cross_attn_dim_head=cross_attn_dim_head,
                    cross_attn_num_heads=cross_attn_num_heads,
                    cross_attn_kv_dim=cross_attn_kv_dim,
                    torch_dtype=kwargs.get("torch_dtype", torch.bfloat16),
                    revision=kwargs.get("revision", None),
                    variant=kwargs.get("variant", None),
                )
            else:
                print(f"🔧 Loading base VideoX-Fun transformer")
                transformer = CogVideoXFunTransformer3DModel.from_pretrained(
                    pretrained_model_name_or_path=pretrained_model_name_or_path,
                    base_model_name_or_path=base_model_name_or_path,
                    subfolder="transformer",
                    torch_dtype=kwargs.get("torch_dtype", torch.bfloat16),
                    revision=kwargs.get("revision", None),
                    variant=kwargs.get("variant", None),
                )
        elif pretrained_model_name_or_path is not None:
            load_dtype = torch.bfloat16 if "5b" in base_model_name_or_path.lower() else torch.float16
            # Override with provided torch_dtype if specified
            if "torch_dtype" in kwargs:
                load_dtype = kwargs["torch_dtype"]
            print(f"🔧 Loading VideoX-Fun pipeline from {pretrained_model_name_or_path}")
            transformer = CogVideoXFunTransformer3DModel.from_pretrained(
                pretrained_model_name_or_path,
                subfolder="transformer",
                torch_dtype=load_dtype,
                revision=kwargs.get("revision", None),
                variant=kwargs.get("variant", None),
            )

        # Create our custom pipeline with the same components
        pipeline = cls(
            tokenizer=original_pipeline.tokenizer,
            text_encoder=original_pipeline.text_encoder,
            vae=original_pipeline.vae,
            transformer=transformer,
            scheduler=original_pipeline.scheduler,
            split_hands=split_hands,
        )
        
        return pipeline

    def save_pretrained(self, save_directory: str, **kwargs):
        """
        Save the pipeline to a directory.
        
        This saves all components (tokenizer, text_encoder, vae, transformer, scheduler)
        to the specified directory.
        """
        os.makedirs(save_directory, exist_ok=True)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(os.path.join(save_directory, "tokenizer"))
        
        # Save text encoder
        self.text_encoder.save_pretrained(os.path.join(save_directory, "text_encoder"))
        
        # Save VAE
        self.vae.save_pretrained(os.path.join(save_directory, "vae"))
        
        # Save transformer
        self.transformer.save_pretrained(os.path.join(save_directory, "transformer"))
        
        # Save scheduler
        self.scheduler.save_pretrained(os.path.join(save_directory, "scheduler"))
        
        # Save pipeline config
        pipeline_config = {
            "pipeline_class": "CogVideoXFunStaticToVideoPipeline",
            "version": "1.0.0",
        }
        with open(os.path.join(save_directory, "pipeline_config.json"), "w") as f:
            import json
            json.dump(pipeline_config, f)


    def prepare_latents(
        self,
        batch_size: int = 1,
        num_channels_latents: int = 16,
        height: int = 60,
        width: int = 90,
        num_frames: int = 13,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,  # For I2V fallback
        static_videos: Optional[torch.Tensor] = None,
        hand_videos: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        is_strength_max: bool = True,
        return_noise: bool = False,
    ):
        """Prepare latents for the pipeline with static conditions."""
        # Calculate latent dimensions
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial
        num_channels_latents = self.vae.config.latent_channels

        shape = (batch_size, num_frames, num_channels_latents, latent_height, latent_width)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            # # if strength is 1. then initialise the latents to noise, else initial to image + noise
            # if static_videos is not None and not is_strength_max:
            #     # Use static_videos as the reference for strength-based initialization
            #     static_videos = static_videos.to(device=device, dtype=self.vae.dtype)
            #     bs = 1
            #     new_static_videos = []
            #     for i in range(0, static_videos.shape[0], bs):
            #         static_videos_bs = static_videos[i : i + bs]
            #         static_videos_bs = self.vae.encode(static_videos_bs)[0]
            #         static_videos_bs = static_videos_bs.sample()
            #         new_static_videos.append(static_videos_bs)
            #     video_latents = torch.cat(new_static_videos, dim=0)
            #     video_latents = video_latents * self.vae.config.scaling_factor
            #     video_latents = video_latents.repeat(batch_size // video_latents.shape[0], 1, 1, 1, 1)
            #     video_latents = video_latents.to(device=device, dtype=dtype)
            #     video_latents = rearrange(video_latents, "b c f h w -> b f c h w")
            #     latents = self.scheduler.add_noise(video_latents, noise, timestep)
            # else:
            #     latents = noise
            # # if pure noise then scale the initial latents by the Scheduler's init sigma
            # latents = latents * self.scheduler.init_noise_sigma if is_strength_max else latents
            latents = noise
            latents = latents * self.scheduler.init_noise_sigma if is_strength_max else latents
        else:
            noise = latents.to(device)
            latents = noise * self.scheduler.init_noise_sigma

        # Prepare condition latents
        static_videos_latents = None
        hand_videos_latents = None
        smpl_pos_map_latents = None
        
        if static_videos is not None:
            # Process static videos with VAE encoding
            static_videos = static_videos.to(device=device, dtype=self.vae.dtype)
            bs = 1
            new_static_videos = []
            for i in range(0, static_videos.shape[0], bs):
                static_videos_bs = static_videos[i : i + bs]
                static_videos_bs = self.vae.encode(static_videos_bs)[0]
                static_videos_bs = static_videos_bs.sample()
                new_static_videos.append(static_videos_bs)
            static_videos_latents = torch.cat(new_static_videos, dim=0)
            static_videos_latents = static_videos_latents * self.vae.config.scaling_factor
            static_videos_latents = static_videos_latents.repeat(batch_size // static_videos_latents.shape[0], 1, 1, 1, 1)
            static_videos_latents = static_videos_latents.to(device=device, dtype=dtype)
            static_videos_latents = rearrange(static_videos_latents, "b c f h w -> b f c h w")

        if hand_videos is not None:
            # Process hand videos with VAE encoding
            hand_videos = hand_videos.to(device=device, dtype=self.vae.dtype)
            bs = 1
            new_hand_videos = []
            for i in range(0, hand_videos.shape[0], bs):
                hand_videos_bs = hand_videos[i : i + bs]
                hand_videos_bs = self.vae.encode(hand_videos_bs)[0]
                hand_videos_bs = hand_videos_bs.sample()
                new_hand_videos.append(hand_videos_bs)
            hand_videos_latents = torch.cat(new_hand_videos, dim=0)
            hand_videos_latents = hand_videos_latents * self.vae.config.scaling_factor
            hand_videos_latents = hand_videos_latents.repeat(batch_size // hand_videos_latents.shape[0], 1, 1, 1, 1)
            hand_videos_latents = hand_videos_latents.to(device=device, dtype=dtype)
            hand_videos_latents = rearrange(hand_videos_latents, "b c f h w -> b f c h w")

        # Prepare outputs
        outputs = (latents,)
        
        if return_noise:
            outputs += (noise,)
            
        outputs += (static_videos_latents, hand_videos_latents)
            
        return outputs

    def preprocess_hand_conditions(
        self,
        hand_videos,
        height,
        width,
        num_frames,
    ):
        """Preprocess hand condition videos.
        
        Args:
            hand_videos: Input video tensor of shape [B, C, F, H, W]
            height: Target height
            width: Target width  
            num_frames: Target number of frames
            
        Returns:
            Preprocessed video tensor of shape [B, C, F, H, W]
        """
        # Convert to torch tensor if needed
        if isinstance(hand_videos, np.ndarray):
            hand_videos = torch.from_numpy(hand_videos)
        
        # Ensure input is [B, C, F, H, W]
        if hand_videos.ndim == 4:  # [C, F, H, W] -> [1, C, F, H, W]
            hand_videos = hand_videos.unsqueeze(0)
        elif hand_videos.ndim == 5:  # Already [B, C, F, H, W]
            pass
        else:
            raise ValueError(f"Expected 4D or 5D tensor, got {hand_videos.ndim}D tensor")
        
        # Input is already [B, C, F, H, W], no need to permute
        
        # Ensure correct number of frames
        if hand_videos.shape[2] != num_frames:
            hand_videos = F.interpolate(
                hand_videos,  # [B, C, F, H, W]
                size=(num_frames, height, width),
                mode='trilinear',
                align_corners=False
            )
        
        return hand_videos

    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt=None,
        callback_on_step_end_tensor_inputs=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        """Check inputs for the static-to-video pipeline."""
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not isinstance(callback_on_step_end_tensor_inputs, list):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be a list but is {type(callback_on_step_end_tensor_inputs)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " either forward `prompt` or `prompt_embeds`."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to either forward `negative_prompt` or"
                " `negative_prompt_embeds`."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    def preprocess_static_conditions(
        self,
        static_videos,  
        height,
        width,
        num_frames,
    ):
        """Preprocess static condition videos.
        
        Args:
            static_videos: Input video tensor of shape [B, C, F, H, W]
            height: Target height
            width: Target width  
            num_frames: Target number of frames
            
        Returns:
            Preprocessed video tensor of shape [B, C, F, H, W]
        """
        # Convert to torch tensor if needed
        if isinstance(static_videos, np.ndarray):
            static_videos = torch.from_numpy(static_videos)
        
        # Ensure input is [B, C, F, H, W]
        if static_videos.ndim == 4:  # [C, F, H, W] -> [1, C, F, H, W]
            static_videos = static_videos.unsqueeze(0)
        elif static_videos.ndim == 5:  # Already [B, C, F, H, W]
            pass
        else:
            raise ValueError(f"Expected 4D or 5D tensor, got {static_videos.ndim}D tensor")
        
        # Input is already [B, C, F, H, W], no need to permute
        
        # Ensure correct number of frames
        if static_videos.shape[2] != num_frames:
            static_videos = F.interpolate(
                static_videos,  # [B, C, F, H, W]
                size=(num_frames, height, width),
                mode='trilinear',
                align_corners=False
            )
        
        return static_videos

    def _preprocess_images(self, images, height, width):
        """Preprocess images to the required format."""
        if isinstance(images, torch.Tensor):
            images = images.cpu().numpy()
        
        if images.dtype == np.uint8:
            images = images.astype(np.float32) / 255.0
            
        if images.ndim == 3:
            images = [images]
            
        images = imcrop_center(images, height, width)
        images = self.video_processor.preprocess(images, height, width)
        return images

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 480,
        width: int = 720,
        video: Union[torch.FloatTensor] = None,
        mask_video: Union[torch.FloatTensor] = None,
        masked_video_latents: Union[torch.FloatTensor] = None,
        static_videos: Optional[Union[torch.FloatTensor, np.ndarray, List[PIL.Image.Image]]] = None,
        hand_videos: Optional[Union[torch.FloatTensor, np.ndarray, List[PIL.Image.Image]]] = None,
        num_frames: int = 49,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6,
        use_dynamic_cfg: bool = False,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "np",
        return_dict: bool = False,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 226,
        strength: float = 1,
        noise_aug_strength: float = 0.0563,
        comfyui_progressbar: bool = False,
    ) -> Union[CogVideoXFunPipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_frames (`int`, defaults to `48`):
                Number of frames to generate. Must be divisible by self.vae_scale_factor_temporal. Generated video will
                contain 1 extra frame because CogVideoX_Fun is conditioned with (num_seconds * fps + 1) frames where
                num_seconds is 6 and fps is 4. However, since videos can be saved at any fps, the only condition that
                needs to be satisfied is that of divisibility mentioned above.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of videos to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
                of a plain tuple.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int`, defaults to `226`):
                Maximum sequence length in encoded prompt. Must be consistent with
                `self.transformer.config.max_text_seq_length` otherwise may lead to poor results.

        Examples:

        Returns:
            [`~pipelines.cogvideo.pipeline_cogvideox.CogVideoXFunPipelineOutput`] or `tuple`:
            [`~pipelines.cogvideo.pipeline_cogvideox.CogVideoXFunPipelineOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is a list with the generated images.
        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        height = height or self.transformer.config.sample_height * self.vae_scale_factor_spatial
        width = width or self.transformer.config.sample_width * self.vae_scale_factor_spatial
        num_frames = num_frames or self.transformer.config.sample_frames

        num_videos_per_prompt = 1

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt=prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        # 2. Default call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            negative_prompt,
            do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        # 4. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(
            num_inference_steps=num_inference_steps, strength=strength, device=device
        )
        self._num_timesteps = len(timesteps)
        if comfyui_progressbar:
            try:
                from comfy.utils import ProgressBar
                pbar = ProgressBar(num_inference_steps + 2)
            except ImportError:
                # Fallback if comfy.utils is not available
                pbar = None
        # at which timestep to set the initial noise (n.b. 50% if strength is 0.5)
        latent_timestep = timesteps[:1].repeat(batch_size * num_videos_per_prompt)
        # create a boolean to check if the strength is set to 1. if so then initialise the latents with pure noise
        is_strength_max = strength == 1.0

        # 5. Prepare latents.
        # Restore original CogVideoXFunInpaintPipeline init_video handling using video parameter
        if video is not None:
            video_length = video.shape[2]
            init_video = self.image_processor.preprocess(rearrange(video, "b c f h w -> (b f) c h w"), height=height, width=width) 
            init_video = init_video.to(dtype=torch.float32)
            init_video = rearrange(init_video, "(b f) c h w -> b c f h w", f=video_length)
        else:
            init_video = None

        # Magvae needs the number of frames to be 4n + 1.
        local_latent_length = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        # For CogVideoX 1.5, the latent frames should be clipped to make it divisible by patch_size_t
        patch_size_t = self.transformer.config.patch_size_t
        additional_frames = 0
        if patch_size_t is not None and local_latent_length % patch_size_t != 0:
            additional_frames = local_latent_length % patch_size_t
            num_frames -= additional_frames * self.vae_scale_factor_temporal
        if num_frames <= 0:
            num_frames = 1
        if video is not None and video_length > num_frames:
            logger.warning("The length of condition video is not right, the latent frames should be clipped to make it divisible by patch_size_t. ")
            video_length = num_frames
            video = video[:, :, :video_length]
            init_video = init_video[:, :, :video_length]
            mask_video = mask_video[:, :, :video_length]

        num_channels_latents = self.vae.config.latent_channels
        num_channels_transformer = self.transformer.config.in_channels

        latents_outputs = self.prepare_latents(
            batch_size=batch_size * num_videos_per_prompt,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            num_frames=local_latent_length,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
            latents=latents,
            static_videos=static_videos,
            hand_videos=hand_videos,
            timestep=latent_timestep,
            is_strength_max=is_strength_max,
            return_noise=True,
        )

        latents, noise, static_videos_latents, hand_videos_latents = latents_outputs
        
        if comfyui_progressbar and pbar is not None:
            pbar.update(1)
        
        # Use preprocessed latents from prepare_latents (no need to re-encode)
        control_latents = hand_videos_latents
        ref_latents = static_videos_latents
        
        # Apply classifier-free guidance if needed
        if do_classifier_free_guidance:
            if control_latents is not None:
                control_latents = torch.cat([control_latents] * 2)
            if ref_latents is not None:
                ref_latents = torch.cat([ref_latents] * 2)
        
        if mask_video is not None:
            if (mask_video == 255).all():  # redraw all frames
                mask_latents = torch.zeros_like(latents)[:, :, :1].to(latents.device, latents.dtype)
                masked_video_latents = torch.zeros_like(latents).to(latents.device, latents.dtype)

                mask_input = torch.cat([mask_latents] * 2) if do_classifier_free_guidance else mask_latents
                masked_video_latents_input = (
                    torch.cat([masked_video_latents] * 2) if do_classifier_free_guidance else masked_video_latents
                )
                inpaint_latents = torch.cat([mask_input, masked_video_latents_input], dim=2).to(latents.dtype)
            else:
                # Prepare mask latent variables
                mask_condition = self.mask_processor.preprocess(rearrange(mask_video, "b c f h w -> (b f) h w c"), height=height, width=width) 
                mask_condition = mask_condition.to(dtype=torch.float32)
                mask_condition = rearrange(mask_condition, "(b f) c h w -> b c f h w", f=video_length)

                if num_channels_transformer != num_channels_latents:
                    mask_condition_tile = torch.tile(mask_condition, [1, 3, 1, 1, 1])
                    if masked_video_latents is None:
                        masked_video = init_video * (mask_condition_tile < 0.5) + torch.ones_like(init_video) * (mask_condition_tile > 0.5) * -1
                    else:
                        masked_video = masked_video_latents

                    _, masked_video_latents = self.prepare_mask_latents(
                        None,
                        masked_video,
                        batch_size,
                        height,
                        width,
                        prompt_embeds.dtype,
                        device,
                        generator,
                        do_classifier_free_guidance,
                        noise_aug_strength=noise_aug_strength,
                    )
                    mask_latents = resize_mask(1 - mask_condition, masked_video_latents)
                    mask_latents = mask_latents.to(masked_video_latents.device) * self.vae.config.scaling_factor

                    mask = torch.tile(mask_condition, [1, num_channels_latents, 1, 1, 1])
                    mask = F.interpolate(mask, size=latents.size()[-3:], mode='trilinear', align_corners=True).to(latents.device, latents.dtype)
                    
                    mask_input = torch.cat([mask_latents] * 2) if do_classifier_free_guidance else mask_latents
                    masked_video_latents_input = (
                        torch.cat([masked_video_latents] * 2) if do_classifier_free_guidance else masked_video_latents
                    )

                    mask = rearrange(mask, "b c f h w -> b f c h w")
                    mask_input = rearrange(mask_input, "b c f h w -> b f c h w")
                    masked_video_latents_input = rearrange(masked_video_latents_input, "b c f h w -> b f c h w")

                    inpaint_latents = torch.cat([mask_input, masked_video_latents_input], dim=2).to(latents.dtype)
                else:
                    mask = torch.tile(mask_condition, [1, num_channels_latents, 1, 1, 1])
                    mask = F.interpolate(mask, size=latents.size()[-3:], mode='trilinear', align_corners=True).to(latents.device, latents.dtype)
                    mask = rearrange(mask, "b c f h w -> b f c h w")
                    
                    inpaint_latents = None
        else:
            if num_channels_transformer != num_channels_latents:
                # Create 1-channel mask
                mask = torch.zeros_like(latents[:, :, :1]).to(latents.device, latents.dtype)  # [B, 1, F, H, W]
                
                if static_videos_latents is not None:
                    # Use static_videos_latents as masked_video_latents
                    masked_video_latents = static_videos_latents.to(latents.device, latents.dtype)
                else:
                    # No static videos, use zeros
                    masked_video_latents = torch.zeros_like(latents).to(latents.device, latents.dtype)
                masked_video_latents = rearrange(masked_video_latents, "b f c h w -> b c f h w")

                # Create mask_latents using resize_mask function (similar to mask_video is not None case)
                mask_condition = self.mask_processor.preprocess(rearrange(mask, "b f c h w -> (b f) c h w"), height=height, width=width) 
                mask_condition = mask_condition.to(dtype=torch.float32)
                mask_condition = rearrange(mask_condition, "(b f) c h w -> b c f h w", f=local_latent_length)
                mask_latents = resize_mask(1 - mask_condition, masked_video_latents)
                mask_latents = mask_latents.to(masked_video_latents.device) * self.vae.config.scaling_factor

                # Handle classifier-free guidance
                if do_classifier_free_guidance:
                    mask_input = torch.cat([mask_latents, mask_latents], dim=0)  # [2B, 1, F, H, W]
                    masked_video_latents_input = torch.cat([masked_video_latents, masked_video_latents], dim=0)  # [2B, C, F, H, W]
                else:
                    mask_input = mask_latents
                    masked_video_latents_input = masked_video_latents
                
                # Concatenate mask (1 channel) + masked_video_latents (C channels) = [B, C+1, F, H, W]
                mask_input = rearrange(mask_input, "b c f h w -> b f c h w")
                masked_video_latents_input = rearrange(masked_video_latents_input, "b c f h w -> b f c h w")
                inpaint_latents = torch.cat([mask_input, masked_video_latents_input], dim=2).to(latents.dtype)
            else:
                # Standard case without additional channels
                mask = torch.zeros_like(init_video[:, :1])
                mask = torch.tile(mask, [1, num_channels_latents, 1, 1, 1])
                mask = F.interpolate(mask, size=latents.size()[-3:], mode='trilinear', align_corners=True).to(latents.device, latents.dtype)
                mask = rearrange(mask, "b c f h w -> b f c h w")

                inpaint_latents = None
        if comfyui_progressbar and pbar is not None:
            pbar.update(1)

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Create rotary embeds if required
        image_rotary_emb = (
            self._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
            if self.transformer.config.use_rotary_positional_embeddings
            else None
        )
        
        # Initialize adapter_control_for_transformer (used by all pipelines)
        if not hasattr(self, 'add_control_adapter') or not self.add_control_adapter:
            adapter_control_for_transformer = None

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            # for DPM-solver++
            old_pred_original_sample = None
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                # predict noise model_output
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    image_rotary_emb=image_rotary_emb,
                    return_dict=False,
                    inpaint_latents=inpaint_latents,
                    control_latents=control_latents,
                    ref_latents=ref_latents,
                )[0]
                noise_pred = noise_pred.float()

                # perform guidance
                if use_dynamic_cfg:
                    self._guidance_scale = 1 + guidance_scale * (
                        (1 - math.cos(math.pi * ((num_inference_steps - t.item()) / num_inference_steps) ** 5.0)) / 2
                    )
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                if not isinstance(self.scheduler, CogVideoXDPMScheduler):
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                else:
                    latents, old_pred_original_sample = self.scheduler.step(
                        noise_pred,
                        old_pred_original_sample,
                        t,
                        timesteps[i - 1] if i > 0 else None,
                        latents,
                        **extra_step_kwargs,
                        return_dict=False,
                    )
                latents = latents.to(prompt_embeds.dtype)

                # call the callback, if provided
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                if comfyui_progressbar:
                    pbar.update(1)

        if not output_type == "latent":
            video = self.decode_latents(latents)
            video = rearrange(video, "b c f h w -> b f h w c")
            # video = self.video_processor.postprocess_video(video=video, output_type=output_type)
            # video = rearrange(video, "b f h w c -> b c f h w")
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        # if not return_dict:
        #     if isinstance(video, np.ndarray):
        #         video = torch.from_numpy(video)
        return CogVideoXFunPipelineOutput(frames=video)

    @classmethod
    def save_lora_weights(
        cls,
        save_directory: str,
        transformer_lora_layers: Optional[Dict[str, torch.Tensor]] = None,
        is_main_process: bool = True,
        weight_name: str = "pytorch_lora_weights.safetensors",
        save_function: Optional[Callable] = None,
        safe_serialization: bool = True,
    ):
        """Save LoRA weights to a directory."""
        if transformer_lora_layers is not None:
            # Save LoRA weights
            os.makedirs(save_directory, exist_ok=True)
            
            if save_function is not None:
                save_function(transformer_lora_layers, os.path.join(save_directory, weight_name))
            elif safe_serialization:
                from safetensors.torch import save_file
                save_file(transformer_lora_layers, os.path.join(save_directory, weight_name))
            else:
                torch.save(transformer_lora_layers, os.path.join(save_directory, weight_name))

    @classmethod
    def lora_state_dict(cls, input_dir: str) -> Dict[str, torch.Tensor]:
        """Load LoRA state dict from a directory."""
        lora_state_dict = {}
        
        # Try to load from safetensors first
        safetensors_path = os.path.join(input_dir, "pytorch_lora_weights.safetensors")
        if os.path.exists(safetensors_path):
            from safetensors.torch import load_file
            lora_state_dict = load_file(safetensors_path)
        else:
            # Try to load from pytorch format
            pytorch_path = os.path.join(input_dir, "pytorch_lora_weights.bin")
            if os.path.exists(pytorch_path):
                lora_state_dict = torch.load(pytorch_path, map_location="cpu")
        
        return lora_state_dict
