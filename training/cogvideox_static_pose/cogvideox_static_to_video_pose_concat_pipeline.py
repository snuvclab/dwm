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
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDDIMScheduler,
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXTransformer3DModel,
)
from diffusers.image_processor import PipelineImageInput
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.utils import BaseOutput, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from einops import rearrange
from transformers import T5EncoderModel, T5Tokenizer

# Import our custom transformer with concat support
from cogvideox_transformer_with_conditions import CogVideoXTransformer3DModelWithConcat, CrossTransformer3DModelWithAdapter

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

    return new_img

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


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    sample_mode: str = "sample",
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


@dataclass
class CogVideoXStaticToVideoPipelineOutput(BaseOutput):
    frames: Union[List[PIL.Image.Image], np.ndarray]


@dataclass
class CogVideoXStaticToVideoPoseConcatPipelineOutput(BaseOutput):
    frames: Union[List[PIL.Image.Image], np.ndarray]


@dataclass
class CogVideoXStaticToVideoCrossPipelineOutput(BaseOutput):
    """
    Output class for CogVideoX static-to-video cross pipeline.
    """
    frames: Union[List[np.ndarray], np.ndarray]

class CogVideoXStaticToVideoPipeline(CogVideoXImageToVideoPipeline):
    """
    Pipeline for static-to-video generation using CogVideoX.
    
    This pipeline extends CogVideoXImageToVideoPipeline by replacing the original I2V's
    image latents (first frame + zero padding) with static scene video latents.
    
    Original I2V: Noisy latents (16) + Image latents (16) = 32 channels
    This pipeline: Noisy latents (16) + Static scene latents (16) = 32 channels
    
    Uses standard CogVideoXTransformer3DModel with in_channels=32.
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
                        transformer=None, *args, **kwargs):
        """
        Load a CogVideoXStaticToVideoPipeline from a saved directory or base model.
        
        This method loads all components (tokenizer, text_encoder, vae, transformer, scheduler)
        from the specified directory, or uses provided components if specified.
        
        Args:
            pretrained_model_name_or_path: Path to saved pipeline or base model
            base_model_name_or_path: Optional base model path for creating I2V static-to-video pipeline
            transformer: Optional transformer to use (for validation with trained transformer)
        """
        # Check if this is a base model path (for creating I2V static-to-video pipeline)
        if base_model_name_or_path is not None:
            print(f"🔧 Creating I2V static-to-video pipeline from base model: {base_model_name_or_path}")
            # Load the original CogVideoX I2V pipeline from base model
            original_pipeline = CogVideoXImageToVideoPipeline.from_pretrained(base_model_name_or_path, *args, **kwargs)
            
        if transformer is None:
            # Use standard CogVideoXTransformer3DModel with in_channels=32
            print(f"🔧 Creating/loading I2V transformer with in_channels=32")
            from diffusers import CogVideoXTransformer3DModel
            if pretrained_model_name_or_path is not None:
                transformer = CogVideoXTransformer3DModel.from_pretrained(
                    pretrained_model_name_or_path=pretrained_model_name_or_path,
                    subfolder="transformer",
                    torch_dtype=kwargs.get("torch_dtype", torch.bfloat16),
                    revision=kwargs.get("revision", None),
                    variant=kwargs.get("variant", None),
                )
            else:
                transformer = CogVideoXTransformer3DModel.from_pretrained(
                    base_model_name_or_path,
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
            print(f"🔧 Loading I2V static-to-video pipeline from {pretrained_model_name_or_path}")
            from diffusers import CogVideoXTransformer3DModel
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
            "pipeline_class": "CogVideoXStaticToVideoPipeline",
            "version": "1.0.0",
        }
        with open(os.path.join(save_directory, "pipeline_config.json"), "w") as f:
            import json
            json.dump(pipeline_config, f)

    def check_inputs(
        self,
        static_videos,
        prompt,
        height,
        width,
        negative_prompt,
        callback_on_step_end_tensor_inputs,
        latents=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        num_frames=None,
        image=None,
    ):
        # Either static_videos or image must be provided
        if static_videos is None and image is None:
            raise ValueError("Either static_videos or image must be provided")
        
        # Call parent check_inputs with image (for I2V fallback) or dummy image
        check_image = image if image is not None else torch.zeros((1, 3, height, width))
        super().check_inputs(
            image=check_image,
            prompt=prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            latents=latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # Check static scene frames
        if static_videos is not None:
            if not isinstance(static_videos, (torch.Tensor, np.ndarray, list)):
                raise ValueError(
                    "`static_videos` has to be of type `torch.Tensor` or `np.ndarray` or `List[PIL.Image.Image]` but is"
                    f" {type(static_videos)}"
                )
            
            if isinstance(static_videos, list):
                if not all(isinstance(img, PIL.Image.Image) for img in static_videos):
                    raise ValueError("All elements in `static_videos` list must be PIL.Image.Image")
                if len(static_videos) != num_frames:
                    raise ValueError(f"`static_videos` must have {num_frames} frames, got {len(static_videos)}")

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

    def preprocess_static_conditions(
        self,
        static_videos,
        height,
        width,
        num_frames,
    ):
        """Preprocess static scene frames."""
        if static_videos is not None:
            if isinstance(static_videos, PIL.Image.Image):
                static_videos = self.video_processor.preprocess(
                    static_videos, height, width, resize_mode="crop"
                ).to(device=self._execution_device, dtype=torch.bfloat16)
            else:
                static_videos = self._preprocess_images(static_videos, height, width).to(
                    device=self._execution_device, dtype=torch.bfloat16
                )

        return static_videos

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

    def prepare_latents(
        self,
        image: Optional[torch.Tensor] = None,  # For I2V fallback
        static_videos: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        num_channels_latents: int = 16,
        num_frames: int = 13,
        height: int = 60,
        width: int = 90,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        orig_num_frames = num_frames
        num_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            batch_size,
            num_frames,
            num_channels_latents,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )

        # For CogVideoX1.5, the latent should add 1 for padding (Not use)
        if self.transformer.config.patch_size_t is not None:
            shape = shape[:1] + (shape[1] + shape[1] % self.transformer.config.patch_size_t,) + shape[2:]

        # Prepare condition latents (either static videos or image-based I2V)
        if static_videos is not None:
            # Static-to-video mode: use full static video latents
            # Convert to torch tensor if needed
            if isinstance(static_videos, np.ndarray):
                static_videos = torch.from_numpy(static_videos).to(dtype=dtype, device=device)
            
            # Check if static_videos are already latents (when load_tensors=True)
            if static_videos.shape[2] == self.vae.config.latent_channels:
                # Already latents, just ensure correct format
                static_videos_latents = static_videos.to(dtype=dtype, device=device)
            else:
                # Raw videos, need to encode
                if static_videos.ndim == 4:
                    static_videos = static_videos.unsqueeze(0)
                
                # static_videos = static_videos.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
                
                # Encode entire batch of videos at once
                latent_dist = self.vae.encode(static_videos)
                static_videos_latents = retrieve_latents(latent_dist, generator)
                static_videos_latents = static_videos_latents.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
                if not self.vae.config.invert_scale_latents:
                    static_videos_latents = self.vae_scaling_factor_image * static_videos_latents
                else:
                    static_videos_latents = 1 / self.vae_scaling_factor_image * static_videos_latents
                static_videos_latents = static_videos_latents.to(dtype=dtype, device=device)
        elif image is not None:
            # I2V fallback mode: use original I2V logic (first frame + zero padding)
            # Preprocess the image first
            if isinstance(image, PIL.Image.Image):
                # Convert PIL Image to tensor
                image = self.video_processor.preprocess(
                    image, height, width, resize_mode="crop"
                ).to(device=device, dtype=dtype)
            # Use parent class's prepare_latents method for I2V logic
            latents, image_latents = super().prepare_latents(
                image=image,
                batch_size=batch_size,
                num_channels_latents=num_channels_latents,
                num_frames=orig_num_frames,
                height=height,
                width=width,
                dtype=dtype,
                device=device,
                generator=generator,
                latents=latents,
            )
            # Return both latents and image_latents (which will be used as static_videos_latents)
            return latents, image_latents
        else:
            # If neither static videos nor image provided, create zero latents
            static_videos_latents = torch.zeros(
                (batch_size, num_frames, num_channels_latents, 
                 height // self.vae_scale_factor_spatial, width // self.vae_scale_factor_spatial),
                device=device, dtype=dtype
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents, static_videos_latents

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
        guidance_scale: float = 6,
        use_dynamic_cfg: bool = False,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None]]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 226,
    ) -> Union[CogVideoXStaticToVideoPipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for generation.

        Args:
            static_videos (`torch.Tensor` or `np.ndarray` or `List[PIL.Image.Image]`, *optional*):
                Static scene frames to condition the generation. If not provided, `image` must be provided for I2V mode.
            image (`torch.Tensor` or `np.ndarray` or `PIL.Image.Image`, *optional*):
                Image to condition the generation. Used for I2V fallback mode (first frame + zero padding).
                If not provided, `static_videos` must be provided for static-to-video mode.
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            height (`int`, *optional*, defaults to self.transformer.config.sample_height * self.vae_scale_factor_spatial):
                The height in pixels of the generated image. This is set to 480 by default for the best results.
            width (`int`, *optional*, defaults to self.transformer.config.sample_height * self.vae_scale_factor_spatial):
                The width in pixels of the generated image. This is set to 720 by default for the best results.
            num_frames (`int`, defaults to `48`):
                Number of frames to generate. Must be divisible by self.vae_scale_factor_temporal. Generated video will
                contain 1 extra frame because CogVideoX is conditioned with (num_seconds * fps + 1) frames where
                num_seconds is 6 and fps is 8. However, since videos can be saved at any fps, the only condition that
                needs to be satisfied is that of divisibility mentioned above.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality.
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
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
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

        Returns:
            [`CogVideoXStaticToVideoPipelineOutput`] or `tuple`:
            [`CogVideoXStaticToVideoPipelineOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is a list with the generated images.
        """

        height = height or self.transformer.config.sample_height * self.vae_scale_factor_spatial
        width = width or self.transformer.config.sample_width * self.vae_scale_factor_spatial
        num_frames = num_frames or self.transformer.config.sample_frames

        num_videos_per_prompt = 1

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            static_videos=static_videos,
            prompt=prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            latents=latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            num_frames=num_frames,
            image=image,
        )
        self._guidance_scale = guidance_scale
        self._current_timestep = None
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
        # of the Imagen paper: https://huggingface.co/papers/2205.11487 . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

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
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        self._num_timesteps = len(timesteps)

        # 5. Prepare latents
        latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1

        # For CogVideoX 1.5, the latent frames should be padded to make it divisible by patch_size_t
        patch_size_t = self.transformer.config.patch_size_t
        additional_frames = 0
        if patch_size_t is not None and latent_frames % patch_size_t != 0:
            additional_frames = patch_size_t - latent_frames % patch_size_t
            num_frames += additional_frames * self.vae_scale_factor_temporal

        latent_channels = self.transformer.config.in_channels // 2  # 32 // 2 = 16 (noisy latents)
        latents, static_videos_latents = self.prepare_latents(
            image=image,  # For I2V fallback
            static_videos=static_videos,
            batch_size=batch_size * num_videos_per_prompt,
            num_channels_latents=latent_channels,
            num_frames=num_frames,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
            latents=latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Create rotary embeds if required
        image_rotary_emb = (
            self._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
            if self.transformer.config.use_rotary_positional_embeddings
            else None
        )

        # 8. Create ofs embeds if required
        ofs_emb = None if self.transformer.config.ofs_embed_dim is None else latents.new_full((1,), fill_value=2.0)

        # 9. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            # for DPM-solver++
            old_pred_original_sample = None
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # Use static video latents instead of image latents (replacing I2V's image_latents)
                latent_static_input = torch.cat([static_videos_latents] * 2) if do_classifier_free_guidance else static_videos_latents
                latent_model_input = torch.cat([latent_model_input, latent_static_input], dim=2)

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                # predict noise model_output
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    ofs=ofs_emb,
                    image_rotary_emb=image_rotary_emb,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
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

        self._current_timestep = None

        if not output_type == "latent":
            # Discard any padding frames that were added for CogVideoX 1.5
            latents = latents[:, additional_frames:]
            video = self.decode_latents(latents)
            video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return CogVideoXStaticToVideoPipelineOutput(frames=video)


class CogVideoXStaticToVideoPoseConcatPipeline(CogVideoXImageToVideoPipeline):
    """
    Pipeline for static-to-video generation with hand pose conditioning using CogVideoX.
    
    This pipeline extends CogVideoXImageToVideoPipeline by concatenating both static scene
    and hand pose video latents as conditions.
    
    Total channels: Noisy latents (16) + Static scene latents (16) + Hand pose latents (16) = 48 channels
    
    Uses CogVideoXTransformer3DModelWithConcat with condition_channels=32 (16 static + 16 hand pose).
    """

    def __init__(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        vae: AutoencoderKLCogVideoX,
        transformer: CogVideoXTransformer3DModelWithConcat,
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
                        transformer=None, *args, **kwargs):
        """
        Load a CogVideoXStaticToVideoPoseConcatPipeline from a saved directory or base model.
        
        This method loads all components (tokenizer, text_encoder, vae, transformer, scheduler)
        from the specified directory, or uses provided components if specified.
        
        Args:
            pretrained_model_name_or_path: Path to saved pipeline or base model
            base_model_name_or_path: Optional base model path for creating I2V dual concat pipeline
            transformer: Optional transformer to use (for validation with trained transformer)
        """
        # Check if this is a base model path (for creating I2V dual concat pipeline)
        if base_model_name_or_path is not None:
            print(f"🔧 Creating I2V dual concat pipeline from base model: {base_model_name_or_path}")
            # Load the original CogVideoX I2V pipeline from base model
            original_pipeline = CogVideoXImageToVideoPipeline.from_pretrained(base_model_name_or_path, *args, **kwargs)
            
        if transformer is None:
            # Create or load concat transformer with 32 condition channels
            print(f"🔧 Creating/loading concat transformer with 32 condition channels")
            transformer = CogVideoXTransformer3DModelWithConcat.from_pretrained(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                base_model_name_or_path=base_model_name_or_path,
                subfolder="transformer",
                condition_channels=16,
                torch_dtype=kwargs.get("torch_dtype", torch.bfloat16),
                revision=kwargs.get("revision", None),
                variant=kwargs.get("variant", None),
            )
        elif pretrained_model_name_or_path is not None:
            load_dtype = torch.bfloat16 if "5b" in base_model_name_or_path.lower() else torch.float16
            # Override with provided torch_dtype if specified
            if "torch_dtype" in kwargs:
                load_dtype = kwargs["torch_dtype"]
            print(f"🔧 Loading concat pipeline from {pretrained_model_name_or_path}")
            transformer = CogVideoXTransformer3DModelWithConcat.from_pretrained(
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
            "pipeline_class": "CogVideoXStaticToVideoPoseConcatPipeline",
            "version": "1.0.0",
        }
        with open(os.path.join(save_directory, "pipeline_config.json"), "w") as f:
            import json
            json.dump(pipeline_config, f)

    def check_inputs(
        self,
        static_videos,
        hand_videos,
        prompt,
        height,
        width,
        negative_prompt,
        callback_on_step_end_tensor_inputs,
        latents=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        num_frames=None,
        image=None,
    ):
        # Either (static_videos + hand_videos) or image must be provided
        if static_videos is None and hand_videos is None and image is None:
            raise ValueError("Either (static_videos + hand_videos) or image must be provided")
        
        # If using video mode, both static_videos and hand_videos must be provided
        if static_videos is not None and hand_videos is None:
            raise ValueError("hand_videos must be provided when static_videos is provided")
        if hand_videos is not None and static_videos is None:
            raise ValueError("static_videos must be provided when hand_videos is provided")
        
        # Call parent check_inputs with image (for I2V fallback) or dummy image
        check_image = image if image is not None else torch.zeros((1, 3, height, width))
        super().check_inputs(
            image=check_image,
            prompt=prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            latents=latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # Check static scene frames
        if static_videos is not None:
            if not isinstance(static_videos, (torch.Tensor, np.ndarray, list)):
                raise ValueError(
                    "`static_videos` has to be of type `torch.Tensor` or `np.ndarray` or `List[PIL.Image.Image]` but is"
                    f" {type(static_videos)}"
                )
            
            if isinstance(static_videos, list):
                if not all(isinstance(img, PIL.Image.Image) for img in static_videos):
                    raise ValueError("All elements in `static_videos` list must be PIL.Image.Image")
                if len(static_videos) != num_frames:
                    raise ValueError(f"`static_videos` must have {num_frames} frames, got {len(static_videos)}")

        # Check hand frames
        if hand_videos is not None:
            if not isinstance(hand_videos, (torch.Tensor, np.ndarray, list)):
                raise ValueError(
                    "`hand_videos` has to be of type `torch.Tensor` or `np.ndarray` or `List[PIL.Image.Image]` but is"
                    f" {type(hand_videos)}"
                )
            
            if isinstance(hand_videos, list):
                if not all(isinstance(img, PIL.Image.Image) for img in hand_videos):
                    raise ValueError("All elements in `hand_videos` list must be PIL.Image.Image")
                if len(hand_videos) != num_frames:
                    raise ValueError(f"`hand_videos` must have {num_frames} frames, got {len(hand_videos)}")

    def preprocess_hand_conditions(
        self,
        hand_videos,
        height,
        width,
        num_frames,
    ):
        """Preprocess hand frames."""
        if hand_videos is not None:
            if isinstance(hand_videos, PIL.Image.Image):
                hand_videos = self.video_processor.preprocess(
                    hand_videos, height, width, resize_mode="crop"
                ).to(device=self._execution_device, dtype=torch.bfloat16)
            else:
                hand_videos = self._preprocess_images(hand_videos, height, width).to(
                    device=self._execution_device, dtype=torch.bfloat16
                )

        return hand_videos

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


    def preprocess_conditions(
        self,
        hand_videos,
        static_videos,
        height,
        width,
        num_frames,
    ):
        """Preprocess hand mesh images and static scene frames."""
        if hand_videos is not None:
            if isinstance(hand_videos, PIL.Image.Image):
                hand_videos = self.video_processor.preprocess(
                    hand_videos, height, width, resize_mode="crop"
                ).to(device=self._execution_device, dtype=torch.bfloat16)
            else:
                hand_videos = self._preprocess_images(hand_videos, height, width).to(
                    device=self._execution_device, dtype=torch.bfloat16
                )
                
        if static_videos is not None:
            if isinstance(static_videos, PIL.Image.Image):
                static_videos = self.video_processor.preprocess(
                    static_videos, height, width, resize_mode="crop"
                ).to(device=self._execution_device, dtype=torch.bfloat16)
            else:
                static_videos = self._preprocess_images(static_videos, height, width).to(
                    device=self._execution_device, dtype=torch.bfloat16
                )

        return hand_videos, static_videos

    def prepare_latents(
        self,
        image: Optional[torch.Tensor] = None,  # For I2V fallback
        static_videos: Optional[torch.Tensor] = None,
        hand_videos: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        num_channels_latents: int = 16,
        num_frames: int = 13,
        height: int = 60,
        width: int = 90,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        orig_num_frames = num_frames
        num_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            batch_size,
            num_frames,
            num_channels_latents,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )

        # # For CogVideoX1.5, the latent should add 1 for padding (Not use)
        # if self.transformer.config.patch_size_t is not None:
        #     shape = shape[:1] + (shape[1] + shape[1] % self.transformer.config.patch_size_t,) + shape[2:]

        # Prepare condition latents (either static+hand videos or image-based I2V)
        if static_videos is not None and hand_videos is not None:
            # Dual video mode: use both static and hand video latents
            # Convert to torch tensor if needed
            if isinstance(static_videos, np.ndarray):
                static_videos = torch.from_numpy(static_videos).to(dtype=dtype, device=device)
            if isinstance(hand_videos, np.ndarray):
                hand_videos = torch.from_numpy(hand_videos).to(dtype=dtype, device=device)
            
            # Process static_videos
            if static_videos.shape[2] == self.vae.config.latent_channels:
                # Already latents, just ensure correct format
                static_videos_latents = static_videos.to(dtype=dtype, device=device)
            else:
                # Raw videos, need to encode
                if static_videos.ndim == 4:
                    static_videos = static_videos.unsqueeze(0)
                
                static_videos = static_videos.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
                
                # Encode entire batch of videos at once
                latent_dist = self.vae.encode(static_videos)
                static_videos_latents = retrieve_latents(latent_dist, generator)
                
                if not self.vae.config.invert_scale_latents:
                    static_videos_latents = self.vae_scaling_factor_image * static_videos_latents
                else:
                    static_videos_latents = 1 / self.vae_scaling_factor_image * static_videos_latents
                static_videos_latents = static_videos_latents.to(dtype=dtype, device=device)
            
            # Process hand_videos
            if hand_videos.shape[2] == self.vae.config.latent_channels:
                # Already latents, just ensure correct format
                hand_video_latents = hand_videos.to(dtype=dtype, device=device)
            else:
                # Raw videos, need to encode
                if hand_videos.ndim == 4:
                    hand_videos = hand_videos.unsqueeze(0)
                
                hand_videos = hand_videos.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
                
                # Encode entire batch of videos at once
                latent_dist = self.vae.encode(hand_videos)
                hand_video_latents = retrieve_latents(latent_dist, generator)
                
                if not self.vae.config.invert_scale_latents:
                    hand_video_latents = self.vae_scaling_factor_image * hand_video_latents
                else:
                    hand_video_latents = 1 / self.vae_scaling_factor_image * hand_video_latents
                hand_video_latents = hand_video_latents.to(dtype=dtype, device=device)
        elif image is not None:
            # I2V fallback mode: use original I2V logic (first frame + zero padding)
            # Preprocess the image first
            if isinstance(image, PIL.Image.Image):
                # Convert PIL Image to tensor
                image = self.video_processor.preprocess(
                    image, height, width, resize_mode="crop"
                ).to(device=device, dtype=dtype)
            
            # Use parent class's prepare_latents method for I2V logic
            latents, image_latents = super().prepare_latents(
                image=image,
                batch_size=batch_size,
                num_channels_latents=num_channels_latents,
                num_frames=orig_num_frames,
                height=height,
                width=width,
                dtype=dtype,
                device=device,
                generator=generator,
                latents=latents,
            )
            # For I2V mode, we need to duplicate image_latents to simulate both static and hand latents
            # This is a hack to make the dual concat pipeline work with I2V
            static_videos_latents = image_latents
            hand_video_latents = image_latents  # Duplicate for dual concat
            # Return both latents and the duplicated image_latents
            return latents, static_videos_latents, hand_video_latents
        else:
            # If neither videos nor image provided, create zero latents
            static_videos_latents = torch.zeros(
                (batch_size, num_frames, num_channels_latents, 
                 height // self.vae_scale_factor_spatial, width // self.vae_scale_factor_spatial),
                device=device, dtype=dtype
            )
            
            # Handle hand_videos (can be None in I2V mode)
            if hand_videos is not None:
                # Check if hand_videos are already latents (when load_tensors=True)
                if hand_videos.shape[2] == self.vae.config.latent_channels:
                    # Already latents, just ensure correct format
                    hand_video_latents = hand_videos.to(dtype=dtype, device=device)
                else:
                    # Raw videos, need to encode
                    if hand_videos.ndim == 4:
                        hand_videos = hand_videos.unsqueeze(0)
                    
                    hand_videos = hand_videos.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
                    
                    # Encode entire batch of videos at once
                    latent_dist = self.vae.encode(hand_videos)
                    hand_video_latents = retrieve_latents(latent_dist, generator)
                    hand_video_latents = hand_video_latents.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
                    
                    if not self.vae.config.invert_scale_latents:
                        hand_video_latents = self.vae_scaling_factor_image * hand_video_latents
                    else:
                        hand_video_latents = 1 / self.vae_scaling_factor_image * hand_video_latents
                    hand_video_latents = hand_video_latents.to(dtype=dtype, device=device)
            else:
                # If hand_videos is None (I2V mode), create zero latents
                hand_video_latents = torch.zeros(
                    (batch_size, num_frames, num_channels_latents, 
                     height // self.vae_scale_factor_spatial, width // self.vae_scale_factor_spatial),
                    device=device, dtype=dtype
                )

        # No padding needed for static-to-video pose concat pipeline
        # Static and hand videos are used directly as conditions without initial frame + zero padding

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents, static_videos_latents, hand_video_latents

    @torch.no_grad()
    def __call__(
        self,
        static_videos: Optional[Union[torch.Tensor, np.ndarray, List[PIL.Image.Image]]] = None,
        hand_videos: Optional[Union[torch.Tensor, np.ndarray, List[PIL.Image.Image]]] = None,
        image: Optional[Union[torch.Tensor, np.ndarray, PIL.Image.Image]] = None,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
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
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None]]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 226,
    ) -> Union[CogVideoXStaticToVideoPoseConcatPipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for generation.

        Args:
            static_videos (`torch.Tensor` or `np.ndarray` or `List[PIL.Image.Image]`, *optional*):
                Static scene frames to condition the generation. If not provided, `image` must be provided for I2V mode.
            hand_videos (`torch.Tensor` or `np.ndarray` or `List[PIL.Image.Image]`, *optional*):
                Hand pose frames to condition the generation. If not provided, `image` must be provided for I2V mode.
            image (`torch.Tensor` or `np.ndarray` or `PIL.Image.Image`, *optional*):
                The input image to condition the generation on. Must be an image, a list of images or a `torch.Tensor`.
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            static_videos (`torch.Tensor` or `np.ndarray` or `List[PIL.Image.Image]`, *optional*):
                Static scene frames to condition the generation.
            hand_videos (`torch.Tensor` or `np.ndarray` or `List[PIL.Image.Image]`, *optional*):
                Hand frames to condition the generation.
            height (`int`, *optional*, defaults to self.transformer.config.sample_height * self.vae_scale_factor_spatial):
                The height in pixels of the generated image. This is set to 480 by default for the best results.
            width (`int`, *optional*, defaults to self.transformer.config.sample_height * self.vae_scale_factor_spatial):
                The width in pixels of the generated image. This is set to 720 by default for the best results.
            num_frames (`int`, defaults to `48`):
                Number of frames to generate. Must be divisible by self.vae_scale_factor_temporal. Generated video will
                contain 1 extra frame because CogVideoX is conditioned with (num_seconds * fps + 1) frames where
                num_seconds is 6 and fps is 8. However, since videos can be saved at any fps, the only condition that
                needs to be satisfied is that of divisibility mentioned above.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality.
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
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
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

        Returns:
            [`CogVideoXStaticToVideoPoseConcatPipelineOutput`] or `tuple`:
            [`CogVideoXStaticToVideoPoseConcatPipelineOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is a list with the generated images.
        """

        height = height or self.transformer.config.sample_height * self.vae_scale_factor_spatial
        width = width or self.transformer.config.sample_width * self.vae_scale_factor_spatial
        num_frames = num_frames or self.transformer.config.sample_frames

        num_videos_per_prompt = 1

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            static_videos=static_videos,
            hand_videos=hand_videos,
            prompt=prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            latents=latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            num_frames=num_frames,
            image=image,
        )
        self._guidance_scale = guidance_scale
        self._current_timestep = None
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
        # of the Imagen paper: https://huggingface.co/papers/2205.11487 . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

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
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        # # 4. Preprocess conditions
        if hand_videos is not None and static_videos is not None:
            hand_videos, static_videos = self.preprocess_conditions(
                rearrange(hand_videos, "b c f h w -> b f h w c"),
                rearrange(static_videos, "b c f h w -> b f h w c"),
                height,
                width,
                num_frames,
            )

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        self._num_timesteps = len(timesteps)

        # 5. Prepare latents
        latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1

        # For CogVideoX 1.5, the latent frames should be padded to make it divisible by patch_size_t
        patch_size_t = self.transformer.config.patch_size_t
        additional_frames = 0
        if patch_size_t is not None and latent_frames % patch_size_t != 0:
            additional_frames = patch_size_t - latent_frames % patch_size_t
            num_frames += additional_frames * self.vae_scale_factor_temporal

        latent_channels = (self.transformer.config.in_channels - self.transformer.config.condition_channels) // 2
        latents, static_videos_latents, hand_video_latents = self.prepare_latents(
            image=image,  # For I2V fallback
            static_videos=static_videos,
            hand_videos=hand_videos,
            batch_size=batch_size * num_videos_per_prompt,
            num_channels_latents=latent_channels,
            num_frames=num_frames,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
            latents=latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Create rotary embeds if required
        image_rotary_emb = (
            self._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
            if self.transformer.config.use_rotary_positional_embeddings
            else None
        )

        # 8. Create ofs embeds if required
        ofs_emb = None if self.transformer.config.ofs_embed_dim is None else latents.new_full((1,), fill_value=2.0)

        # 9. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            # for DPM-solver++
            old_pred_original_sample = None
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # Concatenate all three types of latents: noisy + static + hand
                latent_static_input = torch.cat([static_videos_latents] * 2) if do_classifier_free_guidance else static_videos_latents
                latent_hand_input = torch.cat([hand_video_latents] * 2) if do_classifier_free_guidance else hand_video_latents
                latent_model_input = torch.cat([latent_model_input, latent_static_input, latent_hand_input], dim=2)

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                # predict noise model_output
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    ofs=ofs_emb,
                    image_rotary_emb=image_rotary_emb,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
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

        self._current_timestep = None

        if not output_type == "latent":
            # Discard any padding frames that were added for CogVideoX 1.5
            latents = latents[:, additional_frames:]
            video = self.decode_latents(latents)
            video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return CogVideoXStaticToVideoPoseConcatPipelineOutput(frames=video)

@dataclass
class CogVideoXFunPipelineOutput(BaseOutput):
    """
    Output class for CogVideoX Fun pipeline.

    Args:
        frames (`torch.Tensor`):
            The generated video frames.
    """

    frames: torch.Tensor

class CogVideoXStaticToVideoCrossPoseAdapterPipeline(CogVideoXImageToVideoPipeline):
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
                        use_zero_proj: bool = False,
                        is_train_cross: bool = True,
                        cross_attn_interval: int = 2,
                        cross_attn_dim_head: int = 128,
                        cross_attn_num_heads: int = 16,
                        cross_attn_kv_dim: int = None,
                        *args, **kwargs):
        """
        Load a CogVideoXStaticToVideoCrossPipeline from a saved directory or base model.
        
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
                transformer = CrossTransformer3DModelWithAdapter.from_pretrained(
                    pretrained_model_name_or_path=pretrained_model_name_or_path,
                    base_model_name_or_path=base_model_name_or_path,
                    subfolder="transformer",
                    condition_channels=condition_channels,
                    use_zero_proj=use_zero_proj,
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
            "pipeline_class": "CogVideoXStaticToVideoCrossPipeline",
            "version": "1.0.0",
        }
        with open(os.path.join(save_directory, "pipeline_config.json"), "w") as f:
            import json
            json.dump(pipeline_config, f)


    def prepare_latents(
        self,
        image: Optional[torch.Tensor] = None,  # For I2V fallback
        static_videos: Optional[torch.Tensor] = None,
        hand_videos: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        num_channels_latents: int = 16,
        num_frames: int = 13,
        height: int = 60,
        width: int = 90,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
    ):
        """Prepare latents for the pipeline with I2V style image processing."""
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        shape = (
            batch_size,
            num_frames,
            num_channels_latents,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )

        # For CogVideoX1.5, the latent should add 1 for padding (Not use)
        if self.transformer.config.patch_size_t is not None:
            shape = shape[:1] + (shape[1] + shape[1] % self.transformer.config.patch_size_t,) + shape[2:]

        # I2V style: process image as first frame + zero padding
        image_latents = None
        if image is not None:
            # Check if image is already latents by checking channel dimension
            if image.shape[1] == self.vae.config.latent_channels:
                # Already latents, just ensure correct format
                image_latents = image.to(dtype=dtype, device=device)
                # Add frame dimension: [B, C, H, W] -> [B, C, 1, H, W]
                image_latents = image_latents.unsqueeze(2)
                # Permute to [B, F, C, H, W]
                image_latents = image_latents.permute(0, 2, 1, 3, 4)
            elif image.shape[1] == 2 * self.vae.config.latent_channels:
                # 2 * latent_channels means it's a latent distribution, use retrieve_latents
                image_latent_dist = DiagonalGaussianDistribution(image)
                image_latents = image_latent_dist.sample().to(dtype).permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
            else:
                # Raw image, need to encode
                image = image.unsqueeze(2)  # [B, C, F, H, W]

                if isinstance(generator, list):
                    image_latents = [
                        retrieve_latents(self.vae.encode(image[i].unsqueeze(0)), generator[i]) for i in range(batch_size)
                    ]
                else:
                    image_latents = [retrieve_latents(self.vae.encode(img.unsqueeze(0)), generator) for img in image]

                image_latents = torch.cat(image_latents, dim=0).to(dtype).permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]

            # Apply scaling factor (only for encoded latents, not for pre-computed latents)
            if image.shape[1] != self.vae.config.latent_channels:
                if not self.vae.config.invert_scale_latents:
                    image_latents = self.vae_scaling_factor_image * image_latents
                else:
                    # This is awkward but required because the CogVideoX team forgot to multiply the
                    # scaling factor during training :)
                    image_latents = 1 / self.vae_scaling_factor_image * image_latents

            padding_shape = (
                batch_size,
                num_frames - 1,
                num_channels_latents,
                height // self.vae_scale_factor_spatial,
                width // self.vae_scale_factor_spatial,
            )
            latent_padding = torch.zeros(padding_shape, device=device, dtype=dtype)
            image_latents = torch.cat([image_latents, latent_padding], dim=1)

            # Select the first frame along the second dimension
            if self.transformer.config.patch_size_t is not None:
                first_frame = image_latents[:, : image_latents.size(1) % self.transformer.config.patch_size_t, ...]
                image_latents = torch.cat([first_frame, image_latents], dim=1)

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        # Prepare condition latents
        static_videos_latents = None
        hand_videos_latents = None
        
        if static_videos is not None:
            # Check if static_videos are already latents (when load_tensors=True)
            if isinstance(static_videos, np.ndarray):
                static_videos = torch.from_numpy(static_videos).to(dtype=dtype, device=device)
            else:
                static_videos = static_videos.to(device=device, dtype=self.vae.dtype)
            
            # Check if already latents by checking channel dimension
            if static_videos.shape[2] == self.vae.config.latent_channels:
                # Already latents, just ensure correct format
                static_videos_latents = static_videos.to(dtype=dtype, device=device)
                static_videos_latents = static_videos_latents.repeat(batch_size // static_videos_latents.shape[0], 1, 1, 1, 1)
            else:
                # Raw videos, need to encode
                bs = 1
                new_static_videos = []
                with torch.no_grad():
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
            # Check if hand_videos are already latents (when load_tensors=True)
            if isinstance(hand_videos, np.ndarray):
                hand_videos = torch.from_numpy(hand_videos).to(dtype=dtype, device=device)
            else:
                hand_videos = hand_videos.to(device=device, dtype=self.vae.dtype)
            
            # Check if already latents by checking channel dimension
            if hand_videos.shape[2] == self.vae.config.latent_channels:
                # Already latents, just ensure correct format
                hand_videos_latents = hand_videos.to(dtype=dtype, device=device)
                hand_videos_latents = hand_videos_latents.repeat(batch_size // hand_videos_latents.shape[0], 1, 1, 1, 1)
            else:
                # Raw videos, need to encode
                bs = 1
                new_hand_videos = []
                with torch.no_grad():
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

        return latents, image_latents, static_videos_latents, hand_videos_latents

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
        image,
        prompt,
        height,
        width,
        negative_prompt=None,
        callback_on_step_end_tensor_inputs=None,
        latents=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        """Check inputs for the static-to-video pipeline."""
        if (
            image is not None
            and not isinstance(image, torch.Tensor)
            and not isinstance(image, PIL.Image.Image)
            and not isinstance(image, list)
        ):
            raise ValueError(
                "`image` has to be of type `torch.Tensor` or `PIL.Image.Image` or `List[PIL.Image.Image]` but is"
                f" {type(image)}"
            )

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )
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
    def __call__(
        self,
        image: Optional[Union[torch.Tensor, np.ndarray, PIL.Image.Image]] = None,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
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
    ) -> Union[CogVideoXStaticToVideoCrossPipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for generation.

        Args:
            image (`torch.Tensor` or `np.ndarray` or `PIL.Image.Image`, *optional*):
                The input image to condition the generation on. Must be an image, a list of images or a `torch.Tensor`.
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            height (`int`, *optional*, defaults to self.transformer.config.sample_height * self.vae_scale_factor_spatial):
                The height in pixels of the generated image. This is set to 480 by default for the best results.
            width (`int`, *optional*, defaults to self.transformer.config.sample_width * self.vae_scale_factor_spatial):
                The width in pixels of the generated image. This is set to 720 by default for the best results.
            static_videos (`torch.FloatTensor`, `np.ndarray`, `List[PIL.Image.Image]`, *optional*):
                Static video frames to condition the generation (used as ref_latents).
            hand_videos (`torch.FloatTensor`, `np.ndarray`, `List[PIL.Image.Image]`, *optional*):
                Hand pose video frames to condition the generation (used as control_latents).
            num_frames (`int`, defaults to `48`):
                Number of frames to generate. Must be divisible by self.vae_scale_factor_temporal.
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
            image=image,
            prompt=prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            latents=latents,
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
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        self._num_timesteps = len(timesteps)
        
        # at which timestep to set the initial noise
        latent_timestep = timesteps[:1].repeat(batch_size * num_videos_per_prompt)
        # initialize latents with pure noise
        is_strength_max = True

        # 5. Prepare latents
        latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1

        # For CogVideoX 1.5, the latent frames should be padded to make it divisible by patch_size_t
        patch_size_t = self.transformer.config.patch_size_t
        additional_frames = 0
        if patch_size_t is not None and latent_frames % patch_size_t != 0:
            additional_frames = patch_size_t - latent_frames % patch_size_t
            num_frames += additional_frames * self.vae_scale_factor_temporal

        # Preprocess image if provided (I2V style)
        if image is not None:
            image = self.video_processor.preprocess(image, height=height, width=width).to(
                device, dtype=prompt_embeds.dtype
            )

        latent_channels = self.transformer.config.in_channels // 2
        latents, image_latents, static_videos_latents, hand_videos_latents = self.prepare_latents(
            image=image,
            static_videos=static_videos,
            hand_videos=hand_videos,
            batch_size=batch_size * num_videos_per_prompt,
            num_channels_latents=latent_channels,
            num_frames=num_frames,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
            latents=latents,
        )
        
        # Use preprocessed latents from prepare_latents
        control_latents = hand_videos_latents
        ref_latents = static_videos_latents
        
        # Apply classifier-free guidance if needed
        if do_classifier_free_guidance:
            if control_latents is not None:
                control_latents = torch.cat([control_latents] * 2)
            if ref_latents is not None:
                ref_latents = torch.cat([ref_latents] * 2)

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Create rotary embeds if required
        image_rotary_emb = (
            self._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
            if self.transformer.config.use_rotary_positional_embeddings
            else None
        )

        # if static_videos_latents is not None:
        #     image_rotary_emb_static = (
        #         self._prepare_rotary_positional_embeddings(
        #             height, 
        #             width, 
        #             1,   # Design choice. If the static video-generated video consistency is bad,
        #             device)
        #         if self.transformer.config.use_rotary_positional_embeddings
        #         else None
        #     )
        # else:
        #     image_rotary_emb_static = None
        image_rotary_emb_static = None  # Not implemented yet

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            # for DPM-solver++
            old_pred_original_sample = None
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # I2V style: concat image latents (first frame + zero padding) with noisy latents
                if image_latents is not None:
                    latent_image_input = torch.cat([image_latents] * 2) if do_classifier_free_guidance else image_latents
                    latent_model_input = torch.cat([latent_model_input, latent_image_input], dim=2)

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                # predict noise model_output
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    image_rotary_emb=image_rotary_emb,
                    return_dict=False,
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

        if not output_type == "latent":
            video = self.decode_latents(latents)
            video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        return CogVideoXStaticToVideoCrossPipelineOutput(frames=video)


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