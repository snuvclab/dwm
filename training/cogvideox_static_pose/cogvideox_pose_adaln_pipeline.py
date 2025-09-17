import inspect
import math
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL
import torch
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDDIMScheduler,
    CogVideoXDPMScheduler,
    CogVideoXPipeline,
    CogVideoXImageToVideoPipeline
)
from diffusers.image_processor import PipelineImageInput
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import BaseOutput, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from transformers import T5EncoderModel, T5Tokenizer

# Import our custom transformer with AdaLN pose conditioning
from cogvideox_transformer_with_conditions import CogVideoXTransformer3DModelWithAdaLNPose, CogVideoXTransformer3DModelWithAdaLNPosePerFrame

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
class CogVideoXPoseAdaLNPipelineOutput(BaseOutput):
    frames: Union[List[PIL.Image.Image], np.ndarray]


class CogVideoXPoseAdaLNPipeline(CogVideoXImageToVideoPipeline):
    """
    Pipeline for image-to-video generation using CogVideoX with SMPL pose conditioning via AdaLN-zero.
    
    This pipeline extends CogVideoXImageToVideoPipeline to handle SMPL pose parameters through AdaLN-zero conditioning:
    1. SMPL pose parameters are sampled with stride 4 (e.g., frames 0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48)
    2. Flattened pose parameters (63 * 13 = 819 dimensions) are encoded into embeddings
    3. AdaLN-zero conditioning is applied through extended LayerNormZero layers
    4. The transformer (CogVideoXTransformer3DModelWithAdaLNPose) handles the pose conditioning
    5. Inherits image-to-video functionality from CogVideoXImageToVideoPipeline
    """

    def __init__(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        vae: AutoencoderKLCogVideoX,
        transformer: CogVideoXTransformer3DModelWithAdaLNPose,
        scheduler: Union[CogVideoXDDIMScheduler, CogVideoXDPMScheduler],
    ):
        super().__init__(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
        )
        
        # Initialize empty prompt embeddings for unconditional generation
        self.empty_prompt_embeds, _ = self.encode_prompt(
            prompt="",
            negative_prompt=None,
            do_classifier_free_guidance=False,
            num_videos_per_prompt=1,
            prompt_embeds=None,
        )
        self.empty_prompt_embeds = self.empty_prompt_embeds.to(dtype=torch.bfloat16)
        
        # Verify that the transformer has AdaLN pose conditioning support
        if not hasattr(self.transformer, 'smpl_pose_embedding'):
            raise ValueError(
                "Transformer must be an instance of CogVideoXTransformer3DModelWithAdaLNPose "
                "to support AdaLN pose conditioning"
            )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path=None, 
                        base_model_name_or_path="THUDM/CogVideoX-5b-I2V", 
                        transformer=None, smpl_pose_dim=63, smpl_embed_dim=512, *args, **kwargs):
        """
        Load a CogVideoXPoseAdaLNPipeline from a saved directory or base model.
        
        This method loads all components (tokenizer, text_encoder, vae, transformer, scheduler)
        from the specified directory, or uses provided components if specified.
        
        Args:
            pretrained_model_name_or_path: Path to saved pipeline or base model
            base_model_name_or_path: Optional base model path for creating pose-conditioned pipeline
            transformer: Optional transformer to use (for validation with trained transformer)
            smpl_pose_dim: SMPL pose parameter dimension (default: 63)
            smpl_embed_dim: SMPL embedding dimension (default: 512)
        """
        # Check if this is a base model path (for creating pose-conditioned pipeline)
        if base_model_name_or_path is not None:
            print(f"🔧 Creating AdaLN pose-conditioned pipeline from base model: {base_model_name_or_path}")
            # Load the original CogVideoX ImageToVideo pipeline from base model
            original_pipeline = CogVideoXImageToVideoPipeline.from_pretrained(base_model_name_or_path, *args, **kwargs)
            
        if transformer is None:
            # Create or load AdaLN pose-conditioned transformer
            print(f"🔧 Creating/loading AdaLN pose-conditioned transformer with pose_dim={smpl_pose_dim}, embed_dim={smpl_embed_dim}")
            transformer = CogVideoXTransformer3DModelWithAdaLNPose.from_pretrained(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                base_model_name_or_path=base_model_name_or_path,
                subfolder="transformer",
                smpl_pose_dim=smpl_pose_dim,
                smpl_embed_dim=smpl_embed_dim,
                torch_dtype=kwargs.get("torch_dtype", torch.bfloat16),
                revision=kwargs.get("revision", None),
                variant=kwargs.get("variant", None),
            )
        elif pretrained_model_name_or_path is not None:
            load_dtype = torch.bfloat16 if "5b" in base_model_name_or_path.lower() else torch.float16
            # Override with provided torch_dtype if specified
            if "torch_dtype" in kwargs:
                load_dtype = kwargs["torch_dtype"]
            print(f"🔧 Loading AdaLN pose-conditioned pipeline from {pretrained_model_name_or_path}")
            transformer = CogVideoXTransformer3DModelWithAdaLNPose.from_pretrained(
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
            "pipeline_class": "CogVideoXPoseAdaLNPipeline",
            "version": "1.0.0",
        }
        with open(os.path.join(save_directory, "pipeline_config.json"), "w") as f:
            import json
            json.dump(pipeline_config, f)

    # Use parent class _prepare_rotary_positional_embeddings method

    def check_inputs(
        self,
        image,
        prompt,
        height,
        width,
        negative_prompt,
        callback_on_step_end_tensor_inputs,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pose_params=None,  # New parameter for SMPL pose
        pose_params_mask=None,  # New parameter to track None pose_params
        num_frames=None,
        latents=None,
    ):
        # Call parent check_inputs
        super().check_inputs(
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

        # Additional validation for SMPL pose parameters
        if pose_params is not None:
            if not isinstance(pose_params, (torch.Tensor, np.ndarray)):
                raise ValueError(f"pose_params must be a torch.Tensor or np.ndarray, got {type(pose_params)}")
            
            if isinstance(pose_params, np.ndarray):
                pose_params = torch.from_numpy(pose_params)
            
            # Validate pose_params shape: should be (batch_size, num_frames, 63)
            if pose_params.dim() != 3:
                raise ValueError(f"pose_params must have 3 dimensions (batch_size, num_frames, 63), got {pose_params.shape}")
            
            if pose_params.shape[-1] != 63:
                raise ValueError(f"pose_params last dimension must be 63 (SMPL pose parameters), got {pose_params.shape[-1]}")
        
        # Validate pose_params_mask if provided
        if pose_params_mask is not None:
            if not isinstance(pose_params_mask, (torch.Tensor, np.ndarray)):
                raise ValueError(f"pose_params_mask must be a torch.Tensor or np.ndarray, got {type(pose_params_mask)}")
            
            if isinstance(pose_params_mask, np.ndarray):
                pose_params_mask = torch.from_numpy(pose_params_mask)
            
            # Validate pose_params_mask shape: should be (batch_size,)
            if pose_params_mask.dim() != 1:
                raise ValueError(f"pose_params_mask must have 1 dimension (batch_size,), got {pose_params_mask.shape}")
            
            # Validate that pose_params_mask is boolean
            if pose_params_mask.dtype != torch.bool:
                raise ValueError(f"pose_params_mask must be boolean tensor, got {pose_params_mask.dtype}")
            
            # If pose_params is provided, validate that mask length matches batch size
            if pose_params is not None and pose_params_mask.shape[0] != pose_params.shape[0]:
                raise ValueError(f"pose_params_mask batch size ({pose_params_mask.shape[0]}) must match pose_params batch size ({pose_params.shape[0]})")

        if num_frames is None:
            raise ValueError("`num_frames` is required.")

    def preprocess_pose_params(
        self,
        pose_params: Optional[Union[torch.Tensor, np.ndarray]] = None,
        pose_params_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
        height: int = 480,
        width: int = 720,
        num_frames: int = 48,
    ):
        """Preprocess SMPL pose parameters."""
        if pose_params is not None:
            if isinstance(pose_params, np.ndarray):
                pose_params = torch.from_numpy(pose_params)
            
            # Ensure pose_params is on the correct device and dtype
            pose_params = pose_params.to(device=self._execution_device, dtype=self.transformer.dtype)
        
        if pose_params_mask is not None:
            if isinstance(pose_params_mask, np.ndarray):
                pose_params_mask = torch.from_numpy(pose_params_mask)
            
            # Ensure pose_params_mask is on the correct device and dtype
            pose_params_mask = pose_params_mask.to(device=self._execution_device, dtype=self.transformer.dtype)
        
        return pose_params, pose_params_mask

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

    # Use parent class prepare_latents method which handles image-to-video latents

    @torch.no_grad()
    def __call__(
        self,
        image: PipelineImageInput,  # Required for image-to-video
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        pose_params: Optional[Union[torch.Tensor, np.ndarray]] = None,  # New parameter for SMPL pose
        pose_params_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,  # New parameter to track None pose_params
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: int = 49,  # Default for I2V pipeline
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
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None]]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 226,
    ) -> Union[CogVideoXPoseAdaLNPipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for image-to-video generation with SMPL pose conditioning.

        Args:
            image (`PipelineImageInput`):
                The input image to condition the generation on. Must be an image, a list of images or a `torch.Tensor`.
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the video generation.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the video generation.
            pose_params (`torch.Tensor` or `np.ndarray`, *optional*):
                SMPL pose parameters to condition the generation.
            pose_params_mask (`torch.Tensor` or `np.ndarray`, *optional*):
                Mask to zero out SMPL conditioning for certain samples.
            height (`int`, *optional*):
                The height in pixels of the generated video.
            width (`int`, *optional*):
                The width in pixels of the generated video.
            num_frames (`int`, defaults to `49`):
                Number of frames to generate. Must be divisible by self.vae_scale_factor_temporal.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process.
            guidance_scale (`float`, *optional*, defaults to 6.0):
                Guidance scale for classifier-free guidance.
            use_dynamic_cfg (`bool`, *optional*, defaults to False):
                Whether to use dynamic classifier-free guidance.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of videos to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Controls the amount of noise that should be added to the sample.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of torch generators to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated video.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a dictionary.
            attention_kwargs (`dict`, *optional*):
                Arguments passed to the attention processor.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising step.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the callback function.
            max_sequence_length (`int`, defaults to `226`):
                Maximum sequence length in encoded prompt.

        Returns:
            [`CogVideoXPoseAdaLNPipelineOutput`] or `tuple`:
            [`CogVideoXPoseAdaLNPipelineOutput`] if `return_dict` is True, otherwise a tuple.
        """
        height = height or self.transformer.config.sample_height * self.vae_scale_factor_spatial
        width = width or self.transformer.config.sample_width * self.vae_scale_factor_spatial
        num_frames = num_frames or self.transformer.config.sample_frames

        num_videos_per_prompt = 1

        # 1. Check inputs
        self.check_inputs(
            image=image,
            prompt=prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pose_params=pose_params,
            pose_params_mask=pose_params_mask,
            num_frames=num_frames,
            latents=latents,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 2. Default call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0] if prompt_embeds is not None else 1

        device = self._execution_device

        # 3. Encode input prompt
        if prompt is not None:
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt,
                negative_prompt,
                guidance_scale > 1.0,
                num_videos_per_prompt=num_videos_per_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                max_sequence_length=max_sequence_length,
                device=device,
            )
            prompt_embeds = prompt_embeds.to(torch.bfloat16)
        else:
            prompt_embeds = self.empty_prompt_embeds.to(device)

        if guidance_scale > 1.0:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        # 4. Preprocess pose parameters
        pose_params, pose_params_mask = self.preprocess_pose_params(
            pose_params,
            pose_params_mask,
            height,
            width,
            num_frames,
        )

        # 5. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps
        )
        self._num_timesteps = len(timesteps)

        # 6. Prepare latents using parent class method
        # First preprocess the image
        image = self.video_processor.preprocess(image, height=height, width=width).to(
            device, dtype=prompt_embeds.dtype
        )
        
        # Calculate latent channels for I2V (half of transformer input channels)
        latent_channels = self.transformer.config.in_channels // 2
        
        # Use parent class prepare_latents method
        latents, image_latents = self.prepare_latents(
            image,
            batch_size * num_videos_per_prompt,
            latent_channels,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 7. Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 8. Create rotary embeds if required
        image_rotary_emb = (
            self._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
            if self.transformer.config.use_rotary_positional_embeddings
            else None
        )

        # 9. Create ofs embeds if required
        ofs_emb = None if self.transformer.config.ofs_embed_dim is None else latents.new_full((1,), fill_value=2.0)

        # 10. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        do_classifier_free_guidance = guidance_scale > 1.0

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            old_pred_original_sample = None
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # broadcast to batch dimension
                timestep = t.expand(latent_model_input.shape[0])

                # Concatenate image latents with noisy latents (I2V approach)
                latent_image_input = torch.cat([image_latents] * 2) if do_classifier_free_guidance else image_latents
                latent_model_input = torch.cat([latent_model_input, latent_image_input], dim=2)

                # Handle pose_params for classifier-free guidance
                if do_classifier_free_guidance and pose_params is not None:
                    # For classifier-free guidance, we need to duplicate pose_params
                    # The first half will be for unconditional generation (no SMPL effect)
                    # The second half will be for conditional generation (with SMPL effect)
                    pose_params_input = torch.cat([pose_params] * 2, dim=0)
                    
                    # Also duplicate pose_params_mask if provided
                    if pose_params_mask is not None:
                        pose_params_mask_input = torch.cat([pose_params_mask] * 2, dim=0)
                    else:
                        pose_params_mask_input = None
                else:
                    pose_params_input = pose_params
                    pose_params_mask_input = pose_params_mask

                # Predict the noise residual (with or without SMPL pose conditioning)
                transformer_kwargs = {
                    "hidden_states": latent_model_input,
                    "encoder_hidden_states": prompt_embeds,
                    "timestep": timestep,
                    "ofs": ofs_emb,
                    "image_rotary_emb": image_rotary_emb,
                    "attention_kwargs": attention_kwargs,
                    "return_dict": False,
                }
                
                # Add pose_params only if SMPL conditioning is enabled
                if pose_params_input is not None:
                    transformer_kwargs["pose_params"] = pose_params_input
                if pose_params_mask_input is not None:
                    transformer_kwargs["pose_params_mask"] = pose_params_mask_input
                
                noise_pred = self.transformer(**transformer_kwargs)[0]
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

        # 11. Decode latents
        if not output_type == "latent":
            # Discard any padding frames that were added for CogVideoX 1.5
            # This logic is handled in parent class prepare_latents
            video = self.decode_latents(latents)
            video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return CogVideoXPoseAdaLNPipelineOutput(frames=video)


@dataclass
class CogVideoXPoseAdaLNPerFramePipelineOutput(BaseOutput):
    frames: Union[List[PIL.Image.Image], np.ndarray]


class CogVideoXPoseAdaLNPerFramePipeline(CogVideoXImageToVideoPipeline):
    """
    Pipeline for image-to-video generation using CogVideoX with per-frame SMPL pose conditioning via AdaLN-zero.
    
    This pipeline extends CogVideoXImageToVideoPipeline to handle SMPL pose parameters through per-frame AdaLN-zero conditioning:
    1. SMPL pose parameters are sampled with stride 4 (e.g., frames 0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48)
    2. Each sampled frame's pose parameters are processed through a shared MLP independently
    3. Per-frame embeddings are created for AdaLN-zero conditioning
    4. The transformer (CogVideoXTransformer3DModelWithAdaLNPosePerFrame) handles the per-frame pose conditioning
    5. Inherits image-to-video functionality from CogVideoXImageToVideoPipeline
    """

    def __init__(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        vae: AutoencoderKLCogVideoX,
        transformer: CogVideoXTransformer3DModelWithAdaLNPosePerFrame,
        scheduler: Union[CogVideoXDDIMScheduler, CogVideoXDPMScheduler],
    ):
        super().__init__(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
        )
        
        # Initialize empty prompt embeddings for unconditional generation
        self.empty_prompt_embeds, _ = self.encode_prompt(
            prompt="",
            negative_prompt=None,
            do_classifier_free_guidance=False,
            num_videos_per_prompt=1,
            prompt_embeds=None,
        )
        self.empty_prompt_embeds = self.empty_prompt_embeds.to(dtype=torch.bfloat16)
        
        # Verify that the transformer has per-frame AdaLN pose conditioning support
        if not hasattr(self.transformer, 'smpl_pose_embedding'):
            raise ValueError(
                "Transformer must be an instance of CogVideoXTransformer3DModelWithAdaLNPosePerFrame "
                "to support per-frame AdaLN pose conditioning"
            )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path=None, 
                        base_model_name_or_path="THUDM/CogVideoX-5b-I2V", 
                        transformer=None, smpl_pose_dim=63, smpl_embed_dim=512, *args, **kwargs):
        """
        Load a CogVideoXPoseAdaLNPerFramePipeline from a saved directory or base model.
        
        This method loads all components (tokenizer, text_encoder, vae, transformer, scheduler)
        from the specified directory, or uses provided components if specified.
        
        Args:
            pretrained_model_name_or_path: Path to saved pipeline or base model
            base_model_name_or_path: Optional base model path for creating pose-conditioned pipeline
            transformer: Optional transformer to use (for validation with trained transformer)
            smpl_pose_dim: SMPL pose parameter dimension (default: 63)
            smpl_embed_dim: SMPL embedding dimension (default: 512)
        """
        # Check if this is a base model path (for creating pose-conditioned pipeline)
        if base_model_name_or_path is not None:
            print(f"🔧 Creating per-frame AdaLN pose-conditioned pipeline from base model: {base_model_name_or_path}")
            # Load the original CogVideoX ImageToVideo pipeline from base model
            original_pipeline = CogVideoXImageToVideoPipeline.from_pretrained(base_model_name_or_path, *args, **kwargs)
            
        if transformer is None:
            # Create or load per-frame AdaLN pose-conditioned transformer
            print(f"🔧 Creating/loading per-frame AdaLN pose-conditioned transformer with pose_dim={smpl_pose_dim}, embed_dim={smpl_embed_dim}")
            transformer = CogVideoXTransformer3DModelWithAdaLNPosePerFrame.from_pretrained(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                base_model_name_or_path=base_model_name_or_path,
                subfolder="transformer",
                smpl_pose_dim=smpl_pose_dim,
                smpl_embed_dim=smpl_embed_dim,
                torch_dtype=kwargs.get("torch_dtype", torch.bfloat16),
                revision=kwargs.get("revision", None),
                variant=kwargs.get("variant", None),
            )
        elif pretrained_model_name_or_path is not None:
            load_dtype = torch.bfloat16 if "5b" in base_model_name_or_path.lower() else torch.float16
            # Override with provided torch_dtype if specified
            if "torch_dtype" in kwargs:
                load_dtype = kwargs["torch_dtype"]
            print(f"🔧 Loading per-frame AdaLN pose-conditioned pipeline from {pretrained_model_name_or_path}")
            transformer = CogVideoXTransformer3DModelWithAdaLNPosePerFrame.from_pretrained(
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
            "pipeline_class": "CogVideoXPoseAdaLNPerFramePipeline",
            "version": "1.0.0",
        }
        with open(os.path.join(save_directory, "pipeline_config.json"), "w") as f:
            import json
            json.dump(pipeline_config, f)

    # Use parent class _prepare_rotary_positional_embeddings method

    def check_inputs(
        self,
        image,
        prompt,
        height,
        width,
        negative_prompt,
        callback_on_step_end_tensor_inputs,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pose_params=None,  # New parameter for SMPL pose
        pose_params_mask=None,  # New parameter to track None pose_params
        num_frames=None,
        latents=None,
    ):
        # Call parent check_inputs
        super().check_inputs(
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

        # Additional validation for SMPL pose parameters
        if pose_params is not None:
            if not isinstance(pose_params, (torch.Tensor, np.ndarray)):
                raise ValueError(f"pose_params must be a torch.Tensor or np.ndarray, got {type(pose_params)}")
            
            if isinstance(pose_params, np.ndarray):
                pose_params = torch.from_numpy(pose_params)
            
            # Validate pose_params shape: should be (batch_size, num_frames, 63)
            if pose_params.dim() != 3:
                raise ValueError(f"pose_params must have 3 dimensions (batch_size, num_frames, 63), got {pose_params.shape}")
            
            if pose_params.shape[-1] != 63:
                raise ValueError(f"pose_params last dimension must be 63 (SMPL pose parameters), got {pose_params.shape[-1]}")
        
        # Validate pose_params_mask if provided
        if pose_params_mask is not None:
            if not isinstance(pose_params_mask, (torch.Tensor, np.ndarray)):
                raise ValueError(f"pose_params_mask must be a torch.Tensor or np.ndarray, got {type(pose_params_mask)}")
            
            if isinstance(pose_params_mask, np.ndarray):
                pose_params_mask = torch.from_numpy(pose_params_mask)
            
            # Validate pose_params_mask shape: should be (batch_size,)
            if pose_params_mask.dim() != 1:
                raise ValueError(f"pose_params_mask must have 1 dimension (batch_size,), got {pose_params_mask.shape}")
            
            # Validate that pose_params_mask is boolean
            if pose_params_mask.dtype != torch.bool:
                raise ValueError(f"pose_params_mask must be boolean tensor, got {pose_params_mask.dtype}")
            
            # If pose_params is provided, validate that mask length matches batch size
            if pose_params is not None and pose_params_mask.shape[0] != pose_params.shape[0]:
                raise ValueError(f"pose_params_mask batch size ({pose_params_mask.shape[0]}) must match pose_params batch size ({pose_params.shape[0]})")

        if num_frames is None:
            raise ValueError("`num_frames` is required.")

    def preprocess_pose_params(
        self,
        pose_params: Optional[Union[torch.Tensor, np.ndarray]] = None,
        pose_params_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
        height: int = 480,
        width: int = 720,
        num_frames: int = 48,
    ):
        """Preprocess SMPL pose parameters."""
        if pose_params is not None:
            if isinstance(pose_params, np.ndarray):
                pose_params = torch.from_numpy(pose_params)
            
            # Ensure pose_params is on the correct device and dtype
            pose_params = pose_params.to(device=self._execution_device, dtype=self.transformer.dtype)
        
        if pose_params_mask is not None:
            if isinstance(pose_params_mask, np.ndarray):
                pose_params_mask = torch.from_numpy(pose_params_mask)
            
            # Ensure pose_params_mask is on the correct device and dtype
            pose_params_mask = pose_params_mask.to(device=self._execution_device, dtype=self.transformer.dtype)
        
        return pose_params, pose_params_mask

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

    # Use parent class prepare_latents method which handles image-to-video latents

    @torch.no_grad()
    def __call__(
        self,
        image: PipelineImageInput,  # Required for image-to-video
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        pose_params: Optional[Union[torch.Tensor, np.ndarray]] = None,  # New parameter for SMPL pose
        pose_params_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,  # New parameter to track None pose_params
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: int = 49,  # Default for I2V pipeline
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
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None]]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 226,
    ) -> Union[CogVideoXPoseAdaLNPerFramePipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for image-to-video generation with per-frame SMPL pose conditioning.

        Args:
            image (`PipelineImageInput`):
                The input image to condition the generation on. Must be an image, a list of images or a `torch.Tensor`.
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the video generation.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the video generation.
            pose_params (`torch.Tensor` or `np.ndarray`, *optional*):
                SMPL pose parameters to condition the generation.
            pose_params_mask (`torch.Tensor` or `np.ndarray`, *optional*):
                Mask to zero out SMPL conditioning for certain samples.
            height (`int`, *optional*):
                The height in pixels of the generated video.
            width (`int`, *optional*):
                The width in pixels of the generated video.
            num_frames (`int`, defaults to `49`):
                Number of frames to generate. Must be divisible by self.vae_scale_factor_temporal.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process.
            guidance_scale (`float`, *optional*, defaults to 6.0):
                Guidance scale for classifier-free guidance.
            use_dynamic_cfg (`bool`, *optional*, defaults to False):
                Whether to use dynamic classifier-free guidance.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of videos to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Controls the amount of noise that should be added to the sample.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of torch generators to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated video.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a dictionary.
            attention_kwargs (`dict`, *optional*):
                Arguments passed to the attention processor.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising step.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the callback function.
            max_sequence_length (`int`, defaults to `226`):
                Maximum sequence length in encoded prompt.

        Returns:
            [`CogVideoXPoseAdaLNPerFramePipelineOutput`] or `tuple`:
            [`CogVideoXPoseAdaLNPerFramePipelineOutput`] if `return_dict` is True, otherwise a tuple.
        """
        height = height or self.transformer.config.sample_height * self.vae_scale_factor_spatial
        width = width or self.transformer.config.sample_width * self.vae_scale_factor_spatial
        num_frames = num_frames or self.transformer.config.sample_frames

        num_videos_per_prompt = 1

        # 1. Check inputs
        self.check_inputs(
            image=image,
            prompt=prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pose_params=pose_params,
            pose_params_mask=pose_params_mask,
            num_frames=num_frames,
            latents=latents,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 2. Default call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0] if prompt_embeds is not None else 1

        device = self._execution_device

        # 3. Encode input prompt
        if prompt is not None:
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt,
                negative_prompt,
                guidance_scale > 1.0,
                num_videos_per_prompt=num_videos_per_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                max_sequence_length=max_sequence_length,
                device=device,
            )
            prompt_embeds = prompt_embeds.to(torch.bfloat16)
        else:
            prompt_embeds = self.empty_prompt_embeds.to(device)

        if guidance_scale > 1.0:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        # 4. Preprocess pose parameters
        pose_params, pose_params_mask = self.preprocess_pose_params(
            pose_params,
            pose_params_mask,
            height,
            width,
            num_frames,
        )

        # 5. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps
        )
        self._num_timesteps = len(timesteps)

        # 6. Prepare latents using parent class method
        # First preprocess the image
        image = self.video_processor.preprocess(image, height=height, width=width).to(
            device, dtype=prompt_embeds.dtype
        )
        
        # Calculate latent channels for I2V (half of transformer input channels)
        latent_channels = self.transformer.config.in_channels // 2
        
        # Use parent class prepare_latents method
        latents, image_latents = self.prepare_latents(
            image,
            batch_size * num_videos_per_prompt,
            latent_channels,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 7. Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 8. Create rotary embeds if required
        image_rotary_emb = (
            self._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
            if self.transformer.config.use_rotary_positional_embeddings
            else None
        )

        # 9. Create ofs embeds if required
        ofs_emb = None if self.transformer.config.ofs_embed_dim is None else latents.new_full((1,), fill_value=2.0)

        # 10. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        do_classifier_free_guidance = guidance_scale > 1.0

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            old_pred_original_sample = None
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # broadcast to batch dimension
                timestep = t.expand(latent_model_input.shape[0])

                # Concatenate image latents with noisy latents (I2V approach)
                latent_image_input = torch.cat([image_latents] * 2) if do_classifier_free_guidance else image_latents
                latent_model_input = torch.cat([latent_model_input, latent_image_input], dim=2)

                # Handle pose_params for classifier-free guidance
                if do_classifier_free_guidance and pose_params is not None:
                    # For classifier-free guidance, we need to duplicate pose_params
                    # The first half will be for unconditional generation (no SMPL effect)
                    # The second half will be for conditional generation (with SMPL effect)
                    pose_params_input = torch.cat([pose_params] * 2, dim=0)
                    
                    # Also duplicate pose_params_mask if provided
                    if pose_params_mask is not None:
                        pose_params_mask_input = torch.cat([pose_params_mask] * 2, dim=0)
                    else:
                        pose_params_mask_input = None
                else:
                    pose_params_input = pose_params
                    pose_params_mask_input = pose_params_mask

                # Predict the noise residual (with or without per-frame SMPL pose conditioning)
                transformer_kwargs = {
                    "hidden_states": latent_model_input,
                    "encoder_hidden_states": prompt_embeds,
                    "timestep": timestep,
                    "ofs": ofs_emb,
                    "image_rotary_emb": image_rotary_emb,
                    "attention_kwargs": attention_kwargs,
                    "return_dict": False,
                }
                
                # Add pose_params only if SMPL conditioning is enabled
                if pose_params_input is not None:
                    transformer_kwargs["pose_params"] = pose_params_input
                if pose_params_mask_input is not None:
                    transformer_kwargs["pose_params_mask"] = pose_params_mask_input
                
                noise_pred = self.transformer(**transformer_kwargs)[0]
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

        # 11. Decode latents
        if not output_type == "latent":
            # Discard any padding frames that were added for CogVideoX 1.5
            # This logic is handled in parent class prepare_latents
            video = self.decode_latents(latents)
            video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return CogVideoXPoseAdaLNPerFramePipelineOutput(frames=video)
