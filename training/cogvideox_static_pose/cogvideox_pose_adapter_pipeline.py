import inspect
import math
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL
import torch
import torch.nn as nn
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDDIMScheduler,
    CogVideoXDPMScheduler,
    CogVideoXPipeline,
    CogVideoXTransformer3DModel,
)
from diffusers.image_processor import PipelineImageInput
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import BaseOutput, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer

# Import output class from pose pipeline
from .cogvideox_pose_pipeline import CogVideoXPosePipelineOutput

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

class LatentCondAdapter3D(nn.Module):
    """
    Residual conditioning adapter for DiT/CogVideoX latents.
    - Expects latents and condition tensors in [B, T, C, H, W].
    - Uses 1x1x1 Conv3d per condition (zero-init) + learnable gates (zero-init).
    - Returns either the fused latents (latents + residual) or the residual only.
    - Supports selective freezing of hand/static branches.
    """
    def __init__(
        self,
        channels: int,
        *,
        shared: bool = False,
        dropout_p: float = 0.0,
        norm: str = "none",          # "none" | "group"
        groups: int = 32,            # used if norm="group"
        return_residual: bool = False,
        freeze_hand: bool = False,   # Freeze hand branch parameters
        freeze_static: bool = False, # Freeze static branch parameters
    ):
        super().__init__()
        self.channels = channels
        self.shared = shared
        self.dropout_p = float(dropout_p)
        self.return_residual = return_residual
        self.freeze_hand = freeze_hand
        self.freeze_static = freeze_static

        def make_path():
            conv = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
            nn.init.zeros_(conv.weight)
            return conv

        if shared:
            self.shared_conv = make_path()
            self.hand_conv = None
            self.static_conv = None
        else:
            self.shared_conv = None
            self.hand_conv = make_path()
            self.static_conv = make_path()

        # learnable scalar gates (start at 0 → preserves base behavior)
        self.scale_hand = nn.Parameter(torch.zeros(1))
        self.scale_static = nn.Parameter(torch.zeros(1))

        if norm == "group":
            g = min(groups, channels)
            self.norm = nn.GroupNorm(g, channels, affine=True)
        else:
            self.norm = nn.Identity()

        self.dropout = nn.Dropout3d(p=self.dropout_p) if self.dropout_p > 0 else nn.Identity()
        
        # Apply freezing after initialization
        self._apply_freezing()

    def _apply_freezing(self):
        """Apply freezing to specified branches."""
        if self.freeze_hand:
            self._freeze_branch("hand")
        if self.freeze_static:
            self._freeze_branch("static")
    
    def _freeze_branch(self, branch: str):
        """Freeze parameters for a specific branch."""
        if branch == "hand":
            if self.hand_conv is not None:
                for param in self.hand_conv.parameters():
                    param.requires_grad = False
            self.scale_hand.requires_grad = False
            print(f"🔒 Frozen hand branch parameters")
        elif branch == "static":
            if self.static_conv is not None:
                for param in self.static_conv.parameters():
                    param.requires_grad = False
            self.scale_static.requires_grad = False
            print(f"🔒 Frozen static branch parameters")
        else:
            raise ValueError(f"Unknown branch: {branch}")
    
    def _unfreeze_branch(self, branch: str):
        """Unfreeze parameters for a specific branch."""
        if branch == "hand":
            if self.hand_conv is not None:
                for param in self.hand_conv.parameters():
                    param.requires_grad = True
            self.scale_hand.requires_grad = True
            print(f"🔓 Unfrozen hand branch parameters")
        elif branch == "static":
            if self.static_conv is not None:
                for param in self.static_conv.parameters():
                    param.requires_grad = True
            self.scale_static.requires_grad = True
            print(f"🔓 Unfrozen static branch parameters")
        else:
            raise ValueError(f"Unknown branch: {branch}")
    
    def freeze_hand_branch(self):
        """Freeze hand branch parameters."""
        self.freeze_hand = True
        self._freeze_branch("hand")
    
    def freeze_static_branch(self):
        """Freeze static branch parameters."""
        self.freeze_static = True
        self._freeze_branch("static")
    
    def unfreeze_hand_branch(self):
        """Unfreeze hand branch parameters."""
        self.freeze_hand = False
        self._unfreeze_branch("hand")
    
    def unfreeze_static_branch(self):
        """Unfreeze static branch parameters."""
        self.freeze_static = False
        self._unfreeze_branch("static")
    
    def get_freeze_status(self) -> Dict[str, bool]:
        """Get current freeze status of branches."""
        return {
            "hand": self.freeze_hand,
            "static": self.freeze_static
        }

    @staticmethod
    def _to_c3d(x: torch.Tensor) -> torch.Tensor:
        # [B,T,C,H,W] → [B,C,T,H,W]
        return x.permute(0, 2, 1, 3, 4)

    @staticmethod
    def _to_tfirst(x: torch.Tensor) -> torch.Tensor:
        # [B,C,T,H,W] → [B,T,C,H,W]
        return x.permute(0, 2, 1, 3, 4)

    def _apply_path(self, x_c3d: torch.Tensor, which: str) -> torch.Tensor:
        if self.shared_conv is not None:
            out = self.shared_conv(x_c3d)
        else:
            conv = self.hand_conv if which == "hand" else self.static_conv
            if conv is None:
                raise RuntimeError("Invalid conv path")
            out = conv(x_c3d)
        return out

    def forward(
        self,
        latents: torch.Tensor,               # [B,T,C,H,W]
        hand: torch.Tensor = None,           # [B,T,C,H,W] or None
        static: torch.Tensor = None,         # [B,T,C,H,W] or None
        hand_mask: torch.Tensor = None,      # [B,T,1|C,H,W] in [0,1], optional
        static_mask: torch.Tensor = None,    # [B,T,1|C,H,W] in [0,1], optional
        return_residual: bool = None
    ) -> torch.Tensor:
        """
        If return_residual is True → returns residual; else returns fused latents = latents + residual.
        """
        assert latents.dim() == 5, "latents must be [B,T,C,H,W]"
        B, T, C, H, W = latents.shape
        device = latents.device
        dtype = latents.dtype

        residual = torch.zeros_like(latents)

        if hand is not None:
            assert hand.shape == (B, T, C, H, W), "hand shape mismatch"
            h = self._to_c3d(hand)
            h = self._apply_path(h, "hand")
            h = self.dropout(h)
            h = self._to_tfirst(h)
            if hand_mask is not None:
                # broadcast to [B,T,C,H,W]
                if hand_mask.dim() == 5 and hand_mask.shape[2] in (1, C):
                    pass
                elif hand_mask.dim() == 4:
                    hand_mask = hand_mask.unsqueeze(2)
                else:
                    raise ValueError("hand_mask must be [B,T,1|C,H,W] or [B,T,H,W]")
                hand_mask = hand_mask.to(device=device, dtype=dtype)
                h = h * hand_mask
            residual = residual + self.scale_hand * h

        if static is not None:
            assert static.shape == (B, T, C, H, W), "static shape mismatch"
            s = self._to_c3d(static)
            s = self._apply_path(s, "static")
            s = self.dropout(s)
            s = self._to_tfirst(s)
            if static_mask is not None:
                if static_mask.dim() == 5 and static_mask.shape[2] in (1, C):
                    pass
                elif static_mask.dim() == 4:
                    static_mask = static_mask.unsqueeze(2)
                else:
                    raise ValueError("static_mask must be [B,T,1|C,H,W] or [B,T,H,W]")
                static_mask = static_mask.to(device=device, dtype=dtype)
                s = s * static_mask
            residual = residual + self.scale_static * s

        # light normalization of residual to roughly match latent scale (optional but stabilizing)
        residual = self._to_c3d(residual)
        residual = self.norm(residual)
        residual = self._to_tfirst(residual)

        if return_residual if return_residual is not None else self.return_residual:
            return residual
        return latents + residual


@dataclass
class CogVideoXPoseAdapterPipelineOutput(BaseOutput):
    frames: Union[List[PIL.Image.Image], np.ndarray]


class CogVideoXPoseAdapterPipeline(CogVideoXPipeline):
    """
    Pipeline for text-to-video generation using CogVideoX with egocentric hand mesh and static scene conditions.
    
    This pipeline extends CogVideoXPipeline to handle two types of per-frame conditions:
    1. Egocentric hand mesh images
    2. Static scene frames
    
    The conditions are processed through a LatentCondAdapter3D that adds residual conditioning
    to the noisy latents. The adapter supports selective freezing of hand/static branches
    for fine-grained control during training.
    """

    def __init__(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        vae: AutoencoderKLCogVideoX,
        transformer: CogVideoXTransformer3DModel,
        scheduler: Union[CogVideoXDDIMScheduler, CogVideoXDPMScheduler],
        freeze_hand_branch: bool = False,
        freeze_static_branch: bool = False,
        adapter_norm: str = "group",
        adapter_groups: int = 32,
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
        
        # Initialize conditional adapter with freeze options
        self.cond_adapter = LatentCondAdapter3D(
            self.vae.config.latent_channels, 
            norm=adapter_norm, 
            groups=adapter_groups,
            freeze_hand=freeze_hand_branch,
            freeze_static=freeze_static_branch
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path=None, 
                        base_model_name_or_path="THUDM/CogVideoX-5b", 
                        transformer=None, 
                        freeze_hand_branch: bool = False,
                        freeze_static_branch: bool = False,
                        adapter_norm: str = "group",
                        adapter_groups: int = 32,
                        *args, **kwargs):
        """
        Load a CogVideoXPosePipeline with base components and optionally a trained transformer.
        
        This method loads base components from base_model_name_or_path and optionally
        loads a trained transformer from pretrained_model_name_or_path.
        
        Args:
            pretrained_model_name_or_path: Path to trained transformer checkpoint (optional)
            base_model_name_or_path: Base model path for tokenizer, text_encoder, vae, scheduler
            transformer: Optional transformer to use (for validation with trained transformer)
            freeze_hand_branch: Whether to freeze hand branch parameters in the adapter
            freeze_static_branch: Whether to freeze static branch parameters in the adapter
            adapter_norm: Normalization type for adapter ("none" or "group")
            adapter_groups: Number of groups for group normalization
        """
        print(f"🔧 Loading base components from: {base_model_name_or_path}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name_or_path,
            subfolder="tokenizer",
            revision=kwargs.get("revision", None),
        )
        
        # Load text encoder
        text_encoder = T5EncoderModel.from_pretrained(
            base_model_name_or_path,
            subfolder="text_encoder",
            revision=kwargs.get("revision", None),
        )
        
        # Load VAE
        vae = AutoencoderKLCogVideoX.from_pretrained(
            base_model_name_or_path,
            subfolder="vae",
            revision=kwargs.get("revision", None),
            variant=kwargs.get("variant", None),
        )
        
        # Load scheduler
        scheduler = CogVideoXDPMScheduler.from_pretrained(
            base_model_name_or_path, 
            subfolder="scheduler"
        )
        
        # Load transformer
        if transformer is None:
            if pretrained_model_name_or_path is not None:
                # Load trained transformer from checkpoint
                print(f"📥 Loading trained transformer from: {pretrained_model_name_or_path}")
                # Determine load dtype based on model size
                load_dtype = torch.bfloat16 if "5b" in base_model_name_or_path.lower() else torch.float16
                # Override with provided torch_dtype if specified
                if "torch_dtype" in kwargs:
                    load_dtype = kwargs["torch_dtype"]
                
                transformer = CogVideoXTransformer3DModel.from_pretrained(
                    pretrained_model_name_or_path,
                    subfolder="transformer",
                    torch_dtype=load_dtype,
                    revision=kwargs.get("revision", None),
                    variant=kwargs.get("variant", None),
                )
            else:
                # Load base transformer from base model
                print(f"📥 Loading base transformer from: {base_model_name_or_path}")
                load_dtype = torch.bfloat16 if "5b" in base_model_name_or_path.lower() else torch.float16
                # Override with provided torch_dtype if specified
                if "torch_dtype" in kwargs:
                    load_dtype = kwargs["torch_dtype"]
                
                transformer = CogVideoXTransformer3DModel.from_pretrained(
                    base_model_name_or_path,
                    subfolder="transformer",
                    torch_dtype=load_dtype,
                    revision=kwargs.get("revision", None),
                    variant=kwargs.get("variant", None),
                )

        # Create our custom pipeline with base components + trained transformer
        pipeline = cls(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
            freeze_hand_branch=freeze_hand_branch,
            freeze_static_branch=freeze_static_branch,
            adapter_norm=adapter_norm,
            adapter_groups=adapter_groups,
        )
        
        return pipeline

    # Example usage:
    # pipeline = CogVideoXPoseAdapterPipeline.from_pretrained(
    #     base_model_name_or_path="THUDM/CogVideoX-5b",
    #     freeze_hand_branch=True,    # Freeze hand branch during training
    #     freeze_static_branch=False, # Keep static branch trainable
    # )
    # 
    # # Check freeze status
    # pipeline.print_adapter_freeze_status()
    # 
    # # Dynamically change freeze status during training
    # pipeline.unfreeze_adapter_hand_branch()  # Unfreeze hand branch
    # pipeline.freeze_adapter_static_branch()  # Freeze static branch

    def get_adapter_freeze_status(self) -> Dict[str, bool]:
        """Get current freeze status of adapter branches."""
        return self.cond_adapter.get_freeze_status()
    
    def freeze_adapter_hand_branch(self):
        """Freeze hand branch parameters in the adapter."""
        self.cond_adapter.freeze_hand_branch()
    
    def freeze_adapter_static_branch(self):
        """Freeze static branch parameters in the adapter."""
        self.cond_adapter.freeze_static_branch()
    
    def unfreeze_adapter_hand_branch(self):
        """Unfreeze hand branch parameters in the adapter."""
        self.cond_adapter.unfreeze_hand_branch()
    
    def unfreeze_adapter_static_branch(self):
        """Unfreeze static branch parameters in the adapter."""
        self.cond_adapter.unfreeze_static_branch()
    
    def print_adapter_freeze_status(self):
        """Print current freeze status of adapter branches."""
        status = self.get_adapter_freeze_status()
        print("🔒 Adapter Freeze Status:")
        print(f"  Hand branch: {'Frozen' if status['hand'] else 'Unfrozen'}")
        print(f"  Static branch: {'Frozen' if status['static'] else 'Unfrozen'}")

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
            "pipeline_class": "CogVideoXPosePipeline",
            "version": "1.0.0",
        }
        with open(os.path.join(save_directory, "pipeline_config.json"), "w") as f:
            import json
            json.dump(pipeline_config, f)

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
                device=device,
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
                device=device,
            )

        return freqs_cos, freqs_sin

    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt,
        callback_on_step_end_tensor_inputs,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        hand_videos=None,
        static_videos=None,
        num_frames=None,
    ):
        # Call parent check_inputs
        super().check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            callback_on_step_end_tensor_inputs,
            prompt_embeds,
            negative_prompt_embeds,
        )

        # Check hand mesh images
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

        if num_frames is None:
            raise ValueError("`num_frames` is required.")

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
        hand_videos: Optional[torch.Tensor] = None,
        static_videos: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        num_frames: int = 48,
        height: int = 480,
        width: int = 720,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
    ):
        """Prepare latents with condition latents concatenated."""
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # Calculate latent dimensions
        latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial

        # For CogVideoX 1.5, the latent frames should be padded to make it divisible by patch_size_t
        patch_size_t = self.transformer.config.patch_size_t
        additional_frames = 0
        if patch_size_t is not None and latent_frames % patch_size_t != 0:
            additional_frames = patch_size_t - latent_frames % patch_size_t
            num_frames += additional_frames * self.vae_scale_factor_temporal

        # Base shape for noisy latents
        base_shape = (
            batch_size,
            latent_frames,
            self.vae.config.latent_channels,
            latent_height,
            latent_width,
        )

        # Encode hand videos if provided
        hand_videos_latents = None
        if hand_videos is not None:
            # Check if hand_videos are already latents (when load_tensors=True)
            if hand_videos.ndim == 4:
                hand_videos = hand_videos.unsqueeze(0)

            if hand_videos.shape[2] == self.vae.config.latent_channels:
                # Already latents, just ensure correct format
                hand_videos_latents = hand_videos.to(dtype=dtype, device=device)
            else:
                # Raw videos, need to encode
                hand_videos = hand_videos.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
                
                # Encode entire batch of videos at once
                latent_dist = self.vae.encode(hand_videos)
                hand_videos_latents = retrieve_latents(latent_dist, generator)
                hand_videos_latents = hand_videos_latents.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
                
                if not self.vae.config.invert_scale_latents:
                    hand_videos_latents = self.vae_scaling_factor_image * hand_videos_latents
                else:
                    hand_videos_latents = 1 / self.vae_scaling_factor_image * hand_videos_latents

        # Encode static scene frames if provided
        static_videos_latents = None
        if static_videos is not None:
            # Check if static_videos are already latents (when load_tensors=True)
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
                static_videos_latents = static_videos_latents.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
                
                if not self.vae.config.invert_scale_latents:
                    static_videos_latents = self.vae_scaling_factor_image * static_videos_latents
                else:
                    static_videos_latents = 1 / self.vae_scaling_factor_image * static_videos_latents

        # Generate noisy latents
        latents = randn_tensor(base_shape, device=device, generator=generator, dtype=dtype)
        latents = latents * self.scheduler.init_noise_sigma

        return latents, hand_videos_latents, static_videos_latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        hand_videos: Optional[Union[torch.Tensor, np.ndarray, List[PIL.Image.Image]]] = None,
        static_videos: Optional[Union[torch.Tensor, np.ndarray, List[PIL.Image.Image]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
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
    ) -> Union[CogVideoXPosePipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the video generation.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the video generation.
            hand_videos (`torch.Tensor` or `np.ndarray` or `List[PIL.Image.Image]`, *optional*):
                Egocentric hand mesh images to condition the generation.
            static_videos (`torch.Tensor` or `np.ndarray` or `List[PIL.Image.Image]`, *optional*):
                Static scene frames to condition the generation.
            height (`int`, *optional*):
                The height in pixels of the generated video.
            width (`int`, *optional*):
                The width in pixels of the generated video.
            num_frames (`int`, *optional*):
                Number of frames to generate.
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
            [`CogVideoXPosePipelineOutput`] or `tuple`:
            [`CogVideoXPosePipelineOutput`] if `return_dict` is True, otherwise a tuple.
        """
        height = height or self.transformer.config.sample_height * self.vae_scale_factor_spatial
        width = width or self.transformer.config.sample_width * self.vae_scale_factor_spatial
        num_frames = num_frames or self.transformer.config.sample_frames

        num_videos_per_prompt = 1

        # 1. Check inputs
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            callback_on_step_end_tensor_inputs,
            prompt_embeds,
            negative_prompt_embeds,
            hand_videos,
            static_videos,
            num_frames,
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

        # 4. Preprocess conditions
        hand_videos, static_videos = self.preprocess_conditions(
            hand_videos,
            static_videos,
            height,
            width,
            num_frames,
        )

        # 5. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps
        )
        self._num_timesteps = len(timesteps)

        # 6. Prepare latents
        latents, hand_videos_latents, static_videos_latents = self.prepare_latents(
            hand_videos,
            static_videos,
            batch_size * num_videos_per_prompt,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
        )

        # 7. Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 8. Create rotary embeds if required
        image_rotary_emb = (
            self._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
            if self.transformer.config.use_rotary_positional_embeddings
            else None
        )

        # 9. Denoising loop
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

                # Apply conditional adapter to add residual conditioning
                if do_classifier_free_guidance:
                    # For classifier-free guidance, we need to handle both unconditional and conditional branches
                    uncond_latents, cond_latents = latent_model_input.chunk(2)
                    
                    # Apply adapter to conditional branch
                    if hand_videos_latents is not None or static_videos_latents is not None:
                        cond_latents = self.cond_adapter(
                            cond_latents, 
                            hand=hand_videos_latents, 
                            static=static_videos_latents
                        )
                    
                    # Unconditional branch gets no conditioning
                    latent_model_input = torch.cat([uncond_latents, cond_latents], dim=0)
                else:
                    # Single branch - apply adapter directly
                    if hand_videos_latents is not None or static_videos_latents is not None:
                        latent_model_input = self.cond_adapter(
                            latent_model_input, 
                            hand=hand_videos_latents, 
                            static=static_videos_latents
                        )

                # broadcast to batch dimension
                timestep = t.expand(latent_model_input.shape[0])

                # predict noise model_output
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
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

        # 10. Decode latents
        if not output_type == "latent":
            video = self.decode_latents(latents)
            video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return CogVideoXPosePipelineOutput(frames=video)