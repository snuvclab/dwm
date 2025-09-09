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

import os
import torch
import torch.nn as nn
from typing import Optional, Union, Tuple, Dict, Any
from diffusers.models import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import is_torch_version
from diffusers.models.embeddings import CogVideoXPatchEmbed, TimestepEmbedding, Timesteps
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.attention_processor import AttentionProcessor
from diffusers.models.normalization import AdaLayerNorm, CogVideoXLayerNormZero

from diffusers import CogVideoXTransformer3DModel

class CogVideoXFunTransformer3DModel(CogVideoXTransformer3DModel):
    """
    VideoX-Fun Transformer model for video-like data.
    
    This class extends the standard CogVideoXTransformer3DModel to support VideoX-Fun specific features:
    1. Handles `inpaint_latents` and `control_latents` inputs
    2. Supports single frame processing with padding
    3. VideoX-Fun specific normalization logic
    """
    
    def __init__(self, *args, **kwargs):
        # Accept VideoX-Fun specific parameters but don't use them
        self.add_noise_in_inpaint_model = kwargs.pop("add_noise_in_inpaint_model", False)
        super().__init__(*args, **kwargs)
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path=None,
        base_model_name_or_path="alibaba-pai/CogVideoX-Fun-V1.1-5b-InP",
        subfolder="transformer",
        **kwargs
    ):
        """
        Load a CogVideoXFunTransformer3DModel from a pretrained model.

        Args:
            pretrained_model_name_or_path (str): Fine-tuned checkpoint path
            base_model_name_or_path (str): Base CogVideoX-Fun model path
        """
        if pretrained_model_name_or_path is not None:
            # === Load fine-tuned checkpoint ===
            print(f"📥 Loading fine-tuned VideoX-Fun transformer: {pretrained_model_name_or_path}")

            # 1) Create child class structure first
            config_path = os.path.join(pretrained_model_name_or_path, subfolder, "config.json")
            if os.path.exists(config_path):
                import json
                with open(config_path, "r") as f:
                    config = json.load(f)
                model = cls(**config)
            else:
                # Use base config
                model = cls()

            # 2) Load checkpoint state_dict directly
            state_dict_path = os.path.join(pretrained_model_name_or_path, subfolder, "diffusion_pytorch_model.safetensors")
            if not os.path.exists(state_dict_path):
                # Try alternative paths
                state_dict_path = os.path.join(pretrained_model_name_or_path, subfolder, "pytorch_model.bin")
                if not os.path.exists(state_dict_path):
                    raise FileNotFoundError(f"No model file found in {pretrained_model_name_or_path}")
            
            if state_dict_path.endswith(".safetensors"):
                from safetensors.torch import load_file
                state_dict = load_file(state_dict_path)
            else:
                state_dict = torch.load(state_dict_path, map_location="cpu")

            # 3) Load with strict=False for flexibility
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"⚠️ Missing keys (new layers, need initialization during training): {missing}")
            if unexpected:
                print(f"⚠️ Unexpected keys (replaced layers, etc.): {unexpected}")

            return model

        else:
            # === Initialize from base model ===
            print(f"🔧 Initializing from base model: {base_model_name_or_path}")
            # First load the base CogVideoX transformer
            base_model = CogVideoXTransformer3DModel.from_pretrained(
                base_model_name_or_path,
                subfolder=subfolder,
                torch_dtype=kwargs.get("torch_dtype", torch.bfloat16),
                revision=kwargs.get("revision", None),
                variant=kwargs.get("variant", None),
            )
            
            # Then create CogVideoXFun transformer and copy state_dict
            fun_model = CogVideoXFunTransformer3DModel(**base_model.config)
            fun_model.load_state_dict(base_model.state_dict())
            
            return fun_model
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
        inpaint_latents: Optional[torch.Tensor] = None,
        control_latents: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_dict: bool = True,
        **kwargs
    ):
        """
        Forward pass with VideoX-Fun specific inputs.
        
        Args:
            hidden_states: Input tensor with shape [batch, frames, channels, height, width]
            encoder_hidden_states: Text embeddings
            timestep: Timestep tensor
            timestep_cond: Timestep condition tensor
            inpaint_latents: Inpainting latents (VideoX-Fun specific)
            control_latents: Control latents (VideoX-Fun specific)
            image_rotary_emb: Rotary embeddings
            return_dict: Whether to return a dict
            **kwargs: Additional arguments
        """
        batch_size, num_frames, channels, height, width = hidden_states.shape
        
        # Handle single frame case (VideoX-Fun specific)
        if num_frames == 1 and self.patch_size_t is not None:
            hidden_states = torch.cat([hidden_states, torch.zeros_like(hidden_states)], dim=1)
            if inpaint_latents is not None:
                inpaint_latents = torch.concat([inpaint_latents, torch.zeros_like(inpaint_latents)], dim=1)
            if control_latents is not None:
                control_latents = torch.concat([control_latents, torch.zeros_like(control_latents)], dim=1)
            local_num_frames = num_frames + 1
        else:
            local_num_frames = num_frames
        
        # Time embedding
        timesteps = timestep
        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)
        
        # Patch embedding with VideoX-Fun specific concatenation
        if inpaint_latents is not None:
            hidden_states = torch.concat([hidden_states, inpaint_latents], 2)
        if control_latents is not None:
            hidden_states = torch.concat([hidden_states, control_latents], 2)
        
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)
        
        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]
        
        # Transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                
                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    emb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=emb,
                    image_rotary_emb=image_rotary_emb,
                )
        
        # Final normalization (VideoX-Fun specific)
        if not self.config.use_rotary_positional_embeddings:
            # CogVideoX-2B
            hidden_states = self.norm_final(hidden_states)
        else:
            # CogVideoX-5B
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            hidden_states = self.norm_final(hidden_states)
            hidden_states = hidden_states[:, text_seq_length:]
        
        # Final block
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)
        
        # Unpatchify
        p = self.config.patch_size
        p_t = self.config.patch_size_t
        
        if p_t is None:
            output = hidden_states.reshape(batch_size, local_num_frames, height // p, width // p, -1, p, p)
            output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)
        else:
            output = hidden_states.reshape(
                batch_size, (local_num_frames + p_t - 1) // p_t, height // p, width // p, -1, p_t, p, p
            )
            output = output.permute(0, 1, 5, 4, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(1, 2)
        
        # Handle single frame case (VideoX-Fun specific)
        if num_frames == 1:
            output = output[:, :num_frames, :]
        
        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)


class CogVideoXFunTransformer3DModelWithConcat(CogVideoXFunTransformer3DModel):
    """
    VideoX-Fun Transformer with conditional input support via channel concatenation.
    
    This class extends CogVideoXFunTransformer3DModel to support conditional inputs
    by concatenating condition channels directly to the input channels.
    
    Key differences from CogVideoXFunTransformer3DModel:
    1. Supports `add_noise_in_inpaint_model` parameter (VideoX-Fun specific)
    2. Handles `inpaint_latents` and `control_latents` in forward pass
    3. Extends input channels to accommodate condition channels
    """
    
    def __init__(self, *args, condition_channels: int = 0, **kwargs):
        # Store original in_channels before modification
        original_in_channels = kwargs.get("in_channels", 16)
        self.original_in_channels = original_in_channels
        
        self.add_noise_in_inpaint_model = kwargs.pop("add_noise_in_inpaint_model", False)
        
        super().__init__(*args, **kwargs)
        
        # Setup conditional channels if specified
        if condition_channels > 0:
            self._setup_condition_channels(condition_channels)
            self.condition_channels = condition_channels
        else:
            self.condition_channels = 0
    
    def _setup_condition_channels(self, condition_channels: int, original_proj: Optional[nn.Conv2d] = None):
        """Extend the transformer to handle conditional input channels.
        
        If `original_proj` is provided, use its weights as the source (for base→finetune case).
        Otherwise fall back to self.patch_embed.proj (for reload case).
        """
        if original_proj is None:
            original_proj = self.patch_embed.proj

        original_in_channels = original_proj.in_channels
        new_in_channels = original_in_channels + condition_channels

        print(f"Extending transformer input channels for concat approach:")
        print(f"  Original channels: {original_in_channels}")
        print(f"  Condition channels: {condition_channels}")
        print(f"  Total channels: {new_in_channels}")

        new_proj = nn.Conv2d(
            in_channels=new_in_channels,
            out_channels=original_proj.out_channels,
            kernel_size=original_proj.kernel_size,
            stride=original_proj.stride,
            padding=original_proj.padding,
            bias=original_proj.bias is not None,
        )

        with torch.no_grad():
            # copy pretrained weights into the original channels
            new_proj.weight[:, :original_in_channels] = original_proj.weight
            # init condition channels to zeros
            new_proj.weight[:, original_in_channels:] = 0.0
            if original_proj.bias is not None:
                new_proj.bias.data = original_proj.bias.data

        self.patch_embed.proj = new_proj
        self.register_to_config(
            in_channels=new_in_channels,
            original_in_channels=original_in_channels,
            condition_channels=condition_channels,
        )
        self._conditioning_channels_added = True
        print(f"✅ Successfully extended transformer for {condition_channels} condition channels")
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path=None,
        base_model_name_or_path="alibaba-pai/CogVideoX-Fun-V1.1-5b-InP",
        subfolder="transformer",
        condition_channels: Optional[int] = None,
        **kwargs
    ):
        """
        Load a CogVideoXFunTransformer3DModelWithConcat from a pretrained model.

        Args:
            pretrained_model_name_or_path (str): Fine-tuned checkpoint path
            base_model_name_or_path (str): Base CogVideoX-Fun model path
            condition_channels (int): Number of extra condition channels to add
        """
        if pretrained_model_name_or_path is not None:
            # === Load fine-tuned checkpoint ===
            print(f"📥 Loading fine-tuned concat pose-conditioned transformer: {pretrained_model_name_or_path}")

            # 1) Create child class structure first
            config_path = os.path.join(pretrained_model_name_or_path, subfolder, "config.json")
            if os.path.exists(config_path):
                import json
                with open(config_path, "r") as f:
                    config = json.load(f)
                model = cls(**config)
            else:
                # Use base config
                if condition_channels is None:
                    condition_channels = 32  # default
                model = cls(condition_channels=condition_channels)

            # 2) Load checkpoint state_dict directly
            state_dict_path = os.path.join(pretrained_model_name_or_path, subfolder, "diffusion_pytorch_model.safetensors")
            if not os.path.exists(state_dict_path):
                # Try alternative paths
                state_dict_path = os.path.join(pretrained_model_name_or_path, subfolder, "pytorch_model.bin")
                if not os.path.exists(state_dict_path):
                    raise FileNotFoundError(f"No model file found in {pretrained_model_name_or_path}")
            
            if state_dict_path.endswith(".safetensors"):
                from safetensors.torch import load_file
                state_dict = load_file(state_dict_path)
            else:
                state_dict = torch.load(state_dict_path, map_location="cpu")

            # 3) Load with strict=False for flexibility
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"⚠️ Missing keys (new layers, need initialization during training): {missing}")
            if unexpected:
                print(f"⚠️ Unexpected keys (replaced layers, etc.): {unexpected}")

            return model

        else:
            # === Initialize from base model ===
            print(f"🔧 Initializing from base model: {base_model_name_or_path}")
            # First load the base CogVideoX transformer
            base_model = CogVideoXTransformer3DModel.from_pretrained(
                base_model_name_or_path,
                subfolder=subfolder,
                torch_dtype=kwargs.get("torch_dtype", torch.bfloat16),
                revision=kwargs.get("revision", None),
                variant=kwargs.get("variant", None),
            )
            
            # Then create CogVideoXFun transformer and copy state_dict
            fun_model = CogVideoXFunTransformer3DModel(**base_model.config)
            fun_model.load_state_dict(base_model.state_dict())
            base_model = fun_model

            # Determine condition_channels
            if condition_channels is None:
                condition_channels = getattr(base_model.config, "condition_channels", 32)

            # Create extended model
            model = cls(**base_model.config, condition_channels=0)

            # Load base weights into extended model
            missing, unexpected = model.load_state_dict(base_model.state_dict(), strict=False)
            if missing:
                print(f"⚠️ Missing keys (new condition layers): {missing}")
            if unexpected:
                print(f"⚠️ Unexpected keys (replaced existing layers): {unexpected}")

            # Expand input projection for condition channels
            if condition_channels > 0:
                print(f"🔗 Expanding projection layer for {condition_channels} condition channels")
                model._setup_condition_channels(
                    condition_channels,
                    original_proj=base_model.patch_embed.proj
                )

            return model
    
    def get_condition_info(self) -> Dict[str, Any]:
        """Get information about the conditional setup."""
        info = {
            "approach": "concat",
            "has_conditions": hasattr(self, 'condition_channels') and self.condition_channels > 0,
            "condition_channels": getattr(self, 'condition_channels', 0),
            "total_input_channels": self.patch_embed.proj.in_channels,
            "base_channels": getattr(self, 'original_in_channels', self.patch_embed.proj.in_channels),
        }
        return info
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
        inpaint_latents: Optional[torch.Tensor] = None,
        control_latents: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_dict: bool = True,
        **kwargs
    ):
        """
        Forward pass with VideoX-Fun specific inputs and conditional concatenation.
        
        Args:
            hidden_states: Input tensor with shape [batch, frames, channels, height, width]
            encoder_hidden_states: Text embeddings
            timestep: Timestep tensor
            timestep_cond: Timestep condition tensor
            inpaint_latents: Inpainting latents (VideoX-Fun specific)
            control_latents: Control latents (VideoX-Fun specific)
            image_rotary_emb: Rotary embeddings
            return_dict: Whether to return a dict
            **kwargs: Additional arguments
        """
        # Validate input dimensions if condition channels are expected
        if hasattr(self, 'condition_channels') and self.condition_channels > 0:
            expected_channels = self.patch_embed.proj.in_channels
            actual_channels = hidden_states.shape[2]
            
            if actual_channels != expected_channels:
                raise ValueError(
                    f"Expected {expected_channels} channels (base + condition), "
                    f"but got {actual_channels} channels in hidden_states"
                )
        
        # Use parent class forward method which handles VideoX-Fun specific logic
        return super().forward(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            timestep_cond=timestep_cond,
            inpaint_latents=inpaint_latents,
            control_latents=control_latents,
            image_rotary_emb=image_rotary_emb,
            return_dict=return_dict,
            **kwargs
        )


