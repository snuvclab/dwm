# Copyright 2024 The Wan team.
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

import torch
import torch.nn as nn
from typing import Optional, Union, Tuple, Dict, Any
from .wan_transformer3d import WanTransformer3DModel


class WanTransformer3DModelWithConcat(WanTransformer3DModel):
    """
    Wan Transformer with conditional input support via channel concatenation.
    
    This class extends WanTransformer3DModel to support conditional inputs
    by concatenating condition channels directly to the input channels.
    
    Key differences from WanTransformer3DModel:
    1. Supports conditional input channels via concatenation
    2. Extends input channels to accommodate condition channels
    3. Maintains compatibility with existing Wan model architecture
    """
    
    def __init__(self, *args, condition_channels: int = 0, **kwargs):
        # Store original in_channels before modification
        original_in_channels = kwargs.get("in_channels", 16)
        self.original_in_channels = original_in_channels
        
        super().__init__(*args, **kwargs)
        
        # Setup conditional channels if specified
        if condition_channels > 0:
            self._setup_condition_channels(condition_channels)
            self.condition_channels = condition_channels
        else:
            self.condition_channels = 0
    
    def _setup_condition_channels(self, condition_channels: int, original_proj: Optional[nn.Conv3d] = None):
        """Extend the transformer to handle conditional input channels.
        
        If `original_proj` is provided, use its weights as the source (for base→finetune case).
        Otherwise fall back to self.patch_embedding (for reload case).
        """
        if original_proj is None:
            original_proj = self.patch_embedding

        original_in_channels = original_proj.in_channels
        new_in_channels = original_in_channels + condition_channels

        print(f"Extending Wan transformer input channels for concat approach:")
        print(f"  Original channels: {original_in_channels}")
        print(f"  Condition channels: {condition_channels}")
        print(f"  Total channels: {new_in_channels}")

        new_proj = nn.Conv3d(
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

        self.patch_embedding = new_proj
        self.register_to_config(
            in_channels=new_in_channels,
            original_in_channels=original_in_channels,
            condition_channels=condition_channels,
        )
        self._conditioning_channels_added = True
        print(f"✅ Successfully extended Wan transformer for {condition_channels} condition channels")
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path=None,
        base_model_name_or_path="alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP",
        condition_channels: Optional[int] = None,
        transformer_additional_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Load a WanTransformer3DModelWithConcat from a pretrained model.

        Args:
            pretrained_model_name_or_path (str): Fine-tuned checkpoint path (should include subfolder)
            base_model_name_or_path (str): Base Wan model path
            condition_channels (int): Number of extra condition channels to add
            transformer_additional_kwargs (dict): Additional kwargs for transformer initialization
        """
        if pretrained_model_name_or_path is not None:
            # === Load fine-tuned checkpoint ===
            print(f"📥 Loading fine-tuned concat pose-conditioned Wan transformer: {pretrained_model_name_or_path}")

            # 1) Create child class structure first
            import os
            import json
            config_path = os.path.join(pretrained_model_name_or_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = json.load(f)
                model = cls(**config)
            else:
                # Use base config
                if condition_channels is None:
                    condition_channels = 32  # default
                model = cls(condition_channels=condition_channels)

            # 2) Load checkpoint state_dict directly
            import glob
            from safetensors.torch import load_file
            
            # Try to find all safetensors files first
            safetensors_files = glob.glob(os.path.join(pretrained_model_name_or_path, "*.safetensors"))
            if safetensors_files:
                print(f"🔧 Loading weights from {len(safetensors_files)} safetensors files")
                # Load and merge all safetensors files
                state_dict = {}
                for file_path in safetensors_files:
                    print(f"   Loading: {os.path.basename(file_path)}")
                    file_state_dict = load_file(file_path)
                    state_dict.update(file_state_dict)
            else:
                # Fallback to single file approach
                state_dict_path = os.path.join(pretrained_model_name_or_path, "diffusion_pytorch_model.safetensors")
                if not os.path.exists(state_dict_path):
                    # Try alternative paths
                    state_dict_path = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
                    if not os.path.exists(state_dict_path):
                        raise FileNotFoundError(f"No model file found in {pretrained_model_name_or_path}")
                
                if state_dict_path.endswith(".safetensors"):
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
            
            # Prepare transformer_additional_kwargs
            if transformer_additional_kwargs is None:
                transformer_additional_kwargs = {}
            
            # Add condition_channels to transformer_additional_kwargs if specified
            if condition_channels is not None:
                transformer_additional_kwargs['condition_channels'] = condition_channels
            
            # First load the base Wan transformer with additional kwargs
            base_model = WanTransformer3DModel.from_pretrained(
                base_model_name_or_path,
                transformer_additional_kwargs=transformer_additional_kwargs,
                torch_dtype=kwargs.get("torch_dtype", torch.bfloat16),
                revision=kwargs.get("revision", None),
                variant=kwargs.get("variant", None),
            )
            
            # Then create Wan transformer and copy state_dict
            wan_model = WanTransformer3DModel(**base_model.config)
            wan_model.load_state_dict(base_model.state_dict())
            base_model = wan_model

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
                    original_proj=base_model.patch_embedding
                )

            return model
    
    def get_condition_info(self) -> Dict[str, Any]:
        """Get information about the conditional setup."""
        info = {
            "approach": "concat",
            "has_conditions": hasattr(self, 'condition_channels') and self.condition_channels > 0,
            "condition_channels": getattr(self, 'condition_channels', 0),
            "total_input_channels": self.patch_embedding.in_channels,
            "base_channels": getattr(self, 'original_in_channels', self.patch_embedding.in_channels),
        }
        return info
    
    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        y_camera=None,
        full_ref=None,
        subject_ref=None,
        cond_flag=True,
        control_latents=None,
    ):
        """
        Forward pass with Wan specific inputs and conditional concatenation.
        
        Args:
            x: Input tensor with shape [batch, frames, channels, height, width]
            t: Timestep tensor
            context: Text embeddings
            seq_len: Sequence length
            clip_fea: CLIP features (optional)
            y: Conditional video inputs (optional)
            y_camera: Camera parameters (optional)
            full_ref: Full reference (optional)
            subject_ref: Subject reference (optional)
            cond_flag: Condition flag
            control_latents: Control latents for conditioning (optional)
            **kwargs: Additional arguments
        """
        # Validate input dimensions if condition channels are expected
        if hasattr(self, 'condition_channels') and self.condition_channels > 0:
            expected_channels = self.patch_embedding.in_channels
            actual_channels = x.shape[1]  # Assuming x is [B, C, F, H, W]
            
            if actual_channels != expected_channels:
                raise ValueError(
                    f"Expected {expected_channels} channels (base + condition), "
                    f"but got {actual_channels} channels in x"
                )
        
        # Use parent class forward method which handles Wan specific logic
        return super().forward(
            x=x,
            t=t,
            context=context,
            seq_len=seq_len,
            clip_fea=clip_fea,
            y=y,
            y_camera=y_camera,
            full_ref=full_ref,
            subject_ref=subject_ref,
            cond_flag=cond_flag,
        )
