import torch
import torch.nn as nn
from diffusers import CogVideoXTransformer3DModel
from diffusers.models.attention_processor import AttnProcessor2_0
from typing import Optional, Dict, Any


class CogVideoXTransformer3DModelWithConditions(CogVideoXTransformer3DModel):
    """
    Extended CogVideoXTransformer3DModel that supports conditional inputs.
    
    This transformer extends the base CogVideoX transformer to handle additional
    conditional inputs (e.g., hand mesh videos, static scene videos) by:
    1. Modifying the input projection layer to accept additional channels
    2. Processing concatenated conditional latents along with base latents
    
    The conditional inputs are concatenated channel-wise with the base noisy latents
    before being processed by the transformer.
    """
    
    def __init__(self, *args, condition_channels: int = 0, **kwargs):
        """
        Initialize the transformer with conditional support.
        
        Args:
            *args: Arguments passed to the parent CogVideoXTransformer3DModel
            condition_channels: Number of additional channels for conditional inputs
            **kwargs: Keyword arguments passed to the parent
        """
        super().__init__(*args, **kwargs)
        
        # Setup conditional channels if specified
        if condition_channels > 0:
            self._setup_condition_channels(condition_channels)
            self.condition_channels = condition_channels
        else:
            self.condition_channels = 0
    
    def _setup_condition_channels(self, condition_channels: int):
        """
        Modify the transformer to handle conditional input channels.
        
        This method extends the patch embedding projection layer to accept
        additional channels for conditional inputs (e.g., hand mesh + static scene).
        
        Args:
            condition_channels: Number of additional channels to add
        """
        # Get the original input projection layer
        original_proj = self.patch_embed.proj
        
        # Calculate new input channels
        original_in_channels = original_proj.in_channels
        new_in_channels = original_in_channels + condition_channels
        
        print(f"Extending transformer input channels:")
        print(f"  Original channels: {original_in_channels}")
        print(f"  Condition channels: {condition_channels}")
        print(f"  Total channels: {new_in_channels}")
        
        # Create new projection layer with extended input channels
        new_proj = nn.Conv2d(
            in_channels=new_in_channels,
            out_channels=original_proj.out_channels,
            kernel_size=original_proj.kernel_size,
            stride=original_proj.stride,
            padding=original_proj.padding,
            bias=original_proj.bias is not None,
        )
        
        # Initialize the new channels with AdaLN-zero style initialization
        with torch.no_grad():
            # Copy original weights for base channels
            new_proj.weight[:, :original_in_channels] = original_proj.weight
            
            # Zero initialize the new condition channels
            new_proj.weight[:, original_in_channels:] = 0.0
            
            # Copy bias if it exists
            if original_proj.bias is not None:
                new_proj.bias.data = original_proj.bias.data
        
        # Replace the projection layer
        self.patch_embed.proj = new_proj
        
        # Update the transformer config
        self.register_to_config(
            in_channels=new_in_channels,
            original_in_channels=original_in_channels,
            condition_channels=condition_channels
        )
        
        # Mark that conditioning channels have been added
        self._conditioning_channels_added = True
        
        print(f"✅ Successfully extended transformer for {condition_channels} condition channels")
    
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        image_rotary_emb: Optional[torch.FloatTensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ):
        """
        Forward pass with conditional input support.
        
        The hidden_states should already contain concatenated base latents and
        conditional latents in the channel dimension. This method processes
        them through the extended transformer.
        
        Args:
            hidden_states: Input tensor with shape [batch_size, num_frames, channels, height, width]
                          where channels = base_channels + condition_channels
            encoder_hidden_states: Text embeddings from the text encoder
            timestep: Current timestep for temporal conditioning
            image_rotary_emb: Rotary positional embeddings
            attention_kwargs: Additional arguments for attention processing
            return_dict: Whether to return a dictionary or tuple
            
        Returns:
            Transformer output (same as parent class)
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
        
        # Process through the parent transformer
        return super().forward(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            image_rotary_emb=image_rotary_emb,
            attention_kwargs=attention_kwargs,
            return_dict=return_dict
        )
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, condition_channels: int = 0, **kwargs):
        """
        Load a pretrained transformer and extend it for conditional inputs.
        
        Args:
            pretrained_model_name_or_path: Path to the pretrained model
            condition_channels: Number of condition channels to add
            **kwargs: Additional arguments for loading
            
        Returns:
            CogVideoXTransformer3DModelWithConditions instance
        """
        # Load the base transformer
        transformer = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        
        # If condition channels are requested, extend the transformer
        if condition_channels > 0:
            transformer._setup_condition_channels(condition_channels)
        
        return transformer
    
    def save_pretrained(self, save_directory: str, **kwargs):
        """
        Save the transformer with conditional configuration.
        
        Args:
            save_directory: Directory to save the model
            **kwargs: Additional arguments for saving
        """
        # Save the model with conditional configuration
        super().save_pretrained(save_directory, **kwargs)
        
        # Save additional config for condition channels
        if hasattr(self, 'condition_channels') and self.condition_channels > 0:
            config_path = f"{save_directory}/condition_config.json"
            import json
            condition_config = {
                "condition_channels": self.condition_channels,
                "original_in_channels": getattr(self, 'original_in_channels', None),
                "in_channels": getattr(self, 'in_channels', None)
            }
            with open(config_path, 'w') as f:
                json.dump(condition_config, f, indent=2)
    
    def get_condition_info(self) -> Dict[str, Any]:
        """
        Get information about the conditional setup.
        
        Returns:
            Dictionary containing condition channel information
        """
        info = {
            "has_conditions": hasattr(self, 'condition_channels') and self.condition_channels > 0,
            "condition_channels": getattr(self, 'condition_channels', 0),
            "total_input_channels": self.patch_embed.proj.in_channels,
            "base_channels": getattr(self, 'original_in_channels', self.patch_embed.proj.in_channels)
        }
        return info


def create_conditioned_transformer(
    base_model_path: str,
    condition_channels: int,
    torch_dtype: Optional[torch.dtype] = None,
    **kwargs
) -> CogVideoXTransformer3DModelWithConditions:
    """
    Convenience function to create a conditioned transformer from a base model.
    
    Args:
        base_model_path: Path to the base CogVideoX model
        condition_channels: Number of condition channels to add
        torch_dtype: Data type for the model
        **kwargs: Additional arguments for model loading
        
    Returns:
        CogVideoXTransformer3DModelWithConditions instance
    """
    print(f"Creating conditioned transformer from: {base_model_path}")
    print(f"Adding {condition_channels} condition channels")
    
    # Load the base transformer
    transformer = CogVideoXTransformer3DModelWithConditions.from_pretrained(
        base_model_path,
        subfolder="transformer",
        condition_channels=condition_channels,
        torch_dtype=torch_dtype,
        **kwargs
    )
    
    # Print condition information
    condition_info = transformer.get_condition_info()
    print(f"Transformer condition info: {condition_info}")
    
    return transformer

