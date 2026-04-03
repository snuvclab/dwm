import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from diffusers.configuration_utils import register_to_config
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel as DiffusersWanTransformer3DModel


class WanI2VTransformer3DModelWithConcat(DiffusersWanTransformer3DModel):
    """
    Diffusers WAN I2V transformer with optional hand-condition channel concatenation.

    This class supports two forward styles:
    1) Diffusers-native: hidden_states/timestep/encoder_hidden_states
    2) Legacy WAN-Fun style: x/t/context/y/condition_latents
    """
    _keep_in_fp32_modules = None

    @register_to_config
    def __init__(
        self,
        patch_size=(1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: Optional[int] = None,
        added_kv_proj_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
        pos_embed_seq_len: Optional[int] = None,
        condition_channels: int = 0,
    ):
        inner_dim = int(num_attention_heads) * int(attention_head_dim)
        # Wan2.1 1.3B diffusers checkpoints may contain `added_kv_proj_dim=5120`
        # while actual attn2 added-kv projection weights are [inner_dim, inner_dim].
        # Force projection input dim to model inner dim for compatibility.
        if added_kv_proj_dim is not None and int(added_kv_proj_dim) != inner_dim:
            added_kv_proj_dim = inner_dim

        super().__init__(
            patch_size=patch_size,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            text_dim=text_dim,
            freq_dim=freq_dim,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            cross_attn_norm=cross_attn_norm,
            qk_norm=qk_norm,
            eps=eps,
            image_dim=image_dim,
            added_kv_proj_dim=added_kv_proj_dim,
            rope_max_seq_len=rope_max_seq_len,
            pos_embed_seq_len=pos_embed_seq_len,
        )
        self.condition_channels = int(condition_channels)
        configured_in_channels = self.config.in_channels
        configured_original_in_channels = getattr(self.config, "original_in_channels", None)
        already_extended = (
            configured_in_channels is not None
            and configured_original_in_channels is not None
            and int(configured_in_channels) == int(configured_original_in_channels) + self.condition_channels
        )
        if self.condition_channels > 0 and not already_extended:
            self._extend_patch_embedding(self.condition_channels)

    def _extend_patch_embedding(self, condition_channels: int) -> None:
        original_proj = self.patch_embedding
        original_in_channels = original_proj.in_channels
        new_in_channels = original_in_channels + condition_channels

        new_proj = nn.Conv3d(
            in_channels=new_in_channels,
            out_channels=original_proj.out_channels,
            kernel_size=original_proj.kernel_size,
            stride=original_proj.stride,
            padding=original_proj.padding,
            bias=original_proj.bias is not None,
        )

        with torch.no_grad():
            new_proj.weight[:, :original_in_channels] = original_proj.weight
            new_proj.weight[:, original_in_channels:] = 0.0
            if original_proj.bias is not None:
                new_proj.bias.data = original_proj.bias.data

        self.patch_embedding = new_proj
        self.register_to_config(
            in_channels=new_in_channels,
            original_in_channels=original_in_channels,
            condition_channels=condition_channels,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        condition_channels: int = 0,
        **kwargs,
    ):
        """
        Load base diffusers WAN transformer first, then expand patch embedding channels.

        This avoids relying on `ignore_mismatched_sizes=True` for channel-expanded variants.
        """
        # Accept both root path (+subfolder) and direct transformer directory.
        if kwargs.get("subfolder") is None:
            candidate_subfolder = os.path.join(pretrained_model_name_or_path, "transformer")
            if os.path.isdir(candidate_subfolder):
                kwargs["subfolder"] = "transformer"

        # Diffusers WAN enforces low_cpu_mem_usage=True when keep_in_fp32_modules is enabled.
        if kwargs.get("low_cpu_mem_usage") is False:
            print(
                "[WanI2VTransformer3DModelWithConcat] low_cpu_mem_usage=False is incompatible with "
                "keep_in_fp32_modules; overriding to True."
            )
            kwargs["low_cpu_mem_usage"] = True

        # Important: load through `cls` (via super()) so our __init__ compatibility fix
        # (added_kv_proj_dim normalization) is applied before state dict loading.
        model = super().from_pretrained(
            pretrained_model_name_or_path,
            condition_channels=0,
            **kwargs,
        )

        base_in_channels = int(model.patch_embedding.in_channels)
        original_in_channels = getattr(model.config, "original_in_channels", None)
        if original_in_channels is not None:
            target_in_channels = int(original_in_channels) + int(condition_channels)
        else:
            target_in_channels = base_in_channels + int(condition_channels)

        if target_in_channels > base_in_channels:
            expand_by = target_in_channels - base_in_channels
            print(
                f"[WanI2VTransformer3DModelWithConcat] Expanding patch_embedding channels: "
                f"{base_in_channels} -> {target_in_channels} (delta={expand_by})"
            )
            model._extend_patch_embedding(expand_by)
            model.condition_channels = int(condition_channels)
        else:
            model.condition_channels = max(0, int(target_in_channels - base_in_channels))
            if model.condition_channels > 0:
                model.register_to_config(
                    in_channels=base_in_channels,
                    original_in_channels=base_in_channels - model.condition_channels,
                    condition_channels=model.condition_channels,
                )
            elif target_in_channels < base_in_channels:
                print(
                    "[WanI2VTransformer3DModelWithConcat] Requested condition_channels would shrink input "
                    f"({base_in_channels} -> {target_in_channels}); keeping loaded channels unchanged."
                )

        return model

    @staticmethod
    def _pad_context_list(context_list: List[torch.Tensor], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if len(context_list) == 0:
            raise ValueError("Empty context list")
        max_len = max(int(t.shape[0]) for t in context_list)
        dim = int(context_list[0].shape[-1])
        out = torch.zeros(len(context_list), max_len, dim, device=device, dtype=dtype)
        for i, t in enumerate(context_list):
            t = t.to(device=device, dtype=dtype)
            out[i, : t.shape[0]] = t
        return out

    def forward(
        self,
        x: Optional[torch.Tensor] = None,
        t: Optional[torch.LongTensor] = None,
        context: Optional[Any] = None,
        seq_len: Optional[int] = None,  # kept for signature compatibility
        y: Optional[torch.Tensor] = None,
        clip_fea: Optional[torch.Tensor] = None,
        condition_latents: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        legacy_mode = x is not None or context is not None or y is not None or condition_latents is not None

        if hidden_states is None:
            hidden_states = x
        if hidden_states is None:
            raise ValueError("Either `hidden_states` or `x` must be provided.")

        if y is not None:
            hidden_states = torch.cat([hidden_states, y], dim=1)
        if condition_latents is not None:
            hidden_states = torch.cat([hidden_states, condition_latents], dim=1)

        if timestep is None:
            timestep = t
        if timestep is None:
            raise ValueError("Either `timestep` or `t` must be provided.")

        if encoder_hidden_states is None:
            encoder_hidden_states = context
        if isinstance(encoder_hidden_states, list):
            encoder_hidden_states = self._pad_context_list(
                encoder_hidden_states,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
        if encoder_hidden_states is None:
            raise ValueError("`encoder_hidden_states`/`context` is required.")

        if encoder_hidden_states_image is None and clip_fea is not None:
            encoder_hidden_states_image = clip_fea

        outputs = super().forward(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_image=encoder_hidden_states_image,
            return_dict=(False if legacy_mode else return_dict),
            attention_kwargs=attention_kwargs,
            **kwargs,
        )

        if legacy_mode:
            if isinstance(outputs, tuple):
                return outputs[0]
            if hasattr(outputs, "sample"):
                return outputs.sample
            return outputs
        return outputs
