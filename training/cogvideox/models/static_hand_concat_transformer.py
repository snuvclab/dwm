from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from training.cogvideox.diffusers_compat import disable_broken_torchao

disable_broken_torchao()

from diffusers import CogVideoXTransformer3DModel


class CogVideoXFunStaticHandConcatTransformer3DModel(CogVideoXTransformer3DModel):
    """CogVideoX transformer with an expanded input projection for hand-latent concat conditioning."""

    def __init__(
        self,
        *args,
        condition_channels: int = 16,
        original_in_channels: Optional[int] = None,
        **kwargs,
    ) -> None:
        kwargs.pop("add_noise_in_inpaint_model", None)
        total_in_channels = kwargs.get("in_channels", 16)
        if original_in_channels is None:
            if condition_channels > 0 and total_in_channels > condition_channels:
                original_in_channels = total_in_channels - condition_channels
            else:
                original_in_channels = total_in_channels

        kwargs["in_channels"] = original_in_channels
        super().__init__(*args, **kwargs)

        self.original_in_channels = original_in_channels
        self.condition_channels = condition_channels
        target_in_channels = original_in_channels + condition_channels
        if condition_channels > 0 and self.patch_embed.proj.in_channels != target_in_channels:
            self._expand_patch_embedding(
                original_proj=self.patch_embed.proj,
                target_in_channels=target_in_channels,
                original_in_channels=original_in_channels,
            )

        self.register_to_config(
            original_in_channels=original_in_channels,
            condition_channels=condition_channels,
            in_channels=self.patch_embed.proj.in_channels,
        )

    def _expand_patch_embedding(
        self,
        original_proj: nn.Conv2d,
        target_in_channels: int,
        original_in_channels: int,
    ) -> None:
        new_proj = nn.Conv2d(
            in_channels=target_in_channels,
            out_channels=original_proj.out_channels,
            kernel_size=original_proj.kernel_size,
            stride=original_proj.stride,
            padding=original_proj.padding,
            bias=original_proj.bias is not None,
        )

        with torch.no_grad():
            new_proj.weight.zero_()
            new_proj.weight[:, :original_in_channels].copy_(original_proj.weight)
            if original_proj.bias is not None:
                new_proj.bias.copy_(original_proj.bias)

        self.patch_embed.proj = new_proj

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[str] = None,
        base_model_name_or_path: Optional[str] = None,
        subfolder: str = "transformer",
        condition_channels: Optional[int] = None,
        **kwargs,
    ) -> "CogVideoXFunStaticHandConcatTransformer3DModel":
        if pretrained_model_name_or_path is not None:
            return super().from_pretrained(pretrained_model_name_or_path, subfolder=subfolder, **kwargs)

        if base_model_name_or_path is None:
            raise ValueError("Either pretrained_model_name_or_path or base_model_name_or_path must be provided.")

        base_transformer = CogVideoXTransformer3DModel.from_pretrained(
            base_model_name_or_path,
            subfolder=subfolder,
            **kwargs,
        )
        resolved_condition_channels = (
            condition_channels
            if condition_channels is not None
            else getattr(base_transformer.config, "condition_channels", 16)
        )
        model = cls(
            **base_transformer.config,
            condition_channels=resolved_condition_channels,
            original_in_channels=base_transformer.config.in_channels,
        )
        state_dict = base_transformer.state_dict()
        filtered_state_dict = {}
        for name, value in state_dict.items():
            if name in {"patch_embed.proj.weight", "patch_embed.proj.bias"}:
                continue
            filtered_state_dict[name] = value
        model.load_state_dict(filtered_state_dict, strict=False)
        with torch.no_grad():
            model.patch_embed.proj.weight[:, : base_transformer.patch_embed.proj.in_channels].copy_(
                base_transformer.patch_embed.proj.weight
            )
            if (
                base_transformer.patch_embed.proj.bias is not None
                and model.patch_embed.proj.bias is not None
            ):
                model.patch_embed.proj.bias.copy_(base_transformer.patch_embed.proj.bias)
        return model


__all__ = ["CogVideoXFunStaticHandConcatTransformer3DModel"]
