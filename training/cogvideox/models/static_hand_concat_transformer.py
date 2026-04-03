from __future__ import annotations

from typing import Any

from training.cogvideox.models.cogvideox_fun_transformer_with_conditions import (
    CogVideoXFunTransformer3DModelWithConcat,
)


class CogVideoXFunStaticHandConcatTransformer3DModel(CogVideoXFunTransformer3DModelWithConcat):
    """DWM CogVideoX transformer restricted to static+hand concat conditioning."""

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any):
        blocked_keys = {
            "use_adapter",
            "adapter_version",
            "use_zero_proj",
            "add_control_adapter",
            "in_dim_control_adapter",
            "adapter_control_type",
        }
        invalid = blocked_keys.intersection(kwargs.keys())
        if invalid:
            raise ValueError(f"Adapter-related args are not supported in DWM transformer: {sorted(invalid)}")

        return super().from_pretrained(*args, **kwargs)


__all__ = ["CogVideoXFunStaticHandConcatTransformer3DModel"]
