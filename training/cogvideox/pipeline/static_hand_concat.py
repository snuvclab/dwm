from __future__ import annotations

from typing import Any

from training.cogvideox.pipeline.cogvideox_fun_static_to_video_pose_concat_pipeline import (
    CogVideoXFunStaticToVideoPipeline,
)


class CogVideoXFunStaticHandConcatPipeline(CogVideoXFunStaticToVideoPipeline):
    """DWM CogVideoX pipeline restricted to static+hand concat conditioning."""

    @classmethod
    def from_pretrained(
        cls,
        *args: Any,
        condition_channels: int = 16,
        split_hands: bool = False,
        **kwargs: Any,
    ):
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
            raise ValueError(f"Adapter-related args are not supported in DWM pipeline: {sorted(invalid)}")

        return super().from_pretrained(
            *args,
            condition_channels=condition_channels,
            use_adapter=False,
            split_hands=split_hands,
            add_control_adapter=False,
            **kwargs,
        )


__all__ = ["CogVideoXFunStaticHandConcatPipeline"]
