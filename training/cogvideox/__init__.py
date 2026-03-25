from .config_loader import (
    CANONICAL_PIPELINE_CLASS,
    CANONICAL_PIPELINE_TYPE,
    CANONICAL_TRANSFORMER_CLASS,
    load_experiment_config,
)


def __getattr__(name):
    if name == "CogVideoXFunStaticHandConcatPipeline":
        from .pipeline import CogVideoXFunStaticHandConcatPipeline

        return CogVideoXFunStaticHandConcatPipeline
    if name == "CogVideoXFunStaticHandConcatTransformer3DModel":
        from .models import CogVideoXFunStaticHandConcatTransformer3DModel

        return CogVideoXFunStaticHandConcatTransformer3DModel
    raise AttributeError(name)


__all__ = [
    "CANONICAL_PIPELINE_CLASS",
    "CANONICAL_PIPELINE_TYPE",
    "CANONICAL_TRANSFORMER_CLASS",
    "CogVideoXFunStaticHandConcatPipeline",
    "CogVideoXFunStaticHandConcatTransformer3DModel",
    "load_experiment_config",
]
