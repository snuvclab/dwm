from __future__ import annotations

import importlib.util
import warnings


_PATCHED = False
_ORIGINAL_FIND_SPEC = None
_HIDDEN_OPTIONAL_PACKAGES = {"torchao", "bitsandbytes", "xformers", "deepspeed"}


def disable_problematic_optional_backends() -> None:
    """Disable unsupported optional diffusers integrations in the validated release env."""

    global _PATCHED, _ORIGINAL_FIND_SPEC
    if _PATCHED:
        return

    if _ORIGINAL_FIND_SPEC is None:
        _ORIGINAL_FIND_SPEC = importlib.util.find_spec

        def _patched_find_spec(name, package=None):
            top_level = name.split(".", 1)[0]
            if top_level in _HIDDEN_OPTIONAL_PACKAGES:
                return None
            return _ORIGINAL_FIND_SPEC(name, package)

        importlib.util.find_spec = _patched_find_spec

    try:
        import diffusers.utils.import_utils as import_utils
    except Exception:
        return

    disabled_any = False

    if getattr(import_utils, "_torchao_available", False):
        import_utils._torchao_available = False
        disabled_any = True

    if disabled_any:
        warnings.warn(
            "Disabled unsupported optional acceleration backends for this process to keep the validated "
            "CogVideoX release path stable.",
            RuntimeWarning,
            stacklevel=2,
        )

    _PATCHED = True


__all__ = ["disable_problematic_optional_backends"]
