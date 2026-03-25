from __future__ import annotations

import warnings


_PATCHED = False


def disable_broken_torchao() -> None:
    """Disable optional torchao integration that breaks diffusers imports in the current dwm env."""

    global _PATCHED
    if _PATCHED:
        return

    try:
        import diffusers.utils.import_utils as import_utils
    except Exception:
        return

    if getattr(import_utils, "_torchao_available", False):
        import_utils._torchao_available = False
        warnings.warn(
            "Disabled diffusers torchao integration for this process because the installed torchao package "
            "breaks CogVideoX imports in the dwm environment.",
            RuntimeWarning,
            stacklevel=2,
        )

    _PATCHED = True


__all__ = ["disable_broken_torchao"]
