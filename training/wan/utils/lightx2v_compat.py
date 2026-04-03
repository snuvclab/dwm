import re
from collections import Counter
from typing import Any, Dict, Optional, Tuple

import torch
from safetensors.torch import load_file


LIGHTX2V_BLOCK_PATTERN = re.compile(r"^diffusion_model\.blocks\.(\d+)\.")
PEFT_AB_BLOCK_PATTERN = re.compile(r"^(?:base_model\.model\.|transformer\.)?blocks\.(\d+)\.")


def _get_logger(logger=None):
    if logger is not None:
        return logger
    import logging

    return logging.getLogger(__name__)


def detect_lightx2v_format(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    counts = Counter()
    block_ids_delta = set()
    block_ids_peft = set()
    for key in state_dict.keys():
        if key.startswith("diffusion_model.blocks."):
            m = LIGHTX2V_BLOCK_PATTERN.match(key)
            if m is not None:
                block_ids_delta.add(int(m.group(1)))
        m_peft = PEFT_AB_BLOCK_PATTERN.match(key)
        if m_peft is not None:
            block_ids_peft.add(int(m_peft.group(1)))
        if key.endswith(".lora_down.weight"):
            counts["lora_down"] += 1
        if key.endswith(".lora_up.weight"):
            counts["lora_up"] += 1
        if key.endswith(".lora_A.weight"):
            counts["lora_A"] += 1
        if key.endswith(".lora_B.weight"):
            counts["lora_B"] += 1
        if key.endswith(".diff"):
            counts["diff"] += 1
        if key.endswith(".diff_b"):
            counts["diff_b"] += 1
        if key.endswith(".alpha"):
            counts["alpha"] += 1

    is_delta = (
        len(block_ids_delta) > 0
        and counts["lora_down"] > 0
        and counts["lora_up"] > 0
    )
    is_peft_ab = (
        len(block_ids_peft) > 0
        and counts["lora_A"] > 0
        and counts["lora_B"] > 0
    )
    format_name = None
    block_ids = []
    if is_delta:
        format_name = "delta_lora_diff"
        block_ids = sorted(block_ids_delta)
    elif is_peft_ab:
        format_name = "peft_ab"
        block_ids = sorted(block_ids_peft)

    return {
        "is_lightx2v": bool(format_name is not None),
        "format": format_name,
        "counts": dict(counts),
        "block_ids": block_ids,
    }


def _target_from_base(base: str) -> str:
    return base.replace("diffusion_model.", "")


def _target_from_peft_base(base: str) -> str:
    for prefix in ("base_model.model.", "transformer.", "model."):
        if base.startswith(prefix):
            return base[len(prefix) :]
    return base


def _peft_legacy_renames(base: str) -> str:
    renamed = base
    renamed = renamed.replace(".attn1.to_k", ".self_attn.k")
    renamed = renamed.replace(".attn1.to_q", ".self_attn.q")
    renamed = renamed.replace(".attn1.to_v", ".self_attn.v")
    renamed = renamed.replace(".attn1.to_out.0", ".self_attn.o")
    renamed = renamed.replace(".attn2.to_k", ".cross_attn.k")
    renamed = renamed.replace(".attn2.to_q", ".cross_attn.q")
    renamed = renamed.replace(".attn2.to_v", ".cross_attn.v")
    renamed = renamed.replace(".attn2.to_out.0", ".cross_attn.o")
    renamed = renamed.replace(".ffn.net.0.proj", ".ffn.0")
    renamed = renamed.replace(".ffn.net.2", ".ffn.2")
    return renamed


def _candidate_target_weight_keys_from_peft_base(base: str) -> list[str]:
    normalized = _target_from_peft_base(base)
    candidates = [f"{normalized}.weight"]
    legacy = _peft_legacy_renames(normalized)
    legacy_weight = f"{legacy}.weight"
    if legacy_weight not in candidates:
        candidates.append(legacy_weight)
    return candidates


def _compute_lora_delta(weight_up: torch.Tensor, weight_down: torch.Tensor) -> torch.Tensor:
    if weight_up.ndim == 2 and weight_down.ndim == 2:
        return torch.mm(weight_up, weight_down)
    if weight_up.ndim == 4 and weight_down.ndim == 4:
        return torch.mm(
            weight_up.squeeze(3).squeeze(2),
            weight_down.squeeze(3).squeeze(2),
        ).unsqueeze(2).unsqueeze(3)
    raise ValueError(f"Unsupported LoRA weight dims: up={weight_up.shape}, down={weight_down.shape}")


def _add_to_param(
    params: Dict[str, torch.nn.Parameter],
    key: str,
    delta: torch.Tensor,
    scale: float,
) -> bool:
    if key not in params:
        return False

    p = params[key]
    if p.data.shape != delta.shape:
        raise ValueError(f"Shape mismatch for {key}: param={tuple(p.data.shape)} delta={tuple(delta.shape)}")
    p.data.add_(delta.to(device=p.data.device, dtype=p.data.dtype), alpha=float(scale))
    return True


def apply_lightx2v_state_dict_to_transformer(
    transformer: torch.nn.Module,
    state_dict: Dict[str, torch.Tensor],
    lora_scale: float = 1.0,
    strict: bool = True,
    expected_blocks: Optional[int] = 40,
    apply_non_block_diff: bool = False,
    logger=None,
) -> Dict[str, Any]:
    log = _get_logger(logger)
    params = dict(transformer.named_parameters())
    stats: Dict[str, Any] = {
        "expected_blocks": expected_blocks,
        "lora_scale": float(lora_scale),
        "applied_lora_pairs": 0,
        "applied_diff": 0,
        "applied_diff_b": 0,
        "missing_lora_pairs": [],
        "missing_diff_targets": [],
        "shape_mismatch": [],
        "skipped_other_keys": 0,
        "skipped_non_block_diff": 0,
    }

    fmt = detect_lightx2v_format(state_dict)
    stats["format"] = fmt
    if not fmt["is_lightx2v"]:
        raise ValueError("State dict does not look like LightX2V format.")

    block_ids = set(fmt["block_ids"])
    if expected_blocks is not None:
        expected = set(range(int(expected_blocks)))
        missing_blocks = sorted(expected - block_ids)
        extra_blocks = sorted(block_ids - expected)
        stats["missing_blocks"] = missing_blocks
        stats["extra_blocks"] = extra_blocks
        if strict and (missing_blocks or extra_blocks):
            raise ValueError(
                f"LightX2V block coverage mismatch. expected=0..{expected_blocks-1}, "
                f"missing={missing_blocks[:10]}, extra={extra_blocks[:10]}"
            )

    with torch.no_grad():
        if fmt["format"] == "delta_lora_diff":
            lora_bases = set()
            for key in state_dict.keys():
                if key.endswith(".lora_down.weight"):
                    lora_bases.add(key[: -len(".lora_down.weight")])
                elif key.endswith(".lora_up.weight"):
                    lora_bases.add(key[: -len(".lora_up.weight")])

            for base in sorted(lora_bases):
                up_key = f"{base}.lora_up.weight"
                down_key = f"{base}.lora_down.weight"
                if up_key not in state_dict or down_key not in state_dict:
                    stats["missing_lora_pairs"].append(base)
                    continue

                target_weight_key = f"{_target_from_base(base)}.weight"
                if target_weight_key not in params:
                    stats["missing_lora_pairs"].append(base)
                    continue

                w_up = state_dict[up_key]
                w_down = state_dict[down_key]
                alpha_key = f"{base}.alpha"
                if alpha_key in state_dict:
                    alpha_value = float(state_dict[alpha_key].item()) / float(w_up.shape[1])
                else:
                    alpha_value = 1.0

                try:
                    delta = _compute_lora_delta(w_up, w_down)
                    _add_to_param(params, target_weight_key, delta, lora_scale * alpha_value)
                    stats["applied_lora_pairs"] += 1
                except Exception as exc:
                    stats["shape_mismatch"].append(f"{base}: {exc}")

            for key, value in state_dict.items():
                if key.endswith(".diff_b"):
                    if not apply_non_block_diff and not key.startswith("diffusion_model.blocks."):
                        stats["skipped_non_block_diff"] += 1
                        continue
                    base = key[: -len(".diff_b")]
                    target = _target_from_base(base)
                    target_candidates = [f"{target}.bias", target]
                    applied = False
                    for candidate in target_candidates:
                        if candidate in params:
                            try:
                                applied = _add_to_param(params, candidate, value, lora_scale)
                                if applied:
                                    stats["applied_diff_b"] += 1
                                break
                            except Exception as exc:
                                stats["shape_mismatch"].append(f"{key}->{candidate}: {exc}")
                                applied = False
                                break
                    if not applied:
                        stats["missing_diff_targets"].append(key)
                elif key.endswith(".diff"):
                    if not apply_non_block_diff and not key.startswith("diffusion_model.blocks."):
                        stats["skipped_non_block_diff"] += 1
                        continue
                    base = key[: -len(".diff")]
                    target = _target_from_base(base)
                    target_candidates = [f"{target}.weight", target]
                    applied = False
                    for candidate in target_candidates:
                        if candidate in params:
                            try:
                                applied = _add_to_param(params, candidate, value, lora_scale)
                                if applied:
                                    stats["applied_diff"] += 1
                                break
                            except Exception as exc:
                                stats["shape_mismatch"].append(f"{key}->{candidate}: {exc}")
                                applied = False
                                break
                    if not applied:
                        stats["missing_diff_targets"].append(key)
                elif (
                    not key.endswith(".lora_down.weight")
                    and not key.endswith(".lora_up.weight")
                    and not key.endswith(".alpha")
                ):
                    stats["skipped_other_keys"] += 1
        elif fmt["format"] == "peft_ab":
            peft_bases = set()
            for key in state_dict.keys():
                if key.endswith(".lora_A.weight"):
                    peft_bases.add(key[: -len(".lora_A.weight")])
                elif key.endswith(".lora_B.weight"):
                    peft_bases.add(key[: -len(".lora_B.weight")])

            for base in sorted(peft_bases):
                a_key = f"{base}.lora_A.weight"
                b_key = f"{base}.lora_B.weight"
                if a_key not in state_dict or b_key not in state_dict:
                    stats["missing_lora_pairs"].append(base)
                    continue

                candidate_keys = _candidate_target_weight_keys_from_peft_base(base)
                target_weight_key = None
                for candidate in candidate_keys:
                    if candidate in params:
                        target_weight_key = candidate
                        break
                if target_weight_key is None:
                    stats["missing_lora_pairs"].append(base)
                    continue

                try:
                    w_a = state_dict[a_key]
                    w_b = state_dict[b_key]
                    delta = torch.mm(w_b, w_a)
                    _add_to_param(params, target_weight_key, delta, lora_scale)
                    stats["applied_lora_pairs"] += 1
                except Exception as exc:
                    stats["shape_mismatch"].append(f"{base}: {exc}")
        else:
            raise ValueError(f"Unsupported detected LightX2V format: {fmt.get('format')}")

    if strict and (
        len(stats["missing_lora_pairs"]) > 0
        or len(stats["missing_diff_targets"]) > 0
        or len(stats["shape_mismatch"]) > 0
    ):
        raise ValueError(
            "LightX2V compatibility apply failed in strict mode. "
            f"missing_lora_pairs={len(stats['missing_lora_pairs'])}, "
            f"missing_diff_targets={len(stats['missing_diff_targets'])}, "
            f"shape_mismatch={len(stats['shape_mismatch'])}, "
            f"sample_missing_diff={stats['missing_diff_targets'][:1]}, "
            f"sample_shape_mismatch={stats['shape_mismatch'][:1]}"
        )

    log.info(
        "LightX2V compat apply summary: lora_pairs=%d, diff=%d, diff_b=%d, "
        "missing_lora_pairs=%d, missing_diff_targets=%d, shape_mismatch=%d, "
        "skipped_non_block_diff=%d, skipped_other=%d",
        stats["applied_lora_pairs"],
        stats["applied_diff"],
        stats["applied_diff_b"],
        len(stats["missing_lora_pairs"]),
        len(stats["missing_diff_targets"]),
        len(stats["shape_mismatch"]),
        stats["skipped_non_block_diff"],
        stats["skipped_other_keys"],
    )
    if stats["missing_lora_pairs"]:
        log.info("LightX2V missing LoRA targets (first 10): %s", stats["missing_lora_pairs"][:10])
    if stats["missing_diff_targets"]:
        log.info("LightX2V missing diff targets (first 10): %s", stats["missing_diff_targets"][:10])
    if stats["shape_mismatch"]:
        log.info("LightX2V shape mismatch entries (first 10): %s", stats["shape_mismatch"][:10])

    return stats


def load_lightx2v_with_fallback(
    pipeline: Any,
    lora_path: str,
    lora_scale: float = 1.0,
    compat_mode: str = "auto",
    compat_strict: bool = True,
    expected_blocks: Optional[int] = 40,
    apply_non_block_diff: bool = False,
    logger=None,
) -> Tuple[str, Dict[str, Any]]:
    log = _get_logger(logger)
    compat_mode = str(compat_mode).lower().strip()
    if compat_mode not in {"auto", "force_compat", "off"}:
        raise ValueError(f"Unsupported lightx2v compat mode: {compat_mode}")

    native_error = None
    if compat_mode in {"auto", "off"}:
        try:
            pipeline.load_lora_weights(lora_path, use_safetensors=True)
            log.info("LightX2V loaded via pipeline.load_lora_weights")
            return "native", {"method": "native"}
        except Exception as exc:
            native_error = exc
            if compat_mode == "off":
                raise
            log.warning("Native load_lora_weights failed. Falling back to compat merge: %s", exc)

    if not hasattr(pipeline, "transformer"):
        raise AttributeError("Pipeline has no `transformer` attribute required for compatibility merge.")

    if expected_blocks is None:
        inferred_blocks = None
        try:
            if hasattr(pipeline.transformer, "blocks"):
                inferred_blocks = len(pipeline.transformer.blocks)
        except Exception:
            inferred_blocks = None
        expected_blocks = inferred_blocks

    state_dict = load_file(lora_path)
    stats = apply_lightx2v_state_dict_to_transformer(
        transformer=pipeline.transformer,
        state_dict=state_dict,
        lora_scale=lora_scale,
        strict=compat_strict,
        expected_blocks=expected_blocks,
        apply_non_block_diff=apply_non_block_diff,
        logger=log,
    )
    if native_error is not None:
        stats["native_error"] = str(native_error)
    stats["method"] = "compat"
    return "compat", stats
