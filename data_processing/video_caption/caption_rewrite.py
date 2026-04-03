#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import re
import time
from pathlib import Path


LOGGER = logging.getLogger("caption_rewrite")


def natural_sort_key(path: Path) -> tuple[int, int | str, str]:
    stem = path.stem
    if stem.isdigit():
        return (0, int(stem), str(path))
    return (1, stem, str(path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rewrite prompts for TASTE-Rob style root directory.")
    parser.add_argument("--root_dir", type=Path, required=True)
    parser.add_argument("--prompt_subdir", type=str, default="prompts")
    parser.add_argument("--output_folder_name", type=str, default="prompts_rewrite")
    parser.add_argument("--output_extension", type=str, choices=["auto", "txt", "json"], default="auto")
    parser.add_argument("--prompt_file", type=Path, required=True)
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--array_index", type=int, default=None)
    parser.add_argument("--num_splits", type=int, default=8)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--coordinated_jobs_manifest", type=Path, default=None)
    parser.add_argument("--coordinated_jobs_manifest_timeout_sec", type=int, default=1800)
    parser.add_argument("--max_retry_count", type=int, default=10)
    parser.add_argument("--prefix", type=str, default='"rewritten description": ')
    parser.add_argument("--answer_template", type=str, default="your rewritten description here")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--tensor_parallel_size", type=int, default=None)
    parser.add_argument("--engine", type=str, choices=["auto", "vllm", "transformers"], default="auto")
    parser.add_argument("--fail_on_any_error", action="store_true")
    return parser.parse_args()


def extract_output(text: str, prefix: str) -> str | None:
    match = re.search(r"{(.+?)}", text, re.DOTALL)
    if not match:
        return None

    output = match.group(1).strip()
    if not output.startswith(prefix):
        return None

    output = output[len(prefix) :]
    if len(output) < 2 or output[0] != '"' or output[-1] != '"':
        return None
    return output[1:-1]


def discover_prompt_files(root_dir: Path, prompt_subdir: str) -> list[Path]:
    files: list[Path] = []
    for pattern in (f"*/*/{prompt_subdir}/*.txt", f"*/*/{prompt_subdir}/*.json"):
        files.extend(root_dir.glob(pattern))
    files = sorted(files, key=natural_sort_key)
    return [p for p in files if p.is_file()]


def split_items(items: list[Path], array_index: int | None, num_splits: int) -> list[Path]:
    if array_index is None:
        return items
    if num_splits < 1:
        raise ValueError(f"num_splits must be >= 1, got {num_splits}")
    if array_index < 0 or array_index >= num_splits:
        raise ValueError(f"array_index out of range: {array_index} for num_splits={num_splits}")

    chunk = (len(items) + num_splits - 1) // num_splits
    start = array_index * chunk
    end = min(start + chunk, len(items))
    return items[start:end]


def load_prompt_template(prompt_file: Path) -> str:
    if not prompt_file.exists():
        raise FileNotFoundError(f"prompt_file not found: {prompt_file}")
    prompt = prompt_file.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"prompt_file is empty: {prompt_file}")
    return prompt


def load_input_prompt(input_path: Path) -> str | None:
    try:
        if input_path.suffix == ".json":
            data = json.loads(input_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                prompt = data.get("prompt", "")
                if isinstance(prompt, str) and prompt.strip():
                    return prompt.strip()
                environment = data.get("environment", None)
                action = data.get("action", None)
                if isinstance(environment, str) and environment.strip() and isinstance(action, str) and action.strip():
                    structured_payload = {
                        "environment": environment.strip(),
                        "action": action.strip(),
                    }
                    return json.dumps(structured_payload, ensure_ascii=False, indent=2)
                for value in data.values():
                    if isinstance(value, str) and value.strip():
                        return value.strip()
                return None
            text = str(data).strip()
            return text or None
        text = input_path.read_text(encoding="utf-8").strip()
        return text or None
    except Exception as exc:
        LOGGER.warning("Failed to read input prompt %s: %s", input_path, exc)
        return None


def output_path_for_input(
    input_path: Path,
    prompt_subdir: str,
    output_folder_name: str,
    output_extension: str,
) -> Path:
    prompt_dir = input_path.parent
    if prompt_dir.name != prompt_subdir:
        raise ValueError(f"Unexpected prompt directory: {prompt_dir}")
    action_dir = prompt_dir.parent
    output_dir = action_dir / output_folder_name
    if output_extension == "auto":
        suffix = input_path.suffix
    else:
        suffix = f".{output_extension}"
    return output_dir / f"{input_path.stem}{suffix}"


def atomic_write_text(save_path: Path, content: str, *, require_nonempty: bool = True) -> None:
    if require_nonempty and not content.strip():
        raise RuntimeError(f"Refusing to save empty text output: {save_path}")

    tmp_path = save_path.with_name(f".{save_path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, save_path)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise


def relative_input_path_from_root(input_path: Path, root_dir: Path) -> str:
    return input_path.relative_to(root_dir).as_posix()


def build_runnable_files(
    all_prompt_files: list[Path],
    root_dir: Path,
    prompt_subdir: str,
    output_folder_name: str,
    output_extension: str,
    skip_existing: bool,
) -> tuple[list[Path], int]:
    pre_skipped_existing = 0
    runnable_files: list[Path] = []
    for input_path in all_prompt_files:
        output_path = output_path_for_input(
            input_path,
            prompt_subdir,
            output_folder_name,
            output_extension,
        )
        if skip_existing and output_path.exists():
            pre_skipped_existing += 1
            continue
        runnable_files.append(input_path)
    return runnable_files, pre_skipped_existing


def load_manifest_files(
    manifest_path: Path,
    all_prompt_files: list[Path],
    root_dir: Path,
) -> tuple[list[Path], int]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    relpaths = payload.get("runnable_files", [])
    file_map = {
        relative_input_path_from_root(path, root_dir): path
        for path in all_prompt_files
    }
    runnable_files: list[Path] = []
    missing_relpaths: list[str] = []
    for relpath in relpaths:
        path = file_map.get(relpath)
        if path is None:
            missing_relpaths.append(relpath)
            continue
        runnable_files.append(path)
    if missing_relpaths:
        LOGGER.warning(
            "manifest_missing_files=%d first_missing=%s",
            len(missing_relpaths),
            missing_relpaths[:5],
        )
    return runnable_files, int(payload.get("pre_skipped_existing", 0))


def build_or_load_coordinated_manifest(
    manifest_path: Path,
    all_prompt_files: list[Path],
    root_dir: Path,
    prompt_subdir: str,
    output_folder_name: str,
    output_extension: str,
    skip_existing: bool,
    timeout_sec: int,
) -> tuple[list[Path], int]:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    lock_dir = manifest_path.with_suffix(f"{manifest_path.suffix}.lock")
    waited_sec = 0

    while True:
        if manifest_path.exists():
            LOGGER.info("using_existing_manifest=%s", manifest_path)
            return load_manifest_files(manifest_path, all_prompt_files, root_dir)
        try:
            lock_dir.mkdir()
            break
        except FileExistsError:
            if waited_sec >= timeout_sec:
                raise TimeoutError(
                    f"Timed out waiting for coordinated manifest {manifest_path} "
                    f"after {timeout_sec} seconds"
                )
            time.sleep(1.0)
            waited_sec += 1

    try:
        runnable_files, pre_skipped_existing = build_runnable_files(
            all_prompt_files=all_prompt_files,
            root_dir=root_dir,
            prompt_subdir=prompt_subdir,
            output_folder_name=output_folder_name,
            output_extension=output_extension,
            skip_existing=skip_existing,
        )
        payload = {
            "root_dir": str(root_dir),
            "prompt_subdir": prompt_subdir,
            "output_folder_name": output_folder_name,
            "output_extension": output_extension,
            "prompt_files_total": len(all_prompt_files),
            "pre_skipped_existing": pre_skipped_existing,
            "runnable_files": [
                relative_input_path_from_root(path, root_dir)
                for path in runnable_files
            ],
        }
        atomic_write_text(
            manifest_path,
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            require_nonempty=False,
        )
        LOGGER.info(
            "created_manifest=%s prompt_files_total=%d pre_skipped_existing=%d prompt_files_runnable=%d",
            manifest_path,
            len(all_prompt_files),
            pre_skipped_existing,
            len(runnable_files),
        )
    finally:
        try:
            lock_dir.rmdir()
        except OSError:
            pass

    return load_manifest_files(manifest_path, all_prompt_files, root_dir)


def patch_tokenizer_for_vllm(tokenizer) -> None:
    cls = tokenizer.__class__
    if not hasattr(cls, "all_special_tokens_extended"):
        cls.all_special_tokens_extended = property(lambda self: list(self.all_special_tokens))
    if not hasattr(tokenizer, "all_special_tokens_extended"):
        tokenizer.all_special_tokens_extended = list(tokenizer.all_special_tokens)


def resolve_tensor_parallel_size(args, torch_module) -> int:
    cuda_visible = os.getenv("CUDA_VISIBLE_DEVICES", None)
    if args.tensor_parallel_size is not None:
        return args.tensor_parallel_size
    if cuda_visible is None or not cuda_visible.strip():
        return torch_module.cuda.device_count() if torch_module.cuda.is_available() else 1
    tensor_parallel_size = len([x for x in cuda_visible.split(",") if x.strip()])
    return max(tensor_parallel_size, 1)


def build_vllm_rewriter(args):
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    tensor_parallel_size = resolve_tensor_parallel_size(args, torch)
    tokenizer_name = (
        "NousResearch/Meta-Llama-3-8B-Instruct"
        if "Meta-Llama-3-70B" in args.model_name
        else args.model_name
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    patch_tokenizer_for_vllm(tokenizer)
    llm = LLM(
        model=args.model_name,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    if "Meta-Llama-3" in args.model_name:
        stop_token_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        stop_token_ids = [token_id for token_id in stop_token_ids if token_id is not None]
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=1,
            max_tokens=args.max_tokens,
            stop_token_ids=stop_token_ids,
        )
    else:
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=1,
            max_tokens=args.max_tokens,
        )

    def rewrite(messages: list[dict[str, str]]) -> str:
        llm_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = llm.generate([llm_prompt], sampling_params)
        return outputs[0].outputs[0].text.rstrip()

    def cleanup() -> None:
        torch.cuda.empty_cache()
        gc.collect()

    return rewrite, cleanup, "vllm"


def build_transformers_rewriter(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer_name = (
        "NousResearch/Meta-Llama-3-8B-Instruct"
        if "Meta-Llama-3-70B" in args.model_name
        else args.model_name
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    model.eval()

    terminators = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    eot_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if eot_token_id is not None:
        terminators.append(eot_token_id)

    def rewrite(messages: list[dict[str, str]]) -> str:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer(prompt, return_tensors="pt")
        model_device = next(model.parameters()).device
        model_inputs = {key: value.to(model_device) for key, value in model_inputs.items()}
        with torch.inference_mode():
            generated = model.generate(
                **model_inputs,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=1.0,
                do_sample=args.temperature > 0,
                eos_token_id=terminators if terminators else None,
                pad_token_id=tokenizer.pad_token_id,
            )
        prompt_length = model_inputs["input_ids"].shape[-1]
        generated_tokens = generated[0][prompt_length:]
        return tokenizer.decode(generated_tokens, skip_special_tokens=True).rstrip()

    def cleanup() -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return rewrite, cleanup, "transformers"


def build_rewriter(args):
    backends = [args.engine] if args.engine != "auto" else ["vllm", "transformers"]
    last_error = None
    for backend in backends:
        try:
            if backend == "vllm":
                return build_vllm_rewriter(args)
            if backend == "transformers":
                return build_transformers_rewriter(args)
        except Exception as exc:
            last_error = exc
            LOGGER.warning("failed_to_initialize_%s_rewriter: %s", backend, exc)
            if args.engine != "auto":
                raise
    if last_error is not None:
        raise RuntimeError(f"Failed to initialize any rewrite backend for model={args.model_name}") from last_error
    raise RuntimeError("No rewrite backend configured")


def main() -> None:
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    if not args.root_dir.exists():
        raise SystemExit(f"root_dir not found: {args.root_dir}")
    if args.max_retry_count < 1:
        raise SystemExit(f"max_retry_count must be >= 1, got {args.max_retry_count}")
    if args.output_extension == "auto":
        args.output_extension = "txt"

    base_prompt = load_prompt_template(args.prompt_file)
    all_prompt_files = discover_prompt_files(args.root_dir, args.prompt_subdir)
    if not all_prompt_files:
        raise SystemExit(
            f"No prompt files found under {args.root_dir}/*/*/{args.prompt_subdir}/*.txt or *.json"
        )

    if args.coordinated_jobs_manifest is not None:
        runnable_files, pre_skipped_existing = build_or_load_coordinated_manifest(
            manifest_path=args.coordinated_jobs_manifest,
            all_prompt_files=all_prompt_files,
            root_dir=args.root_dir,
            prompt_subdir=args.prompt_subdir,
            output_folder_name=args.output_folder_name,
            output_extension=args.output_extension,
            skip_existing=args.skip_existing,
            timeout_sec=args.coordinated_jobs_manifest_timeout_sec,
        )
    else:
        runnable_files, pre_skipped_existing = build_runnable_files(
            all_prompt_files=all_prompt_files,
            root_dir=args.root_dir,
            prompt_subdir=args.prompt_subdir,
            output_folder_name=args.output_folder_name,
            output_extension=args.output_extension,
            skip_existing=args.skip_existing,
        )

    selected_files = split_items(runnable_files, args.array_index, args.num_splits)
    LOGGER.info("prompt_files_total=%d", len(all_prompt_files))
    LOGGER.info("pre_skipped_existing=%d", pre_skipped_existing)
    LOGGER.info("prompt_files_runnable=%d", len(runnable_files))
    LOGGER.info("prompt_files_selected=%d", len(selected_files))
    if args.array_index is not None:
        LOGGER.info("array_index=%d, num_splits=%d", args.array_index, args.num_splits)
    if not selected_files:
        LOGGER.info("No files selected for this split. Exiting.")
        return

    rewrite_once, cleanup_once, backend_name = build_rewriter(args)
    LOGGER.info("rewrite_backend=%s", backend_name)

    files_seen = 0
    rewritten = 0
    failed = 0

    for input_path in selected_files:
        files_seen += 1
        output_path = output_path_for_input(
            input_path,
            args.prompt_subdir,
            args.output_folder_name,
            args.output_extension,
        )

        input_prompt = load_input_prompt(input_path)
        if not input_prompt:
            failed += 1
            LOGGER.warning("Empty or unreadable input prompt: %s", input_path)
            continue

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": base_prompt + "\n" + input_prompt},
        ]

        success = False
        for _ in range(args.max_retry_count):
            output_text = rewrite_once(messages)
            parsed = extract_output(output_text, args.prefix) if args.prefix is not None else output_text
            if parsed is not None and parsed != args.answer_template:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(str(parsed).strip() + "\n", encoding="utf-8")
                rewritten += 1
                success = True
                break

        if not success:
            failed += 1
            LOGGER.warning("Failed to rewrite after %d retries: %s", args.max_retry_count, input_path)

        cleanup_once()

    LOGGER.info("=== Summary ===")
    LOGGER.info("files_seen=%d", files_seen)
    LOGGER.info("rewritten=%d", rewritten)
    LOGGER.info("skipped_existing=%d", pre_skipped_existing)
    LOGGER.info("failed=%d", failed)
    if failed > 0:
        LOGGER.warning("rewrite had %d failed files; keeping shard exit successful", failed)
    if failed > 0 and args.fail_on_any_error:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
