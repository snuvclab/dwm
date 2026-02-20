#!/usr/bin/env python3
"""Batch render hand-mesh videos via hamer-mediapipe.

Input layout (generic):
  data_root/**/{video_dir_name}/*.mp4

Output layout:
  data_root/**/{output_dir_name}/*.mp4
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

VALID_TYPES = {"SingleHand", "DoubleHand"}


@dataclass(frozen=True)
class Task:
    input_video: str
    output_video: str


def parse_gpus(raw: str) -> list[int]:
    vals = []
    for x in raw.split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(int(x))
    if not vals:
        raise ValueError("--gpus must contain at least one GPU id, e.g. '0' or '0,1'")
    return vals


def discover_videos(data_root: Path, video_dir_name: str) -> Iterable[Path]:
    # Scan all nested video directories to make this script dataset-agnostic.
    pattern = f"**/{video_dir_name}/*.mp4"
    for video_path in sorted(data_root.glob(pattern)):
        if video_path.is_file():
            yield video_path


def classify_type(video_path: Path) -> str | None:
    for part in video_path.parts:
        if part in VALID_TYPES:
            return part
    return None


def extract_scene(video_path: Path, video_dir_name: str) -> str | None:
    if video_path.parent.name != video_dir_name:
        return None
    return video_path.parent.parent.name if video_path.parent.parent else None


def load_video_list_file(data_root: Path, list_file: Path) -> tuple[list[Path], int]:
    videos: list[Path] = []
    missing_count = 0
    if not list_file.exists():
        raise FileNotFoundError(f"video_list_file not found: {list_file}")

    with list_file.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            rel = Path(line)
            if rel.is_absolute():
                print(f"[WARN] {list_file}:{lineno} absolute path is not allowed, skip: {line}")
                continue
            abs_path = (data_root / rel).resolve()
            if not abs_path.exists():
                missing_count += 1
                print(f"[WARN] {list_file}:{lineno} file not found, skip: {line}")
                continue
            videos.append(abs_path)

    # Deduplicate while preserving deterministic order.
    dedup = sorted(set(videos))
    return dedup, missing_count


def relative_to_root(path: Path, data_root: Path) -> str:
    return str(path.resolve().relative_to(data_root.resolve()))


def run_single_task(
    task: Task,
    gpu_id: int,
    python_bin: str,
    hamer_demo_path: Path,
    renderer: str,
    mesh_only: bool,
    checkpoint: str | None,
    fps: int | None,
    hamer_extra_args: list[str],
) -> tuple[bool, str]:
    input_video = Path(task.input_video)
    output_video = Path(task.output_video)
    output_dir = output_video.parent
    output_stem_video = output_dir / f"{input_video.stem}.mp4"

    cmd = [
        python_bin,
        str(hamer_demo_path),
        "--input",
        str(input_video),
        "--out_folder",
        str(output_dir),
        "--device",
        "cuda:0",
        "--renderer",
        renderer,
        "--save_video",
    ]

    if mesh_only:
        cmd.append("--mesh_only")
    if checkpoint:
        cmd.extend(["--checkpoint", checkpoint])
    if fps is not None:
        cmd.extend(["--fps", str(fps)])
    if hamer_extra_args:
        cmd.extend(hamer_extra_args)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    start = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    elapsed = time.time() - start

    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        msg = stderr.splitlines()[-1] if stderr else (stdout.splitlines()[-1] if stdout else "command failed")
        return False, f"rc={proc.returncode}, elapsed={elapsed:.2f}s, err={msg}"

    if not output_stem_video.exists():
        return False, f"rc=0 but output not found: {output_stem_video}"

    if output_stem_video != output_video:
        output_stem_video.rename(output_video)

    return True, f"elapsed={elapsed:.2f}s"


def worker_loop(
    worker_name: str,
    gpu_id: int,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    stop_event: mp.Event,
    python_bin: str,
    hamer_demo_path: str,
    renderer: str,
    mesh_only: bool,
    checkpoint: str | None,
    fps: int | None,
    hamer_extra_args: list[str],
) -> None:
    demo_path = Path(hamer_demo_path)
    while True:
        if stop_event.is_set():
            break
        try:
            item = task_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        if item is None:
            break

        task = Task(**item)
        ok, detail = run_single_task(
            task=task,
            gpu_id=gpu_id,
            python_bin=python_bin,
            hamer_demo_path=demo_path,
            renderer=renderer,
            mesh_only=mesh_only,
            checkpoint=checkpoint,
            fps=fps,
            hamer_extra_args=hamer_extra_args,
        )
        result_queue.put(
            {
                "worker": worker_name,
                "gpu": gpu_id,
                "input": task.input_video,
                "output": task.output_video,
                "status": "ok" if ok else "failed",
                "detail": detail,
            }
        )


def thread_worker_loop(
    worker_name: str,
    gpu_id: int,
    task_queue: queue.Queue,
    result_queue: queue.Queue,
    stop_event: threading.Event,
    python_bin: str,
    hamer_demo_path: str,
    renderer: str,
    mesh_only: bool,
    checkpoint: str | None,
    fps: int | None,
    hamer_extra_args: list[str],
) -> None:
    demo_path = Path(hamer_demo_path)
    while True:
        if stop_event.is_set():
            break
        try:
            item = task_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        if item is None:
            break

        task = Task(**item)
        ok, detail = run_single_task(
            task=task,
            gpu_id=gpu_id,
            python_bin=python_bin,
            hamer_demo_path=demo_path,
            renderer=renderer,
            mesh_only=mesh_only,
            checkpoint=checkpoint,
            fps=fps,
            hamer_extra_args=hamer_extra_args,
        )
        result_queue.put(
            {
                "worker": worker_name,
                "gpu": gpu_id,
                "input": task.input_video,
                "output": task.output_video,
                "status": "ok" if ok else "failed",
                "detail": detail,
            }
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render videos_hands with hamer-mediapipe.")
    parser.add_argument("--data_root", type=Path, default=Path("data/taste_rob_resized"))
    parser.add_argument("--video_dir_name", type=str, default="videos")
    parser.add_argument("--output_dir_name", type=str, default="videos_hands")
    parser.add_argument(
        "--hand_type",
        type=str,
        default="all",
        choices=["all", "SingleHand", "DoubleHand"],
        help="Filter by hand split in path.",
    )
    parser.add_argument("--scene", nargs="+", default=None, help="Scene names to include, e.g. --scene Dinning Kitchen")
    parser.add_argument("--video_list_file", type=Path, default=None, help="Text file of relative paths from data_root")
    parser.add_argument("--save_video_list", type=Path, default=None, help="Save final selected relative paths")
    parser.add_argument("--python_bin", type=str, default="python")
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--workers_per_gpu", type=int, default=1)

    skip_group = parser.add_mutually_exclusive_group()
    skip_group.add_argument("--skip_existing", action="store_true", default=True)
    skip_group.add_argument("--overwrite", action="store_true", help="Re-render even if output exists.")

    parser.add_argument("--fail_fast", action="store_true")
    parser.add_argument("--renderer", type=str, default="pytorch3d", choices=["pyrender", "pytorch3d"])
    parser.add_argument(
        "--mesh_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render only mesh on black background (default: true).",
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--fps", type=int, default=None, help="Optional output fps override")
    parser.add_argument(
        "--hamer_extra_args",
        type=str,
        default="",
        help="Extra args appended to demo_mediapipe.py, e.g. '--hand_colors'",
    )
    parser.add_argument(
        "--log_jsonl",
        type=Path,
        default=None,
        help="Default: <data_root>/_logs/hamer_render.jsonl",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.data_root.exists():
        raise SystemExit(f"data_root not found: {args.data_root}")
    if args.workers_per_gpu < 1:
        raise SystemExit("workers_per_gpu must be >= 1")

    repo_root = Path(__file__).resolve().parents[2]
    hamer_demo_path = repo_root / "third_party" / "hamer-mediapipe" / "demo_mediapipe.py"
    if not hamer_demo_path.exists():
        raise SystemExit(
            f"hamer-mediapipe demo not found: {hamer_demo_path}\n"
            "Did you initialize submodules?\n"
            "  git submodule update --init --recursive"
        )

    log_jsonl = args.log_jsonl or (args.data_root / "_logs" / "hamer_render.jsonl")
    log_jsonl.parent.mkdir(parents=True, exist_ok=True)

    gpus = parse_gpus(args.gpus)
    hamer_extra_args = shlex.split(args.hamer_extra_args) if args.hamer_extra_args else []
    scene_filter = set(args.scene) if args.scene else None

    if args.video_list_file is None:
        candidates = list(discover_videos(data_root=args.data_root, video_dir_name=args.video_dir_name))
        missing_from_list = 0
    else:
        candidates, missing_from_list = load_video_list_file(data_root=args.data_root, list_file=args.video_list_file)

    selected_videos: list[Path] = []
    filtered_out = 0
    for video_path in candidates:
        hand_type = classify_type(video_path)
        if args.hand_type != "all" and hand_type != args.hand_type:
            filtered_out += 1
            continue

        scene_name = extract_scene(video_path, args.video_dir_name)
        if scene_filter is not None and scene_name not in scene_filter:
            filtered_out += 1
            continue

        selected_videos.append(video_path)

    if args.save_video_list is not None:
        args.save_video_list.parent.mkdir(parents=True, exist_ok=True)
        with args.save_video_list.open("w", encoding="utf-8") as f:
            for path in sorted(selected_videos):
                f.write(relative_to_root(path, args.data_root) + "\n")

    all_tasks: list[Task] = []
    skipped_records: list[dict] = []
    for video_path in selected_videos:
        output_video = video_path.parent.parent / args.output_dir_name / video_path.name
        output_video.parent.mkdir(parents=True, exist_ok=True)

        if args.skip_existing and not args.overwrite and output_video.exists():
            skipped_records.append(
                {
                    "gpu": None,
                    "worker": None,
                    "input": str(video_path),
                    "output": str(output_video),
                    "status": "skipped_existing",
                    "detail": "exists",
                }
            )
            continue

        all_tasks.append(Task(input_video=str(video_path), output_video=str(output_video)))

    with log_jsonl.open("a", encoding="utf-8") as f:
        for rec in skipped_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if not all_tasks:
        print("No tasks to run. (all skipped or no input videos found)")
        print(f"selected_total={len(selected_videos)}")
        print(f"filtered_out={filtered_out}")
        print(f"missing_from_list={missing_from_list}")
        print(f"Log: {log_jsonl}")
        return 0

    workers: list = []
    worker_slots: list[tuple[int, int]] = []
    for gpu_id in gpus:
        for worker_idx in range(args.workers_per_gpu):
            worker_slots.append((gpu_id, worker_idx))

    backend = "process"
    try:
        ctx = mp.get_context("spawn")
        task_queue = ctx.Queue()
        result_queue = ctx.Queue()
        stop_event = ctx.Event()
        for gpu_id, worker_idx in worker_slots:
            worker_name = f"gpu{gpu_id}-w{worker_idx}"
            p = ctx.Process(
                target=worker_loop,
                args=(
                    worker_name,
                    gpu_id,
                    task_queue,
                    result_queue,
                    stop_event,
                    args.python_bin,
                    str(hamer_demo_path),
                    args.renderer,
                    args.mesh_only,
                    args.checkpoint,
                    args.fps,
                    hamer_extra_args,
                ),
                daemon=True,
            )
            p.start()
            workers.append(p)
    except (PermissionError, OSError) as exc:
        backend = "thread"
        print(f"[WARN] multiprocessing unavailable ({exc}); fallback to thread backend.")
        task_queue = queue.Queue()
        result_queue = queue.Queue()
        stop_event = threading.Event()
        workers = []
        for gpu_id, worker_idx in worker_slots:
            worker_name = f"gpu{gpu_id}-w{worker_idx}"
            t = threading.Thread(
                target=thread_worker_loop,
                args=(
                    worker_name,
                    gpu_id,
                    task_queue,
                    result_queue,
                    stop_event,
                    args.python_bin,
                    str(hamer_demo_path),
                    args.renderer,
                    args.mesh_only,
                    args.checkpoint,
                    args.fps,
                    hamer_extra_args,
                ),
                daemon=True,
            )
            t.start()
            workers.append(t)

    for task in all_tasks:
        task_queue.put(task.__dict__)
    for _ in workers:
        task_queue.put(None)

    pending_inputs = {t.input_video for t in all_tasks}
    results: list[dict] = []
    failed = 0
    failed_fast_triggered = False

    while pending_inputs:
        try:
            rec = result_queue.get(timeout=1.0)
        except queue.Empty:
            if failed_fast_triggered:
                break
            if not any(p.is_alive() for p in workers):
                break
            continue

        input_path = rec.get("input")
        if input_path in pending_inputs:
            pending_inputs.remove(input_path)
        results.append(rec)

        if rec.get("status") == "failed":
            failed += 1
            if args.fail_fast:
                failed_fast_triggered = True
                stop_event.set()
                for p in workers:
                    if p.is_alive() and backend == "process":
                        p.terminate()
                break

    for p in workers:
        p.join(timeout=1.0)

    if failed_fast_triggered and pending_inputs:
        for input_path in sorted(pending_inputs):
            rec = {
                "gpu": None,
                "worker": None,
                "input": input_path,
                "output": "",
                "status": "skipped_fail_fast",
                "detail": "not executed due to fail_fast",
            }
            results.append(rec)

    with log_jsonl.open("a", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    ok_count = sum(1 for x in results if x.get("status") == "ok")
    failed_count = sum(1 for x in results if x.get("status") == "failed")
    skipped_count = len(skipped_records) + sum(1 for x in results if x.get("status") == "skipped_fail_fast")

    print("=== Summary ===")
    print(f"selected_total={len(selected_videos)}")
    print(f"filtered_out={filtered_out}")
    print(f"missing_from_list={missing_from_list}")
    print(f"queued={len(all_tasks)}")
    print(f"ok={ok_count}")
    print(f"failed={failed_count}")
    print(f"skipped={skipped_count}")
    print(f"log_jsonl={log_jsonl}")

    return 1 if failed_count > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
