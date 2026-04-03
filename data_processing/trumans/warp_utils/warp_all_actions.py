#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import subprocess
from collections import deque
from natsort import natsorted


def find_actions(base_root: Path):
    actions = []
    for scene_dir in natsorted(base_root.iterdir(), key=lambda p: p.name):
        if not scene_dir.is_dir():
            continue
        for action_dir in natsorted(scene_dir.iterdir(), key=lambda p: p.name):
            if not action_dir.is_dir():
                continue
            seq_dir = action_dir / "sequences"
            if not seq_dir.exists():
                continue
            images_dir = seq_dir / "images_static"
            depth_dir = seq_dir / "depth_static"
            cam_dir = action_dir / "cam_params"
            if images_dir.exists() and depth_dir.exists() and cam_dir.exists():
                actions.append((scene_dir.name, action_dir.name, action_dir))
    return actions


def main():
    parser = argparse.ArgumentParser(description="Batch runner for warping all actions with GPU parallelism")
    parser.add_argument("--base-root", type=str, required=True, help="Base path, e.g., ./data/trumans/ego_render_fov90")
    parser.add_argument("--script", type=str, default="data_processing/trumans/warp_utils/warp_single_action.py",
                        help="Path to single-action warp script")
    parser.add_argument("--gpu-indices", type=str, default="0",
                        help="Comma-separated GPU indices, e.g., 0,1,2")
    parser.add_argument("--max-procs", type=int, default=0,
                        help="Max concurrent processes (0 => len(gpu-indices)")
    parser.add_argument("--output", type=str, default="",
                        help="Output subdirectory name (relative to action root); default uses 'warped_depth'")
    parser.add_argument("--no-skip-existing", action="store_true",
                        help="Recreate outputs even if they exist")
    parser.add_argument("--dry-run", action="store_true", help="Only print commands")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-action batch size to pass to warp_single_action.py (default: 4)")
    args = parser.parse_args()

    base_root = Path(args.base_root)
    if not base_root.exists():
        print(f"Base root not found: {base_root}")
        return

    actions = find_actions(base_root)
    if not actions:
        print("No valid actions found (need sequences/images_static & depth_static, and cam_params)")
        return

    print(f"Found {len(actions)} actions to process")

    gpu_indices = [g.strip() for g in args.gpu_indices.split(',') if g.strip()]
    if not gpu_indices:
        gpu_indices = ["0"]
    queue_gpus = deque(gpu_indices)
    max_procs = args.max_procs if args.max_procs > 0 else len(gpu_indices)

    procs = []
    failures = []

    for scene_name, action_name, action_dir in actions:
        # Backpressure: wait until we have a free slot
        while len(procs) >= max_procs:
            still_running = []
            for (p, info) in procs:
                if p.poll() is None:
                    still_running.append((p, info))
                else:
                    if p.returncode != 0:
                        failures.append(info)
                        print(f"❌ Failed: {info}")
                    else:
                        print(f"✅ Done: {info}")
            procs = still_running

        # Assign next GPU
        gpu = queue_gpus[0]
        queue_gpus.rotate(-1)

        cmd = [
            "python", args.script,
            "--data_root", str(action_dir),
        ]
        if args.output:
            cmd.extend(["--output", args.output])
        if args.no_skip_existing:
            cmd.append("--no-skip-existing")
        if args.batch_size:
            cmd.extend(["--batch-size", str(args.batch_size)])

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        # Limit CPU thread over-subscription per process to improve parallel throughput
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        env.setdefault("NUMEXPR_NUM_THREADS", "1")
        env.setdefault("OPENCV_OPENCL_RUNTIME", "")

        info = f"{scene_name}/{action_name} (GPU {gpu})"
        print(f"▶️  {info}: {' '.join(cmd)}")
        if args.dry_run:
            continue
        p = subprocess.Popen(cmd, env=env)
        procs.append((p, info))

    # Drain remaining
    for (p, info) in procs:
        p.wait()
        if p.returncode != 0:
            failures.append(info)
            print(f"❌ Failed: {info}")
        else:
            print(f"✅ Done: {info}")

    if failures:
        print(f"\nFailed actions ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
    else:
        print("\nAll actions processed successfully")


if __name__ == "__main__":
    main()


