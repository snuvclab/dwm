#!/usr/bin/env python3
"""Single-GPU batch renderer for hand-mesh videos via hamer-mediapipe.

Design:
- Single process / single visible GPU (`CUDA_VISIBLE_DEVICES`)
- Load HaMeR + MediaPipe once
- Process many videos without reloading model

For multi-GPU, use `launch_render_hands_hamer_multi_gpu.py`.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

VALID_TYPES = {"SingleHand", "DoubleHand"}


@dataclass(frozen=True)
class Task:
    input_video: str
    output_video: str


def discover_videos(data_root: Path, video_dir_name: str) -> Iterable[Path]:
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

    return sorted(set(videos)), missing_count


def relative_to_root(path: Path, data_root: Path) -> str:
    return str(path.resolve().relative_to(data_root.resolve()))


def load_demo_module(hamer_root: Path):
    demo_path = hamer_root / "demo_mediapipe.py"
    if not demo_path.exists():
        raise FileNotFoundError(f"demo_mediapipe.py not found: {demo_path}")

    # Allow `import hamer...` used inside demo_mediapipe.py
    sys.path.insert(0, str(hamer_root))

    spec = importlib.util.spec_from_file_location("hamer_demo_mediapipe", demo_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from: {demo_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_checkpoint(hamer_root: Path, override: str | None) -> Path:
    if override:
        ckpt = Path(override).resolve()
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint override not found: {ckpt}")
        return ckpt

    new_ckpt = hamer_root / "_DATA" / "hamer_ckpts" / "checkpoints" / "new_hamer_weights.ckpt"
    old_ckpt = hamer_root / "_DATA" / "hamer_ckpts" / "checkpoints" / "hamer.ckpt"
    if new_ckpt.exists():
        return new_ckpt
    if old_ckpt.exists():
        return old_ckpt
    raise FileNotFoundError(
        "No checkpoint found. Expected one of:\n"
        f"  - {new_ckpt}\n"
        f"  - {old_ckpt}\n"
        "Run `bash third_party/hamer-mediapipe/fetch_demo_data.sh` and place MANO/checkpoints."
    )


def run_video_task(task: Task, demo, model, model_cfg, mp_hands, args) -> tuple[bool, str]:
    input_video = Path(task.input_video).resolve()
    output_video = Path(task.output_video).resolve()
    output_dir = output_video.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    cap, frame, is_webcam, is_video, total_frames = demo.setup_input_source(str(input_video))
    if is_webcam or not is_video:
        if cap is not None:
            cap.release()
        raise RuntimeError(f"Expected video input, got: {input_video}")

    img_h, img_w = frame.shape[:2]
    render_res = (img_w, img_h)
    renderer = demo.Renderer(model_cfg, faces=model.mano.faces, render_res=render_res)

    input_fps = cap.get(demo.cv2.CAP_PROP_FPS)
    output_fps = args.fps if args.fps is not None else (input_fps if input_fps > 0 else 30)

    ffmpeg_proc, actual_output_path = demo.start_ffmpeg_writer(str(output_dir), input_video.stem, output_fps, img_w, img_h)

    render_opts = SimpleNamespace(
        mesh_only=args.mesh_only,
        renderer=args.renderer,
        hand_colors=args.hand_colors,
    )

    frame_id = 0
    try:
        while True:
            if frame_id > 0:
                ok, frame = cap.read()
                if not ok:
                    break

            img_rgb = demo.cv2.cvtColor(frame, demo.cv2.COLOR_BGR2RGB)
            results = mp_hands.process(img_rgb)
            bboxes, handed_list = demo.extract_mediapipe_hands(results, frame.shape)

            boxes_np = demo.np.array(bboxes, dtype=demo.np.float32)
            right_np = demo.np.array(handed_list, dtype=demo.np.float32)

            if boxes_np.ndim != 2 or boxes_np.shape[0] == 0:
                out_bgr = demo.np.zeros_like(frame) if args.mesh_only else frame
                ffmpeg_proc.stdin.write(out_bgr.tobytes())
                frame_id += 1
                continue

            img_size = demo.np.array(render_res)
            scaled_focal = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()

            batch_dict = demo.make_batch(model_cfg, frame, boxes_np, right_np, args.device_obj)
            all_verts, all_cam_t, all_right = demo.run_hamer_inference(model, batch_dict, scaled_focal)

            out_img_rgb = demo.render_frame(
                renderer,
                frame,
                all_verts,
                all_cam_t,
                all_right,
                img_size,
                scaled_focal,
                render_opts,
            )
            out_bgr = out_img_rgb[:, :, ::-1]
            ffmpeg_proc.stdin.write(out_bgr.tobytes())
            frame_id += 1

        ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait()
        if ffmpeg_proc.returncode != 0:
            stderr = ffmpeg_proc.stderr.read().decode(errors="replace")
            raise RuntimeError(f"ffmpeg failed: {stderr}")
        ffmpeg_proc.stderr.close()

        produced = output_dir / f"{input_video.stem}.mp4"
        if not produced.exists():
            raise RuntimeError(f"output not found: {produced}")
        if produced != output_video:
            produced.rename(output_video)

        return True, f"frames={frame_id}, out={actual_output_path}"
    finally:
        cap.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render videos_hands with hamer-mediapipe (single GPU, model cached).")
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
    parser.add_argument("--progress_every", type=int, default=10)

    skip_group = parser.add_mutually_exclusive_group()
    skip_group.add_argument("--skip_existing", action="store_true", default=True)
    skip_group.add_argument("--overwrite", action="store_true", help="Re-render even if output exists.")

    parser.add_argument("--fail_fast", action="store_true")
    parser.add_argument("--renderer", type=str, default="pytorch3d", choices=["pyrender", "pytorch3d"])
    parser.add_argument("--mesh_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hand_colors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--log_jsonl", type=Path, default=None, help="Default: <data_root>/_logs/hamer_render.jsonl")

    # split options for multi-gpu launcher
    parser.add_argument("--split_index", type=int, default=0)
    parser.add_argument("--num_splits", type=int, default=1)
    parser.add_argument("--enable_split", action="store_true")

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    data_root = args.data_root.resolve()
    if not data_root.exists():
        raise SystemExit(f"data_root not found: {data_root}")
    if args.progress_every < 1:
        raise SystemExit("progress_every must be >= 1")

    repo_root = Path(__file__).resolve().parents[2]
    hamer_root = repo_root / "third_party" / "hamer-mediapipe"
    if not hamer_root.exists():
        raise SystemExit(f"hamer-mediapipe not found: {hamer_root}")

    log_jsonl = (args.log_jsonl.resolve() if args.log_jsonl else (data_root / "_logs" / "hamer_render.jsonl"))
    log_jsonl.parent.mkdir(parents=True, exist_ok=True)

    if args.video_list_file is None:
        candidates = list(discover_videos(data_root=data_root, video_dir_name=args.video_dir_name))
        missing_from_list = 0
    else:
        candidates, missing_from_list = load_video_list_file(data_root=data_root, list_file=args.video_list_file)

    scene_filter = set(args.scene) if args.scene else None
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
        selected_videos.append(video_path.resolve())

    selected_videos = sorted(selected_videos)
    if args.enable_split:
        if args.num_splits < 1:
            raise SystemExit("num_splits must be >= 1 when --enable_split is used")
        if args.split_index < 0 or args.split_index >= args.num_splits:
            raise SystemExit("split_index must satisfy 0 <= split_index < num_splits")
        selected_videos = [v for i, v in enumerate(selected_videos) if (i % args.num_splits) == args.split_index]

    if args.save_video_list is not None:
        save_list = args.save_video_list.resolve()
        save_list.parent.mkdir(parents=True, exist_ok=True)
        with save_list.open("w", encoding="utf-8") as f:
            for path in selected_videos:
                f.write(relative_to_root(path, data_root) + "\n")

    print("=== Config ===")
    print(f"data_root={data_root}")
    print(f"video_dir_name={args.video_dir_name}")
    print(f"output_dir_name={args.output_dir_name}")
    print(f"hand_type={args.hand_type}")
    print(f"scene_filter={sorted(scene_filter) if scene_filter is not None else 'None'}")
    print(f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}")
    print(f"hamer_root={hamer_root}")
    print(f"split_enabled={args.enable_split}")
    if args.enable_split:
        print(f"split_index={args.split_index}")
        print(f"num_splits={args.num_splits}")
    print(f"log_jsonl={log_jsonl}")

    all_tasks: list[Task] = []
    skipped_records: list[dict] = []
    for video_path in selected_videos:
        output_video = video_path.parent.parent / args.output_dir_name / video_path.name
        if args.skip_existing and not args.overwrite and output_video.exists():
            skipped_records.append(
                {
                    "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
                    "worker": "single-gpu",
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
        print(f"log_jsonl={log_jsonl}")
        return 0

    print("=== Plan ===")
    print(f"selected_total={len(selected_videos)}")
    print(f"filtered_out={filtered_out}")
    print(f"missing_from_list={missing_from_list}")
    print(f"skipped_existing={len(skipped_records)}")
    print(f"queued={len(all_tasks)}")

    # Make all hamer-relative paths (./_DATA/...) resolve under submodule root.
    orig_cwd = Path.cwd()
    os.chdir(hamer_root)

    ok_count = 0
    failed_count = 0
    skipped_fail_fast_count = 0
    total = len(all_tasks)
    start_time = time.time()

    try:
        demo = load_demo_module(hamer_root)

        device_str = "cuda:0" if demo.torch.cuda.is_available() else "cpu"
        args.device_obj = demo.torch.device(device_str)

        # Load models once.
        demo.download_models(str(hamer_root / "_DATA"))
        ckpt = resolve_checkpoint(hamer_root, args.checkpoint)
        model, model_cfg = demo.load_hamer(str(ckpt))
        model = model.to(args.device_obj).eval()

        mp_hands = demo.mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
        )

        for idx, task in enumerate(all_tasks, start=1):
            print(f"[START] {idx}/{total} input={Path(task.input_video).name}")
            try:
                ok, detail = run_video_task(task, demo, model, model_cfg, mp_hands, args)
            except Exception as exc:
                ok = False
                detail = str(exc)

            rec = {
                "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
                "worker": "single-gpu",
                "input": task.input_video,
                "output": task.output_video,
                "status": "ok" if ok else "failed",
                "detail": detail,
            }
            with log_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if ok:
                ok_count += 1
            else:
                failed_count += 1
                print(f"[FAILED] {idx}/{total} input={Path(task.input_video).name} detail={detail}")
                if args.fail_fast:
                    for rem in all_tasks[idx:]:
                        skip_rec = {
                            "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
                            "worker": "single-gpu",
                            "input": rem.input_video,
                            "output": rem.output_video,
                            "status": "skipped_fail_fast",
                            "detail": "not executed due to fail_fast",
                        }
                        with log_jsonl.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(skip_rec, ensure_ascii=False) + "\n")
                        skipped_fail_fast_count += 1
                    break

            if idx == 1 or idx % args.progress_every == 0 or idx == total:
                elapsed = time.time() - start_time
                print(
                    f"[PROGRESS] {idx}/{total} done "
                    f"(ok={ok_count}, failed={failed_count}, skipped_ff={skipped_fail_fast_count}, elapsed={elapsed:.1f}s)"
                )

        try:
            mp_hands.close()
        except Exception:
            pass
    finally:
        os.chdir(orig_cwd)

    skipped_count = len(skipped_records) + skipped_fail_fast_count

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
