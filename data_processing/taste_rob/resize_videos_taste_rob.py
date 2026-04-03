#!/usr/bin/env python3
"""Resize TASTE-Rob videos and create static-video pairs.

Input:
  data/taste_rob/{SingleHand,DoubleHand}/{subdir}/*.mp4

Output:
  data/taste_rob_resized/{SingleHand,DoubleHand}/{subdir}/videos/*.mp4
  data/taste_rob_resized/{SingleHand,DoubleHand}/{subdir}/videos_static/*.mp4
"""

import argparse
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

def run_cmd(cmd: list[str]) -> tuple[bool, str]:
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode == 0:
        return True, "ok"
    msg = result.stderr.strip().splitlines()[-1] if result.stderr else "command failed"
    return False, msg


def ffmpeg_resize_pad_filter(target_w: int, target_h: int, output_fps: int) -> str:
    return (
        f"fps={output_fps},"
        f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
        f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black,"
        "setsar=1"
    )


def build_video_cmd(
    input_video: Path,
    output_video: Path,
    target_w: int,
    target_h: int,
    target_frames: int,
    output_fps: int,
    video_codec: str,
    crf: int,
    preset: str,
) -> list[str]:
    vf = ffmpeg_resize_pad_filter(target_w, target_h, output_fps)
    return [
        "ffmpeg",
        "-y",
        "-i",
        str(input_video),
        "-vf",
        vf,
        "-frames:v",
        str(target_frames),
        "-c:v",
        video_codec,
        "-crf",
        str(crf),
        "-preset",
        preset,
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(output_video),
    ]


def build_extract_first_frame_cmd(
    input_video: Path,
    first_frame_path: Path,
    target_w: int,
    target_h: int,
) -> list[str]:
    vf = (
        "select=eq(n\\,0),"
        f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
        f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black,"
        "setsar=1"
    )
    return [
        "ffmpeg",
        "-y",
        "-i",
        str(input_video),
        "-vf",
        vf,
        "-frames:v",
        "1",
        str(first_frame_path),
    ]


def build_static_video_cmd(
    first_frame_path: Path,
    out_static: Path,
    target_frames: int,
    output_fps: int,
    video_codec: str,
    static_crf: int,
    static_preset: str,
) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-loop",
        "1",
        "-framerate",
        str(output_fps),
        "-i",
        str(first_frame_path),
        "-frames:v",
        str(target_frames),
        "-c:v",
        video_codec,
        "-crf",
        str(static_crf),
        "-preset",
        static_preset,
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(out_static),
    ]


def process_video(
    input_video: Path,
    input_root: Path,
    output_root: Path,
    target_w: int,
    target_h: int,
    target_frames: int,
    output_fps: int,
    video_codec: str,
    crf: int,
    preset: str,
    static_crf: int,
    static_preset: str,
    skip_existing: bool,
    merge_scene_prefix: bool,
    scene_case_map: dict[str, str],
) -> tuple[bool, str]:
    rel = input_video.relative_to(input_root)
    rel_parent = rel.parent
    out_parent = build_output_parent(
        rel_parent=rel_parent,
        input_root=input_root,
        output_root=output_root,
        merge_scene_prefix=merge_scene_prefix,
        scene_case_map=scene_case_map,
    )
    out_video = out_parent / "videos" / rel.name
    out_static = out_parent / "videos_static" / rel.name

    need_video = not (skip_existing and out_video.exists())
    need_static = not (skip_existing and out_static.exists())
    if not need_video and not need_static:
        return True, f"skipped {rel}"

    # Keep existing files on path collisions when scenes are merged.
    if merge_scene_prefix and need_video and out_video.exists():
        need_video = False
    if merge_scene_prefix and need_static and out_static.exists():
        need_static = False
    if not need_video and not need_static:
        return True, f"conflict-skip {rel}"

    if need_video:
        out_video.parent.mkdir(parents=True, exist_ok=True)
        ok, msg = run_cmd(
            build_video_cmd(
                input_video=input_video,
                output_video=out_video,
                target_w=target_w,
                target_h=target_h,
                target_frames=target_frames,
                output_fps=output_fps,
                video_codec=video_codec,
                crf=crf,
                preset=preset,
            )
        )
        if not ok:
            return False, f"video failed {rel}: {msg}"

    if need_static:
        out_static.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="taste_rob_static_") as tmp_dir:
            first_frame = Path(tmp_dir) / "first_frame.png"
            ok, msg = run_cmd(
                build_extract_first_frame_cmd(
                    input_video=input_video,
                    first_frame_path=first_frame,
                    target_w=target_w,
                    target_h=target_h,
                )
            )
            if not ok:
                return False, f"extract first frame failed {rel}: {msg}"
            ok, msg = run_cmd(
                build_static_video_cmd(
                    first_frame_path=first_frame,
                    out_static=out_static,
                    target_frames=target_frames,
                    output_fps=output_fps,
                    video_codec=video_codec,
                    static_crf=static_crf,
                    static_preset=static_preset,
                )
            )
            if not ok:
                return False, f"static failed {rel}: {msg}"

    return True, f"processed {rel}"


def process_video_job(args: tuple) -> tuple[bool, str]:
    return process_video(*args)


def normalize_scene_name(scene_name: str, scene_case_map: dict[str, str]) -> str:
    prefix = scene_name.split("_")[0]
    return scene_case_map.get(prefix.lower(), prefix)


def infer_hand_scene(rel_parent: Path, input_root: Path) -> tuple[str, str]:
    parts = rel_parent.parts
    hand_candidates = {"SingleHand", "DoubleHand"}

    if len(parts) >= 2 and parts[0] in hand_candidates:
        return parts[0], parts[1]

    if len(parts) >= 1 and input_root.name in hand_candidates:
        return input_root.name, parts[0]

    if len(parts) >= 2:
        return parts[0], parts[1]

    raise ValueError(f"Cannot infer hand/scene from relative parent path: {rel_parent}")


def build_output_parent(
    rel_parent: Path,
    input_root: Path,
    output_root: Path,
    merge_scene_prefix: bool,
    scene_case_map: dict[str, str],
) -> Path:
    if not merge_scene_prefix:
        return output_root / rel_parent

    hand, scene = infer_hand_scene(rel_parent=rel_parent, input_root=input_root)
    merged_scene = normalize_scene_name(scene, scene_case_map)

    if output_root.name == hand:
        return output_root / merged_scene
    return output_root / hand / merged_scene


def build_scene_case_map(
    videos: list[Path],
    input_root: Path,
    merge_scene_prefix: bool,
) -> dict[str, str]:
    if not merge_scene_prefix:
        return {}

    scene_case_map: dict[str, str] = {}
    for v in videos:
        rel = v.relative_to(input_root)
        try:
            _, scene = infer_hand_scene(rel_parent=rel.parent, input_root=input_root)
        except ValueError:
            continue
        prefix = scene.split("_")[0]
        key = prefix.lower()
        if key not in scene_case_map:
            scene_case_map[key] = prefix
    return scene_case_map


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Resize TASTE-Rob videos and build videos/videos_static pairs.")
    p.add_argument("--input_dir", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--target_width", type=int, default=720)
    p.add_argument("--target_height", type=int, default=480)
    p.add_argument("--target_frames", type=int, default=49)
    p.add_argument("--output_fps", type=int, default=8, help="Output FPS used by ffmpeg encoding.")
    p.add_argument("--video_codec", type=str, default="libx264")
    p.add_argument("--crf", type=int, default=23)
    p.add_argument("--preset", type=str, default="medium")
    p.add_argument("--static_crf", type=int, default=23)
    p.add_argument("--static_preset", type=str, default="medium")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument(
        "--merge_scene_prefix",
        action="store_true",
        help="Merge scene directories by scene.split('_')[0] in output paths.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_root = args.input_dir
    output_root = args.output_dir
    videos = sorted(input_root.rglob("*.mp4"))
    if not videos:
        raise SystemExit(f"No .mp4 files found under {input_root}")
    scene_case_map = build_scene_case_map(
        videos=videos,
        input_root=input_root,
        merge_scene_prefix=args.merge_scene_prefix,
    )

    jobs = [
        (
            v,
            input_root,
            output_root,
            args.target_width,
            args.target_height,
            args.target_frames,
            args.output_fps,
            args.video_codec,
            args.crf,
            args.preset,
            args.static_crf,
            args.static_preset,
            args.skip_existing,
            args.merge_scene_prefix,
            scene_case_map,
        )
        for v in videos
    ]

    ok_count = 0
    fail_count = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for ok, msg in ex.map(process_video_job, jobs):
            print(msg)
            if ok:
                ok_count += 1
            else:
                fail_count += 1

    print(f"Done. success={ok_count}, failed={fail_count}")


if __name__ == "__main__":
    main()
