#!/usr/bin/env python3

import argparse
import os
import sys
import time
from pathlib import Path

import bpy

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from rendering_common import (  # noqa: E402
    EYE_MESH_NAME,
    FACIAL_BONE_NAME,
    RenderErrorLogger,
    apply_animation_set,
    configure_cycles_devices,
    create_ego_camera,
    discover_armature_name,
    ensure_collection_hidden,
    file_ok,
    get_animation_sets,
    get_output_folder,
    get_required_scene_objects,
    optimize_scene_for_cycles,
    parse_blender_argv,
    reset_armature_pose,
    resolve_animations_to_render,
    set_actor_render_hidden,
    strip_animation_extension,
)
from rendering_static_mode import StaticSceneController  # noqa: E402
from rendering_strategies import build_clip_start_indices, build_frame_numbers  # noqa: E402


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_frame", type=int, default=None)
    parser.add_argument("--end_frame", type=int, default=None)
    parser.add_argument("--animation_index", type=int, default=None, help="Specific animation index")
    parser.add_argument("--animation_name", type=str, default=None, help="Specific animation name")
    parser.add_argument("--samples", type=int, default=32, help="Cycles samples")
    parser.add_argument(
        "--save-path",
        type=str,
        default="/home/byungjun/workspace/trumans_ego/ego_render_new",
        help="Root output dir",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip clips that already exist",
    )
    parser.add_argument("--no-skip-existing", action="store_true", help="Disable skipping existing files")
    parser.add_argument("--frame-skip", type=int, default=3, help="Render every Nth frame")
    parser.add_argument("--stride", "--clip-stride", dest="stride", type=int, default=25)
    parser.add_argument("--clip-length", type=int, default=49)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--fov", type=float, default=90.0, help="Camera FOV in degrees")
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--direct-clips", action="store_true", help="Accepted for CLI compatibility")
    parser.add_argument(
        "--auto-split-clips",
        action="store_true",
        help="Rejected because static rendering is clip-first",
    )
    parser.add_argument("--video-output", action="store_true", help="Accepted for CLI compatibility")
    return parser


def main():
    argv = parse_blender_argv(sys.argv)
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.no_skip_existing:
        args.skip_existing = False

    if args.auto_split_clips:
        print("ERROR: blender_ego_static.py does not support --auto-split-clips")
        print("  Static clips freeze the scene at each clip's first frame, so reusable-frame split is invalid.")
        sys.exit(1)

    blend_filepath = bpy.data.filepath
    output_folder = get_output_folder(args.save_path, blend_filepath)
    error_logger = RenderErrorLogger(args.save_path, blend_filepath)

    def log_error(error_type, error_message, animation_name=None):
        error_logger.log(error_type, error_message, animation_name=animation_name)

    try:
        armature_name = discover_armature_name()
    except RuntimeError as exc:
        log_error("ARMATURE_DISCOVERY_ERROR", str(exc))
        print(f"Error: {exc}")
        sys.exit(1)

    scene = bpy.context.scene
    render = scene.render
    render.engine = "CYCLES"

    ensure_collection_hidden("AugmentAreaCollection", hidden=True)
    configure_cycles_devices()

    render.resolution_x = args.width
    render.resolution_y = args.height
    render.resolution_percentage = 100
    optimize_scene_for_cycles(
        scene,
        args.samples,
        use_adaptive_sampling=True,
        adaptive_threshold=0.05,
        tile_size=256,
        disable_motion_blur=True,
    )

    render.image_settings.file_format = "FFMPEG"
    render.ffmpeg.format = "MPEG4"
    render.ffmpeg.codec = "H264"
    render.ffmpeg.constant_rate_factor = "MEDIUM"
    render.ffmpeg.ffmpeg_preset = "REALTIME"
    render.fps = int(args.fps)

    try:
        armature_obj, eye_mesh_obj = get_required_scene_objects(armature_name, EYE_MESH_NAME)
    except RuntimeError as exc:
        log_error("SCENE_SETUP_ERROR", str(exc))
        print(f"Error: {exc}")
        sys.exit(1)

    reset_armature_pose(scene, armature_obj)

    try:
        camera_obj = create_ego_camera(
            scene,
            armature_obj,
            eye_mesh_obj,
            args.fov,
            parent_bone_name=FACIAL_BONE_NAME,
        )
    except RuntimeError as exc:
        log_error("CAMERA_SETUP_ERROR", str(exc))
        print(f"Error: {exc}")
        sys.exit(1)

    controller = StaticSceneController()

    def get_render_range():
        render_start_frame = scene.frame_start if args.start_frame is None else args.start_frame
        render_end_frame = scene.frame_end if args.end_frame is None else args.end_frame
        return render_start_frame, render_end_frame

    def rename_blender_video_output(expected_path):
        if os.path.exists(expected_path):
            return expected_path
        base_name = os.path.splitext(os.path.basename(expected_path))[0]
        output_dir = os.path.dirname(expected_path)
        for filename in os.listdir(output_dir):
            if filename.startswith(base_name) and filename.endswith(".mp4"):
                temp_path = os.path.join(output_dir, filename)
                if temp_path != expected_path:
                    os.rename(temp_path, expected_path)
                return expected_path
        return expected_path

    def render_animation_sequence(animation_index, animation_name):
        render_start_frame, render_end_frame = get_render_range()
        frames_to_render = build_frame_numbers(render_start_frame, render_end_frame, args.frame_skip)
        clip_starts = build_clip_start_indices(len(frames_to_render), args.clip_length, args.stride)
        if not clip_starts:
            print(f"Skipping {animation_name}: not enough frames for clip_length={args.clip_length}")
            return

        videos_output_path = os.path.join(output_folder, animation_name, "sequences", "videos_static")
        os.makedirs(videos_output_path, exist_ok=True)

        print(f"Rendering animation {animation_name}")
        print(f"  Mode: static_direct_clips")
        print(f"  Videos: {videos_output_path}")
        print(f"  Clip length: {args.clip_length} | stride: {args.stride} | frame_skip: {args.frame_skip}")

        start_time = time.time()
        clips_completed = 0

        for clip_idx, clip_start in enumerate(clip_starts):
            clip_frame_numbers = frames_to_render[clip_start:clip_start + args.clip_length]
            clip_output_path = os.path.join(videos_output_path, f"{clip_idx:05d}.mp4")

            if args.skip_existing and file_ok(clip_output_path, min_size=10240):
                print(f"  Clip {clip_idx:05d}.mp4 already exists, skipping")
                clips_completed += 1
                continue

            controller.restore_animations(camera_obj)
            set_actor_render_hidden(armature_obj, False)

            if not apply_animation_set(animation_index, animation_sets):
                raise RuntimeError(f"Failed to apply animation {animation_index}")

            camera_locations, camera_rotations = controller.sample_camera_world_transforms(
                camera_obj,
                clip_frame_numbers,
            )

            scene.frame_set(clip_frame_numbers[0])
            controller.disable_animations_except_camera(camera_obj)
            set_actor_render_hidden(armature_obj, True)
            controller.bake_camera_keys(
                camera_obj,
                clip_frame_numbers,
                camera_locations,
                camera_rotations,
            )

            original_frame_step = getattr(scene, "frame_step", 1)
            scene.frame_start = clip_frame_numbers[0]
            scene.frame_end = clip_frame_numbers[-1]
            if hasattr(scene, "frame_step"):
                scene.frame_step = max(1, args.frame_skip)

            render.filepath = os.path.splitext(clip_output_path)[0]
            clip_start_time = time.time()
            bpy.ops.render.render(animation=True)
            rename_blender_video_output(clip_output_path)
            if hasattr(scene, "frame_step"):
                scene.frame_step = original_frame_step

            controller.clear_camera_keys(camera_obj)
            controller.restore_animations(camera_obj)
            set_actor_render_hidden(armature_obj, False)

            clips_completed += 1
            clip_time = time.time() - clip_start_time
            elapsed = time.time() - start_time
            avg_time = elapsed / max(1, clips_completed)
            remaining = len(clip_starts) - clips_completed
            print(
                f"  Created clip {clip_idx:05d}.mp4 in {clip_time:.1f}s"
                f" | progress {clips_completed}/{len(clip_starts)} | ETA {remaining * avg_time / 60:.1f} min"
            )

        total_time = time.time() - start_time
        print(f"  Finished static clips in {total_time/60:.1f} min")

    animation_sets = get_animation_sets()
    if not animation_sets:
        print("No animation sets found!")
        sys.exit(0)

    try:
        animations_to_render = resolve_animations_to_render(
            animation_sets,
            animation_index=args.animation_index,
            animation_name=args.animation_name,
        )
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    total_start_time = time.time()
    failed_animations = []
    for anim_idx, anim_name in animations_to_render:
        stripped_name = strip_animation_extension(anim_name)
        print("\n" + "=" * 60)
        print(f"PROCESSING ANIMATION {anim_idx}: {anim_name}")
        print("=" * 60)
        try:
            render_animation_sequence(anim_idx, stripped_name)
        except Exception as exc:
            log_error("RENDERING_ERROR", str(exc), animation_name=anim_name)
            print(f"Error during animation {anim_idx}: {exc}")
            failed_animations.append((anim_idx, anim_name))

    total_time = time.time() - total_start_time
    print("\n" + "=" * 60)
    print("STATIC RENDERING COMPLETE")
    print(f"Total time: {total_time/3600:.1f} hours ({total_time/60:.1f} minutes)")
    print(f"Results saved in: {output_folder}")
    if failed_animations:
        print(f"Failed animations: {len(failed_animations)}")
        for anim_idx, anim_name in failed_animations:
            print(f"  Animation {anim_idx}: {anim_name}")
        print(f"Check error log for details: {error_logger.error_log_file}")
    else:
        print("All animations completed successfully")
    print("=" * 60)


if __name__ == "__main__":
    main()
