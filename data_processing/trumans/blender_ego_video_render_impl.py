#!/usr/bin/env python3

import argparse
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import bpy
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from rendering_common import (  # noqa: E402
    EYE_MESH_NAME,
    FACIAL_BONE_NAME,
    RenderErrorLogger,
    apply_animation_set,
    check_frame_outputs,
    configure_cycles_devices,
    create_ego_camera,
    discover_armature_name,
    ensure_collection_hidden,
    file_ok,
    get_animation_sets,
    get_camera_intrinsics,
    get_camera_to_world_matrix,
    get_output_folder,
    get_required_scene_objects,
    optimize_scene_for_cycles,
    parse_blender_argv,
    reset_armature_pose,
    restore_actor_shadow_visibility,
    resolve_animations_to_render,
    set_actor_render_hidden,
    suppress_actor_shadow_visibility,
    strip_animation_extension,
)
from rendering_strategies import (  # noqa: E402
    build_clip_start_indices,
    build_frame_numbers,
    create_video_clip_from_frames,
    resolve_execution_strategy,
)


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
        default="/home/byungjun/workspace/trumans_ego/ego_render_fov90",
        help="Root output dir",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip frames or clips that already exist",
    )
    parser.add_argument("--no-skip-existing", action="store_true", help="Disable skipping existing files")
    parser.add_argument("--frame-skip", type=int, default=3, help="Render every Nth frame")
    parser.add_argument("--fov", type=float, default=90.0, help="Camera FOV in degrees")
    parser.add_argument("--width", type=int, default=720, help="Render width")
    parser.add_argument("--height", type=int, default=480, help="Render height")
    parser.add_argument("--only-object", action="store_true", help="Hide actor and render only objects")
    parser.add_argument(
        "--no-actor-shadow",
        action="store_true",
        help="Keep the actor visible but disable actor-cast shadows",
    )
    parser.add_argument(
        "--static",
        action="store_true",
        help="Deprecated. Use blender_ego_static.py for static scene rendering.",
    )
    parser.add_argument("--video-output", action="store_true", help="Output MP4 clips or videos")
    parser.add_argument("--auto-split-clips", action="store_true", help="Render reusable frames, then split clips")
    parser.add_argument("--direct-clips", action="store_true", help="Render clips directly for quicker previews")
    parser.add_argument("--clip-length", type=int, default=49, help="Frames per clip")
    parser.add_argument("--clip-stride", type=int, default=25, help="Clip stride in stepped frames")
    parser.add_argument("--fps", type=float, default=8.0, help="FPS for video output")
    return parser


def main():
    argv = parse_blender_argv(sys.argv)
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.no_skip_existing:
        args.skip_existing = False

    if args.static:
        print("ERROR: --static moved to blender_ego_static.py")
        print("  Use data_processing/trumans/blender_ego_static.py for clip-first static rendering.")
        sys.exit(1)

    if not args.video_output and (args.auto_split_clips or args.direct_clips):
        print("ERROR: --auto-split-clips and --direct-clips require --video-output")
        sys.exit(1)

    if args.only_object and args.no_actor_shadow:
        print("WARNING: --no-actor-shadow has no effect with --only-object because the actor is hidden.")

    try:
        execution_strategy = resolve_execution_strategy(
            auto_split=args.auto_split_clips,
            direct_clips=args.direct_clips,
            default="direct_video" if args.video_output else None,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}")
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

    camera_fov_degrees = args.fov
    scene = bpy.context.scene
    render = scene.render
    render.engine = "CYCLES"

    ensure_collection_hidden("AugmentAreaCollection", hidden=True)
    configure_cycles_devices()

    render.resolution_x = args.width
    render.resolution_y = args.height
    render.resolution_percentage = 100
    optimize_scene_for_cycles(scene, args.samples)

    def configure_video_render_settings():
        render.image_settings.file_format = "FFMPEG"
        render.ffmpeg.format = "MPEG4"
        render.ffmpeg.codec = "H264"
        render.ffmpeg.constant_rate_factor = "MEDIUM"
        render.ffmpeg.ffmpeg_preset = "REALTIME"
        render.fps = int(args.fps)
        scene.use_nodes = False

    def configure_frame_render_settings():
        render.image_settings.file_format = "PNG"
        render.image_settings.color_mode = "RGBA"
        scene.use_nodes = True
        tree = scene.node_tree
        for node in list(tree.nodes):
            tree.nodes.remove(node)

        render_layers = tree.nodes.new(type="CompositorNodeRLayers")
        render_layers.location = (0, 0)
        render_layers.layer = bpy.context.view_layer.name

        output_node = tree.nodes.new(type="CompositorNodeOutputFile")
        output_node.label = "RGB Output"
        output_node.base_path = output_folder
        output_node.file_slots[0].path = "####"
        output_node.format.file_format = "PNG"
        output_node.format.color_mode = "RGBA"
        output_node.location = (400, 200)
        tree.links.new(render_layers.outputs["Image"], output_node.inputs[0])
        return output_node

    if args.video_output:
        configure_video_render_settings()
        rgb_output_node = None
    else:
        rgb_output_node = configure_frame_render_settings()

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
            camera_fov_degrees,
            parent_bone_name=FACIAL_BONE_NAME,
        )
    except RuntimeError as exc:
        log_error("CAMERA_SETUP_ERROR", str(exc))
        print(f"Error: {exc}")
        sys.exit(1)

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

    def render_frames_to_temp(frame_numbers, temp_dir):
        original_format = render.image_settings.file_format
        original_color_mode = render.image_settings.color_mode
        original_filepath = render.filepath
        frame_paths = []
        try:
            render.image_settings.file_format = "PNG"
            render.image_settings.color_mode = "RGBA"
            for frame_idx, frame_num in enumerate(frame_numbers):
                scene.frame_set(frame_num)
                frame_path = os.path.join(temp_dir, f"frame_{frame_idx:04d}.png")
                render.filepath = frame_path
                bpy.ops.render.render(write_still=True)
                if os.path.exists(frame_path):
                    frame_paths.append(frame_path)
            return frame_paths
        finally:
            render.filepath = original_filepath
            render.image_settings.file_format = original_format
            render.image_settings.color_mode = original_color_mode

    def render_direct_video(animation_name):
        video_suffix = "_object" if args.only_object else ""
        video_output_path = os.path.join(output_folder, f"{animation_name}{video_suffix}.mp4")

        if args.skip_existing and file_ok(video_output_path, min_size=10240):
            print(f"Rendering animation {animation_name}: {video_output_path} already exists, skipping")
            return

        render_start_frame, render_end_frame = get_render_range()
        print(f"Rendering animation {animation_name}")
        print(f"  Mode: direct_video")
        print(f"  Video: {video_output_path}")
        print(f"  Frames: {render_start_frame}..{render_end_frame} (step {args.frame_skip})")

        original_frame_step = getattr(scene, "frame_step", 1)
        scene.frame_start = render_start_frame
        scene.frame_end = render_end_frame
        if hasattr(scene, "frame_step"):
            scene.frame_step = max(1, args.frame_skip)

        render.filepath = os.path.splitext(video_output_path)[0]
        start_time = time.time()
        bpy.ops.render.render(animation=True)
        rename_blender_video_output(video_output_path)
        if hasattr(scene, "frame_step"):
            scene.frame_step = original_frame_step
        print(f"  Created video in {time.time() - start_time:.1f}s")

    def render_direct_clips(animation_name):
        render_start_frame, render_end_frame = get_render_range()
        frames_to_render = build_frame_numbers(render_start_frame, render_end_frame, args.frame_skip)
        clip_starts = build_clip_start_indices(len(frames_to_render), args.clip_length, args.clip_stride)
        if not clip_starts:
            print(f"Skipping {animation_name}: not enough frames for clip_length={args.clip_length}")
            return

        clips_output_path = os.path.join(
            output_folder,
            animation_name,
            "videos_object" if args.only_object else "videos",
        )
        os.makedirs(clips_output_path, exist_ok=True)
        print(f"Rendering animation {animation_name}")
        print(f"  Mode: direct_clips")
        print(f"  Clips output: {clips_output_path}")
        print(f"  Windows: {len(clip_starts)}")

        temp_dir = tempfile.mkdtemp(prefix="truman_dynamic_frames_")
        frame_buffer = []
        next_clip_index = 0
        start_time = time.time()

        original_format = render.image_settings.file_format
        original_color_mode = render.image_settings.color_mode
        original_filepath = render.filepath
        try:
            render.image_settings.file_format = "PNG"
            render.image_settings.color_mode = "RGBA"

            for frame_idx, frame_num in enumerate(frames_to_render):
                scene.frame_set(frame_num)
                frame_path = os.path.join(temp_dir, f"frame_{frame_idx:04d}.png")
                render.filepath = frame_path
                bpy.ops.render.render(write_still=True)
                if os.path.exists(frame_path):
                    frame_buffer.append((frame_idx, frame_path))

                while next_clip_index < len(clip_starts):
                    clip_start = clip_starts[next_clip_index]
                    clip_end = clip_start + args.clip_length - 1
                    if frame_idx < clip_end:
                        break
                    clip_output_path = os.path.join(clips_output_path, f"{next_clip_index:05d}.mp4")
                    required = [path for idx, path in frame_buffer if clip_start <= idx <= clip_end]
                    if len(required) == args.clip_length:
                        if args.skip_existing and file_ok(clip_output_path, min_size=10240):
                            print(f"  Clip {next_clip_index:05d}.mp4 already exists, skipping")
                        else:
                            if create_video_clip_from_frames(required, clip_output_path, fps=args.fps):
                                print(
                                    f"  Created clip {next_clip_index:05d}.mp4"
                                    f" (frames {clip_start}-{clip_end})"
                                )
                            else:
                                print(f"  Failed to create clip {next_clip_index:05d}.mp4")
                    next_clip_index += 1
                    min_keep_idx = (
                        clip_starts[next_clip_index]
                        if next_clip_index < len(clip_starts)
                        else len(frames_to_render)
                    )
                    frame_buffer = [(idx, path) for idx, path in frame_buffer if idx >= min_keep_idx]

                if (frame_idx + 1) % 10 == 0 or frame_idx == len(frames_to_render) - 1:
                    progress = (frame_idx + 1) / len(frames_to_render) * 100.0
                    print(f"  Rendered {frame_idx + 1}/{len(frames_to_render)} frames ({progress:.1f}%)")
        finally:
            render.filepath = original_filepath
            render.image_settings.file_format = original_format
            render.image_settings.color_mode = original_color_mode
            shutil.rmtree(temp_dir, ignore_errors=True)

        print(f"  Finished direct_clips in {time.time() - start_time:.1f}s")

    def render_auto_split_clips(animation_name):
        render_start_frame, render_end_frame = get_render_range()
        frames_to_render = build_frame_numbers(render_start_frame, render_end_frame, args.frame_skip)
        clip_starts = build_clip_start_indices(len(frames_to_render), args.clip_length, args.clip_stride)
        if not clip_starts:
            print(f"Skipping {animation_name}: not enough frames for clip_length={args.clip_length}")
            return

        clips_output_path = os.path.join(
            output_folder,
            animation_name,
            "videos_object" if args.only_object else "videos",
        )
        os.makedirs(clips_output_path, exist_ok=True)
        print(f"Rendering animation {animation_name}")
        print(f"  Mode: auto_split")
        print(f"  Clips output: {clips_output_path}")
        print(f"  Frames: {len(frames_to_render)} | Windows: {len(clip_starts)}")

        temp_dir = tempfile.mkdtemp(prefix="truman_dynamic_frames_")
        start_time = time.time()
        try:
            frame_paths = render_frames_to_temp(frames_to_render, temp_dir)
            for clip_idx, clip_start in enumerate(clip_starts):
                clip_output_path = os.path.join(clips_output_path, f"{clip_idx:05d}.mp4")
                if args.skip_existing and file_ok(clip_output_path, min_size=10240):
                    print(f"  Clip {clip_idx:05d}.mp4 already exists, skipping")
                    continue
                clip_frames = frame_paths[clip_start:clip_start + args.clip_length]
                if create_video_clip_from_frames(clip_frames, clip_output_path, fps=args.fps):
                    print(
                        f"  Created clip {clip_idx:05d}.mp4"
                        f" (frames {clip_start}-{clip_start + args.clip_length - 1})"
                    )
                else:
                    print(f"  Failed to create clip {clip_idx:05d}.mp4")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        print(f"  Finished auto_split in {time.time() - start_time:.1f}s")

    def render_frame_sequence(animation_name):
        render_start_frame, render_end_frame = get_render_range()
        frames_to_render = build_frame_numbers(render_start_frame, render_end_frame, args.frame_skip)
        anim_output_folder = os.path.join(output_folder, animation_name)

        if args.only_object:
            images_output_path = os.path.join(anim_output_folder, "images_object")
            cam_params_path = None
        else:
            images_output_path = os.path.join(anim_output_folder, "images")
            cam_params_path = os.path.join(anim_output_folder, "cam_params")

        os.makedirs(images_output_path, exist_ok=True)
        if cam_params_path is not None:
            os.makedirs(cam_params_path, exist_ok=True)

        rgb_output_node.base_path = images_output_path
        rgb_output_node.file_slots[0].path = "####"

        if cam_params_path is not None:
            intrinsics = get_camera_intrinsics(camera_obj, camera_fov_degrees)
            np.save(os.path.join(cam_params_path, "intrinsics.npy"), intrinsics)

        print(f"Rendering animation {animation_name}")
        print(f"  Mode: frames")
        print(f"  Images: {images_output_path}")
        if cam_params_path is not None:
            print(f"  Cam params: {cam_params_path}")

        start_time = time.time()
        rendered = 0
        skipped = 0
        cam_params_saved = 0

        for frame_num in frames_to_render:
            _, _, needs_rendering, needs_cam_param = check_frame_outputs(
                frame_num,
                images_output_path,
                cam_params_path,
            )
            scene.frame_set(frame_num)

            if cam_params_path is not None and needs_cam_param:
                c2w = get_camera_to_world_matrix(camera_obj)
                np.save(os.path.join(cam_params_path, f"cam_{frame_num:04d}.npy"), c2w)
                cam_params_saved += 1

            if args.skip_existing and not needs_rendering:
                skipped += 1
                continue

            bpy.ops.render.render(write_still=True)
            rendered += 1

        total_time = time.time() - start_time
        print(
            f"  Completed frames in {total_time/60:.1f} min | rendered={rendered}, skipped={skipped},"
            f" cam_params_saved={cam_params_saved}"
        )

    def render_animation_sequence(animation_name):
        actor_shadow_state = []
        if args.only_object:
            set_actor_render_hidden(armature_obj, True)
        elif args.no_actor_shadow:
            actor_shadow_state = suppress_actor_shadow_visibility(armature_obj)
        try:
            if not args.video_output:
                render_frame_sequence(animation_name)
            elif execution_strategy == "auto_split":
                render_auto_split_clips(animation_name)
            elif execution_strategy == "direct_clips":
                render_direct_clips(animation_name)
            else:
                render_direct_video(animation_name)
        finally:
            if args.only_object:
                set_actor_render_hidden(armature_obj, False)
            elif actor_shadow_state:
                restore_actor_shadow_visibility(actor_shadow_state)

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

    print(f"Found {len(animation_sets)} animation sets")
    print(f"Rendering {len(animations_to_render)} animation(s)")

    total_start_time = time.time()
    failed_animations = []
    successful_frame_count = 0

    for anim_idx, anim_name in animations_to_render:
        stripped_name = strip_animation_extension(anim_name)
        print("\n" + "=" * 60)
        print(f"PROCESSING ANIMATION {anim_idx}: {anim_name}")
        print("=" * 60)
        try:
            if not apply_animation_set(anim_idx, animation_sets):
                raise RuntimeError(f"Failed to apply animation {anim_idx}")
            render_animation_sequence(stripped_name)
            render_start_frame, render_end_frame = get_render_range()
            successful_frame_count += len(
                build_frame_numbers(render_start_frame, render_end_frame, args.frame_skip)
            )
        except Exception as exc:
            log_error("RENDERING_ERROR", str(exc), animation_name=anim_name)
            print(f"Error during animation {anim_idx}: {exc}")
            failed_animations.append((anim_idx, anim_name))

    total_time = time.time() - total_start_time
    overall_fps = successful_frame_count / total_time if total_time > 0 else 0.0

    print("\n" + "=" * 60)
    print("RENDERING COMPLETE")
    print(f"Total time: {total_time/3600:.1f} hours ({total_time/60:.1f} minutes)")
    print(f"Total stepped frames covered: {successful_frame_count}")
    print(f"Overall throughput: {overall_fps:.2f} fps")
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
