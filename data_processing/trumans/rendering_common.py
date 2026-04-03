#!/usr/bin/env python3

import json
import math
import os
import time
import traceback
from datetime import datetime

import bpy
import numpy as np
from mathutils import Vector


EYE_MESH_NAME = "CC_Base_Eye"
FACIAL_BONE_NAME = "CC_Base_FacialBone"


def parse_blender_argv(argv):
    return argv[argv.index("--") + 1:] if "--" in argv else []


def get_output_folder(save_path, blend_filepath):
    directory_name = os.path.basename(os.path.dirname(blend_filepath))
    output_folder = os.path.join(save_path, directory_name)
    os.makedirs(output_folder, exist_ok=True)
    return output_folder


def strip_animation_extension(animation_name):
    return os.path.splitext(animation_name)[0]


class RenderErrorLogger:
    def __init__(self, save_path, blend_filepath):
        self.blend_filepath = blend_filepath
        self.error_log_file = os.path.join(save_path, "rendering_errors.log")

    def log(self, error_type, error_message, blend_file=None, animation_name=None):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        blend_file = blend_file or self.blend_filepath
        animation_info = f" (Animation: {animation_name})" if animation_name else ""
        entry = f"[{timestamp}] {error_type}: {error_message} - File: {blend_file}{animation_info}\n"
        try:
            with open(self.error_log_file, "a") as handle:
                handle.write(entry)
                handle.write(f"Traceback:\n{traceback.format_exc()}\n")
                handle.write("-" * 80 + "\n")
        except Exception as exc:
            print(f"Warning: Could not write to error log: {exc}")
        print(f"ERROR: {error_type}: {error_message}")
        print(f"Error logged to: {self.error_log_file}")


def discover_armature_name():
    try:
        armature_name = bpy.context.scene.hsi_properties.name_armature_CC
        if armature_name:
            return armature_name
        for obj in bpy.data.objects:
            if obj.type == "ARMATURE" and "CC_Base_Hip" in obj.pose.bones:
                return obj.name
    except Exception as exc:
        raise RuntimeError(f"Could not access HSI addon properties: {exc}") from exc
    raise RuntimeError("No armature found. Please import a CC4 character first.")


def get_animation_sets():
    try:
        animation_sets_json = bpy.context.scene.hsi_properties.animation_sets
        animation_sets = json.loads(animation_sets_json)
        return animation_sets
    except Exception as exc:
        print(f"Error getting animation sets: {exc}")
        return {}


def resolve_animation_index(animation_sets, animation_index=None, animation_name=None):
    keys = list(animation_sets.keys())
    if animation_name is not None:
        for index, name in enumerate(keys):
            if name == animation_name or strip_animation_extension(name) == animation_name:
                return index
        raise ValueError(f"Animation name '{animation_name}' was not found")
    if animation_index is None:
        raise ValueError("Either animation_index or animation_name must be provided")
    if animation_index < 0 or animation_index >= len(keys):
        raise ValueError(
            f"Animation index {animation_index} out of range. Available: 0-{len(keys) - 1}"
        )
    return animation_index


def resolve_animations_to_render(animation_sets, animation_index=None, animation_name=None):
    keys = list(animation_sets.keys())
    if animation_name is not None or animation_index is not None:
        resolved_index = resolve_animation_index(
            animation_sets,
            animation_index=animation_index,
            animation_name=animation_name,
        )
        return [(resolved_index, keys[resolved_index])]
    return list(enumerate(keys))


def apply_animation_set(animation_index, animation_sets=None):
    animation_sets = animation_sets or get_animation_sets()
    if not animation_sets:
        print("No animation sets found!")
        return False
    if animation_index < 0 or animation_index >= len(animation_sets):
        print(
            f"Animation index {animation_index} out of range. Available: 0-{len(animation_sets) - 1}"
        )
        return False
    bpy.context.scene.hsi_properties.current_animation_index_display = animation_index
    bpy.ops.hsi.set_animation()
    animation_name = list(animation_sets.keys())[animation_index]
    print(f"Applied animation set {animation_index}: {animation_name}")
    return True


def get_camera_intrinsics(camera_obj, camera_fov_degrees):
    cam = camera_obj.data
    scene = bpy.context.scene
    width = scene.render.resolution_x
    height = scene.render.resolution_y
    cam.lens_unit = "FOV"
    cam.angle = math.radians(camera_fov_degrees)
    fov_rad = cam.angle
    focal_length_px = (width / 2.0) / math.tan(fov_rad / 2.0)
    fx = fy = focal_length_px
    cx = width / 2.0
    cy = height / 2.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)


def get_camera_to_world_matrix(camera_obj):
    return np.array(camera_obj.matrix_world, dtype=np.float32)


def file_ok(path, min_size=1024, freshness_sec=2.0):
    if not os.path.isfile(path):
        return False
    size = os.path.getsize(path)
    if size < min_size:
        return False
    mtime = os.path.getmtime(path)
    if time.time() - mtime < freshness_sec:
        return False
    return True


def get_existing_source_video_clip_indices(action_output_path, source_subdir="videos", min_size=10240):
    source_videos_path = os.path.join(action_output_path, source_subdir)
    if not os.path.isdir(source_videos_path):
        return []

    clip_indices = []
    for entry in sorted(os.listdir(source_videos_path)):
        if not entry.endswith(".mp4"):
            continue
        stem = os.path.splitext(entry)[0]
        if not stem.isdigit():
            continue
        if file_ok(os.path.join(source_videos_path, entry), min_size=min_size):
            clip_indices.append(int(stem))
    return clip_indices


def select_clip_windows_from_source_videos(action_output_path, clip_windows, source_subdir="videos", min_size=10240):
    source_clip_indices = get_existing_source_video_clip_indices(
        action_output_path,
        source_subdir=source_subdir,
        min_size=min_size,
    )
    selected_windows = []
    ignored_indices = []
    for clip_idx in source_clip_indices:
        if 0 <= clip_idx < len(clip_windows):
            selected_windows.append((clip_idx, clip_windows[clip_idx]))
        else:
            ignored_indices.append(clip_idx)
    return selected_windows, ignored_indices


def check_frame_outputs(frame_num, images_output_path, cam_params_path=None):
    image_path = os.path.join(images_output_path, f"{frame_num:04d}.png")
    rgb_exists = file_ok(image_path, min_size=10240)

    cam_param_exists = True
    if cam_params_path is not None:
        cam_param_path = os.path.join(cam_params_path, f"cam_{frame_num:04d}.npy")
        cam_param_exists = file_ok(cam_param_path, min_size=200)

    needs_rendering = not rgb_exists
    needs_cam_param = cam_params_path is not None and not cam_param_exists
    return rgb_exists, cam_param_exists, needs_rendering, needs_cam_param


def check_video_output(video_idx, videos_output_path, min_size=10240):
    video_path = os.path.join(videos_output_path, f"{video_idx:05d}.mp4")
    video_exists = file_ok(video_path, min_size=min_size)
    needs_rendering = not video_exists
    return video_exists, needs_rendering


def configure_cycles_devices():
    try:
        prefs_cycles = bpy.context.preferences.addons["cycles"].preferences
        prefs_cycles.compute_device_type = "NONE"
        device_found = False
        for dev_type in ("OPTIX", "CUDA"):
            try:
                prefs_cycles.compute_device_type = dev_type
                prefs_cycles.get_devices()
                for device in prefs_cycles.devices:
                    if device.type == dev_type:
                        device.use = True
                        print(f"Enabled GPU for Cycles: {device.name} ({device.type})")
                        device_found = True
                        break
                if device_found:
                    break
            except Exception:
                pass
        if not device_found:
            print("No GPU device found or failed to configure. Using CPU.")
    except Exception as exc:
        print(f"Unexpected GPU setup error: {exc}. Using CPU.")


def optimize_scene_for_cycles(
    scene,
    cycles_samples,
    *,
    use_denoising=True,
    use_gpu=True,
    use_adaptive_sampling=False,
    adaptive_threshold=0.05,
    tile_size=None,
    disable_motion_blur=False,
):
    scene.cycles.samples = cycles_samples
    if use_gpu:
        scene.cycles.device = "GPU"

    scene.cycles.use_denoising = use_denoising
    if hasattr(scene.cycles, "denoiser"):
        keys = scene.cycles.bl_rna.properties["denoiser"].enum_items.keys()
        if "OPTIX" in keys:
            scene.cycles.denoiser = "OPTIX"
        elif "OPENIMAGEDENOISE" in keys:
            scene.cycles.denoiser = "OPENIMAGEDENOISE"

    if hasattr(scene.render, "use_persistent_data"):
        scene.render.use_persistent_data = True

    if use_adaptive_sampling and hasattr(scene.cycles, "use_adaptive_sampling"):
        scene.cycles.use_adaptive_sampling = True
        scene.cycles.adaptive_threshold = adaptive_threshold

    if tile_size is not None and hasattr(scene.cycles, "tile_size"):
        scene.cycles.tile_size = tile_size

    scene.render.resolution_percentage = 100
    scene.render.use_border = False
    scene.render.use_crop_to_border = False
    if disable_motion_blur and hasattr(scene.render, "use_motion_blur"):
        scene.render.use_motion_blur = False
    if hasattr(scene.render, "use_free_unused_nodes"):
        scene.render.use_free_unused_nodes = True
    if hasattr(scene.render, "use_free_image_textures"):
        scene.render.use_free_image_textures = True
    print(
        f"Optimized scene: {scene.render.resolution_x}x{scene.render.resolution_y}, {cycles_samples} samples"
    )


def ensure_collection_hidden(collection_name, hidden=True):
    if collection_name in bpy.data.collections:
        bpy.data.collections[collection_name].hide_render = hidden
        state = "Hide" if hidden else "Show"
        print(f"{state} {collection_name} for rendering")
    else:
        print(f"{collection_name} not found - skipping")


def get_required_scene_objects(armature_name, eye_mesh_name=EYE_MESH_NAME):
    armature_obj = bpy.data.objects.get(armature_name)
    eye_mesh_obj = bpy.data.objects.get(eye_mesh_name)
    if not armature_obj:
        raise RuntimeError(f"Armature '{armature_name}' not found")
    if not eye_mesh_obj:
        raise RuntimeError(f"Mesh '{eye_mesh_name}' not found")
    return armature_obj, eye_mesh_obj


def reset_armature_pose(scene, armature_obj, frame=0):
    print(f"Moving to frame {frame} and resetting pose...")
    scene.frame_set(frame)
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode="POSE")
    for bone in armature_obj.pose.bones:
        bone.rotation_mode = "QUATERNION"
        bone.location = (0, 0, 0)
        bone.rotation_quaternion = (1, 0, 0, 0)
        bone.scale = (1, 1, 1)


def compute_eye_camera_world_location(eye_mesh_obj):
    if eye_mesh_obj.type == "MESH" and eye_mesh_obj.data.vertices:
        matrix_world = eye_mesh_obj.matrix_world.copy()
        bbox_min = Vector((float("inf"),) * 3)
        bbox_max = Vector((float("-inf"),) * 3)
        for vertex in eye_mesh_obj.data.vertices:
            bbox_min.x = min(bbox_min.x, vertex.co.x)
            bbox_min.y = min(bbox_min.y, vertex.co.y)
            bbox_min.z = min(bbox_min.z, vertex.co.z)
            bbox_max.x = max(bbox_max.x, vertex.co.x)
            bbox_max.y = max(bbox_max.y, vertex.co.y)
            bbox_max.z = max(bbox_max.z, vertex.co.z)
        local_center = (bbox_min + bbox_max) / 2.0
        return matrix_world @ local_center
    return eye_mesh_obj.matrix_world.translation


def create_ego_camera(
    scene,
    armature_obj,
    eye_mesh_obj,
    camera_fov_degrees,
    *,
    parent_bone_name=FACIAL_BONE_NAME,
    camera_name="POV_Camera",
):
    camera_initial_world_location = compute_eye_camera_world_location(eye_mesh_obj)
    if camera_initial_world_location is None:
        raise RuntimeError("Failed to determine camera position.")

    parent_pose_bone = armature_obj.pose.bones.get(parent_bone_name)
    if not parent_pose_bone:
        raise RuntimeError(
            f"Bone '{parent_bone_name}' not found in armature '{armature_obj.name}'"
        )

    print("Creating camera...")
    camera_data = bpy.data.cameras.new(name=camera_name)
    camera_obj = bpy.data.objects.new(name=camera_name, object_data=camera_data)
    scene.collection.objects.link(camera_obj)
    camera_obj.location = camera_initial_world_location

    target_forward = Vector((0, -1, 0))
    rot_quat = target_forward.to_track_quat("-Z", "Y")
    camera_obj.rotation_mode = "QUATERNION"
    camera_obj.rotation_quaternion = rot_quat

    camera_data.lens_unit = "FOV"
    camera_data.angle = math.radians(camera_fov_degrees)
    bpy.context.scene.camera = camera_obj
    print(f"Set {camera_name} active. FOV {camera_fov_degrees} degrees.")

    for obj in bpy.data.objects:
        obj.select_set(False)
    camera_obj.select_set(True)
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode="POSE")
    for bone in armature_obj.pose.bones:
        bone.bone.select = False
    parent_pose_bone.bone.select = True
    bpy.context.view_layer.objects.active.data.bones.active = parent_pose_bone.bone
    bpy.ops.object.parent_set(type="BONE")
    bpy.ops.object.mode_set(mode="OBJECT")
    return camera_obj


def set_actor_render_hidden(armature_obj, hidden):
    if not armature_obj:
        return
    armature_obj.hide_render = hidden
    action = "Hidden" if hidden else "Shown"
    print(f"{action} armature '{armature_obj.name}' {'from' if hidden else 'in'} rendering")

    for child in armature_obj.children:
        child.hide_render = hidden
        print(f"{action} child '{child.name}' {'from' if hidden else 'in'} rendering")

    for obj in bpy.data.objects:
        if obj.parent_bone and obj.parent == armature_obj:
            obj.hide_render = hidden
            print(f"{action} bone-child '{obj.name}' {'from' if hidden else 'in'} rendering")


def get_actor_render_objects(armature_obj):
    if not armature_obj:
        return []

    actor_objects = []
    seen = set()

    def add(obj):
        if obj is None:
            return
        try:
            key = obj.as_pointer()
        except ReferenceError:
            return
        if key in seen:
            return
        seen.add(key)
        actor_objects.append(obj)

    add(armature_obj)
    for child in armature_obj.children:
        add(child)

    for obj in bpy.data.objects:
        if obj.parent_bone and obj.parent == armature_obj:
            add(obj)

    return actor_objects


def suppress_actor_shadow_visibility(armature_obj):
    shadow_state = []
    actor_objects = get_actor_render_objects(armature_obj)

    for obj in actor_objects:
        obj_state = {"object": obj}
        changed = False

        if hasattr(obj, "visible_shadow"):
            obj_state["visible_shadow"] = obj.visible_shadow
            obj.visible_shadow = False
            changed = True

        cycles_visibility = getattr(obj, "cycles_visibility", None)
        if cycles_visibility is not None and hasattr(cycles_visibility, "shadow"):
            obj_state["cycles_visibility_shadow"] = cycles_visibility.shadow
            cycles_visibility.shadow = False
            changed = True

        if changed:
            shadow_state.append(obj_state)

    if shadow_state:
        print(f"Disabled actor-cast shadows for {len(shadow_state)} actor objects")
    else:
        print("No actor shadow visibility properties found to disable")

    return shadow_state


def restore_actor_shadow_visibility(shadow_state):
    restored = 0

    for obj_state in shadow_state:
        obj = obj_state.get("object")
        if obj is None:
            continue

        try:
            if "visible_shadow" in obj_state and hasattr(obj, "visible_shadow"):
                obj.visible_shadow = obj_state["visible_shadow"]

            cycles_visibility = getattr(obj, "cycles_visibility", None)
            if (
                "cycles_visibility_shadow" in obj_state
                and cycles_visibility is not None
                and hasattr(cycles_visibility, "shadow")
            ):
                cycles_visibility.shadow = obj_state["cycles_visibility_shadow"]

            restored += 1
        except ReferenceError:
            continue

    if shadow_state:
        print(f"Restored actor shadow visibility for {restored}/{len(shadow_state)} actor objects")
