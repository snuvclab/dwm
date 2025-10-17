#!/usr/bin/env python3
import bpy
import math
import sys
import os
import json
import numpy as np
from mathutils import Vector
import argparse
import time
import traceback
from datetime import datetime
from pathlib import Path
# ---------------------------
# CLI args
# ---------------------------
argv = sys.argv
argv = argv[argv.index("--") + 1:] if "--" in argv else []

parser = argparse.ArgumentParser()
parser.add_argument("--start_frame", type=int, default=None)
parser.add_argument("--end_frame", type=int, default=None)
parser.add_argument("--animation_index", type=int, default=None, help="Specific animation index (else: all)")
parser.add_argument("--samples", type=int, default=32, help="Cycles samples")
parser.add_argument("--save-path", type=str, default="/home/byungjun/workspace/trumans_ego/ego_render_new",
                    help="Root output dir")
parser.add_argument("--skip-existing", action="store_true", default=True, help="Skip frames that already exist")
parser.add_argument("--no-skip-existing", action="store_true", help="Disable skipping existing frames")
parser.add_argument("--frame-skip", type=int, default=3, help="Render every Nth frame")
parser.add_argument("--stride", type=int, default=25, help="Stride for video sequences (default: 25)")
parser.add_argument("--fov", type=float, default=90.0, help="Camera FOV in degrees (perspective)")
parser.add_argument("--width", type=int, default=720, help="Render width in pixels (default: 720)")
parser.add_argument("--height", type=int, default=480, help="Render height in pixels (default: 480)")
args = parser.parse_args(argv)
if args.no_skip_existing:
    args.skip_existing = False

# ---------------------------
# Paths / logging
# ---------------------------
blend_filepath = bpy.data.filepath
directory_name = os.path.basename(os.path.dirname(blend_filepath))
output_folder = os.path.join(args.save_path, directory_name)
os.makedirs(output_folder, exist_ok=True)

error_log_file = os.path.join(args.save_path, "rendering_errors.log")
def log_error(error_type, error_message, blend_file=None, animation_name=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    blend_file = blend_file or blend_filepath
    animation_info = f" (Animation: {animation_name})" if animation_name else ""
    entry = f"[{timestamp}] {error_type}: {error_message} - File: {blend_file}{animation_info}\n"
    try:
        with open(error_log_file, 'a') as f:
            f.write(entry)
            f.write(f"Traceback:\n{traceback.format_exc()}\n")
            f.write("-"*80 + "\n")
    except Exception as e:
        print(f"Warning: Could not write to error log: {e}")
    print(f"ERROR: {error_type}: {error_message}")
    print(f"Error logged to: {error_log_file}")

# ---------------------------
# Scene handles / discovery
# ---------------------------
start_frame = args.start_frame
end_frame = args.end_frame
animation_index = args.animation_index

# HSI/armature discovery
try:
    armature_name = bpy.context.scene.hsi_properties.name_armature_CC
    if not armature_name:
        for obj in bpy.data.objects:
            if obj.type == 'ARMATURE' and 'CC_Base_Hip' in obj.pose.bones:
                armature_name = obj.name
                break
        if not armature_name:
            msg = "No armature found. Please import a CC4 character first."
            log_error("NO_ARMATURE_FOUND", msg); print(f"Error: {msg}"); sys.exit(1)
except Exception as e:
    msg = f"Could not access HSI addon properties: {e}"
    log_error("HSI_PROPERTIES_ERROR", msg); print(f"Error: {msg}"); sys.exit(1)

eye_mesh_name = "CC_Base_Eye"
parent_bone_name = "CC_Base_FacialBone"

print(f"Using armature: {armature_name}")
camera_fov_degrees = args.fov
cycles_samples = args.samples

# ---------------------------
# Helpers
# ---------------------------

def get_animation_sets():
    try:
        animation_sets_json = bpy.context.scene.hsi_properties.animation_sets
        animation_sets = json.loads(animation_sets_json)
        return animation_sets
    except Exception as e:
        print(f"Error getting animation sets: {e}")
        return {}

def apply_animation_set(animation_index):
    animation_sets = get_animation_sets()
    if not animation_sets:
        print("No animation sets found!")
        return False
    if animation_index >= len(animation_sets):
        print(f"Animation index {animation_index} out of range. Available: 0-{len(animation_sets)-1}")
        return False
    bpy.context.scene.hsi_properties.current_animation_index_display = animation_index
    bpy.ops.hsi.set_animation()
    animation_name = list(animation_sets.keys())[animation_index]
    print(f"Applied animation set {animation_index}: {animation_name}")
    return True

def get_camera_intrinsics(camera_obj):
    cam = camera_obj.data
    scene = bpy.context.scene
    width = scene.render.resolution_x
    height = scene.render.resolution_y
    cam.lens_unit = 'FOV'
    cam.angle = math.radians(camera_fov_degrees)
    fov_rad = cam.angle
    focal_length_px = (width/2.0) / math.tan(fov_rad/2.0)
    fx = fy = focal_length_px
    cx, cy = width/2.0, height/2.0
    return np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float32)

def get_camera_to_world_matrix(camera_obj):
    return np.array(camera_obj.matrix_world, dtype=np.float32)

def check_video_exists(video_idx, videos_output_path):
    """
    Check if video already exists and is complete.
    Uses filesize threshold to detect incomplete files from interrupted renders.
    """
    def file_ok(path, min_size=10240):  # 10KB minimum for a valid video
        """File must exist, have reasonable size, and not be too recent"""
        if not os.path.isfile(path):
            return False
        size = os.path.getsize(path)
        if size < min_size:  # Too small = likely incomplete
            return False
        # Check if file was modified very recently (< 2 seconds ago)
        import time
        mtime = os.path.getmtime(path)
        if time.time() - mtime < 2.0:
            return False  # File too fresh, might still be writing
        return True
    
    video_path = os.path.join(videos_output_path, f"{video_idx:05d}.mp4")
    video_exists = file_ok(video_path, min_size=10240)  # Videos should be at least 10KB
    needs_rendering = not video_exists
    return video_exists, needs_rendering

def optimize_scene_for_rendering():
    scene = bpy.context.scene
    scene.cycles.samples = cycles_samples
    scene.cycles.device = 'GPU'

    # Persistent data caching (Blender 4.x)
    if hasattr(scene.render, "use_persistent_data"):
        scene.render.use_persistent_data = True

    # Adaptive sampling
    if hasattr(scene.cycles, "use_adaptive_sampling"):
        scene.cycles.use_adaptive_sampling = True
        scene.cycles.adaptive_threshold = 0.05
    if hasattr(scene.cycles, "tile_size"):
        scene.cycles.tile_size = 256

    # ✅ Denoising ON (OptiX GPU based)
    scene.cycles.use_denoising = True
    if hasattr(scene.cycles, "denoiser"):
        keys = scene.cycles.bl_rna.properties['denoiser'].enum_items.keys()
        if "OPTIX" in keys:
            scene.cycles.denoiser = "OPTIX"
            scene.cycles.use_preview_denoising = True
        elif "OPENIMAGEDENOISE" in keys:
            scene.cycles.denoiser = "OPENIMAGEDENOISE"  # CPU fallback
    scene.cycles.preview_denoising = True

    # Other settings
    scene.render.resolution_percentage = 100
    scene.render.use_border = False
    scene.render.use_crop_to_border = False
    scene.render.use_motion_blur = False

def hide_actor_from_rendering():
    """Hide the actor (armature and its children) from rendering for static scenes."""
    if armature_obj:
        # Hide the armature itself
        armature_obj.hide_render = True
        print(f"Hidden armature '{armature_name}' from rendering")
        
        # Hide all children of the armature
        for child in armature_obj.children:
            child.hide_render = True
            print(f"Hidden child '{child.name}' from rendering")
        
        # Also hide any objects that are parented to bones
        for obj in bpy.data.objects:
            if obj.parent_bone and obj.parent == armature_obj:
                obj.hide_render = True
                print(f"Hidden bone-child '{obj.name}' from rendering")

def show_actor_in_rendering():
    """Show the actor in rendering."""
    if armature_obj:
        # Show the armature itself
        armature_obj.hide_render = False
        print(f"Shown armature '{armature_name}' in rendering")
        
        # Show all children of the armature
        for child in armature_obj.children:
            child.hide_render = False
            print(f"Shown child '{child.name}' in rendering")
        
        # Also show any objects that are parented to bones
        for obj in bpy.data.objects:
            if obj.parent_bone and obj.parent == armature_obj:
                obj.hide_render = False
                print(f"Shown bone-child '{obj.name}' in rendering")

def sample_camera_world_transforms(camera_obj, frames):
    """
    Return lists of (location, rotation_quaternion) for the camera's world transform,
    sampled by setting scene.frame_set(f) for each frame in `frames`.
    """
    scene = bpy.context.scene
    locs, rots = [], []
    prev = scene.frame_current
    try:
        for f in frames:
            scene.frame_set(f)
            mw = camera_obj.matrix_world.copy()
            locs.append(mw.to_translation())
            rots.append(mw.to_quaternion())
    finally:
        scene.frame_set(prev)
    return locs, rots

# ---------------------------
# Helpers (stateful freeze/restore)
# ---------------------------

# Save original animation state & camera parenting
_original_animation_state = {
    "camera_parent": None,   # (parent, parent_type, parent_bone, matrix_world_before)
    "actions": {},           # obj_name -> action (or None)
}

def disable_animations_except_camera(camera_obj):
    """
    1) Save camera parenting and clear it (KEEP transform).
    2) Save each object's current action, then detach (freeze) for all objects except the camera.
    """
    global _original_animation_state
    _original_animation_state = {"camera_parent": None, "actions": {}}

    # --- 1) Camera parenting ---
    if camera_obj.parent is not None:
        _original_animation_state["camera_parent"] = (
            camera_obj.parent,
            camera_obj.parent_type,
            camera_obj.parent_bone,
            camera_obj.matrix_world.copy(),
        )
        bpy.context.view_layer.objects.active = camera_obj
        bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')

    # --- 2) Detach actions of all non-camera objects ---
    for obj in bpy.data.objects:
        if obj == camera_obj:
            continue
        if obj.animation_data:
            # record current action (could be None)
            _original_animation_state["actions"][obj.name] = obj.animation_data.action
            # keep the action from being purged
            if obj.animation_data.action:
                obj.animation_data.action.use_fake_user = True
            # detach to freeze at current pose
            obj.animation_data.action = None

    # Depsgraph refresh helps ensure “frozen” state is used during rendering
    bpy.context.view_layer.update()


def restore_animations(camera_obj):
    """
    Restore every object's action and camera parenting that we saved in disable_animations_except_camera().
    This is idempotent-safe and can be called at the end of each video iteration.
    """
    global _original_animation_state
    if not _original_animation_state:
        return

    # --- 1) Restore actions ---
    for obj_name, action in _original_animation_state.get("actions", {}).items():
        obj = bpy.data.objects.get(obj_name)
        if not obj:
            continue
        if not obj.animation_data:
            obj.animation_data_create()
        obj.animation_data.action = action

    # --- 2) Restore camera parenting ---
    cam_parent = _original_animation_state.get("camera_parent")
    if cam_parent is not None:
        parent, parent_type, parent_bone, mw_before = cam_parent
        camera_obj.parent = parent
        camera_obj.parent_type = parent_type
        if parent_type == 'BONE':
            camera_obj.parent_bone = parent_bone
        # keep the same world transform
        camera_obj.matrix_world = mw_before

    # clear for next use
    _original_animation_state = {"camera_parent": None, "actions": {}}
    bpy.context.view_layer.update()

def bake_camera_keys(camera_obj, frames, locs, rots):
    """Bake camera transform keys based on frames/locs/rots."""
    # Prepare action
    if not camera_obj.animation_data:
        camera_obj.animation_data_create()
    if not camera_obj.animation_data.action:
        camera_obj.animation_data.action = bpy.data.actions.new(name="POV_Camera_Baked")
    action = camera_obj.animation_data.action

    # Clear existing FCurves
    for fc in list(action.fcurves):
        action.fcurves.remove(fc)

    # Create location/rotation fcurves
    loc_curves = [action.fcurves.new(data_path="location", index=i) for i in range(3)]
    rot_curves = [action.fcurves.new(data_path="rotation_quaternion", index=i) for i in range(4)]

    # Insert keys
    for f, loc, rot in zip(frames, locs, rots):
        # Set values
        camera_obj.location = loc
        camera_obj.rotation_mode = 'QUATERNION'
        camera_obj.rotation_quaternion = rot
        # Keyframe
        for i, c in enumerate(loc_curves):
            c.keyframe_points.insert(frame=f, value=camera_obj.location[i], options={'FAST'})
        for i, c in enumerate(rot_curves):
            c.keyframe_points.insert(frame=f, value=camera_obj.rotation_quaternion[i], options={'FAST'})

    # Set interpolation to Linear (if desired)
    for fc in action.fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = 'LINEAR'


def clear_camera_keys(camera_obj):
    """Clean up baked camera keys (delete action)."""
    if camera_obj.animation_data and camera_obj.animation_data.action:
        act = camera_obj.animation_data.action
        camera_obj.animation_data_clear()
        try:
            bpy.data.actions.remove(act)
        except Exception:
            pass


# ---------------------------
# Scene setup
# ---------------------------
scene = bpy.context.scene
render = scene.render
render.engine = 'CYCLES'

# Show AugmentAreaCollection if it exists
if "AugmentAreaCollection" in bpy.data.collections:
    bpy.data.collections["AugmentAreaCollection"].hide_render = True
    print("✓ Hide AugmentAreaCollection for rendering")
else:
    print("ℹ️  AugmentAreaCollection not found - skipping")

# Try GPU
try:
    prefs_cycles = bpy.context.preferences.addons['cycles'].preferences
    prefs_cycles.compute_device_type = 'NONE'
    device_found = False
    for dev_type in ['OPTIX','CUDA']:
        try:
            prefs_cycles.compute_device_type = dev_type
            prefs_cycles.get_devices()
            for d in prefs_cycles.devices:
                if d.type == dev_type:
                    d.use = True
                    print(f"Enabled GPU for Cycles: {d.name} ({d.type})")
                    device_found = True
                    break
            if device_found: break
        except Exception:
            pass
    if not device_found:
        print("No GPU device found or failed to configure. Using CPU.")
except Exception as e:
    print(f"Unexpected GPU setup error: {e}. Using CPU.")

# Resolution & formats
render.resolution_x = args.width
render.resolution_y = args.height
render.resolution_percentage = 100
render.image_settings.file_format = 'PNG'
render.image_settings.color_mode = 'RGBA'

optimize_scene_for_rendering()

# Direct video output (H.264 MP4)
render.image_settings.file_format = 'FFMPEG'
render.ffmpeg.format = 'MPEG4'
render.ffmpeg.codec = 'H264'
render.ffmpeg.constant_rate_factor = 'MEDIUM'   # Quality/speed balance: 18~23 range
render.ffmpeg.ffmpeg_preset = 'REALTIME'       # Speed up in light I/O environments
render.fps = 8


# ---------------------------
# Locate objects & camera setup
# ---------------------------
armature_obj = bpy.data.objects.get(armature_name)
eye_mesh_obj = bpy.data.objects.get(eye_mesh_name)
if not armature_obj:
    msg = f"Armature '{armature_name}' not found"
    log_error("MISSING_ARMATURE", msg); print(f"Error: {msg}"); sys.exit(1)
if not eye_mesh_obj:
    msg = f"Mesh '{eye_mesh_name}' not found"
    log_error("MISSING_EYE_MESH", msg); print(f"Error: {msg}"); sys.exit(1)

# Reset pose at frame 0
print("Moving to frame 0 and resetting pose...")
scene.frame_set(0)
bpy.context.view_layer.objects.active = armature_obj
bpy.ops.object.mode_set(mode='POSE')
for bone in armature_obj.pose.bones:
    bone.rotation_mode = 'QUATERNION'
    bone.location = (0,0,0)
    bone.rotation_quaternion = (1,0,0,0)
    bone.scale = (1,1,1)

# Camera position from eye mesh center
if eye_mesh_obj.type == 'MESH' and eye_mesh_obj.data.vertices:
    mw = eye_mesh_obj.matrix_world.copy()
    bbox_min = Vector((float('inf'),)*3)
    bbox_max = Vector((float('-inf'),)*3)
    for v in eye_mesh_obj.data.vertices:
        bbox_min.x = min(bbox_min.x, v.co.x)
        bbox_min.y = min(bbox_min.y, v.co.y)
        bbox_min.z = min(bbox_min.z, v.co.z)
        bbox_max.x = max(bbox_max.x, v.co.x)
        bbox_max.y = max(bbox_max.y, v.co.y)
        bbox_max.z = max(bbox_max.z, v.co.z)
    local_center = (bbox_min + bbox_max) / 2.0
    camera_initial_world_location = mw @ local_center
else:
    camera_initial_world_location = eye_mesh_obj.matrix_world.translation

if camera_initial_world_location is None:
    print("Error: Failed to determine camera position."); sys.exit(1)

# Parent camera to facial bone
parent_pose_bone = armature_obj.pose.bones.get(parent_bone_name)
if not parent_pose_bone:
    msg = f"Bone '{parent_bone_name}' not found in armature '{armature_name}'"
    log_error("MISSING_BONE", msg); print(f"Error: {msg}"); sys.exit(1)

print("Creating camera...")
camera_data = bpy.data.cameras.new(name="POV_Camera")
camera_obj = bpy.data.objects.new(name="POV_Camera", object_data=camera_data)
scene.collection.objects.link(camera_obj)
camera_obj.location = camera_initial_world_location

# look -Y, up Z
target_forward = Vector((0,-1,0))
rot_quat = target_forward.to_track_quat('-Z','Y')
camera_obj.rotation_mode = 'QUATERNION'
camera_obj.rotation_quaternion = rot_quat

camera_data.lens_unit = 'FOV'
camera_data.angle = math.radians(camera_fov_degrees)
bpy.context.scene.camera = camera_obj
print(f"Set POV_Camera active. FOV {camera_fov_degrees}°.")

# Parent to bone
for obj in bpy.data.objects: obj.select_set(False)
camera_obj.select_set(True)
bpy.context.view_layer.objects.active = armature_obj
bpy.ops.object.mode_set(mode='POSE')
for b in armature_obj.pose.bones: b.bone.select = False
parent_pose_bone.bone.select = True
bpy.context.view_layer.objects.active.data.bones.active = parent_pose_bone.bone
bpy.ops.object.parent_set(type='BONE')
bpy.ops.object.mode_set(mode='OBJECT')

# ---------------------------
# Render sequence
# ---------------------------
def render_animation_sequence(animation_index, animation_name):
    anim_output_folder = os.path.join(output_folder, f"{animation_name}")
    sequences_folder = os.path.join(anim_output_folder, "sequences")
    videos_output_path = os.path.join(sequences_folder, "videos_static")
    os.makedirs(videos_output_path, exist_ok=True)
    
    # Create temp directory using scene name and animation info
    # Extract scene code from output_folder (first 8 characters)
    scene_code = os.path.basename(output_folder)
    
    # Simple temp directory naming
    base_temp_dir = Path.cwd() / "temp_static_images"
    base_temp_dir.mkdir(exist_ok=True)
    
    temp_dir = base_temp_dir / f"{scene_code}_{animation_name}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    print(f"Temporary directory for images: {temp_dir}")
    print(f"  Scene code: {scene_code}, Animation: {animation_name}")

    print(f"Rendering animation {animation_index}: {animation_name}")
    print(f"  Videos: {videos_output_path}")
    print(f"  Temp images: {temp_dir}")

    # Frame range and video sequence parameters
    scene = bpy.context.scene
    render_start_frame = scene.frame_start if start_frame is None else start_frame
    render_end_frame   = scene.frame_end   if end_frame   is None else end_frame
    
    # Video sequence parameters
    clip_length = 49  # 49 frames per video
    stride = args.stride
    frame_skip = args.frame_skip
    fps = 8
    
    # Calculate video start frames
    effective_stride = stride * frame_skip  # Actual frame stride
    video_start_frames = []
    current_frame = render_start_frame
    while current_frame + (clip_length - 1) * frame_skip <= render_end_frame:
        video_start_frames.append(current_frame)
        current_frame += effective_stride
    
    print(f"Animation frames: {render_start_frame}..{render_end_frame}")
    print(f"Video parameters: {clip_length} frames, stride {stride}, frame_skip {frame_skip}")
    print(f"Effective stride: {effective_stride} frames (50% overlap)")
    print(f"Creating {len(video_start_frames)} video sequences")

    start_time = time.time()
    videos_completed = 0
    total_render_time = 0.0

    for video_idx, start_frame_num in enumerate(video_start_frames):
        video_end_frame = start_frame_num + (clip_length - 1) * frame_skip
        frames_to_render = list(range(start_frame_num, video_end_frame + 1, frame_skip))

        # Check if video already exists
        video_exists, needs_video_rendering = check_video_exists(video_idx, videos_output_path)
        
        if args.skip_existing and not needs_video_rendering:
            print(f"\n========== VIDEO {video_idx + 1}/{len(video_start_frames)} ==========")
            print(f"SKIPPED: Video {video_idx:05d}.mp4 already exists")
            videos_completed += 1
            continue

        # === (A) Loop start: restore clean state ===
        restore_animations(camera_obj)    # Restore what was separated from previous clip
        show_actor_in_rendering()         # Show actor (armature) again

        # Apply animation for current clip (in dynamic state)
        if not apply_animation_set(animation_index):
            print(f"Failed to apply animation set for video {video_idx}")
            continue

        # === (B) Sample camera poses for this clip ===
        cam_locs, cam_rots = sample_camera_world_transforms(camera_obj, frames_to_render)

        # Set scene to start frame of clip
        scene.frame_set(start_frame_num)

        # === (C) Freeze scene: separate all animations except camera & unparent camera ===
        disable_animations_except_camera(camera_obj)

        # Hide actor (static scene)
        hide_actor_from_rendering()

        print(f"\n========== VIDEO {video_idx + 1}/{len(video_start_frames)} ==========")
        print(f"Frames: {start_frame_num}..{video_end_frame} (step {frame_skip}) -> {len(frames_to_render)} frames")

        # ---- Camera key baking followed by animation render ----
        # 1) Bake camera keys
        bake_camera_keys(camera_obj, frames_to_render, cam_locs, cam_rots)

        # 2) Set frame range/step
        orig_frame_step = getattr(scene, "frame_step", 1)
        if hasattr(scene, "frame_step"):
            scene.frame_step = frame_skip  # Render as 0,3,6,... if 3

        scene.frame_start = start_frame_num
        scene.frame_end   = video_end_frame

        # 3) Output file path (direct mp4)
        video_output_path = os.path.join(videos_output_path, f"{video_idx:05d}.mp4")
        render.filepath = os.path.splitext(video_output_path)[0]  # Blender adds extension

        # 4) Render (continuous)
        video_start_time = time.time()
        bpy.ops.render.render(animation=True)
        final_path = os.path.join(videos_output_path, f"{video_idx:05d}.mp4")
        # Search for actual file name created by Blender
        for f in os.listdir(videos_output_path):
            if f.startswith(f"{video_idx:05d}") and f.endswith(".mp4") and "-" in f:
                os.rename(os.path.join(videos_output_path, f), final_path)
                print(f"Renamed {f} -> {os.path.basename(final_path)}")
                break
        clip_time = time.time() - video_start_time
        print(f"  ✅ Created video: {os.path.basename(video_output_path)} ({clip_time:.1f}s)")

        # 5) Cleanup (camera keys, restore frame_step)
        clear_camera_keys(camera_obj)
        if hasattr(scene, "frame_step"):
            scene.frame_step = orig_frame_step

        # 6) Restore for next clip
        restore_animations(camera_obj)
        show_actor_in_rendering()

        videos_completed += 1

        # Progress output
        total_elapsed = time.time() - start_time
        avg_video_time = total_elapsed / max(1, videos_completed)
        remaining_videos = len(video_start_frames) - videos_completed
        eta = remaining_videos * avg_video_time
        print(f"  📊 Progress: {videos_completed}/{len(video_start_frames)} videos")
        print(f"  ⏱️  Video time: {clip_time:.1f}s | Avg: {avg_video_time:.1f}s")
        print(f"  🎯 ETA: {eta/60:.1f} min | Elapsed: {total_elapsed/60:.1f} min")

    total_time = time.time() - start_time
    avg_fps = (videos_completed * clip_length) / total_time if total_time > 0 else 0

    # Final cleanup of temporary directory
    print(f"\n🧹 Final cleanup of temporary directory: {temp_dir}")
    try:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"  ✅ Cleaned up temporary directory")
    except Exception as e:
        print(f"  ⚠️  Cleanup warning: {e}")

    print("\n" + "="*50)
    print(f"COMPLETED: Animation {animation_index} ({animation_name})")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Videos created: {videos_completed}/{len(video_start_frames)}")
    print(f"Frame step: {frame_skip} | Stride: {stride} | Effective stride: {effective_stride}")
    print(f"Average throughput: {avg_fps:.2f} fps")
    print(f"Skip existing: {args.skip_existing}")
    print("="*50)

# ---------------------------
# Anim sets & main loop
# ---------------------------
animation_sets = get_animation_sets()
if not animation_sets:
    print("No animation sets found!"); sys.exit(0)
print(f"Found {len(animation_sets)} animation sets:")
for i, name in enumerate(animation_sets.keys()):
    print(f"  {i}: {name}")

if animation_index is not None:
    if animation_index >= len(animation_sets):
        print(f"Error: Animation index {animation_index} out of range."); sys.exit(1)
    animations_to_render = [(animation_index, list(animation_sets.keys())[animation_index])]
    print(f"Rendering specific animation: {animation_index}")
else:
    animations_to_render = [(i, name) for i, name in enumerate(animation_sets.keys())]
    print(f"Rendering all {len(animations_to_render)} animations")

print("Starting animation rendering loop...")
total_start_time = time.time()
failed_animations = []
for anim_idx, anim_name in animations_to_render:
    print("\n" + "="*60)
    print(f"PROCESSING ANIMATION {anim_idx}: {anim_name}")
    print("="*60)
    try:
        if not apply_animation_set(anim_idx):
            msg = f"Failed to apply animation {anim_idx}"
            log_error("ANIMATION_APPLY_FAILED", msg, animation_name=anim_name)
            print(f"Failed to apply animation {anim_idx}. Skipping...")
            failed_animations.append((anim_idx, anim_name, "ANIMATION_APPLY_FAILED"))
            continue
        render_animation_sequence(anim_idx, anim_name[:-4])  # strip e.g., ".fbx"
        print(f"Completed animation {anim_idx}: {anim_name}")
    except Exception as e:
        msg = f"Unexpected error during animation {anim_idx}: {str(e)}"
        log_error("RENDERING_ERROR", msg, animation_name=anim_name)
        print(f"Error during animation {anim_idx}: {str(e)}")
        failed_animations.append((anim_idx, anim_name, "RENDERING_ERROR"))
        continue

total_time = time.time() - total_start_time
# rendered frames count with step
total_frames_rendered = 0
for idx, _name in animations_to_render:
    if idx not in [f[0] for f in failed_animations]:
        scene = bpy.context.scene
        frames_rendered = len(range(scene.frame_start, scene.frame_end + 1, max(1, args.frame_skip)))
        total_frames_rendered += frames_rendered
overall_fps = total_frames_rendered / total_time if total_time > 0 else 0

print("\n" + "="*60)
print("RENDERING COMPLETE!")
print(f"Total time: {total_time/3600:.1f} hours ({total_time/60:.1f} minutes)")
print(f"Total frames rendered (stepped): {total_frames_rendered}")
print(f"Overall throughput: {overall_fps:.2f} fps")
print(f"Results saved in: '{output_folder}'")
print("Each animation has its own subfolder: {animation_name}/")
print("Output structure:")
print("  {animation_name}/")
print("    └── sequences/")
print("        └── videos_static/         # Static scene video sequences (MP4)")
if failed_animations:
    print(f"\nFAILED ANIMATIONS ({len(failed_animations)}):")
    for anim_idx, anim_name, error_type in failed_animations:
        print(f"  Animation {anim_idx} ({anim_name}): {error_type}")
    print(f"Check error log for details: {error_log_file}")
else:
    print("\nAll animations completed successfully!")
print("="*60)
