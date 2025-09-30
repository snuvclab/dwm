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
parser.add_argument("--fov", type=float, default=90.0, help="Camera FOV in degrees (perspective)")
parser.add_argument("--width", type=int, default=720, help="Render width in pixels (default: 720)")
parser.add_argument("--height", type=int, default=480, help="Render height in pixels (default: 480)")
parser.add_argument("--no-depth", action="store_true", help="Skip depth rendering (RGB only)")
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

def check_frame_exists(frame_num, images_output_path, depth_output_path, cam_params_path, no_depth=False):
    def file_ok(path): return os.path.isfile(path) and os.path.getsize(path) > 0
    image_path = os.path.join(images_output_path, f"{frame_num:04d}.png")
    depth_path = os.path.join(depth_output_path,  f"{frame_num:04d}.exr")
    cam_param_path = os.path.join(cam_params_path, f"cam_{frame_num:04d}.npy")
    rgb_exists = file_ok(image_path)
    depth_exists = file_ok(depth_path) if not no_depth else True  # Skip depth check if no_depth
    cam_param_exists = file_ok(cam_param_path)
    needs_rendering = not rgb_exists or (not depth_exists and not no_depth)
    needs_cam_param = not cam_param_exists
    return rgb_exists, depth_exists, cam_param_exists, needs_rendering, needs_cam_param

def optimize_scene_for_rendering():
    scene = bpy.context.scene
    scene.cycles.samples = cycles_samples
    scene.cycles.use_denoising = True
    if hasattr(scene.cycles, 'denoiser'):
        keys = scene.cycles.bl_rna.properties['denoiser'].enum_items.keys()
        if 'OPTIX' in keys: scene.cycles.denoiser = 'OPTIX'
        elif 'OPENIMAGEDENOISE' in keys: scene.cycles.denoiser = 'OPENIMAGEDENOISE'
    scene.render.resolution_percentage = 100
    scene.render.use_border = False
    scene.render.use_crop_to_border = False
    if hasattr(scene.render, 'use_free_unused_nodes'): scene.render.use_free_unused_nodes = True
    if hasattr(scene.render, 'use_free_image_textures'): scene.render.use_free_image_textures = True
    print(f"Optimized scene: {scene.render.resolution_x}x{scene.render.resolution_y}, {cycles_samples} samples")

# ---------------------------
# Scene setup
# ---------------------------
scene = bpy.context.scene
render = scene.render
render.engine = 'CYCLES'
optimize_scene_for_rendering()

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

# ---------------------------
# Make Clean-Depth View Layer (opaque override) BEFORE building nodes
# ---------------------------
mat_depth = None
depth_layer = None

if not args.no_depth:
    # Create solid diffuse material
    mat_depth = bpy.data.materials.get("DepthOverride")
    if mat_depth is None:
        mat_depth = bpy.data.materials.new("DepthOverride")
    mat_depth.use_nodes = True
    nodes = mat_depth.node_tree.nodes
    links = mat_depth.node_tree.links
    for n in list(nodes): nodes.remove(n)
    bsdf = nodes.new("ShaderNodeBsdfDiffuse")
    out  = nodes.new("ShaderNodeOutputMaterial")
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    # Create / configure depth view layer
    depth_layer = scene.view_layers.get("DepthSolid") or scene.view_layers.new("DepthSolid")
    depth_layer.material_override = mat_depth
    depth_layer.use_pass_z = True

# ---------------------------
# Compositor nodes
# ---------------------------
scene.use_nodes = True
tree = scene.node_tree
for node in list(tree.nodes):
    tree.nodes.remove(node)

# Default ViewLayer (RGB only, no Depth)
rl_rgb = tree.nodes.new(type='CompositorNodeRLayers')
rl_rgb.location = 0, 0
rl_rgb.layer = bpy.context.view_layer.name  # usually "ViewLayer"

rgb_output_node = tree.nodes.new(type='CompositorNodeOutputFile')
rgb_output_node.label = "RGB Output"
rgb_output_node.base_path = output_folder
rgb_output_node.file_slots[0].path = "####"
rgb_output_node.format.file_format = 'PNG'
rgb_output_node.format.color_mode = 'RGBA'
rgb_output_node.location = 400, 200
tree.links.new(rl_rgb.outputs['Image'], rgb_output_node.inputs[0])

# DepthSolid layer for clean Z (meters) - only if depth rendering is enabled
rl_depth = None
depth_output_node = None

if not args.no_depth and depth_layer:
    rl_depth = tree.nodes.new(type='CompositorNodeRLayers')
    rl_depth.location = 0, -200
    rl_depth.layer = depth_layer.name

    depth_output_node = tree.nodes.new(type='CompositorNodeOutputFile')
    depth_output_node.label = "Depth Clean (EXR)"
    depth_output_node.base_path = output_folder
    depth_output_node.file_slots[0].path = "####"
    depth_output_node.format.file_format = 'OPEN_EXR'
    depth_output_node.format.color_depth = '32'
    depth_output_node.format.exr_codec = 'ZIP'
    depth_output_node.location = 400, -200
    if 'Depth' in rl_depth.outputs:
        tree.links.new(rl_depth.outputs['Depth'], depth_output_node.inputs[0])
    else:
        print("Warning: Depth output not found in DepthSolid layer.")

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
    images_output_path = os.path.join(anim_output_folder, "images")
    depth_output_path  = os.path.join(anim_output_folder, "depth")
    cam_params_path    = os.path.join(anim_output_folder, "cam_params")
    
    # Create directories - only create depth folder if depth rendering is enabled
    os.makedirs(images_output_path, exist_ok=True)
    os.makedirs(cam_params_path, exist_ok=True)
    if not args.no_depth:
        os.makedirs(depth_output_path, exist_ok=True)

    print(f"Rendering animation {animation_index}: {animation_name}")
    print(f"  Images: {images_output_path}")
    if not args.no_depth:
        print(f"  Depth : {depth_output_path}")
    print(f"  Cam   : {cam_params_path}")

    # Set base paths per animation
    rgb_output_node.base_path   = images_output_path
    rgb_output_node.file_slots[0].path   = "####"
    if not args.no_depth and depth_output_node:
        depth_output_node.base_path = depth_output_path
        depth_output_node.file_slots[0].path = "####"

    # Save intrinsics (same for all frames in render res)
    intrinsics = get_camera_intrinsics(camera_obj)
    np.save(os.path.join(cam_params_path, "intrinsics.npy"), intrinsics)

    # Frame range
    scene = bpy.context.scene
    render_start_frame = scene.frame_start if start_frame is None else start_frame
    render_end_frame   = scene.frame_end   if end_frame   is None else end_frame
    frames_to_render = list(range(render_start_frame, render_end_frame + 1, max(1, args.frame_skip)))
    total_frames = len(frames_to_render)
    print(f"Render frames: {render_start_frame}..{render_end_frame} (step {args.frame_skip}) -> {total_frames} frames")

    start_time = time.time()
    frames_completed = 0
    frames_skipped   = 0
    cam_params_saved_count = 0
    total_render_time = 0.0

    for frame_num in frames_to_render:
        rgb_exists, depth_exists, cam_param_exists, needs_rendering, needs_cam_param = \
            check_frame_exists(frame_num, images_output_path, depth_output_path, cam_params_path, args.no_depth)

        scene.frame_set(frame_num)

        # Save cam2world per frame if missing
        if needs_cam_param:
            c2w = get_camera_to_world_matrix(camera_obj)
            np.save(os.path.join(cam_params_path, f"cam_{frame_num:04d}.npy"), c2w)
            cam_params_saved_count += 1
            cam_param_exists = True

        if args.skip_existing and not needs_rendering:
            frames_skipped += 1
            status = []
            if rgb_exists: status.append("RGB")
            if depth_exists and not args.no_depth: status.append("Depth")
            if cam_param_exists: status.append("Cam")
            print(f"[ANIM {animation_index}] Frame {frame_num}: SKIPPED ({', '.join(status)})")
            continue

        # Render (compositor writes RGB+clean depth directly to anim folders)
        frame_start_time = time.time()
        bpy.ops.render.render(write_still=True)

        frame_time = time.time() - frame_start_time
        total_render_time += frame_time
        frames_completed += 1

        # Progress
        total_processed = frames_completed + frames_skipped
        progress = total_processed / total_frames * 100.0
        elapsed = time.time() - start_time

        if frames_completed > 1:
            avg = total_render_time / frames_completed
            remaining = total_frames - total_processed
            eta = remaining * avg
            fps = 1.0 / avg if avg > 0 else 0
            if (frame_num % 5 == 0) or (frame_num == frames_to_render[0] + 1):
                print(f"\n========== PROGRESS ==========")
                print(f"[ANIM {animation_index}] Frame {frame_num}/{frames_to_render[-1]} ({progress:.1f}%)")
                print(f"  Rendered: {frames_completed} | Skipped: {frames_skipped} | Remaining: {remaining}")
                print(f"  Frame time: {frame_time:.1f}s | Avg: {avg:.1f}s | Throughput: {fps:.2f} fps")
                print(f"  ETA: {eta/60:.1f} min | Elapsed: {elapsed/60:.1f} min")
                print(f"==============================")
        else:
            print(f"\n========== PROGRESS ==========")
            print(f"[ANIM {animation_index}] Frame {frame_num}/{frames_to_render[-1]} ({progress:.1f}%)")
            print(f"  Rendered: {frames_completed} | Skipped: {frames_skipped}")
            print(f"  Frame time: {frame_time:.1f}s")
            print(f"==============================")

    total_time = time.time() - start_time
    avg_fps = frames_completed / total_time if total_time > 0 and frames_completed > 0 else 0

    # Count saved cam params
    cam_params_saved = 0
    for frame_num in frames_to_render:
        if os.path.exists(os.path.join(cam_params_path, f"cam_{frame_num:04d}.npy")):
            cam_params_saved += 1

    print("\n" + "="*50)
    print(f"COMPLETED: Animation {animation_index} ({animation_name})")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Frame step: {args.frame_skip} | Rendered: {frames_completed} | Skipped: {frames_skipped}")
    print(f"Camera parameters saved: {cam_params_saved_count} this run | Total present: {cam_params_saved}/{total_frames}")
    print(f"Average throughput: {avg_fps:.2f} fps")
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
if failed_animations:
    print(f"\nFAILED ANIMATIONS ({len(failed_animations)}):")
    for anim_idx, anim_name, error_type in failed_animations:
        print(f"  Animation {anim_idx} ({anim_name}): {error_type}")
    print(f"Check error log for details: {error_log_file}")
else:
    print("\nAll animations completed successfully!")
print("="*60)
