import bpy
import math
import sys
import os
import json
import numpy as np
from mathutils import Vector, Matrix
import argparse
import time
import traceback
from datetime import datetime

# --- Command-line arguments ---
argv = sys.argv
if "--" in argv:
    argv = argv[argv.index("--") + 1:]  # Only keep arguments after '--'
else:
    argv = []

parser = argparse.ArgumentParser()
parser.add_argument("--start_frame", type=int, default=None)
parser.add_argument("--end_frame", type=int, default=None)
parser.add_argument("--animation_index", type=int, default=None, help="Specific animation index to render (if not specified, will loop through all)")
parser.add_argument("--samples", type=int, default=64, help="Cycles samples (reduced for speed)")
parser.add_argument("--resolution", type=int, default=720, help="Max resolution")
parser.add_argument("--save-path", type=str, default="/home/byungjun/workspace/trumans_ego/ego_render_new", help="Path to save rendered outputs")
parser.add_argument("--skip-existing", action="store_true", default=True, help="Skip frames that already exist (default: True)")
parser.add_argument("--no-skip-existing", action="store_true", help="Disable skipping existing frames")
args = parser.parse_args(argv)

# Handle skip-existing arguments
if args.no_skip_existing:
    args.skip_existing = False

# --- Output folder naming based on directory and .blend file ---
blend_filepath = bpy.data.filepath
blend_filename = os.path.splitext(os.path.basename(blend_filepath))[0]
# Get the directory name from the blend file path
directory_name = os.path.basename(os.path.dirname(blend_filepath))
# Create unique output folder using directory name only
output_folder = os.path.join(args.save_path, directory_name)

# Set up error logging
error_log_file = os.path.join(args.save_path, "rendering_errors.log")

def log_error(error_type, error_message, blend_file=None, animation_name=None):
    """Log rendering errors to file with timestamp and context."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    blend_file = blend_file or blend_filepath
    animation_info = f" (Animation: {animation_name})" if animation_name else ""
    
    error_entry = f"[{timestamp}] {error_type}: {error_message} - File: {blend_file}{animation_info}\n"
    
    try:
        with open(error_log_file, 'a') as f:
            f.write(error_entry)
            f.write(f"Traceback:\n{traceback.format_exc()}\n")
            f.write("-" * 80 + "\n")
    except Exception as e:
        print(f"Warning: Could not write to error log: {e}")
    
    print(f"ERROR: {error_type}: {error_message}")
    print(f"Error logged to: {error_log_file}")

# --- Apply parsed args ---
start_frame = args.start_frame
end_frame = args.end_frame
animation_index = args.animation_index

# Get armature name from HSI addon properties
try:
    armature_name = bpy.context.scene.hsi_properties.name_armature_CC
    if not armature_name:
        # Fallback: try to find any armature with CC bones
        for obj in bpy.data.objects:
            if obj.type == 'ARMATURE' and 'CC_Base_Hip' in obj.pose.bones:
                armature_name = obj.name
                break
        if not armature_name:
            error_msg = "No armature found. Please import a CC4 character first."
            log_error("NO_ARMATURE_FOUND", error_msg)
            print(f"Error: {error_msg}")
            exit(1)
except Exception as e:
    error_msg = f"Could not access HSI addon properties: {e}"
    log_error("HSI_PROPERTIES_ERROR", error_msg)
    print(f"Error: {error_msg}")
    exit(1)

eye_mesh_name = "CC_Base_Eye"
parent_bone_name = "CC_Base_FacialBone"

print(f"Using armature: {armature_name}")

camera_fov_degrees = 60.0
cycles_samples = args.samples  # Reduced from 128 to 64 for speed
max_resolution = args.resolution
# --- End of User Configuration ---

def get_animation_sets():
    """Get all available animation sets from the scene properties"""
    try:
        animation_sets_json = bpy.context.scene.hsi_properties.animation_sets
        animation_sets = json.loads(animation_sets_json)
        return animation_sets
    except Exception as e:
        print(f"Error getting animation sets: {e}")
        return {}

def apply_animation_set(animation_index):
    """Apply a specific animation set by index"""
    animation_sets = get_animation_sets()
    
    if not animation_sets:
        print("No animation sets found!")
        return False
    
    if animation_index >= len(animation_sets):
        print(f"Animation index {animation_index} out of range. Available: 0-{len(animation_sets)-1}")
        return False
    
    # Set the animation index display
    bpy.context.scene.hsi_properties.current_animation_index_display = animation_index
    
    # Apply the animation
    bpy.ops.hsi.set_animation()
    
    animation_name = list(animation_sets.keys())[animation_index]
    print(f"Applied animation set {animation_index}: {animation_name}")
    
    return True

def get_camera_intrinsics(camera_obj):
    """Extract camera intrinsics matrix from Blender camera"""
    camera_data = camera_obj.data
    
    # Get render resolution
    scene = bpy.context.scene
    width = scene.render.resolution_x
    height = scene.render.resolution_y
    
    # Calculate focal length in pixels
    if camera_data.lens_unit == 'FOV':
        # Convert FOV to focal length
        fov_rad = camera_data.angle
        focal_length_px = (width / 2.0) / math.tan(fov_rad / 2.0)
    else:
        # Focal length in mm, convert to pixels
        sensor_width = camera_data.sensor_width
        focal_length_px = (camera_data.lens * width) / sensor_width
    
    # Create intrinsics matrix
    fx = focal_length_px
    fy = focal_length_px  # Assuming square pixels
    cx = width / 2.0
    cy = height / 2.0
    
    intrinsics = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    
    return intrinsics

def get_world_to_camera_matrix(camera_obj):
    """Get 4x4 world-to-camera transformation matrix"""
    # Get camera world matrix
    camera_world_matrix = camera_obj.matrix_world
    
    # Convert to numpy array
    world_to_camera = np.array(camera_world_matrix, dtype=np.float32)
    
    return world_to_camera

def check_frame_exists(frame_num, images_output_path, depth_output_path, cam_params_path):
    """Check if required files exist for a given frame.
    Returns (rgb_exists, depth_exists, cam_param_exists, needs_rendering, needs_cam_param)"""
    
    # Check image file (PNG format)
    image_filename = f"{frame_num:04d}.png"
    image_path = os.path.join(images_output_path, image_filename)
    
    # Check depth file (EXR format)
    depth_filename = f"{frame_num:04d}.exr"
    depth_path = os.path.join(depth_output_path, depth_filename)
    
    # Check camera parameters file
    cam_param_filename = f"cam_{frame_num:04d}.npy"
    cam_param_path = os.path.join(cam_params_path, cam_param_filename)
    
    # Check what exists
    rgb_exists = os.path.exists(image_path)
    depth_exists = os.path.exists(depth_path)
    cam_param_exists = os.path.exists(cam_param_path)
    
    # Need rendering if either RGB OR depth is missing (or both)
    needs_rendering = not rgb_exists or not depth_exists
    
    # Always need camera parameter if it doesn't exist
    needs_cam_param = not cam_param_exists
    
    return rgb_exists, depth_exists, cam_param_exists, needs_rendering, needs_cam_param

def optimize_scene_for_rendering():
    """Optimize scene settings for faster rendering"""
    scene = bpy.context.scene
    
    # Optimize Cycles settings
    scene.cycles.samples = cycles_samples
    scene.cycles.use_denoising = True
    # Use available denoiser (OpenImageDenoise is the default and most compatible)
    if hasattr(scene.cycles, 'denoiser'):
        available_denoisers = scene.cycles.bl_rna.properties['denoiser'].enum_items.keys()
        if 'OPTIX' in available_denoisers:
            scene.cycles.denoiser = 'OPTIX'  # Use OptiX denoiser for speed
        elif 'OPENIMAGEDENOISE' in available_denoisers:
            scene.cycles.denoiser = 'OPENIMAGEDENOISE'  # Fallback to OpenImageDenoise
        # If neither is available, just use denoising without specifying type
    
    # Optimize render settings (don't override resolution, it will be set later)
    scene.render.resolution_percentage = 100
    
    # Optimize viewport settings
    scene.render.use_border = False
    scene.render.use_crop_to_border = False
    
    # Optimize memory usage (check if properties exist first)
    if hasattr(scene.render, 'use_free_unused_nodes'):
        scene.render.use_free_unused_nodes = True
    if hasattr(scene.render, 'use_free_image_textures'):
        scene.render.use_free_image_textures = True
    
    print(f"Optimized scene: {scene.render.resolution_x}x{scene.render.resolution_y}, {cycles_samples} samples")

def render_animation_sequence(animation_index, animation_name):
    """Render the current animation sequence"""
    # Create animation-specific output directories
    anim_output_folder = os.path.join(output_folder, f"{animation_name}")
    images_output_path = os.path.join(anim_output_folder, "images")  # Changed from "rgb" to "images"
    depth_output_path = os.path.join(anim_output_folder, "depth")
    cam_params_path = os.path.join(anim_output_folder, "cam_params")
    os.makedirs(images_output_path, exist_ok=True)
    os.makedirs(depth_output_path, exist_ok=True)
    os.makedirs(cam_params_path, exist_ok=True)
    
    print(f"Rendering animation {animation_index}: {animation_name}")
    print(f"Output paths: images: {images_output_path}, depth: {depth_output_path}, cam_params: {cam_params_path}")
    
    # Update output node paths for this animation
    tree = bpy.context.scene.node_tree
    rgb_output_node = None
    depth_output_node = None
    
    for node in tree.nodes:
        if node.label == "RGB Output":
            rgb_output_node = node
        elif node.label == "Depth Output (EXR)":
            depth_output_node = node
    
    if rgb_output_node:
        rgb_output_node.base_path = images_output_path  # Changed from rgb_output_path to images_output_path
    if depth_output_node:
        depth_output_node.base_path = depth_output_path
    
    # Get camera intrinsics (same for all frames in this animation)
    camera_obj = bpy.context.scene.camera
    intrinsics = get_camera_intrinsics(camera_obj)
    intrinsics_path = os.path.join(cam_params_path, "intrinsics.npy")  # Changed to save in cam_params/intrinsics.npy
    np.save(intrinsics_path, intrinsics)
    print(f"Saved camera intrinsics to: {intrinsics_path}")
    print(f"Intrinsics matrix:\n{intrinsics}")
    
    # Determine render frame range for this animation
    scene = bpy.context.scene
    if start_frame is None:
        render_start_frame = scene.frame_start
    else:
        render_start_frame = start_frame

    if end_frame is None:
        render_end_frame = scene.frame_end
    else:
        render_end_frame = end_frame

    total_frames = render_end_frame - render_start_frame + 1
    print(f"Render range for animation {animation_index}: {render_start_frame} to {render_end_frame} ({total_frames} frames)")
    
    # Rendering loop for this animation
    start_time = time.time()
    frames_completed = 0
    frames_skipped = 0
    cam_params_saved_count = 0
    total_render_time = 0
    
    print(f"\n{'='*50}")
    print(f"STARTING RENDER: Animation {animation_index} ({animation_name})")
    print(f"Total frames: {total_frames}")
    print(f"Resolution: {scene.render.resolution_x}x{scene.render.resolution_y}")
    print(f"Samples: {scene.cycles.samples}")
    print(f"{'='*50}")
    
    for frame_num in range(render_start_frame, render_end_frame + 1):
        # Check what files exist and what needs to be done
        rgb_exists, depth_exists, cam_param_exists, needs_rendering, needs_cam_param = check_frame_exists(
            frame_num, images_output_path, depth_output_path, cam_params_path
        )
        
        # Set frame for camera parameter calculation (always needed)
        scene.frame_set(frame_num)
        
        # Always save camera parameters if missing (regardless of rendering status)
        if needs_cam_param:
            world_to_camera = get_world_to_camera_matrix(camera_obj)
            cam_param_filename = f"cam_{frame_num:04d}.npy"
            cam_param_path = os.path.join(cam_params_path, cam_param_filename)
            np.save(cam_param_path, world_to_camera)
            cam_params_saved_count += 1
            print(f"[ANIM {animation_index}] Frame {frame_num}: Saved camera parameters")
            # Update the status since we just saved the camera parameters
            cam_param_exists = True
        
        # Skip rendering if not needed and skip-existing is enabled
        if args.skip_existing and not needs_rendering:
            frames_skipped += 1
            status = []
            if rgb_exists: status.append("RGB")
            if depth_exists: status.append("Depth")
            if cam_param_exists: status.append("Cam")
            print(f"[ANIM {animation_index}] Frame {frame_num}: SKIPPED (RGB and Depth exist: {', '.join(status)})")
            continue
        
        # Render if needed
        if needs_rendering:
            frame_start_time = time.time()
            
            # Log what's missing
            missing = []
            if not rgb_exists: missing.append("RGB")
            if not depth_exists: missing.append("Depth")
            print(f"[ANIM {animation_index}] Frame {frame_num}: RENDERING (missing: {', '.join(missing)})")
            
            # Render RGB and Depth
            if rgb_output_node:
                rgb_output_node.file_slots[0].path = f""
            if depth_output_node:
                depth_output_node.file_slots[0].path = f""
            bpy.ops.render.render(write_still=True)
            
            # Calculate timing and throughput
            frame_time = time.time() - frame_start_time
            total_render_time += frame_time
            frames_completed += 1
            
            # Progress reporting with throughput metrics
            total_processed = frames_completed + frames_skipped
            progress = total_processed / total_frames * 100
            elapsed_time = time.time() - start_time
            
            if frames_completed > 1:
                avg_time_per_frame = total_render_time / frames_completed
                remaining_frames = total_frames - total_processed
                eta = remaining_frames * avg_time_per_frame if frames_completed > 0 else 0
                fps_throughput = 1.0 / avg_time_per_frame
                
                # Show progress every 5 frames for better visibility
                if frame_num % 5 == 0 or frame_num == render_start_frame + 1:
                    print(f"\n{'='*20} PROGRESS {'='*20}")
                    print(f"[ANIM {animation_index}] Frame {frame_num}/{render_end_frame} ({progress:.1f}%)")
                    print(f"  Rendered: {frames_completed} | Skipped: {frames_skipped} | Remaining: {remaining_frames}")
                    print(f"  Frame time: {frame_time:.1f}s | Avg: {avg_time_per_frame:.1f}s | Throughput: {fps_throughput:.2f} fps")
                    print(f"  ETA: {eta/60:.1f}min | Elapsed: {elapsed_time/60:.1f}min")
                    print(f"{'='*50}")
            else:
                print(f"\n{'='*20} PROGRESS {'='*20}")
                print(f"[ANIM {animation_index}] Frame {frame_num}/{render_end_frame} ({progress:.1f}%)")
                print(f"  Rendered: {frames_completed} | Skipped: {frames_skipped}")
                print(f"  Frame time: {frame_time:.1f}s")
                print(f"{'='*50}")
        else:
            # No rendering needed, just count as processed
            frames_skipped += 1
            print(f"[ANIM {animation_index}] Frame {frame_num}: SKIPPED (RGB and Depth both exist)")
    
    total_time = time.time() - start_time
    avg_fps = frames_completed / total_time if total_time > 0 and frames_completed > 0 else 0
    
    # Count how many camera parameters were saved
    cam_params_saved = 0
    for frame_num in range(render_start_frame, render_end_frame + 1):
        cam_param_path = os.path.join(cam_params_path, f"cam_{frame_num:04d}.npy")
        if os.path.exists(cam_param_path):
            cam_params_saved += 1
    
    print(f"\n{'='*50}")
    print(f"COMPLETED: Animation {animation_index} ({animation_name})")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Frames rendered: {frames_completed} | Frames skipped: {frames_skipped} | Total: {frames_completed + frames_skipped}")
    print(f"Camera parameters saved: {cam_params_saved_count} during this run | Total: {cam_params_saved}/{total_frames}")
    print(f"Average throughput: {avg_fps:.2f} fps (rendered frames only)")
    print(f"{'='*50}")



# Scene and render setup
scene = bpy.context.scene
render = scene.render
render.engine = 'CYCLES'

# Optimize scene for rendering
optimize_scene_for_rendering()

# Attempt to enable GPU rendering if available
try:
    prefs_cycles = bpy.context.preferences.addons['cycles'].preferences
    prefs_cycles.compute_device_type = 'NONE'

    device_found = False
    for dev_type in ['OPTIX', 'CUDA']:
        try:
            prefs_cycles.compute_device_type = dev_type
            prefs_cycles.get_devices()
            for d in prefs_cycles.devices:
                if d.type == dev_type:
                    d.use = True
                    print(f"Enabled GPU for Cycles rendering: {d.name} (type: {d.type})")
                    device_found = True
                    break
            if device_found:
                break
        except Exception:
            pass

    if not device_found:
        print("No GPU device found or failed to configure. Falling back to CPU rendering.")

except Exception as e:
    print(f"Unexpected error while setting up Cycles GPU rendering: {e}. Falling back to CPU.")
    pass

scene.cycles.samples = cycles_samples
scene.cycles.use_denoising = True

# Set resolution
render.resolution_x = 720
render.resolution_y = 480
render.resolution_percentage = 100

# Set file format (RGB)
render.image_settings.file_format = 'PNG'
render.image_settings.color_mode = 'RGBA'

# Enable compositor nodes (for depth output)
scene.use_nodes = True
tree = scene.node_tree

# Clear existing nodes
for node in tree.nodes:
    tree.nodes.remove(node)

# Enable Z-Pass before creating nodes
bpy.context.scene.view_layers['ViewLayer'].use_pass_z = True

# Create required nodes
render_layers_node = tree.nodes.new(type='CompositorNodeRLayers')
render_layers_node.location = 0, 0

# RGB output node
rgb_output_node = tree.nodes.new(type='CompositorNodeOutputFile')
rgb_output_node.label = "RGB Output"
rgb_output_node.base_path = output_folder  # Use output folder as initial path
rgb_output_node.file_slots[0].path = "#"
rgb_output_node.format.file_format = 'PNG'
rgb_output_node.format.color_mode = 'RGBA'
rgb_output_node.location = 400, 200

# Depth output node
depth_output_node = tree.nodes.new(type='CompositorNodeOutputFile')
depth_output_node.label = "Depth Output (EXR)"
depth_output_node.base_path = output_folder  # Use output folder as initial path
depth_output_node.file_slots[0].path = "#"
depth_output_node.format.file_format = 'OPEN_EXR'
depth_output_node.format.color_depth = '32'
depth_output_node.format.exr_codec = 'ZIP'
depth_output_node.location = 400, 0

# Link nodes
if 'Image' in render_layers_node.outputs:
    tree.links.new(render_layers_node.outputs['Image'], rgb_output_node.inputs[0])
else:
    print("Warning: 'Image' output not found in Render Layers node.")

if 'Depth' in render_layers_node.outputs:
    tree.links.new(render_layers_node.outputs['Depth'], depth_output_node.inputs[0])
else:
    print("Warning: 'Depth' output not found in Render Layers node. Make sure Z-pass is enabled.")

# Locate objects and bone
armature_obj = bpy.data.objects.get(armature_name)
eye_mesh_obj = bpy.data.objects.get(eye_mesh_name)

if not armature_obj:
    error_msg = f"Armature '{armature_name}' not found"
    log_error("MISSING_ARMATURE", error_msg)
    print(f"Error: {error_msg}")
    exit(1)
if not eye_mesh_obj:
    error_msg = f"Mesh '{eye_mesh_name}' not found"
    log_error("MISSING_EYE_MESH", error_msg)
    print(f"Error: {error_msg}")
    exit(1)

# Save original mode and selections
original_mode = bpy.context.object.mode if bpy.context.object else None
original_active_obj = bpy.context.active_object
original_selected_objs = [obj for obj in bpy.context.selected_objects]

# Set frame 0 and T-pose for stable reference
print("Moving to frame 0 and resetting pose for setup...")
scene.frame_set(0)
bpy.context.view_layer.objects.active = armature_obj
bpy.ops.object.mode_set(mode='POSE')
for bone in armature_obj.pose.bones:
    bone.rotation_mode = 'QUATERNION'
    bone.location = (0, 0, 0)
    bone.rotation_quaternion = (1, 0, 0, 0)
    bone.scale = (1, 1, 1)

# Compute camera position from eye mesh geometric center
camera_initial_world_location = None
if eye_mesh_obj.type == 'MESH' and eye_mesh_obj.data.vertices:
    eye_mesh_world_matrix_at_tpose = eye_mesh_obj.matrix_world.copy()
    bbox_min_local = Vector((float('inf'), float('inf'), float('inf')))
    bbox_max_local = Vector((float('-inf'), float('-inf'), float('-inf')))
    for vertex in eye_mesh_obj.data.vertices:
        bbox_min_local.x = min(bbox_min_local.x, vertex.co.x)
        bbox_min_local.y = min(bbox_min_local.y, vertex.co.y)
        bbox_min_local.z = min(bbox_min_local.z, vertex.co.z)
        bbox_max_local.x = max(bbox_max_local.x, vertex.co.x)
        bbox_max_local.y = max(bbox_max_local.y, vertex.co.y)
        bbox_max_local.z = max(bbox_max_local.z, vertex.co.z)
    local_geometric_center = (bbox_min_local + bbox_max_local) / 2.0
    camera_initial_world_location = eye_mesh_world_matrix_at_tpose @ local_geometric_center
    print(f"Camera world location from eye mesh geometric center: {camera_initial_world_location}")
else:
    print("Warning: Eye mesh is invalid or empty. Using object origin instead.")
    camera_initial_world_location = eye_mesh_obj.matrix_world.translation
    print(f"Fallback camera world location: {camera_initial_world_location}")

if camera_initial_world_location is None:
    print("Error: Failed to determine camera position. Aborting script.")
    for obj in bpy.data.objects: obj.select_set(False)
    for obj in original_selected_objs: obj.select_set(True)
    if original_active_obj: bpy.context.view_layer.objects.active = original_active_obj
    if original_mode: bpy.ops.object.mode_set(mode=original_mode)
    exit()

# Locate bone to parent camera to
parent_pose_bone = armature_obj.pose.bones.get(parent_bone_name)
if not parent_pose_bone:
    error_msg = f"Bone '{parent_bone_name}' not found in armature '{armature_name}'"
    log_error("MISSING_BONE", error_msg)
    print(f"Error: {error_msg}")
    for obj in bpy.data.objects: obj.select_set(False)
    for obj in original_selected_objs: obj.select_set(True)
    if original_active_obj: bpy.context.view_layer.objects.active = original_active_obj
    if original_mode: bpy.ops.object.mode_set(mode=original_mode)
    exit(1)

# Create and configure camera
print("Creating and configuring camera...")
camera_data = bpy.data.cameras.new(name="POV_Camera")
camera_obj = bpy.data.objects.new(name="POV_Camera", object_data=camera_data)
scene.collection.objects.link(camera_obj)
camera_obj.location = camera_initial_world_location

# Set camera orientation (look -Y, up Z)
target_forward = Vector((0, -1, 0))
target_up = Vector((0, 0, 1))
rot_quat = target_forward.to_track_quat('-Z', 'Y')
camera_obj.rotation_mode = 'QUATERNION'
camera_obj.rotation_quaternion = rot_quat

# Set FOV
camera_data.lens_unit = 'FOV'
camera_data.angle = math.radians(camera_fov_degrees)
print(f"Camera vertical FOV set to {camera_fov_degrees} degrees")

# Set camera as active
bpy.context.scene.camera = camera_obj
print(f"Set '{camera_obj.name}' as active scene camera.")

# Parent camera to bone
print(f"Parenting camera to bone '{parent_bone_name}'...")
for obj in bpy.data.objects:
    obj.select_set(False)
camera_obj.select_set(True)
bpy.context.view_layer.objects.active = armature_obj
bpy.ops.object.mode_set(mode='POSE')
for b in armature_obj.pose.bones:
    b.bone.select = False
parent_pose_bone.bone.select = True
bpy.context.view_layer.objects.active.data.bones.active = parent_pose_bone.bone
bpy.ops.object.parent_set(type='BONE')

# Confirm parenting
if camera_obj.parent == armature_obj and camera_obj.parent_bone == parent_bone_name:
    print(f"Camera successfully parented to '{parent_bone_name}'")
else:
    print("Warning: Camera parenting may have failed.")

# Restore previous state
for obj in bpy.data.objects:
    obj.select_set(False)
for obj in original_selected_objs:
    obj.select_set(True)
if original_active_obj:
    bpy.context.view_layer.objects.active = original_active_obj
if original_mode:
    try:
        bpy.ops.object.mode_set(mode=original_mode)
    except RuntimeError:
        print("Warning: Failed to restore previous mode. Defaulting to OBJECT.")
        bpy.ops.object.mode_set(mode='OBJECT')
else:
    bpy.ops.object.mode_set(mode='OBJECT')

# Get animation sets and determine what to render
animation_sets = get_animation_sets()

if not animation_sets:
    print("No animation sets found! Please load some animations first.")
    exit()

print(f"Found {len(animation_sets)} animation sets:")
for i, name in enumerate(animation_sets.keys()):
    print(f"  {i}: {name}")

# Determine which animations to render
if animation_index is not None:
    # Render specific animation
    if animation_index >= len(animation_sets):
        print(f"Error: Animation index {animation_index} out of range. Available: 0-{len(animation_sets)-1}")
        exit()
    animations_to_render = [(animation_index, list(animation_sets.keys())[animation_index])]
    print(f"Rendering specific animation: {animation_index}")
else:
    # Render all animations
    animations_to_render = [(i, name) for i, name in enumerate(animation_sets.keys())]
    print(f"Rendering all {len(animations_to_render)} animations")

# Main rendering loop with error handling
print("Starting animation rendering loop...")
total_start_time = time.time()
failed_animations = []

for anim_idx, anim_name in animations_to_render:
    print(f"\n{'='*60}")
    print(f"PROCESSING ANIMATION {anim_idx}: {anim_name}")
    print(f"{'='*60}")
    
    try:
        # Apply the animation set
        if not apply_animation_set(anim_idx):
            error_msg = f"Failed to apply animation {anim_idx}"
            log_error("ANIMATION_APPLY_FAILED", error_msg, animation_name=anim_name)
            print(f"Failed to apply animation {anim_idx}. Skipping...")
            failed_animations.append((anim_idx, anim_name, "ANIMATION_APPLY_FAILED"))
            continue
        
        # Render this animation sequence
        render_animation_sequence(anim_idx, anim_name[:-4])
        
        print(f"Completed animation {anim_idx}: {anim_name}")
        
    except Exception as e:
        error_msg = f"Unexpected error during animation {anim_idx}: {str(e)}"
        log_error("RENDERING_ERROR", error_msg, animation_name=anim_name)
        print(f"Error during animation {anim_idx}: {str(e)}")
        failed_animations.append((anim_idx, anim_name, "RENDERING_ERROR"))
        continue

total_time = time.time() - total_start_time
total_frames_rendered = sum(len(range(scene.frame_start, scene.frame_end + 1)) for _, _ in animations_to_render if _ not in [f[0] for f in failed_animations])
overall_fps = total_frames_rendered / total_time if total_time > 0 else 0

print(f"\n{'='*60}")
print("RENDERING COMPLETE!")
print(f"Total time: {total_time/3600:.1f} hours ({total_time/60:.1f} minutes)")
print(f"Total frames rendered: {total_frames_rendered}")
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

print(f"{'='*60}") 