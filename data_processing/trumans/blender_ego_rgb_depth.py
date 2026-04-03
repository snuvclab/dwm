import bpy
import math
import sys
import os
import json
import numpy as np
from mathutils import Vector, Matrix
import argparse

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
args = parser.parse_args(argv)

# --- Output folder naming based on .blend file ---
blend_filename = os.path.splitext(os.path.basename(bpy.data.filepath))[0]
output_folder = f"/home/byungjun/workspace/trumans_ego/ego_render_new/{blend_filename}"

# --- Apply parsed args ---
start_frame = args.start_frame
end_frame = args.end_frame
animation_index = args.animation_index

armature_name = "zzy3"
eye_mesh_name = "CC_Base_Eye"
parent_bone_name = "CC_Base_FacialBone"

camera_fov_degrees = 60.0
cycles_samples = 128
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

def render_animation_sequence(animation_index, animation_name):
    """Render the current animation sequence"""
    # Create animation-specific output directories
    # anim_output_folder = os.path.join(output_folder, f"animation_{animation_index:03d}_{animation_name}")
    anim_output_folder = os.path.join(output_folder, f"{animation_name}")
    rgb_output_path = os.path.join(anim_output_folder, "rgb")
    depth_output_path = os.path.join(anim_output_folder, "depth")
    cam_params_path = os.path.join(anim_output_folder, "cam_params")
    os.makedirs(rgb_output_path, exist_ok=True)
    os.makedirs(depth_output_path, exist_ok=True)
    os.makedirs(cam_params_path, exist_ok=True)
    
    print(f"Rendering animation {animation_index}: {animation_name}")
    print(f"Output paths: rgb: {rgb_output_path}, depth: {depth_output_path}, cam_params: {cam_params_path}")
    
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
        rgb_output_node.base_path = rgb_output_path
    if depth_output_node:
        depth_output_node.base_path = depth_output_path
    
    # Get camera intrinsics (same for all frames in this animation)
    camera_obj = bpy.context.scene.camera
    intrinsics = get_camera_intrinsics(camera_obj)
    intrinsics_path = os.path.join(anim_output_folder, "cam_intrinsics.npy")
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

    print(f"Render range for animation {animation_index}: {render_start_frame} to {render_end_frame}")
    
    # Rendering loop for this animation
    for frame_num in range(render_start_frame, render_end_frame + 1):
        scene.frame_set(frame_num)
        print(f"Rendering animation {animation_index}, frame {frame_num}...")
        
        # Get world-to-camera matrix for this frame
        world_to_camera = get_world_to_camera_matrix(camera_obj)
        
        # Save camera parameters for this frame
        cam_param_filename = f"cam_{frame_num:04d}.npy"
        cam_param_path = os.path.join(cam_params_path, cam_param_filename)
        np.save(cam_param_path, world_to_camera)
        
        # Render RGB and Depth
        if rgb_output_node:
            rgb_output_node.file_slots[0].path = f"rgb_"
        if depth_output_node:
            depth_output_node.file_slots[0].path = f"depth_"
        bpy.ops.render.render(write_still=True)
        
        print(f"  Saved camera params: {cam_param_filename}")
    
    print(f"Completed rendering animation {animation_index}: {animation_name}")

# Create base output directories
rgb_output_path = os.path.join(output_folder, "RGB")
depth_output_path = os.path.join(output_folder, "Depth")
os.makedirs(rgb_output_path, exist_ok=True)
os.makedirs(depth_output_path, exist_ok=True)

print(f"Base render output paths: RGB: {rgb_output_path}, Depth: {depth_output_path}")

# Scene and render setup
scene = bpy.context.scene
render = scene.render
render.engine = 'CYCLES'

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
                print(d.type)
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
rgb_output_node.base_path = rgb_output_path
rgb_output_node.file_slots[0].path = "#"
rgb_output_node.format.file_format = 'PNG'
rgb_output_node.format.color_mode = 'RGBA'
rgb_output_node.location = 400, 200

# Depth output node
depth_output_node = tree.nodes.new(type='CompositorNodeOutputFile')
depth_output_node.label = "Depth Output (EXR)"
depth_output_node.base_path = depth_output_path
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
    print(f"Error: Armature '{armature_name}' not found.")
    exit()
if not eye_mesh_obj:
    print(f"Error: Mesh '{eye_mesh_name}' not found.")
    exit()

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
    print(f"Error: Bone '{parent_bone_name}' not found in armature.")
    for obj in bpy.data.objects: obj.select_set(False)
    for obj in original_selected_objs: obj.select_set(True)
    if original_active_obj: bpy.context.view_layer.objects.active = original_active_obj
    if original_mode: bpy.ops.object.mode_set(mode=original_mode)
    exit()

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

# Main rendering loop
print("Starting animation rendering loop...")
for anim_idx, anim_name in animations_to_render:
    print(f"\n{'='*60}")
    print(f"PROCESSING ANIMATION {anim_idx}: {anim_name}")
    print(f"{'='*60}")
    
    # Apply the animation set
    if not apply_animation_set(anim_idx):
        print(f"Failed to apply animation {anim_idx}. Skipping...")
        continue
    
    # Render this animation sequence
    render_animation_sequence(anim_idx, anim_name[:-4])
    
    print(f"Completed animation {anim_idx}: {anim_name}")

print(f"\n{'='*60}")
print("ALL RENDERING COMPLETE!")
print(f"Results saved in: '{output_folder}'")
print("Each animation has its own subfolder: {animation_name}/")
print(f"{'='*60}")

