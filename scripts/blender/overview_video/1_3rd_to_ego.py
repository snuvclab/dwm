#!/usr/bin/env python3
"""
Blender script to create a transition video from "Camera" view to POV Camera view.
- Sets up POV Camera (like blender_ego_static.py)
- Interpolates "Camera" transform from its position to POV Camera position over 60 frames
- Starts at frame 571
- Renders video with "Camera" as render camera
"""

import bpy
import math
import os
import signal
import sys
from mathutils import Vector

# ------------------------------------------------------------------
# 설정: 출력 폴더 / 프레임 범위
# ------------------------------------------------------------------
output_dir = "/home/byungjunkim/dwm_teaser/overview_video"
os.makedirs(output_dir, exist_ok=True)

scene = bpy.context.scene
render = scene.render

# Frame range for transition
start_frame = 571
transition_frames = 60
end_frame = start_frame + transition_frames - 1

scene.frame_start = start_frame
scene.frame_end = end_frame

# Ensure frame step is 1
if hasattr(scene, "frame_step"):
    scene.frame_step = 1
print(f"Frame step: {getattr(scene, 'frame_step', 1)}")

# ------------------------------------------------------------------
# Reset pose at frame 0 (from blender_ego_static.py)
# ------------------------------------------------------------------
print("\n" + "="*60)
print("RESETTING POSE AT FRAME 0")
print("="*60)

# HSI/armature discovery
try:
    armature_name = bpy.context.scene.hsi_properties.name_armature_CC
    if not armature_name:
        for obj in bpy.data.objects:
            if obj.type == 'ARMATURE' and 'CC_Base_Hip' in obj.pose.bones:
                armature_name = obj.name
                break
        if not armature_name:
            print("Error: No armature found. Please import a CC4 character first.")
            sys.exit(1)
except Exception as e:
    print(f"Error: Could not access HSI addon properties: {e}")
    sys.exit(1)

armature_obj = bpy.data.objects.get(armature_name)
if not armature_obj:
    print(f"Error: Armature '{armature_name}' not found")
    sys.exit(1)

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
bpy.ops.object.mode_set(mode='OBJECT')
print("✓ Reset pose to zero at frame 0")

# ------------------------------------------------------------------
# POV Camera Setup (from blender_ego_static.py)
# ------------------------------------------------------------------
print("\n" + "="*60)
print("SETTING UP POV CAMERA")
print("="*60)

eye_mesh_name = "CC_Base_Eye"
parent_bone_name = "CC_Base_FacialBone"
camera_fov_degrees = 90.0

eye_mesh_obj = bpy.data.objects.get(eye_mesh_name)
if not eye_mesh_obj:
    print(f"Error: Mesh '{eye_mesh_name}' not found")
    sys.exit(1)

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
    print("Error: Failed to determine camera position.")
    sys.exit(1)

# Parent camera to facial bone
parent_pose_bone = armature_obj.pose.bones.get(parent_bone_name)
if not parent_pose_bone:
    print(f"Error: Bone '{parent_bone_name}' not found in armature '{armature_name}'")
    sys.exit(1)

# Create POV Camera if it doesn't exist
pov_camera_obj = bpy.data.objects.get("POV_Camera")
if not pov_camera_obj:
    print("Creating POV Camera...")
    camera_data = bpy.data.cameras.new(name="POV_Camera")
    pov_camera_obj = bpy.data.objects.new(name="POV_Camera", object_data=camera_data)
    scene.collection.objects.link(pov_camera_obj)
    pov_camera_obj.location = camera_initial_world_location
    
    # look -Y, up Z
    target_forward = Vector((0,-1,0))
    rot_quat = target_forward.to_track_quat('-Z','Y')
    pov_camera_obj.rotation_mode = 'QUATERNION'
    pov_camera_obj.rotation_quaternion = rot_quat
    
    camera_data.lens_unit = 'FOV'
    camera_data.angle = math.radians(camera_fov_degrees)
    
    # Parent to bone
    for obj in bpy.data.objects: obj.select_set(False)
    pov_camera_obj.select_set(True)
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='POSE')
    for b in armature_obj.pose.bones: b.bone.select = False
    parent_pose_bone.bone.select = True
    bpy.context.view_layer.objects.active.data.bones.active = parent_pose_bone.bone
    bpy.ops.object.parent_set(type='BONE')
    bpy.ops.object.mode_set(mode='OBJECT')
    
    print(f"✓ Created and parented POV_Camera to {parent_bone_name}")
else:
    print("✓ POV_Camera already exists - will use existing camera")
    # Ensure FOV is correct
    if pov_camera_obj.data:
        pov_camera_obj.data.lens_unit = 'FOV'
        pov_camera_obj.data.angle = math.radians(camera_fov_degrees)
        print(f"✓ Set POV_Camera FOV to {camera_fov_degrees}°")
    
    # Check if POV_Camera is parented to the correct bone
    if pov_camera_obj.parent and pov_camera_obj.parent_type == 'BONE':
        if pov_camera_obj.parent_bone == parent_bone_name:
            print(f"✓ POV_Camera is already parented to {parent_bone_name}")
        else:
            print(f"⚠️  POV_Camera is parented to {pov_camera_obj.parent_bone}, not {parent_bone_name}")
            print("   Will use existing parent - camera should follow bone correctly")
    elif pov_camera_obj.parent:
        print(f"⚠️  POV_Camera is parented to {pov_camera_obj.parent.name} (not a bone)")
        print("   Camera may not follow character correctly")
    else:
        print("⚠️  POV_Camera is not parented - will parent to bone")
        # Parent to bone
        for obj in bpy.data.objects: obj.select_set(False)
        pov_camera_obj.select_set(True)
        bpy.context.view_layer.objects.active = armature_obj
        bpy.ops.object.mode_set(mode='POSE')
        for b in armature_obj.pose.bones: b.bone.select = False
        parent_pose_bone.bone.select = True
        bpy.context.view_layer.objects.active.data.bones.active = parent_pose_bone.bone
        bpy.ops.object.parent_set(type='BONE')
        bpy.ops.object.mode_set(mode='OBJECT')
        print(f"✓ Parented POV_Camera to {parent_bone_name}")

# ------------------------------------------------------------------
# Move to frame 571 and capture camera transforms (BEFORE freezing)
# ------------------------------------------------------------------
print(f"\nMoving to frame {start_frame} and capturing camera transforms...")

# Find "Camera" object
render_camera_obj = bpy.data.objects.get("Camera")
if not render_camera_obj:
    print("Error: 'Camera' object not found!")
    sys.exit(1)

# Set to start frame (571) - this will apply the animation at that frame
scene.frame_set(start_frame)
bpy.context.view_layer.update()

# Capture "Camera" transform (start position) - at frame 571 with character animated
camera_start_matrix = render_camera_obj.matrix_world.copy()
camera_start_loc = camera_start_matrix.to_translation()
camera_start_rot = camera_start_matrix.to_quaternion()

# Capture POV_Camera transform (end position) - at frame 571 with character animated
# POV_Camera should be parented to bone, so its world matrix will be correct at frame 571
pov_camera_end_matrix = pov_camera_obj.matrix_world.copy()
pov_camera_end_loc = pov_camera_end_matrix.to_translation()
pov_camera_end_rot = pov_camera_end_matrix.to_quaternion()

print(f"  Camera start location: {camera_start_loc}")
print(f"  Camera start rotation: {camera_start_rot}")
print(f"  POV_Camera end location: {pov_camera_end_loc}")
print(f"  POV_Camera end rotation: {pov_camera_end_rot}")

# ------------------------------------------------------------------
# Freeze character animation (except cameras) - AFTER capturing transforms
# ------------------------------------------------------------------
print(f"\nFreezing character animation at frame {start_frame}...")

# Save original animation state
_original_animation_state = {
    "actions": {},  # obj_name -> action (or None)
}

# Detach actions of all objects except cameras
for obj in bpy.data.objects:
    if obj.type == 'CAMERA':
        continue  # Keep camera animations
    if obj.animation_data:
        # Record current action
        _original_animation_state["actions"][obj.name] = obj.animation_data.action
        # Keep the action from being purged
        if obj.animation_data.action:
            obj.animation_data.action.use_fake_user = True
        # Detach to freeze at current pose
        obj.animation_data.action = None

# Update view layer to ensure frozen state
bpy.context.view_layer.update()

print("✓ Frozen all character animations (cameras excluded)")
print("  Character will remain static during camera transition")

# ------------------------------------------------------------------
# Create interpolation keyframes for "Camera"
# ------------------------------------------------------------------
print(f"\nCreating camera transition keyframes ({start_frame}..{end_frame})...")

# Clear existing animation on "Camera"
if render_camera_obj.animation_data:
    render_camera_obj.animation_data_clear()

render_camera_obj.animation_data_create()
action = bpy.data.actions.new(name="Camera_Transition")
render_camera_obj.animation_data.action = action

# Create fcurves
loc_curves = [action.fcurves.new(data_path="location", index=i) for i in range(3)]
rot_curves = [action.fcurves.new(data_path="rotation_quaternion", index=i) for i in range(4)]

# Set rotation mode to quaternion
render_camera_obj.rotation_mode = 'QUATERNION'

# Create keyframes for each frame in transition
for frame_offset in range(transition_frames):
    frame_num = start_frame + frame_offset
    t = frame_offset / (transition_frames - 1)  # 0.0 to 1.0
    
    # Interpolate location (linear)
    interp_loc = camera_start_loc.lerp(pov_camera_end_loc, t)
    
    # Interpolate rotation with easing (ease-in: slow at start, fast at end)
    # Use cubic ease-in: t^3 for smoother acceleration
    t_eased_rot = t ** 3  # Cubic ease-in: starts slow, accelerates at end
    
    # Interpolate rotation (quaternion slerp with eased t)
    interp_rot = camera_start_rot.slerp(pov_camera_end_rot, t_eased_rot)
    
    # Set values
    render_camera_obj.location = interp_loc
    render_camera_obj.rotation_quaternion = interp_rot
    
    # Insert keyframes
    for i, c in enumerate(loc_curves):
        c.keyframe_points.insert(frame=frame_num, value=render_camera_obj.location[i], options={'FAST'})
    for i, c in enumerate(rot_curves):
        c.keyframe_points.insert(frame=frame_num, value=render_camera_obj.rotation_quaternion[i], options={'FAST'})

# Set interpolation to smooth (BEZIER) for smooth transition
for fc in action.fcurves:
    for kp in fc.keyframe_points:
        kp.interpolation = 'BEZIER'
        # Make handles auto for smooth curve
        kp.handle_left_type = 'AUTO'
        kp.handle_right_type = 'AUTO'

print(f"✓ Created {transition_frames} keyframes for camera transition")

# Set "Camera" as render camera
scene.camera = render_camera_obj
print(f"✓ Set 'Camera' as render camera")

# Set Camera FOV to 90 degrees
if render_camera_obj.data:
    render_camera_obj.data.lens_unit = 'FOV'
    render_camera_obj.data.angle = math.radians(90.0)
    print(f"✓ Set Camera FOV to 90°")

# ------------------------------------------------------------------
# Render settings: Resolution, FOV, Cycles samples
# ------------------------------------------------------------------
# Resolution: 1500x1000
render.resolution_x = 1500
render.resolution_y = 1000
render.resolution_percentage = 100
render.use_border = False
render.use_crop_to_border = False
render.use_motion_blur = False

# Cycles settings: 128 samples
cycles_samples = 128
scene.cycles.samples = cycles_samples
scene.cycles.device = 'GPU'

# Persistent data caching
if hasattr(scene.render, "use_persistent_data"):
    scene.render.use_persistent_data = True

# Adaptive sampling
if hasattr(scene.cycles, "use_adaptive_sampling"):
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.05
if hasattr(scene.cycles, "tile_size"):
    scene.cycles.tile_size = 256

# Denoising
scene.cycles.use_denoising = True
if hasattr(scene.cycles, "denoiser"):
    keys = scene.cycles.bl_rna.properties['denoiser'].enum_items.keys()
    if "OPTIX" in keys:
        scene.cycles.denoiser = "OPTIX"
        scene.cycles.use_preview_denoising = True
    elif "OPENIMAGEDENOISE" in keys:
        scene.cycles.denoiser = "OPENIMAGEDENOISE"
scene.cycles.preview_denoising = True

# Video output settings
render.image_settings.file_format = 'FFMPEG'
render.ffmpeg.format = 'MPEG4'
render.ffmpeg.codec = 'H264'
render.ffmpeg.constant_rate_factor = 'MEDIUM'
render.ffmpeg.ffmpeg_preset = 'REALTIME'
render.fps = 30  # 30fps video

# Output file path
video_output_path = os.path.join(output_dir, "3rd_to_ego")
render.filepath = video_output_path

print(f"\n" + "="*60)
print(f"RENDERING CAMERA TRANSITION VIDEO")
print("="*60)
print(f"Resolution: {render.resolution_x}x{render.resolution_y}")
print(f"FOV: 90°")
print(f"Cycles samples: {cycles_samples}")
print(f"Frame range: {start_frame}..{end_frame} ({transition_frames} frames)")
print(f"FPS: {render.fps}")
print(f"Output: {video_output_path}.mp4")
print("\n⚠️  To stop rendering: Press Ctrl+C in the terminal")
print("   Or close Blender window to abort\n")

# Signal handler for graceful shutdown
interrupted = False

def signal_handler(sig, frame):
    global interrupted
    interrupted = True
    print("\n\n⚠️  INTERRUPTED: Rendering stopped by user (Ctrl+C)")
    print("   Partial video may be saved. Check output directory.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

try:
    # Render video
    bpy.ops.render.render(animation=True)
    
    if not interrupted:
        print(f"\n✅ Video rendering complete: {video_output_path}.mp4")
        print(f"   Transition from 'Camera' view to POV Camera view over {transition_frames} frames")
except KeyboardInterrupt:
    print("\n\n⚠️  INTERRUPTED: Rendering stopped by user")
    print("   Partial video may be saved. Check output directory.")
except Exception as e:
    print(f"\n❌ ERROR during rendering: {e}")
    import traceback
    traceback.print_exc()
    raise
finally:
    # Restore character animations
    print("\nRestoring character animations...")
    for obj_name, action in _original_animation_state.get("actions", {}).items():
        obj = bpy.data.objects.get(obj_name)
        if not obj:
            continue
        if not obj.animation_data:
            obj.animation_data_create()
        obj.animation_data.action = action
    print("✓ Restored character animations")

