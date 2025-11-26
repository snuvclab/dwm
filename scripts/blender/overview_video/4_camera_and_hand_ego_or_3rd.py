#!/usr/bin/env python3
"""
Blender script to render embodied actions from POV Camera view.
- Sets up POV Camera (like blender_ego_static.py)
- Renders frames 571-744 from POV Camera view
- Resolution: 1500x1000 (higher than blender_ego_static.py's 720x480)
- FOV: same as blender_ego_static.py (90 degrees)
"""

import bpy
import math
import os
import signal
import sys
from mathutils import Vector

# ------------------------------------------------------------------
# 설정: 출력 폴더 / 프레임 범위 / 카메라 선택
# ------------------------------------------------------------------
output_dir = "/home/byungjunkim/dwm_teaser/overview_video"
os.makedirs(output_dir, exist_ok=True)

# Camera selection: True = POV_Camera (ego view), False = Camera (third person view)
USE_POV_CAMERA = True  # Set to False to use "Camera" (third person view)

# Output format: 'VIDEO' or 'IMAGES'
OUTPUT_FORMAT = 'VIDEO'  # Set to 'IMAGES' to save individual PNG frames

scene = bpy.context.scene
render = scene.render

# Frame range
start_frame = 571
end_frame = 744
frame_skip = 1  # Render every Nth frame

scene.frame_start = start_frame
scene.frame_end = end_frame

# Set frame step for skipping frames
if hasattr(scene, "frame_step"):
    scene.frame_step = frame_skip
print(f"Frame step: {frame_skip} (rendering every {frame_skip} frames)")

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
# Camera Setup: POV_Camera or Camera (third person)
# ------------------------------------------------------------------
if USE_POV_CAMERA:
    print("\n" + "="*60)
    print("SETTING UP POV CAMERA (EGO VIEW)")
    print("="*60)
else:
    print("\n" + "="*60)
    print("USING CAMERA (THIRD PERSON VIEW)")
    print("="*60)

camera_fov_degrees = 90.0  # Same FOV as blender_ego_static.py

# Select camera based on USE_POV_CAMERA setting
if USE_POV_CAMERA:
    # POV Camera setup (ego view)
    eye_mesh_name = "CC_Base_Eye"
    parent_bone_name = "CC_Base_FacialBone"
    
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

    # Set POV_Camera as render camera
    render_camera_obj = pov_camera_obj
    scene.camera = render_camera_obj
    print(f"✓ Set POV_Camera as render camera")
else:
    # Use existing "Camera" (third person view)
    render_camera_obj = bpy.data.objects.get("Camera")
    if not render_camera_obj:
        print("Error: 'Camera' object not found!")
        sys.exit(1)
    
    # Set Camera FOV to 90 degrees
    if render_camera_obj.data:
        render_camera_obj.data.lens_unit = 'FOV'
        render_camera_obj.data.angle = math.radians(camera_fov_degrees)
        print(f"✓ Set Camera FOV to {camera_fov_degrees}°")
    
    scene.camera = render_camera_obj
    print(f"✓ Set 'Camera' as render camera (third person view)")

# ------------------------------------------------------------------
# Move to start frame and freeze non-character objects (like blender_ego_static_with_agent.py)
# ------------------------------------------------------------------
print(f"\nMoving to frame {start_frame} and freezing non-character objects...")

# Set to start frame - this will apply the animation at that frame
scene.frame_set(start_frame)
bpy.context.view_layer.update()

# Freeze non-character objects at start frame (like blender_ego_static_with_agent.py)
print(f"Freezing non-character objects at frame {start_frame}...")

# Save original animation state
_original_animation_state = {
    "camera_parent": None,   # (parent, parent_type, parent_bone, matrix_world_before)
    "actions": {},           # obj_name -> action (or None)
}

def disable_animations_except_camera_and_armature(camera_obj, armature_obj):
    """
    Freeze all objects except camera and armature (character).
    Camera needs to follow character, armature is the character itself.
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

    # --- 2) Detach actions of all objects except camera and armature ---
    for obj in bpy.data.objects:
        if obj == camera_obj or obj == armature_obj:
            continue  # Keep camera and armature animations
        if obj.animation_data:
            # Record current action
            _original_animation_state["actions"][obj.name] = obj.animation_data.action
            # Keep the action from being purged
            if obj.animation_data.action:
                obj.animation_data.action.use_fake_user = True
            # Detach to freeze at current pose
            obj.animation_data.action = None

    # Depsgraph refresh helps ensure "frozen" state is used during rendering
    bpy.context.view_layer.update()

def restore_animations(camera_obj):
    """
    Restore every object's action and camera parenting that we saved.
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
        # Keep the same world transform
        camera_obj.matrix_world = mw_before

    # Clear for next use
    _original_animation_state = {"camera_parent": None, "actions": {}}
    bpy.context.view_layer.update()

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
            bpy.context.view_layer.update()
            mw = camera_obj.matrix_world.copy()
            locs.append(mw.to_translation())
            rots.append(mw.to_quaternion())
    finally:
        scene.frame_set(prev)
    return locs, rots

def bake_camera_keys(camera_obj, frames, locs, rots):
    """Bake camera transform keys based on frames/locs/rots."""
    # Prepare action
    if not camera_obj.animation_data:
        camera_obj.animation_data_create()
    if not camera_obj.animation_data.action:
        camera_obj.animation_data.action = bpy.data.actions.new(name="Camera_Baked")
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

    # Set interpolation to Linear
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

# Calculate frames to render
frames_to_render = list(range(start_frame, end_frame + 1, frame_skip))

# Sample camera poses BEFORE freezing (while camera is still parented to bone)
# Only sample if using POV_Camera (it's parented to bone and follows character)
if USE_POV_CAMERA:
    print(f"\nSampling camera poses for {len(frames_to_render)} frames...")
    cam_locs, cam_rots = sample_camera_world_transforms(render_camera_obj, frames_to_render)
    print(f"✓ Sampled camera poses for frames: {frames_to_render[0]}, {frames_to_render[1] if len(frames_to_render) > 1 else ''}, ...")
else:
    # For third person Camera, sample its existing animation
    print(f"\nSampling Camera (third person) poses for {len(frames_to_render)} frames...")
    cam_locs, cam_rots = sample_camera_world_transforms(render_camera_obj, frames_to_render)
    print(f"✓ Sampled Camera poses for frames: {frames_to_render[0]}, {frames_to_render[1] if len(frames_to_render) > 1 else ''}, ...")

# Freeze non-character objects (camera and armature excluded)
# Note: render_camera_obj is either POV_Camera or Camera depending on USE_POV_CAMERA
# This will clear camera parent, so we need to bake camera keys after this
disable_animations_except_camera_and_armature(render_camera_obj, armature_obj)

# Bake camera keys so camera moves during rendering
print(f"\nBaking camera keys for {len(frames_to_render)} frames...")
bake_camera_keys(render_camera_obj, frames_to_render, cam_locs, cam_rots)
print(f"✓ Baked camera keys - camera will move during rendering")

print(f"✓ Frozen non-character objects at frame {start_frame}")
print("  Character (armature) will continue animating during rendering")
print(f"  Scene objects will remain frozen at frame {start_frame} state")

# ------------------------------------------------------------------
# Cycles settings (from blender_ego_static.py)
# ------------------------------------------------------------------
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

# Resolution settings (higher than blender_ego_static.py)
render.resolution_x = 1500
render.resolution_y = 1000
render.resolution_percentage = 100
render.use_border = False
render.use_crop_to_border = False
render.use_motion_blur = False

# Output format settings
if OUTPUT_FORMAT == 'VIDEO':
    # Video output settings
    render.image_settings.file_format = 'FFMPEG'
    render.ffmpeg.format = 'MPEG4'
    render.ffmpeg.codec = 'H264'
    render.ffmpeg.constant_rate_factor = 'MEDIUM'
    render.ffmpeg.ffmpeg_preset = 'REALTIME'
    render.fps = 30
else:
    # Image output settings
    render.image_settings.file_format = 'PNG'
    render.image_settings.color_mode = 'RGB'
    render.fps = 30  # Still needed for frame timing

# Output file path with suffix based on camera type
if USE_POV_CAMERA:
    basename = "embodied_actions_ego"
else:
    basename = "embodied_actions_3rd"

if OUTPUT_FORMAT == 'VIDEO':
    output_path = os.path.join(output_dir, basename)
    render.filepath = output_path
else:
    # For images, create a subdirectory
    images_output_dir = os.path.join(output_dir, f"{basename}_images")
    os.makedirs(images_output_dir, exist_ok=True)
    render.filepath = os.path.join(images_output_dir, basename)

# Calculate actual frames to render (already calculated above)
num_frames = len(frames_to_render)

camera_type = "POV_Camera (ego view)" if USE_POV_CAMERA else "Camera (third person view)"
output_type = "VIDEO" if OUTPUT_FORMAT == 'VIDEO' else "IMAGES"
print(f"\n" + "="*60)
print(f"RENDERING EMBODIED ACTIONS FROM {camera_type.upper()} ({output_type})")
print("="*60)
print(f"Frame range: {start_frame}..{end_frame}")
print(f"Frame skip: {frame_skip} (rendering {num_frames} frames: {frames_to_render[0]}, {frames_to_render[1] if len(frames_to_render) > 1 else ''}, ...)")
print(f"Resolution: {render.resolution_x}x{render.resolution_y}")
print(f"FOV: {camera_fov_degrees}°")
print(f"Cycles samples: {cycles_samples}")
print(f"FPS: {render.fps}")
if OUTPUT_FORMAT == 'VIDEO':
    print(f"Output: {output_path}.mp4")
else:
    print(f"Output directory: {images_output_dir}")
    print(f"Output pattern: {basename}_####.png")
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
    # Render based on output format
    if OUTPUT_FORMAT == 'VIDEO':
        # Render video
        bpy.ops.render.render(animation=True)
        
        if not interrupted:
            print(f"\n✅ Video rendering complete: {output_path}.mp4")
            print(f"   Rendered {num_frames} frames (frame skip: {frame_skip})")
    else:
        # Render individual images
        print(f"\nRendering {num_frames} frames as PNG images...")
        frames_completed = 0
        for frame_idx, frame_num in enumerate(frames_to_render):
            if interrupted:
                break
            scene.frame_set(frame_num)
            bpy.context.view_layer.update()
            
            # Update filepath for each frame (Blender will append frame number)
            frame_filepath = os.path.join(images_output_dir, f"{basename}_{frame_num:04d}")
            render.filepath = frame_filepath
            
            print(f"Rendering frame {frame_num} ({frame_idx + 1}/{num_frames})...")
            bpy.ops.render.render(write_still=True)
            frames_completed += 1
            
            if (frame_idx + 1) % 10 == 0 or frame_idx == len(frames_to_render) - 1:
                progress = (frame_idx + 1) / len(frames_to_render) * 100.0
                print(f"  Progress: {frame_idx + 1}/{num_frames} ({progress:.1f}%)")
        
        if not interrupted:
            print(f"\n✅ Image rendering complete: {frames_completed} frames saved")
            print(f"   Output directory: {images_output_dir}")
except KeyboardInterrupt:
    print("\n\n⚠️  INTERRUPTED: Rendering stopped by user")
    if OUTPUT_FORMAT == 'VIDEO':
        print("   Partial video may be saved. Check output directory.")
    else:
        print("   Partial images may be saved. Check output directory.")
except Exception as e:
    print(f"\n❌ ERROR during rendering: {e}")
    import traceback
    traceback.print_exc()
    raise
finally:
    # Cleanup camera keys
    print("\nCleaning up camera keys...")
    clear_camera_keys(render_camera_obj)
    
    # Restore animations
    print("Restoring animations...")
    restore_animations(render_camera_obj)
    print("✓ Restored animations")

