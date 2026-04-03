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
USE_POV_CAMERA = False  # Set to False to use "Camera" (third person view)

# Output format: 'VIDEO' or 'IMAGES'
OUTPUT_FORMAT = 'VIDEO'  # Set to 'IMAGES' to save individual PNG frames

# Render hands only with transparent background (only works when USE_POV_CAMERA=True)
# When True, only hands will be rendered with transparent background (for overlay in Keynote)
RENDER_HANDS_ONLY = False  # Set to True to render only hands with transparent background

# Character color (from blender_ego_static_with_agent.py)
COLOR = (0.305, 0.437, 0.8, 1.0)

scene = bpy.context.scene
render = scene.render

# Frame range
start_frame = 0
end_frame = 144
#end_frame = 675
frame_skip = 3  # Render every Nth frame

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
# Find or get POV_Camera object for frustum (USE_POV_CAMERA와 관계없이 항상 사용)
# This will be used later to create frustum AFTER freezing
# ------------------------------------------------------------------
pov_camera_for_frustum = None

# First, try to get existing POV_Camera
if USE_POV_CAMERA:
    # If USE_POV_CAMERA=True, we already have pov_camera_obj
    pov_camera_for_frustum = render_camera_obj  # This is pov_camera_obj when USE_POV_CAMERA=True
    print(f"✓ Will use existing POV_Camera for frustum (from USE_POV_CAMERA=True)")
else:
    # If USE_POV_CAMERA=False, find or create POV_Camera for frustum only
    pov_camera_for_frustum = bpy.data.objects.get("POV_Camera")
    if not pov_camera_for_frustum:
        print("POV_Camera not found. Will create POV_Camera for frustum visualization...")
        eye_mesh_name = "CC_Base_Eye"
        parent_bone_name = "CC_Base_FacialBone"
        
        eye_mesh_obj = bpy.data.objects.get(eye_mesh_name)
        if not eye_mesh_obj:
            print(f"Warning: Mesh '{eye_mesh_name}' not found. Cannot create POV_Camera for frustum.")
            pov_camera_for_frustum = None
        else:
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
            
            # Parent camera to facial bone
            parent_pose_bone = armature_obj.pose.bones.get(parent_bone_name)
            if not parent_pose_bone:
                print(f"Warning: Bone '{parent_bone_name}' not found. Cannot create POV_Camera for frustum.")
                pov_camera_for_frustum = None
            else:
                camera_data = bpy.data.cameras.new(name="POV_Camera")
                pov_camera_for_frustum = bpy.data.objects.new(name="POV_Camera", object_data=camera_data)
                scene.collection.objects.link(pov_camera_for_frustum)
                pov_camera_for_frustum.location = camera_initial_world_location
                
                # look -Y, up Z
                target_forward = Vector((0,-1,0))
                rot_quat = target_forward.to_track_quat('-Z','Y')
                pov_camera_for_frustum.rotation_mode = 'QUATERNION'
                pov_camera_for_frustum.rotation_quaternion = rot_quat
                
                camera_data.lens_unit = 'FOV'
                camera_data.angle = math.radians(camera_fov_degrees)
                
                # Parent to bone
                for obj in bpy.data.objects: obj.select_set(False)
                pov_camera_for_frustum.select_set(True)
                bpy.context.view_layer.objects.active = armature_obj
                bpy.ops.object.mode_set(mode='POSE')
                for b in armature_obj.pose.bones: b.bone.select = False
                parent_pose_bone.bone.select = True
                bpy.context.view_layer.objects.active.data.bones.active = parent_pose_bone.bone
                bpy.ops.object.parent_set(type='BONE')
                bpy.ops.object.mode_set(mode='OBJECT')
                
                print(f"✓ Created POV_Camera for frustum visualization")
    else:
        print(f"✓ Found existing POV_Camera for frustum visualization")
        # Ensure FOV is correct
        if pov_camera_for_frustum.data:
            pov_camera_for_frustum.data.lens_unit = 'FOV'
            pov_camera_for_frustum.data.angle = math.radians(camera_fov_degrees)

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

def disable_animations_except_camera_and_armature(camera_obj, armature_obj, preserve_pov_camera_parent=False):
    """
    Freeze all objects except camera and armature (character).
    Camera needs to follow character, armature is the character itself.
    
    Args:
        preserve_pov_camera_parent: If True, don't clear parent for POV_Camera (to preserve bone parenting for frustum)
    """
    global _original_animation_state
    _original_animation_state = {"camera_parent": None, "actions": {}}

    # --- 1) Camera parenting ---
    # Don't clear POV_Camera parent if preserve_pov_camera_parent is True (for frustum visualization)
    if camera_obj.parent is not None:
        if preserve_pov_camera_parent and camera_obj.name == "POV_Camera":
            print(f"  Preserving POV_Camera parent (for frustum visualization)")
        else:
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

def create_character_material(color_rgba):
    """Create material for character with specified color (from blender_ego_static_with_agent.py)."""
    material_name = "CharacterMaterial_CoolEnd"
    material = bpy.data.materials.get(material_name)
    if material is None:
        material = bpy.data.materials.new(material_name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    
    # Clear existing nodes
    for node in list(nodes):
        nodes.remove(node)
    
    # Create nodes
    output_node = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    links.new(bsdf.outputs["BSDF"], output_node.inputs["Surface"])
    
    # Set color
    base_color = (color_rgba[0], color_rgba[1], color_rgba[2], 1.0)
    bsdf.inputs["Base Color"].default_value = base_color
    bsdf.inputs["Alpha"].default_value = 1.0  # No transparency
    bsdf.inputs["Roughness"].default_value = 0.6
    
    # No blend method (opaque material)
    material.use_backface_culling = False
    return material

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
# IMPORTANT: If pov_camera_for_frustum exists and is different from render_camera_obj,
# Freeze non-character objects
# If USE_POV_CAMERA=False, we need to preserve POV_Camera parent for frustum visualization
# If USE_POV_CAMERA=True, frustum is not created, so we can clear parent normally
if not USE_POV_CAMERA and pov_camera_for_frustum and pov_camera_for_frustum != render_camera_obj:
    # POV_Camera for frustum is separate from render camera - preserve its parent
    print(f"  Note: Preserving POV_Camera parent for frustum visualization (separate from render camera)")
    # Freeze render camera normally
    disable_animations_except_camera_and_armature(render_camera_obj, armature_obj, preserve_pov_camera_parent=False)
    # But don't clear parent for pov_camera_for_frustum - it should stay parented to bone
    # (pov_camera_for_frustum is not passed to disable_animations, so its parent won't be cleared)
elif not USE_POV_CAMERA and render_camera_obj.name == "POV_Camera" and pov_camera_for_frustum and render_camera_obj == pov_camera_for_frustum:
    # If render_camera_obj is POV_Camera and it's also used for frustum, preserve its parent
    print(f"  Note: Preserving POV_Camera parent for frustum visualization")
    disable_animations_except_camera_and_armature(render_camera_obj, armature_obj, preserve_pov_camera_parent=True)
else:
    # For USE_POV_CAMERA=True or other cameras, clear parent normally
    disable_animations_except_camera_and_armature(render_camera_obj, armature_obj, preserve_pov_camera_parent=False)

# Bake camera keys so camera moves during rendering
print(f"\nBaking camera keys for {len(frames_to_render)} frames...")
bake_camera_keys(render_camera_obj, frames_to_render, cam_locs, cam_rots)
print(f"✓ Baked camera keys - camera will move during rendering")

print(f"✓ Frozen non-character objects at frame {start_frame}")
print("  Character (armature) will continue animating during rendering")
print(f"  Scene objects will remain frozen at frame {start_frame} state")

# ------------------------------------------------------------------
# Create camera frustum visualization (AFTER freezing, so it follows POV_Camera correctly)
# Frustum is only created when USE_POV_CAMERA=False (for third-person view)
# When USE_POV_CAMERA=True, frustum is not needed (ego view)
# ------------------------------------------------------------------
if not USE_POV_CAMERA:
    print("\n=== Creating camera frustum (using POV_Camera) ===")
body_mesh_name = "CC_Base_Body"
body_mesh_obj = bpy.data.objects.get(body_mesh_name)

# Hide character (body) from rendering
if body_mesh_obj:
    body_mesh_obj.hide_render = True
    body_mesh_obj.hide_viewport = True
    print(f"✓ Hidden character body '{body_mesh_name}' from rendering")
else:
    print(f"Warning: Body mesh '{body_mesh_name}' not found.")

# Setup hand rendering with colors
print("\n=== Setting up hand rendering ===")
def setup_hand_rendering_with_colors():
    """Setup hand rendering with different colors (from blender_ego_static_with_agent_fig2.py)."""
    if not body_mesh_obj:
        print(f"ERROR: Body mesh not found (body_mesh_obj is None)")
        return False
    
    print(f"  Found body mesh: {body_mesh_obj.name}")
    
    left_hand_groups = [
        "CC_Base_L_Thumb1","CC_Base_L_Thumb2","CC_Base_L_Thumb3",
        "CC_Base_L_Index1","CC_Base_L_Index2","CC_Base_L_Index3",
        "CC_Base_L_Mid1","CC_Base_L_Mid2","CC_Base_L_Mid3",
        "CC_Base_L_Ring1","CC_Base_L_Ring2","CC_Base_L_Ring3",
        "CC_Base_L_Pinky1","CC_Base_L_Pinky2","CC_Base_L_Pinky3",
        "CC_Base_L_Hand"
    ]
    right_hand_groups = [
        "CC_Base_R_Thumb1","CC_Base_R_Thumb2","CC_Base_R_Thumb3",
        "CC_Base_R_Index1","CC_Base_R_Index2","CC_Base_R_Index3",
        "CC_Base_R_Mid1","CC_Base_R_Mid2","CC_Base_R_Mid3",
        "CC_Base_R_Ring1","CC_Base_R_Ring2","CC_Base_R_Ring3",
        "CC_Base_R_Pinky1","CC_Base_R_Pinky2","CC_Base_R_Pinky3",
        "CC_Base_R_Hand"
    ]
    
    # Clean previous hand objects
    for name in ("CC_Hand_L", "CC_Hand_R"):
        if name in bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)
    
    # Duplicate body for hands
    def duplicate_object(src_obj, new_name):
        dup = src_obj.copy()
        dup.data = src_obj.data.copy()
        dup.animation_data_clear()
        dup.name = new_name
        bpy.context.scene.collection.objects.link(dup)
        dup.modifiers.clear()
        for m in src_obj.modifiers:
            if m.type == 'ARMATURE':
                m2 = dup.modifiers.new(m.name, m.type)
                m2.object = m.object
                if hasattr(m2, 'use_deform_preserve_volume'):
                    m2.use_deform_preserve_volume = getattr(m, 'use_deform_preserve_volume', True)
        return dup
    
    print("  Duplicating body for hands...")
    hand_L = duplicate_object(body_mesh_obj, "CC_Hand_L")
    hand_R = duplicate_object(body_mesh_obj, "CC_Hand_R")
    print(f"  Created hand objects: {hand_L.name}, {hand_R.name}")
    
    # Ensure armature modifier is properly set for both hands
    for hand_obj in [hand_L, hand_R]:
        for mod in hand_obj.modifiers:
            if mod.type == 'ARMATURE':
                mod.object = armature_obj
                mod.use_deform_preserve_volume = True
    
    def ensure_vgroup(obj, name):
        vg = obj.vertex_groups.get(name)
        if vg is None:
            vg = obj.vertex_groups.new(name=name)
        return vg
    
    def build_union_via_modifiers(obj, target_name, source_group_names):
        ensure_vgroup(obj, target_name)
        for gname in source_group_names:
            if obj.vertex_groups.get(gname) is None:
                continue
            vwm = obj.modifiers.new(name=f"VWM_{target_name}_ADD_{gname}", type='VERTEX_WEIGHT_MIX')
            vwm.vertex_group_a = target_name
            vwm.vertex_group_b = gname
            vwm.mix_mode = 'ADD'
            vwm.mix_set = 'ALL'
            vwm.mask_constant = 1.0
    
    build_union_via_modifiers(hand_L, "Hand_L_All", left_hand_groups)
    build_union_via_modifiers(hand_R, "Hand_R_All", right_hand_groups)
    
    def add_mask(obj, vgname):
        m = obj.modifiers.new(name=f"Mask_{vgname}", type='MASK')
        m.vertex_group = vgname
        m.invert_vertex_group = False
        m.show_viewport = True
        m.show_render = True
    
    print("  Adding mask modifiers...")
    add_mask(hand_L, "Hand_L_All")
    add_mask(hand_R, "Hand_R_All")
    
    # Update depsgraph to evaluate modifiers
    bpy.context.view_layer.update()
    
    # Create materials for hands (Phong-like, matching blender_ego_hand.py)
    print("  Creating hand materials...")
    def create_phong_material(name, color):
        """Create Phong-like material (from blender_ego_hand.py)."""
        if name in bpy.data.materials:
            mat = bpy.data.materials[name]
            bpy.data.materials.remove(mat, do_unlink=True)
        
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        nt = mat.node_tree
        for n in list(nt.nodes):
            nt.nodes.remove(n)
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.inputs["Base Color"].default_value = (color[0], color[1], color[2], 1.0)
        bsdf.inputs["Metallic"].default_value = 0.0
        bsdf.inputs["Roughness"].default_value = 0.3
        if "Specular" in bsdf.inputs:
            bsdf.inputs["Specular"].default_value = 0.8
        elif "Specular IOR Level" in bsdf.inputs:
            bsdf.inputs["Specular IOR Level"].default_value = 0.8
        if "IOR" in bsdf.inputs:
            bsdf.inputs["IOR"].default_value = 1.45
        nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
        return mat
    
    # Use colors from blender_ego_hand.py
    print("  Creating left hand material (vibrant green)...")
    mat_L = create_phong_material("LeftHandMaterial", (0.5, 0.8, 0.5))  # Vibrant green
    print("  Creating right hand material (deep red)...")
    mat_R = create_phong_material("RightHandMaterial", (0.8, 0.4, 0.4))  # Deep red
    
    print("  Assigning materials to hand objects...")
    for obj, mat in ((hand_L, mat_L), (hand_R, mat_R)):
        obj.data.materials.clear()
        obj.data.materials.append(mat)
        obj.hide_render = False
        obj.hide_viewport = False
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.shade_smooth()
    
    # Ensure objects are visible
    for obj in [hand_L, hand_R]:
        obj.hide_render = False
        obj.hide_viewport = False
        obj.hide_set(False)
        for mod in obj.modifiers:
            mod.show_viewport = True
            mod.show_render = True
        for mod in obj.modifiers:
            if mod.type == 'ARMATURE' and mod.object:
                mod.object = armature_obj
                mod.use_deform_preserve_volume = True
    
    bpy.context.view_layer.update()
    print("✓ Built hand objects: CC_Hand_L (vibrant green) / CC_Hand_R (deep red)")
    return True

try:
    result = setup_hand_rendering_with_colors()
    if result:
        print("✓ Hand rendering setup completed successfully")
    else:
        print("✗ Hand rendering setup failed")
except Exception as e:
    print(f"✗ ERROR in setup_hand_rendering_with_colors: {e}")
    import traceback
    traceback.print_exc()

# ------------------------------------------------------------------
# Hide all objects except hands when RENDER_HANDS_ONLY=True
# ------------------------------------------------------------------
if RENDER_HANDS_ONLY and USE_POV_CAMERA:
    print("\n=== Hiding all objects except hands (RENDER_HANDS_ONLY=True) ===")
    
    # Keep only hands and camera/light visible
    keep_objects = {"CC_Hand_L", "CC_Hand_R"}
    kept_count = 0
    hidden_count = 0
    
    for obj in bpy.data.objects:
        if obj.name in keep_objects:
            # Keep hands visible
            obj.hide_render = False
            obj.hide_viewport = False
            kept_count += 1
            print(f"  Keeping: {obj.name}")
        elif obj.type in {'CAMERA', 'LIGHT'}:
            # Keep camera and lights visible (for rendering)
            obj.hide_render = False
            obj.hide_viewport = False
            kept_count += 1
        else:
            # Hide everything else
            obj.hide_render = True
            obj.hide_viewport = True
            hidden_count += 1
    
    # Set World shader to transparent (remove background)
    print("  Setting World shader to transparent...")
    world = scene.world
    if world and world.use_nodes:
        # Remove all existing nodes
        world.node_tree.nodes.clear()
        # Create a simple transparent background
        output_node = world.node_tree.nodes.new(type='ShaderNodeOutputWorld')
        # No background = transparent
        print("  ✓ World shader set to transparent")
    else:
        # Create world with nodes if it doesn't exist
        if not world:
            world = bpy.data.worlds.new("World")
            scene.world = world
        world.use_nodes = True
        world.node_tree.nodes.clear()
        output_node = world.node_tree.nodes.new(type='ShaderNodeOutputWorld')
        print("  ✓ Created transparent World shader")
    
    print(f"✓ Visibility set: Kept {kept_count} objects (hands/camera/lights), hidden {hidden_count} others")
    print("✓ Transparent background enabled - hands will render with alpha channel")
else:
    print(f"✓ Normal rendering mode (RENDER_HANDS_ONLY={RENDER_HANDS_ONLY}, USE_POV_CAMERA={USE_POV_CAMERA})")

# Find or get POV_Camera object for frustum (only when USE_POV_CAMERA=False)
# This will be used later to create frustum AFTER freezing
pov_camera_for_frustum = None

if not USE_POV_CAMERA:
    # If USE_POV_CAMERA=False, find or create POV_Camera for frustum only
    pov_camera_for_frustum = bpy.data.objects.get("POV_Camera")
    if not pov_camera_for_frustum:
        print("POV_Camera not found. Will create POV_Camera for frustum visualization...")
        eye_mesh_name = "CC_Base_Eye"
        parent_bone_name = "CC_Base_FacialBone"
        
        eye_mesh_obj = bpy.data.objects.get(eye_mesh_name)
        if not eye_mesh_obj:
            print(f"Warning: Mesh '{eye_mesh_name}' not found. Cannot create POV_Camera for frustum.")
            pov_camera_for_frustum = None
        else:
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
            
            # Parent camera to facial bone
            parent_pose_bone = armature_obj.pose.bones.get(parent_bone_name)
            if not parent_pose_bone:
                print(f"Warning: Bone '{parent_bone_name}' not found. Cannot create POV_Camera for frustum.")
                pov_camera_for_frustum = None
            else:
                camera_data = bpy.data.cameras.new(name="POV_Camera")
                pov_camera_for_frustum = bpy.data.objects.new(name="POV_Camera", object_data=camera_data)
                scene.collection.objects.link(pov_camera_for_frustum)
                pov_camera_for_frustum.location = camera_initial_world_location
                
                # look -Y, up Z
                target_forward = Vector((0,-1,0))
                rot_quat = target_forward.to_track_quat('-Z','Y')
                pov_camera_for_frustum.rotation_mode = 'QUATERNION'
                pov_camera_for_frustum.rotation_quaternion = rot_quat
                
                camera_data.lens_unit = 'FOV'
                camera_data.angle = math.radians(camera_fov_degrees)
                
                # Parent to bone
                for obj in bpy.data.objects: obj.select_set(False)
                pov_camera_for_frustum.select_set(True)
                bpy.context.view_layer.objects.active = armature_obj
                bpy.ops.object.mode_set(mode='POSE')
                for b in armature_obj.pose.bones: b.bone.select = False
                parent_pose_bone.bone.select = True
                bpy.context.view_layer.objects.active.data.bones.active = parent_pose_bone.bone
                bpy.ops.object.parent_set(type='BONE')
                bpy.ops.object.mode_set(mode='OBJECT')
                
                print(f"✓ Created POV_Camera for frustum visualization: {pov_camera_for_frustum.name}")
    else:
        print(f"✓ Found existing POV_Camera for frustum visualization: {pov_camera_for_frustum.name}")
        # Ensure FOV is correct
        if pov_camera_for_frustum.data:
            pov_camera_for_frustum.data.lens_unit = 'FOV'
            pov_camera_for_frustum.data.angle = math.radians(camera_fov_degrees)

def create_camera_frustum_mesh(camera_obj, frustum_length=0.5, bevel_depth=0.005):
    """Create a pyramid-shaped camera frustum as a Curve with transparent base plane."""
    cam = camera_obj.data
    scene = bpy.context.scene
    
    # Get camera parameters
    fov_rad = cam.angle
    width = scene.render.resolution_x
    height = scene.render.resolution_y
    aspect = width / height
    
    # Calculate frustum dimensions
    near = 0.01
    far = frustum_length
    
    # Calculate frustum corners at far plane only (pyramid shape)
    far_height = 2.0 * math.tan(fov_rad / 2.0) * far
    far_width = far_height * aspect
    
    # In Blender camera space: -Z is forward, Y is up
    # Use local space coordinates (will be parented to camera)
    apex = Vector((0, 0, 0))
    ftl = Vector((-far_width/2, far_height/2, -far))
    ftr = Vector((far_width/2, far_height/2, -far))
    fbr = Vector((far_width/2, -far_height/2, -far))
    fbl = Vector((-far_width/2, -far_height/2, -far))
    
    # Use local space coordinates (frustum will be parented to camera)
    # These are in camera's local space, so when parented, frustum will follow camera
    apex_w = apex
    ftl_w = ftl
    ftr_w = ftr
    fbr_w = fbr
    fbl_w = fbl
    
    curve_name = "CameraFrustum"
    
    # Clean up existing objects/data
    if curve_name in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[curve_name], do_unlink=True)
    if curve_name in bpy.data.curves:
        bpy.data.curves.remove(bpy.data.curves[curve_name])
    
    # Create Curve data
    curve_data = bpy.data.curves.new(curve_name, type='CURVE')
    curve_data.dimensions = '3D'
    
    def add_polyline(points):
        spline = curve_data.splines.new(type='POLY')
        spline.points.add(len(points) - 1)
        for i, p in enumerate(points):
            spline.points[i].co = (p.x, p.y, p.z, 1.0)
        return spline
    
    # Camera apex → four corners (4 lines)
    add_polyline([apex_w, ftl_w])
    add_polyline([apex_w, ftr_w])
    add_polyline([apex_w, fbr_w])
    add_polyline([apex_w, fbl_w])
    
    # Base square loop
    base_spline = add_polyline([ftl_w, ftr_w, fbr_w, fbl_w])
    base_spline.use_cyclic_u = True
    
    # Line thickness settings
    curve_data.bevel_depth = bevel_depth
    curve_data.bevel_resolution = 2
    curve_data.resolution_u = 8
    
    frustum_obj = bpy.data.objects.new(curve_name, curve_data)
    scene.collection.objects.link(frustum_obj)
    
    # Parent frustum to camera so it follows camera movement
    frustum_obj.parent = camera_obj
    frustum_obj.parent_type = 'OBJECT'
    
    # Material: Emission with fixed color (cyan/blue)
    mat_name = "FrustumMaterial"
    if mat_name in bpy.data.materials:
        mat = bpy.data.materials[mat_name]
    else:
        mat = bpy.data.materials.new(mat_name)
    
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    for n in list(nodes):
        nodes.remove(n)
    
    out_node = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    color_node = nodes.new("ShaderNodeRGB")
    
    # Fixed cyan/blue color for frustum
    color_node.outputs[0].default_value = (0.2, 0.8, 1.0, 1.0)  # Cyan/blue
    emission.inputs["Strength"].default_value = 3.0
    
    links.new(color_node.outputs[0], emission.inputs["Color"])
    links.new(emission.outputs[0], out_node.inputs["Surface"])
    
    # Disable light interaction for frustum (no reflections, no shadows)
    frustum_obj.cycles.is_shadow_catcher = False
    frustum_obj.cycles.is_holdout = False
    
    frustum_obj.data.materials.append(mat)
    frustum_obj.hide_render = False
    frustum_obj.hide_viewport = False
    
    # Create transparent base plane (far plane)
    # NOTE: Base plane size is automatically linked to frustum_length via 'far' variable
    base_plane_name = "CameraFrustumBase"
    # Clean up any existing base plane objects
    for obj in list(bpy.data.objects):
        if obj.name == base_plane_name:
            bpy.data.objects.remove(obj, do_unlink=True)
    if base_plane_name in bpy.data.meshes:
        bpy.data.meshes.remove(bpy.data.meshes[base_plane_name])
    
    # Create mesh for base plane using far plane corners (which are based on frustum_length)
    mesh_data = bpy.data.meshes.new(base_plane_name)
    # Create quad from far plane corners (in local space)
    # These corners (ftl_w, ftr_w, fbr_w, fbl_w) are calculated from 'far' which equals 'frustum_length'
    vertices = [ftl_w, ftr_w, fbr_w, fbl_w]
    faces = [[0, 1, 2, 3]]  # Single quad face
    mesh_data.from_pydata(vertices, [], faces)
    mesh_data.update()
    print(f"  Created base plane with size: far={far:.3f}, width={far_width:.3f}, height={far_height:.3f}")
    
    base_plane_obj = bpy.data.objects.new(base_plane_name, mesh_data)
    scene.collection.objects.link(base_plane_obj)
    
    # Parent base plane to camera so it follows camera movement
    base_plane_obj.parent = camera_obj
    base_plane_obj.parent_type = 'OBJECT'
    
    # Material: Transparent base plane with lighter color (Emission only, no light interaction)
    base_mat_name = "FrustumBaseMaterial"
    if base_mat_name in bpy.data.materials:
        base_mat = bpy.data.materials[base_mat_name]
    else:
        base_mat = bpy.data.materials.new(base_mat_name)
    
    base_mat.use_nodes = True
    base_nodes = base_mat.node_tree.nodes
    base_links = base_mat.node_tree.links
    
    for n in list(base_nodes):
        base_nodes.remove(n)
    
    base_out_node = base_nodes.new("ShaderNodeOutputMaterial")
    base_emission = base_nodes.new("ShaderNodeEmission")
    base_color_node = base_nodes.new("ShaderNodeRGB")
    base_transparent = base_nodes.new("ShaderNodeBsdfTransparent")
    base_mix = base_nodes.new("ShaderNodeMixShader")
    
    # Lighter cyan/blue color for base plane
    base_color = (0.3, 0.85, 1.0, 1.0)  # Slightly lighter cyan/blue
    base_color_node.outputs[0].default_value = base_color
    base_emission.inputs["Strength"].default_value = 2.0
    
    # Transparency: alpha 0.4 (60% opaque, 40% transparent)
    base_alpha = 0.4
    base_mix.inputs["Fac"].default_value = base_alpha
    
    base_links.new(base_color_node.outputs[0], base_emission.inputs["Color"])
    base_links.new(base_emission.outputs[0], base_mix.inputs[1])
    base_links.new(base_transparent.outputs[0], base_mix.inputs[2])
    base_links.new(base_mix.outputs[0], base_out_node.inputs["Surface"])
    
    # Enable transparency
    base_mat.blend_method = 'HASHED'
    base_mat.shadow_method = 'HASHED'
    
    # Prevent plane from being reflected in other objects
    # This makes the plane invisible to reflection rays (won't appear in other objects' reflections)
    if hasattr(base_plane_obj, 'cycles_visibility'):
        base_plane_obj.cycles_visibility.camera = True  # Visible to camera
        base_plane_obj.cycles_visibility.diffuse = False  # Not visible in diffuse reflections
        base_plane_obj.cycles_visibility.glossy = False  # Not visible in glossy reflections
        base_plane_obj.cycles_visibility.transmission = False  # Not visible in transmission
        base_plane_obj.cycles_visibility.scatter = False  # Not visible in volume scatter
    
    base_plane_obj.data.materials.append(base_mat)
    base_plane_obj.hide_render = False
    base_plane_obj.hide_viewport = False
    
    return frustum_obj

# Create frustum using POV_Camera (only when USE_POV_CAMERA=False)
if not USE_POV_CAMERA and pov_camera_for_frustum:
    # Verify we're using POV_Camera, not Camera
    if pov_camera_for_frustum.name != "POV_Camera":
        print(f"⚠️  WARNING: pov_camera_for_frustum is '{pov_camera_for_frustum.name}', not 'POV_Camera'!")
        # Try to get POV_Camera explicitly
        pov_camera_for_frustum = bpy.data.objects.get("POV_Camera")
        if not pov_camera_for_frustum:
            print(f"✗ ERROR: Cannot find POV_Camera object!")
            frustum_obj = None
        else:
            print(f"✓ Using POV_Camera: {pov_camera_for_frustum.name}")
    
    if pov_camera_for_frustum and pov_camera_for_frustum.name == "POV_Camera":
        try:
            print(f"Creating frustum using POV_Camera: {pov_camera_for_frustum.name}")
            frustum_obj = create_camera_frustum_mesh(pov_camera_for_frustum, frustum_length=0.15)
            print(f"✓ Created frustum mesh: {frustum_obj.name} (following POV_Camera: {pov_camera_for_frustum.name})")
            # Verify frustum is parented to POV_Camera
            if frustum_obj.parent:
                print(f"  Frustum parent: {frustum_obj.parent.name} (type: {frustum_obj.parent_type})")
            else:
                print(f"  ⚠️  WARNING: Frustum has no parent!")
        except Exception as e:
            print(f"✗ ERROR creating frustum: {e}")
            import traceback
            traceback.print_exc()
            frustum_obj = None
    else:
        print(f"✗ ERROR: pov_camera_for_frustum is not POV_Camera!")
        frustum_obj = None
elif USE_POV_CAMERA:
    print("✓ Skipping frustum creation (USE_POV_CAMERA=True, ego view)")
    frustum_obj = None
else:
    print("⚠️  Cannot create frustum: POV_Camera not available")
    frustum_obj = None

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
render.resolution_x = 1000
render.resolution_y = 1000
render.resolution_percentage = 100
render.use_border = False
render.use_crop_to_border = False
render.use_motion_blur = False

# Output format settings
if OUTPUT_FORMAT == 'VIDEO':
    # Video output settings
    if RENDER_HANDS_ONLY and USE_POV_CAMERA:
        # Transparent video for overlay (Keynote-compatible)
        render.image_settings.file_format = 'FFMPEG'
        render.ffmpeg.format = 'QUICKTIME'  # QuickTime format for Keynote compatibility
        render.ffmpeg.codec = 'PNG'  # PNG codec supports transparency
        render.ffmpeg.constant_rate_factor = 'MEDIUM'
        render.ffmpeg.ffmpeg_preset = 'REALTIME'
        render.film_transparent = True  # Enable transparent background
        print("✓ Transparent video settings: QuickTime format with PNG codec (for Keynote overlay)")
    else:
        # Regular video (opaque)
        render.image_settings.file_format = 'FFMPEG'
        render.ffmpeg.format = 'MPEG4'
        render.ffmpeg.codec = 'H264'
        render.ffmpeg.constant_rate_factor = 'MEDIUM'
        render.ffmpeg.ffmpeg_preset = 'REALTIME'
        render.film_transparent = False
    render.fps = 10
else:
    # Image output settings
    render.image_settings.file_format = 'PNG'
    if RENDER_HANDS_ONLY and USE_POV_CAMERA:
        render.image_settings.color_mode = 'RGBA'  # RGBA for transparent images
        render.film_transparent = True
    else:
        render.image_settings.color_mode = 'RGB'
        render.film_transparent = False
    render.fps = 10  # Still needed for frame timing

# Output file path with suffix based on camera type and rendering mode
if RENDER_HANDS_ONLY and USE_POV_CAMERA:
    basename = "embodied_actions_hands_only"
elif USE_POV_CAMERA:
    basename = "embodied_actions_ego"
else:
    basename = "embodied_actions_3rd"

if OUTPUT_FORMAT == 'VIDEO':
    output_path = os.path.join(output_dir, basename)
    render.filepath = output_path
    # For transparent video, use .mov extension (QuickTime)
    if RENDER_HANDS_ONLY and USE_POV_CAMERA:
        output_path = output_path + ".mov"
    else:
        output_path = output_path + ".mp4"
else:
    # For images, create a subdirectory
    images_output_dir = os.path.join(output_dir, f"{basename}_images")
    os.makedirs(images_output_dir, exist_ok=True)
    render.filepath = os.path.join(images_output_dir, basename)

# Calculate actual frames to render (already calculated above)
num_frames = len(frames_to_render)

camera_type = "POV_Camera (ego view)" if USE_POV_CAMERA else "Camera (third person view)"
output_type = "VIDEO" if OUTPUT_FORMAT == 'VIDEO' else "IMAGES"
rendering_mode = "HANDS ONLY (transparent)" if (RENDER_HANDS_ONLY and USE_POV_CAMERA) else "NORMAL"
print(f"\n" + "="*60)
print(f"RENDERING EMBODIED ACTIONS FROM {camera_type.upper()} ({output_type})")
if RENDER_HANDS_ONLY and USE_POV_CAMERA:
    print(f"MODE: {rendering_mode}")
print("="*60)
print(f"Frame range: {start_frame}..{end_frame}")
print(f"Frame skip: {frame_skip} (rendering {num_frames} frames: {frames_to_render[0]}, {frames_to_render[1] if len(frames_to_render) > 1 else ''}, ...)")
print(f"Resolution: {render.resolution_x}x{render.resolution_y}")
print(f"FOV: {camera_fov_degrees}°")
print(f"Cycles samples: {cycles_samples}")
print(f"FPS: {render.fps}")
if OUTPUT_FORMAT == 'VIDEO':
    print(f"Output: {output_path}")
    if RENDER_HANDS_ONLY and USE_POV_CAMERA:
        print(f"  Format: QuickTime (.mov) with PNG codec (transparent, Keynote-compatible)")
else:
    print(f"Output directory: {images_output_dir}")
    print(f"Output pattern: {basename}_####.png")
    if RENDER_HANDS_ONLY and USE_POV_CAMERA:
        print(f"  Format: PNG with RGBA (transparent)")
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
            print(f"\n✅ Video rendering complete: {output_path}")
            print(f"   Rendered {num_frames} frames (frame skip: {frame_skip})")
            if RENDER_HANDS_ONLY and USE_POV_CAMERA:
                print(f"   Transparent video ready for Keynote overlay")
    else:
        # Render individual images
        print(f"\nRendering {num_frames} frames as PNG images...")
        frames_completed = 0
        for frame_idx, frame_num in enumerate(frames_to_render):
            if interrupted:
                break
            scene.frame_set(frame_num)
            bpy.context.view_layer.update()
            
            # Update frustum position for each frame (if frustum exists)
            # NOTE: Frustum is parented to POV_Camera, so it should follow automatically
            # No need to recreate it for each frame
            if frustum_obj:
                # Frustum is already parented to pov_camera_for_frustum, so it will follow automatically
                # Just verify it's still parented correctly
                if frustum_obj.parent and frustum_obj.parent.name != "POV_Camera":
                    print(f"⚠️  WARNING: Frustum parent is '{frustum_obj.parent.name}', not 'POV_Camera'!")
                try:
                    # Ensure view layer is updated so frustum follows POV_Camera
                    bpy.context.view_layer.update()
                except Exception as e:
                    print(f"  Warning: Could not update frustum for frame {frame_num}: {e}")
            
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

