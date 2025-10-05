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
import torch
import cv2
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PointLights,
    DirectionalLights,
    PerspectiveCameras,
    Materials,
    SoftPhongShader,
    RasterizationSettings,
    MeshRenderer,
    MeshRendererWithFragments,
    MeshRasterizer,
    TexturesVertex
)

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
parser.add_argument("--grayscale", action="store_true", help="Render hands in grayscale (black & white)")
parser.add_argument("--separate", action="store_true", help="Create separate videos for left and right hands (only works with --grayscale)")
parser.add_argument("--save-images", action="store_true", help="Save individual images instead of creating videos")
args = parser.parse_args(argv)
if args.no_skip_existing:
    args.skip_existing = False

print(f"skip_existing: {args.skip_existing}")

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
    """Check if video already exists and has non-zero size."""
    def file_ok(path): return os.path.isfile(path) and os.path.getsize(path) > 0
    video_path = os.path.join(videos_output_path, f"{video_idx:05d}.mp4")
    video_exists = file_ok(video_path)
    needs_rendering = not video_exists
    return video_exists, needs_rendering

def check_frame_exists(frame_num, images_output_path):
    def file_ok(path): return os.path.isfile(path) and os.path.getsize(path) > 0
    image_path = os.path.join(images_output_path, f"{frame_num:04d}.png")
    image_exists = file_ok(image_path)
    needs_rendering = not image_exists
    return image_exists, needs_rendering

# -------------------------------
# Basic config
# -------------------------------
BODY_NAME = "CC_Base_Body"

# Hand vertex groups (left and right hands)
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

# -------------------------------
# Cycles fast setup
# -------------------------------
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
# Scene setup (minimal for PyTorch3D)
# ---------------------------
scene = bpy.context.scene

# Show AugmentAreaCollection if it exists
if "AugmentAreaCollection" in bpy.data.collections:
    bpy.data.collections["AugmentAreaCollection"].hide_render = True
    print("✓ Hide AugmentAreaCollection for rendering")
else:
    print("ℹ️  AugmentAreaCollection not found - skipping")

# Check PyTorch3D availability
try:
    import torch
    import pytorch3d
    print(f"✓ PyTorch3D available: {pytorch3d.__version__}")
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ CUDA device: {torch.cuda.get_device_name()}")
except ImportError as e:
    print(f"✗ PyTorch3D not available: {e}")
    print("Please install PyTorch3D: pip install pytorch3d")
    sys.exit(1)

# ---------------------------
# PyTorch3D setup (no compositor nodes needed)
# ---------------------------
print("✓ PyTorch3D rendering setup complete")

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

# -------------------------------
# Materials (Phong-like with Principled)
# -------------------------------
def create_phong_material(name, color):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes): nt.nodes.remove(n)
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

# -------------------------------
# Hand-only rendering setup
# -------------------------------
def setup_simple_hand_rendering():
    """
    Build hand-only duplicates using only modifiers (no Python vertex loops):
      - Duplicate CC_Base_Body -> CC_Hand_L / CC_Hand_R
      - Chain Vertex Weight Mix modifiers to OR all hand vgroups into Hand_*_All
      - Mask modifier keeps only Hand_*_All
      - Assign solid materials; hide original body
    """
    body = bpy.data.objects.get(BODY_NAME)
    if not body:
        print(f"Error: {BODY_NAME} not found")
        return False

    # Exact group names (case-sensitive)
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

    # Clean any previous hand objects
    for name in ("CC_Hand_L", "CC_Hand_R"):
        if name in bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)

    # Duplicate body -> hands
    def duplicate_object(src_obj, new_name):
        dup = src_obj.copy()
        dup.data = src_obj.data.copy()
        dup.animation_data_clear()
        dup.name = new_name
        bpy.context.scene.collection.objects.link(dup)
        # Keep only Armature modifier (preserve deformation)
        dup.modifiers.clear()
        for m in src_obj.modifiers:
            if m.type == 'ARMATURE':
                m2 = dup.modifiers.new(m.name, m.type)
                m2.object = m.object
                if hasattr(m2, 'use_deform_preserve_volume'):
                    m2.use_deform_preserve_volume = getattr(m, 'use_deform_preserve_volume', True)
        return dup

    hand_L = duplicate_object(body, "CC_Hand_L")
    hand_R = duplicate_object(body, "CC_Hand_R")

    # Helper: add (or get) a vertex group by name
    def ensure_vgroup(obj, name):
        vg = obj.vertex_groups.get(name)
        if vg is None:
            vg = obj.vertex_groups.new(name=name)
        return vg

    # Helper: chain Vertex Weight Mix modifiers to OR groups into target
    def build_union_via_modifiers(obj, target_name, source_group_names):
        ensure_vgroup(obj, target_name)  # target group must exist
        for gname in source_group_names:
            if obj.vertex_groups.get(gname) is None:
                continue
            vwm = obj.modifiers.new(name=f"VWM_{target_name}_ADD_{gname}", type='VERTEX_WEIGHT_MIX')
            vwm.vertex_group_a = target_name
            vwm.vertex_group_b = gname
            vwm.mix_mode = 'ADD'           # add weights
            vwm.mix_set = 'ALL'            # apply to all vertices
            vwm.mask_constant = 1.0        # full strength
            # No need to move order; we create in correct sequence already

    # Build unions entirely in modifier stack (fast, no Python loops)
    build_union_via_modifiers(hand_L, "Hand_L_All", left_hand_groups)
    build_union_via_modifiers(hand_R, "Hand_R_All", right_hand_groups)

    # Mask: keep only the union group
    def add_mask(obj, vgname):
        m = obj.modifiers.new(name=f"Mask_{vgname}", type='MASK')
        m.vertex_group = vgname
        m.invert_vertex_group = False
        m.show_viewport = True
        m.show_render = True

    add_mask(hand_L, "Hand_L_All")
    add_mask(hand_R, "Hand_R_All")

    # Materials (Phong-like)
    mat_L = create_phong_material("LeftHandMaterial",  (0.0, 1.0, 0.0))
    mat_R = create_phong_material("RightHandMaterial", (1.0, 0.0, 0.0))
    for obj, mat in ((hand_L, mat_L), (hand_R, mat_R)):
        obj.data.materials.clear()
        obj.data.materials.append(mat)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.shade_smooth()

    # Hide original body completely
    body.hide_viewport = True
    body.hide_render = True

    print("✓ Built hand-only objects: CC_Hand_L / CC_Hand_R via modifiers (no Python loops).")
    return True
# -------------------------------
# Hide non-hand objects
# -------------------------------
def hide_non_hand_objects():
    keep = {"CC_Hand_L", "CC_Hand_R"}
    kept = 0; hidden = 0
    for obj in bpy.data.objects:
        if obj.name in keep or obj.type in {'CAMERA','LIGHT'}:
            obj.hide_viewport = False
            obj.hide_render = False
            kept += 1
        else:
            obj.hide_viewport = True
            obj.hide_render = True
            hidden += 1
    print(f"✓ Visibility set. Kept {kept} (hands/cam/lights), hidden {hidden} others.")

# -------------------------------
# PyTorch3D rendering functions
# -------------------------------
def get_hand_mesh_data(hand_obj):
    """
    Extract world-space vertices & triangulated faces for PyTorch3D *after* modifiers.
    We evaluate the depsgraph so that Mask/Armature modifiers are applied.
    """
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = hand_obj.evaluated_get(depsgraph)
    eval_mesh = eval_obj.to_mesh(preserve_all_data_layers=False, depsgraph=depsgraph)

    try:
        # World transform (object space -> world space)
        M = hand_obj.matrix_world

        verts = []
        for v in eval_mesh.vertices:
            w = M @ v.co
            verts.append([w.x, w.y, w.z])

        # Ensure triangles (triangulate polygons on the fly)
        faces = []
        for poly in eval_mesh.polygons:
            if len(poly.vertices) == 3:
                i0, i1, i2 = poly.vertices
                faces.append([i0, i1, i2])
            elif len(poly.vertices) == 4:
                i0, i1, i2, i3 = poly.vertices
                faces.append([i0, i1, i2])
                faces.append([i0, i2, i3])

        V = np.asarray(verts, dtype=np.float32)
        F = np.asarray(faces, dtype=np.int64)

        return V, F
    finally:
        # Important: free evaluated mesh to avoid memory leaks
        eval_obj.to_mesh_clear()


def render_hands_pytorch3d(camera_obj, hand_objects, render_shape=None, grayscale=False, separate_hands=False, target_hand=None):
    """
    Render CC_Hand_L/CC_Hand_R with Soft Phong shading via PyTorch3D.
    If grayscale=True, render in black & white with enhanced contrast.
    If separate_hands=True and target_hand is specified, render only that hand.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Intrinsics (pixels) ---
    if render_shape is None:
        # Use default resolution if not specified
        H, W = 480, 720
    else:
        H, W = render_shape
    fov_rad = camera_obj.data.angle
    fx = fy = (W / 2.0) / math.tan(fov_rad / 2.0)
    cx, cy = W / 2.0, H / 2.0
    focal   = torch.tensor([[fx, fy]], device=device, dtype=torch.float32)
    princpt = torch.tensor([[cx, cy]], device=device, dtype=torch.float32)
    image_size = torch.tensor([[H, W]], device=device, dtype=torch.int64)

    # --- World->Camera from Blender, then to PyTorch3D coords (+Z forward) ---
    M_wc = np.array(camera_obj.matrix_world.inverted(), dtype=np.float32)  # world->blenderCam
    R_wc = M_wc[:3, :3]
    t_wc = M_wc[:3, 3]
    # Convert Blender -> PyTorch3D
    C = np.diag([-1.0, 1.0, -1.0]).astype(np.float32)
    R_p3d = C @ R_wc
    t_p3d = C @ t_wc
    R_p3d_t = torch.from_numpy(R_p3d).to(device)           # (3,3)
    t_p3d_t = torch.from_numpy(t_p3d).to(device)           # (3,)

         # Camera setup complete

    # Identity extrinsics for the camera; we already moved vertices into camera space
    cameras = PerspectiveCameras(
        R=torch.eye(3, device=device).unsqueeze(0),
        T=torch.zeros(1, 3, device=device),
        focal_length=focal,
        principal_point=princpt,
        in_ndc=False,
        image_size=image_size,
        device=device,
    )

    raster_settings = RasterizationSettings(
        image_size=(H, W),
        blur_radius=0.0,
        faces_per_pixel=1,
        perspective_correct=True,
    )
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings).to(device)

    # Lighting setup - enhanced for grayscale contrast if needed
    if grayscale:
        # Stronger directional lighting for better grayscale contrast
        lights = DirectionalLights(
            device=device,
            direction=[[0.0, 0.0, -1.0]],        # pointing towards camera
            ambient_color=((0.3, 0.3, 0.3),),     # reduced ambient
            diffuse_color=((0.8, 0.8, 0.8),),     # stronger diffuse
            specular_color=((0.2, 0.2, 0.2),),   # some specular for highlights
        )
    else:
        # Gentle lighting so it never goes black (original)
        lights = PointLights(
            device=device,
            location=[[0.0, 0.0, 0.0]],           # camera center
            ambient_color=((0.7, 0.7, 0.7),),
            diffuse_color=((0.6, 0.6, 0.6),),
            specular_color=((0.0, 0.0, 0.0),),
        )
    shader = SoftPhongShader(device=device, cameras=cameras, lights=lights)
    renderer = MeshRendererWithFragments(rasterizer=rasterizer, shader=shader)

    final_rgb = torch.zeros((H, W, 3), device=device, dtype=torch.float32)
    final_z   = torch.full((H, W), float('inf'), device=device, dtype=torch.float32)

    for obj in hand_objects:
        if not obj.visible_get():
            continue
        
        # If separate_hands mode, only render the target hand
        if separate_hands and target_hand is not None:
            if target_hand == "left" and "CC_Hand_L" not in obj.name:
                continue
            elif target_hand == "right" and "CC_Hand_R" not in obj.name:
                continue

        V_np, F_np = get_hand_mesh_data(obj)  # world space (after modifiers)
        if V_np.size == 0 or F_np.size == 0:
            continue

        # --- Explicit world->camera transform ---
        # V_cam = R_p3d @ V_world + t_p3d
        Vw = torch.from_numpy(V_np).to(device)                     # (V,3)
        Vc = (Vw @ R_p3d_t.t()) + t_p3d_t                          # (V,3)  z should be > 0 if in front
        V = Vc.unsqueeze(0)                                        # (1,V,3)
        F = torch.from_numpy(F_np).to(device).unsqueeze(0).long()  # (1,F,3)

        # Per-vertex color - different for grayscale vs color
        if grayscale:
            # For grayscale: use white/light gray for both hands
            # In separate mode, use consistent white color
            if separate_hands:
                col = (1.0, 1.0, 1.0)  # Pure white for separate hand rendering
            else:
                # Left hand slightly brighter to distinguish when both hands are rendered
                if "CC_Hand_L" in obj.name:
                    col = (0.9, 0.9, 0.9)  # Slightly brighter for left hand
                else:
                    col = (0.8, 0.8, 0.8)  # Slightly darker for right hand
        else:
            # Original color scheme for color mode
            if "CC_Hand_L" in obj.name:
                col = (0.5, 0.8, 0.5)  # Vibrant green with slight blue tint
            else:
                col = (0.8, 0.4, 0.4)  # Deep, rich red
        Cverts = torch.tensor(col, device=device).view(1, 1, 3).expand(1, V.shape[1], 3)
        mesh = Meshes(verts=V, faces=F, textures=TexturesVertex(Cverts))

        with torch.no_grad():
            images, frags = renderer(mesh)     # images: (1,H,W,4)

        rgb  = images[0, :, :, :3]             # (H,W,3)
        zbuf = frags.zbuf[0, :, :, 0].float()  # (H,W)
        valid = frags.pix_to_face[0, :, :, 0] >= 0

        closer = valid & (zbuf < final_z)
        closer3 = closer.unsqueeze(-1)
        final_rgb = torch.where(closer3, rgb, final_rgb)
        final_z   = torch.where(closer,  zbuf, final_z)

    # Convert to grayscale if needed
    if grayscale:
        # Convert RGB to grayscale using luminance weights
        gray = 0.299 * final_rgb[:, :, 0] + 0.587 * final_rgb[:, :, 1] + 0.114 * final_rgb[:, :, 2]
        # Enhance contrast for better grayscale appearance
        gray = torch.clamp((gray - 0.3) * 1.5 + 0.3, 0, 1)
        # Convert back to RGB format (grayscale)
        out = (gray.unsqueeze(-1).expand(-1, -1, 3).clamp(0, 1) * 255.0).byte().cpu().numpy()
    else:
        out = (final_rgb.clamp(0, 1) * 255.0).byte().cpu().numpy()
    
    return out, final_z.cpu().numpy()


def render_hands_pytorch3d_channel_separated(camera_obj, hand_objects, render_shape=None, grayscale=False):
    """
    Render CC_Hand_L/CC_Hand_R with channel separation for efficient separate hand rendering.
    Left hand: Red channel (1,0,0), Right hand: Green channel (0,1,0)
    Returns the combined image and separate left/right hand images.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Intrinsics (pixels) ---
    if render_shape is None:
        # Use default resolution if not specified
        H, W = 480, 720
    else:
        H, W = render_shape
    fov_rad = camera_obj.data.angle
    fx = fy = (W / 2.0) / math.tan(fov_rad / 2.0)
    cx, cy = W / 2.0, H / 2.0
    focal   = torch.tensor([[fx, fy]], device=device, dtype=torch.float32)
    princpt = torch.tensor([[cx, cy]], device=device, dtype=torch.float32)
    image_size = torch.tensor([[H, W]], device=device, dtype=torch.int64)

    # --- World->Camera from Blender, then to PyTorch3D coords (+Z forward) ---
    M_wc = np.array(camera_obj.matrix_world.inverted(), dtype=np.float32)  # world->blenderCam
    R_wc = M_wc[:3, :3]
    t_wc = M_wc[:3, 3]
    # Convert Blender -> PyTorch3D
    C = np.diag([-1.0, 1.0, -1.0]).astype(np.float32)
    R_p3d = C @ R_wc
    t_p3d = C @ t_wc
    R_p3d_t = torch.from_numpy(R_p3d).to(device)           # (3,3)
    t_p3d_t = torch.from_numpy(t_p3d).to(device)           # (3,)

         # Camera setup complete

    # Identity extrinsics for the camera; we already moved vertices into camera space
    cameras = PerspectiveCameras(
        R=torch.eye(3, device=device).unsqueeze(0),
        T=torch.zeros(1, 3, device=device),
        focal_length=focal,
        principal_point=princpt,
        in_ndc=False,
        image_size=image_size,
        device=device,
    )

    raster_settings = RasterizationSettings(
        image_size=(H, W),
        blur_radius=0.0,
        faces_per_pixel=1,
        perspective_correct=True,
    )
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings).to(device)

    # Lighting setup - enhanced for grayscale contrast if needed
    if grayscale:
        # Stronger directional lighting for better grayscale contrast
        lights = DirectionalLights(
            device=device,
            direction=[[0.0, 0.0, -1.0]],        # pointing towards camera
            ambient_color=((0.3, 0.3, 0.3),),     # reduced ambient
            diffuse_color=((0.8, 0.8, 0.8),),     # stronger diffuse
            specular_color=((0.2, 0.2, 0.2),),   # some specular for highlights
        )
    else:
        # Gentle lighting so it never goes black (original)
        lights = PointLights(
            device=device,
            location=[[0.0, 0.0, 0.0]],           # camera center
            ambient_color=((0.7, 0.7, 0.7),),
            diffuse_color=((0.6, 0.6, 0.6),),
            specular_color=((0.0, 0.0, 0.0),),
        )
    shader = SoftPhongShader(device=device, cameras=cameras, lights=lights)
    renderer = MeshRendererWithFragments(rasterizer=rasterizer, shader=shader)

    final_rgb = torch.zeros((H, W, 3), device=device, dtype=torch.float32)
    final_z   = torch.full((H, W), float('inf'), device=device, dtype=torch.float32)

    for obj in hand_objects:
        if not obj.visible_get():
            continue

        V_np, F_np = get_hand_mesh_data(obj)  # world space (after modifiers)
        if V_np.size == 0 or F_np.size == 0:
            continue

        # --- Explicit world->camera transform ---
        # V_cam = R_p3d @ V_world + t_p3d
        Vw = torch.from_numpy(V_np).to(device)                     # (V,3)
        Vc = (Vw @ R_p3d_t.t()) + t_p3d_t                          # (V,3)  z should be > 0 if in front
        V = Vc.unsqueeze(0)                                        # (1,V,3)
        F = torch.from_numpy(F_np).to(device).unsqueeze(0).long()  # (1,F,3)

        # Per-vertex color - channel separation for left/right hands
        if "CC_Hand_L" in obj.name:
            col = (1.0, 0.0, 0.0)  # Red channel for left hand
        else:
            col = (0.0, 1.0, 0.0)  # Green channel for right hand
            
        Cverts = torch.tensor(col, device=device).view(1, 1, 3).expand(1, V.shape[1], 3)
        mesh = Meshes(verts=V, faces=F, textures=TexturesVertex(Cverts))

        with torch.no_grad():
            images, frags = renderer(mesh)     # images: (1,H,W,4)

        rgb  = images[0, :, :, :3]             # (H,W,3)
        zbuf = frags.zbuf[0, :, :, 0].float()  # (H,W)
        valid = frags.pix_to_face[0, :, :, 0] >= 0

        closer = valid & (zbuf < final_z)
        closer3 = closer.unsqueeze(-1)
        final_rgb = torch.where(closer3, rgb, final_rgb)
        final_z   = torch.where(closer,  zbuf, final_z)

    # Convert to numpy
    combined_image = (final_rgb.clamp(0, 1) * 255.0).byte().cpu().numpy()
    
    # Extract separate channels
    left_hand_image = combined_image[:, :, 0:1]  # Red channel only
    right_hand_image = combined_image[:, :, 1:2]  # Green channel only
    
    # Convert to grayscale if needed
    if grayscale:
        # Convert single channel to grayscale RGB
        left_hand_gray = np.repeat(left_hand_image, 3, axis=2)
        right_hand_gray = np.repeat(right_hand_image, 3, axis=2)
        
        # Apply grayscale conversion and contrast enhancement
        left_hand_gray = (left_hand_gray.astype(np.float32) / 255.0)
        left_hand_gray = np.clip((left_hand_gray - 0.3) * 1.5 + 0.3, 0, 1)
        left_hand_gray = (left_hand_gray * 255.0).astype(np.uint8)
        
        right_hand_gray = (right_hand_gray.astype(np.float32) / 255.0)
        right_hand_gray = np.clip((right_hand_gray - 0.3) * 1.5 + 0.3, 0, 1)
        right_hand_gray = (right_hand_gray * 255.0).astype(np.uint8)
        
        return left_hand_gray, right_hand_gray, final_z.cpu().numpy()
    else:
        # For color mode, just return the single channels as RGB
        left_hand_rgb = np.repeat(left_hand_image, 3, axis=2)
        right_hand_rgb = np.repeat(right_hand_image, 3, axis=2)
        
        return left_hand_rgb, right_hand_rgb, final_z.cpu().numpy()



# -------------------------------
# Simple lighting setup (kept for compatibility)
# -------------------------------
def setup_lighting_for_hands():
    """Setup simple lighting for hand rendering."""
    # Remove existing lights
    for obj in list(bpy.data.objects):
        if obj.type == 'LIGHT':
            bpy.data.objects.remove(obj, do_unlink=True)
    
    # Create main light (sun)
    sun_data = bpy.data.lights.new("MainLight", type='SUN')
    sun_data.energy = 3.0
    sun_data.angle = 0.1
    sun = bpy.data.objects.new("MainLight", sun_data)
    bpy.context.scene.collection.objects.link(sun)
    sun.location = (2.0, -2.0, 3.0)
    sun.rotation_euler = (math.radians(45), math.radians(-45), 0)
    
    # Create fill light (area)
    fill_data = bpy.data.lights.new("FillLight", type='AREA')
    fill_data.energy = 1.0
    fill_data.size = 2.0
    fill = bpy.data.objects.new("FillLight", fill_data)
    bpy.context.scene.collection.objects.link(fill)
    fill.location = (-1.0, 1.0, 2.0)
    fill.rotation_euler = (math.radians(30), math.radians(45), 0)
    
    # Setup world ambient lighting
    world = bpy.context.scene.world
    if not world:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    
    world.use_nodes = True
    world_nodes = world.node_tree.nodes
    world_links = world.node_tree.links
    
    # Clear existing nodes
    for node in list(world_nodes):
        world_nodes.remove(node)
    
    # Add background node
    bg_node = world_nodes.new(type='ShaderNodeBackground')
    bg_node.inputs['Color'].default_value = (0.1, 0.1, 0.1, 1.0)
    bg_node.inputs['Strength'].default_value = 0.2
    bg_node.location = (0, 0)
    
    # Add output node
    output_node = world_nodes.new(type='ShaderNodeOutputWorld')
    output_node.location = (200, 0)
    
    # Link nodes
    world_links.new(bg_node.outputs['Background'], output_node.inputs['Surface'])
    
    print("✓ Lighting setup complete")

# ---------------------------
# Render sequence
# ---------------------------
def render_animation_sequence(animation_index, animation_name):
    anim_output_folder = os.path.join(output_folder, f"{animation_name}")
    
    # Create local temporary directory for images (only for video mode)
    temp_dir = None
    if not args.save_images:
        # Video mode: create temp directory in current working directory
        temp_dir = Path.cwd() / "temp_hand_images" / f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        print(f"Temporary directory for images: {temp_dir}")
        print(f"Directory exists: {temp_dir.exists()}")
        print(f"Directory is writable: {temp_dir.is_dir() and os.access(temp_dir, os.W_OK)}")
    
    # Validate separate flag usage
    if args.separate and not args.grayscale:
        print("Warning: --separate flag only works with --grayscale. Ignoring --separate.")
        args.separate = False
    
    if args.save_images:
        # Image mode: save to images_hands folder
        if args.separate:
            # Separate mode: create subfolders for left and right hands
            images_output_path_left = os.path.join(anim_output_folder, "images_hands_left")
            images_output_path_right = os.path.join(anim_output_folder, "images_hands_right")
            os.makedirs(images_output_path_left, exist_ok=True)
            os.makedirs(images_output_path_right, exist_ok=True)
            print(f"Rendering animation {animation_index}: {animation_name}")
            print(f"  Mode: Grayscale Separate (Images)")
            print(f"  Left hand images: {images_output_path_left}")
            print(f"  Right hand images: {images_output_path_right}")
        else:
            images_output_path = os.path.join(anim_output_folder, "images_hands")
            os.makedirs(images_output_path, exist_ok=True)
            print(f"Rendering animation {animation_index}: {animation_name}")
            print(f"  Mode: {'Grayscale' if args.grayscale else 'Color'} (Images)")
            print(f"  Images: {images_output_path}")
    else:
        # Video mode: save to sequences/videos_hands folder
        sequences_folder = os.path.join(anim_output_folder, "sequences")
        if args.separate:
            # Separate mode: create subfolders for left and right hands
            videos_output_path_left = os.path.join(sequences_folder, "videos_hands_gray_left")
            videos_output_path_right = os.path.join(sequences_folder, "videos_hands_gray_right")
            os.makedirs(videos_output_path_left, exist_ok=True)
            os.makedirs(videos_output_path_right, exist_ok=True)
            print(f"Rendering animation {animation_index}: {animation_name}")
            print(f"  Mode: Grayscale Separate (Videos)")
            print(f"  Left hand videos: {videos_output_path_left}")
            print(f"  Right hand videos: {videos_output_path_right}")
            print(f"  Temp images: {temp_dir}")
        else:
            # Use different path for grayscale vs color videos
            video_folder_suffix = "hands_gray" if args.grayscale else "hands"
            videos_output_path = os.path.join(sequences_folder, f"videos_{video_folder_suffix}")
            os.makedirs(videos_output_path, exist_ok=True)
            print(f"Rendering animation {animation_index}: {animation_name}")
            print(f"  Mode: {'Grayscale' if args.grayscale else 'Color'} (Videos)")
            print(f"  Videos: {videos_output_path}")
            print(f"  Temp images: {temp_dir}")

    # Get hand objects
    hand_objects = []
    for obj_name in ["CC_Hand_L", "CC_Hand_R"]:
        obj = bpy.data.objects.get(obj_name)
        if obj and obj.visible_get():
            hand_objects.append(obj)
    
    if not hand_objects:
        print("Warning: No visible hand objects found!")
        return

    # Get camera
    camera_obj = bpy.context.scene.camera
    if not camera_obj:
        print("Error: No active camera found!")
        return

    # Frame range
    scene = bpy.context.scene
    render_start_frame = scene.frame_start if start_frame is None else start_frame
    render_end_frame   = scene.frame_end   if end_frame   is None else end_frame
    
    if args.save_images:
        # Image mode: render all frames directly
        frames_to_render = list(range(render_start_frame, render_end_frame + 1, args.frame_skip))
        print(f"Animation frames: {render_start_frame}..{render_end_frame}")
        print(f"Frame step: {args.frame_skip} -> {len(frames_to_render)} frames")
        
        if args.separate:
            # Separate mode: render both hands in one pass with channel separation
            print(f"\n--- Rendering BOTH HANDS with channel separation ---")
            frames_completed = 0
            frames_skipped = 0
            
            for frame_idx, frame_num in enumerate(frames_to_render):
                scene.frame_set(frame_num)
                
                # Check if both frames already exist
                left_image_path = os.path.join(images_output_path_left, f"{frame_num:04d}.png")
                right_image_path = os.path.join(images_output_path_right, f"{frame_num:04d}.png")
                
                left_exists, left_needs_rendering = check_frame_exists(frame_num, images_output_path_left)
                right_exists, right_needs_rendering = check_frame_exists(frame_num, images_output_path_right)
                
                if args.skip_existing and not left_needs_rendering and not right_needs_rendering:
                    frames_skipped += 1
                    print(f"[SEPARATE] Frame {frame_num}: SKIPPED (both hands already exist)")
                    continue
                
                # Render with PyTorch3D (both hands in one pass with channel separation)
                frame_render_start = time.time()
                print(f"[SEPARATE] Rendering frame {frame_num} ({frame_idx + 1}/{len(frames_to_render)})...")
                
                # Render hands using channel separation
                left_image, right_image, depth = render_hands_pytorch3d_channel_separated(
                    camera_obj, hand_objects, 
                    render_shape=(args.height, args.width), 
                    grayscale=args.grayscale
                )
                
                # Save images directly to NAS (images_only mode)
                import cv2
                if left_needs_rendering:
                    cv2.imwrite(left_image_path, cv2.cvtColor(left_image, cv2.COLOR_RGB2BGR))
                if right_needs_rendering:
                    cv2.imwrite(right_image_path, cv2.cvtColor(right_image, cv2.COLOR_RGB2BGR))
                
                frame_time = time.time() - frame_render_start
                total_render_time += frame_time
                frames_completed += 1
                
                if frame_idx % 10 == 0 or frame_idx == len(frames_to_render) - 1:
                    progress = (frame_idx + 1) / len(frames_to_render) * 100.0
                    print(f"  Frame {frame_idx + 1}/{len(frames_to_render)} ({progress:.1f}%) - {frame_time:.1f}s")
            
            total_time = time.time() - start_time
            avg_fps = frames_completed / total_time if total_time > 0 and frames_completed > 0 else 0
            
            print("\n" + "="*50)
            print(f"COMPLETED: Animation {animation_index} ({animation_name}) - SEPARATE IMAGE MODE")
            print(f"Total time: {total_time/60:.1f} minutes")
            print(f"Frames rendered: {frames_completed} | Skipped: {frames_skipped}")
            print(f"Frame step: {args.frame_skip}")
            print(f"Average throughput: {avg_fps:.2f} fps")
            print(f"Left hand images: {images_output_path_left}")
            print(f"Right hand images: {images_output_path_right}")
            print("="*50)
        else:
            # Normal mode: render both hands together
            frames_completed = 0
            frames_skipped = 0
            
            for frame_idx, frame_num in enumerate(frames_to_render):
                scene.frame_set(frame_num)
                
                # Check if frame already exists
                image_path = os.path.join(images_output_path, f"{frame_num:04d}.png")
                image_exists, needs_rendering = check_frame_exists(frame_num, images_output_path)
                
                if args.skip_existing and not needs_rendering:
                    frames_skipped += 1
                    print(f"[IMAGE] Frame {frame_num}: SKIPPED (already exists)")
                    continue
                
                # Render with PyTorch3D
                frame_render_start = time.time()
                print(f"[IMAGE] Rendering frame {frame_num} ({frame_idx + 1}/{len(frames_to_render)})...")
                
                # Render hands using PyTorch3D
                image, depth = render_hands_pytorch3d(camera_obj, hand_objects, render_shape=(args.height, args.width), grayscale=args.grayscale)
                
                # Save image directly to NAS (images_only mode)
                import cv2
                cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                
                frame_time = time.time() - frame_render_start
                total_render_time += frame_time
                frames_completed += 1
                
                if frame_idx % 10 == 0 or frame_idx == len(frames_to_render) - 1:
                    progress = (frame_idx + 1) / len(frames_to_render) * 100.0
                    avg_frame_time = total_render_time / frames_completed if frames_completed > 0 else 0
                    print(f"  Frame {frame_idx + 1}/{len(frames_to_render)} ({progress:.1f}%) - {frame_time:.1f}s")
            
            total_time = time.time() - start_time
            avg_fps = frames_completed / total_time if total_time > 0 and frames_completed > 0 else 0
            
            print("\n" + "="*50)
            print(f"COMPLETED: Animation {animation_index} ({animation_name}) - IMAGE MODE")
            print(f"Total time: {total_time/60:.1f} minutes")
            print(f"Frames rendered: {frames_completed} | Skipped: {frames_skipped}")
            print(f"Frame step: {args.frame_skip}")
            print(f"Average throughput: {avg_fps:.2f} fps")
            print(f"Images saved to: {images_output_path}")
            print("="*50)
    else:
        # Video mode: video sequence parameters
        clip_length = 49  # 49 frames per video
        stride = args.stride  # Use command line argument
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
    total_render_time = 0.0

    if args.save_images:
        # Image mode: render all frames directly
        frames_completed = 0
        frames_skipped = 0
        
        for frame_idx, frame_num in enumerate(frames_to_render):
            scene.frame_set(frame_num)
            
            # Check if frame already exists
            image_path = os.path.join(images_output_path, f"{frame_num:04d}.png")
            image_exists, needs_rendering = check_frame_exists(frame_num, images_output_path)
            
            if args.skip_existing and not needs_rendering:
                frames_skipped += 1
                print(f"[IMAGE] Frame {frame_num}: SKIPPED (already exists)")
                continue
            
            # Render with PyTorch3D
            frame_render_start = time.time()
            print(f"[IMAGE] Rendering frame {frame_num} ({frame_idx + 1}/{len(frames_to_render)})...")
            
            # Render hands using PyTorch3D
            image, depth = render_hands_pytorch3d(camera_obj, hand_objects, render_shape=(args.height, args.width), grayscale=args.grayscale)
            
            # Save image directly to NAS (images_only mode)
            import cv2
            cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            
            frame_time = time.time() - frame_render_start
            total_render_time += frame_time
            frames_completed += 1
            
            if frame_idx % 10 == 0 or frame_idx == len(frames_to_render) - 1:
                progress = (frame_idx + 1) / len(frames_to_render) * 100.0
                avg_frame_time = total_render_time / frames_completed if frames_completed > 0 else 0
                print(f"  Frame {frame_idx + 1}/{len(frames_to_render)} ({progress:.1f}%) - {frame_time:.1f}s")
        
        total_time = time.time() - start_time
        avg_fps = frames_completed / total_time if total_time > 0 and frames_completed > 0 else 0
        
        print("\n" + "="*50)
        print(f"COMPLETED: Animation {animation_index} ({animation_name}) - IMAGE MODE")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Frames rendered: {frames_completed} | Skipped: {frames_skipped}")
        print(f"Frame step: {args.frame_skip}")
        print(f"Average throughput: {avg_fps:.2f} fps")
        print(f"Images saved to: {images_output_path}")
        print("="*50)
        
    else:
        # Video mode: optimized - render all frames once, then create videos
        videos_completed = 0
        
        if args.separate:
            # Separate mode: progressive rendering - render frames and create videos incrementally
            print(f"\n--- PROGRESSIVE SEPARATE MODE: Render frames and create videos incrementally ---")
            
            # Create temp directory for current video frames
            current_video_temp_dir = temp_dir / "current_video_frames"
            current_video_temp_dir.mkdir(exist_ok=True)
            
            frames_rendered_total = 0
            frames_skipped_total = 0
            
            for video_idx, start_frame_num in enumerate(video_start_frames):
                video_end_frame = start_frame_num + (clip_length - 1) * frame_skip
                frames_for_video = list(range(start_frame_num, video_end_frame + 1, frame_skip))

                # Define output paths first
                left_video_output_path = os.path.join(videos_output_path_left, f"{video_idx:05d}.mp4")
                right_video_output_path = os.path.join(videos_output_path_right, f"{video_idx:05d}.mp4")
                
                # Check if both videos already exist
                left_video_exists, left_needs_video_rendering = check_video_exists(video_idx, videos_output_path_left)
                right_video_exists, right_needs_video_rendering = check_video_exists(video_idx, videos_output_path_right)
                
                # Force re-rendering if --no-skip-existing is used
                if not args.skip_existing:
                    left_needs_video_rendering = True
                    right_needs_video_rendering = True
                    print(f"    [FORCE] --no-skip-existing: Will re-render both videos")
                    
                    # Delete existing video files to ensure clean re-rendering
                    if left_video_exists:
                        try:
                            os.remove(left_video_output_path)
                            print(f"    [DELETE] Removed existing left video: {left_video_output_path}")
                        except OSError as e:
                            print(f"    [WARNING] Could not remove left video: {e}")
                    
                    if right_video_exists:
                        try:
                            os.remove(right_video_output_path)
                            print(f"    [DELETE] Removed existing right video: {right_video_output_path}")
                        except OSError as e:
                            print(f"    [WARNING] Could not remove right video: {e}")
            
                if args.skip_existing and not left_needs_video_rendering and not right_needs_video_rendering:
                    print(f"[VIDEO {video_idx + 1}] SKIPPED: Both videos already exist")
                    videos_completed += 2
                    continue

                print(f"\n[VIDEO {video_idx + 1}] Processing frames {start_frame_num}..{video_end_frame} ({len(frames_for_video)} frames)")
                print(f"  Frame sequence: {frames_for_video[:5]}{'...' if len(frames_for_video) > 5 else ''} (showing first 5)")
                
                # Step 1: Render frames for this video
                frames_rendered = 0
                frames_skipped = 0
                left_video_frames = []
                right_video_frames = []
                
                for frame_idx, frame_num in enumerate(frames_for_video):
                    scene.frame_set(frame_num)
                    
                    # Check if frame already exists
                    # Use 1-based indexing for ffmpeg compatibility
                    left_frame_path = current_video_temp_dir / f"left_frame_{frame_idx + 1:04d}.png"
                    right_frame_path = current_video_temp_dir / f"right_frame_{frame_idx + 1:04d}.png"
                    
                    if args.skip_existing and left_frame_path.exists() and right_frame_path.exists():
                        frames_skipped += 1
                        if left_needs_video_rendering:
                            left_video_frames.append(left_frame_path)
                        if right_needs_video_rendering:
                            right_video_frames.append(right_frame_path)
                        continue
                    
                    # Render with PyTorch3D (both hands in one pass with channel separation)
                    frame_render_start = time.time()
                    if frame_idx % 10 == 0 or frame_idx == len(frames_for_video) - 1:
                        print(f"  [RENDER] Frame {frame_num} ({frame_idx + 1}/{len(frames_for_video)})...")
                    
                    # Render hands using channel separation
                    left_image, right_image, depth = render_hands_pytorch3d_channel_separated(
                        camera_obj, hand_objects, 
                        render_shape=(args.height, args.width), 
                        grayscale=args.grayscale
                    )
                    
                    # Save images to current video temp directory
                    import cv2
                    if left_needs_video_rendering:
                        cv2.imwrite(str(left_frame_path), cv2.cvtColor(left_image, cv2.COLOR_RGB2BGR))
                        left_video_frames.append(left_frame_path)
                    
                    if right_needs_video_rendering:
                        cv2.imwrite(str(right_frame_path), cv2.cvtColor(right_image, cv2.COLOR_RGB2BGR))
                        right_video_frames.append(right_frame_path)
                    
                    frame_time = time.time() - frame_render_start
                    total_render_time += frame_time
                    frames_rendered += 1
                    
                    if frame_idx % 10 == 0 or frame_idx == len(frames_for_video) - 1:
                        progress = (frame_idx + 1) / len(frames_for_video) * 100.0
                        print(f"    Frame {frame_idx + 1}/{len(frames_for_video)} ({progress:.1f}%) - {frame_time:.1f}s")
                
                frames_rendered_total += frames_rendered
                frames_skipped_total += frames_skipped
                print(f"  [RENDER COMPLETE] Rendered: {frames_rendered} frames, Skipped: {frames_skipped} frames")
                
                # Step 2: Create videos from rendered frames
                print(f"  [VIDEO CREATION] Creating videos...")
                
                try:
                    import subprocess
                    
                    # Create left hand video
                    if left_needs_video_rendering and left_video_frames:
                        # Debug: Check if frame files actually exist
                        frame_files_exist = 0
                        for frame_path in left_video_frames[:5]:  # Check first 5 frames
                            if frame_path.exists():
                                frame_files_exist += 1
                        print(f"    [DEBUG] {frame_files_exist}/{min(5, len(left_video_frames))} frame files exist in temp directory")
                        left_rgb_cmd = [
                            'ffmpeg', '-y', '-framerate', str(fps),
                            '-i', str(current_video_temp_dir / 'left_frame_%04d.png'),
                            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                            '-crf', '18', left_video_output_path
                        ]
                        print(f"    [FFMPEG] Creating left hand video with {len(left_video_frames)} frames")
                        print(f"    [FFMPEG] Command: {' '.join(left_rgb_cmd)}")
                        result = subprocess.run(left_rgb_cmd, check=True, capture_output=True, text=True)
                        if os.path.exists(left_video_output_path) and os.path.getsize(left_video_output_path) > 0:
                            print(f"    ✅ Created left hand video: {left_video_output_path}")
                        else:
                            print(f"    ❌ Left hand video file not created or empty: {left_video_output_path}")
                    
                    # Create right hand video
                    if right_needs_video_rendering and right_video_frames:
                        right_rgb_cmd = [
                            'ffmpeg', '-y', '-framerate', str(fps),
                            '-i', str(current_video_temp_dir / 'right_frame_%04d.png'),
                            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                            '-crf', '18', right_video_output_path
                        ]
                        print(f"    [FFMPEG] Creating right hand video with {len(right_video_frames)} frames")
                        print(f"    [FFMPEG] Command: {' '.join(right_rgb_cmd)}")
                        result = subprocess.run(right_rgb_cmd, check=True, capture_output=True, text=True)
                        if os.path.exists(right_video_output_path) and os.path.getsize(right_video_output_path) > 0:
                            print(f"    ✅ Created right hand video: {right_video_output_path}")
                        else:
                            print(f"    ❌ Right hand video file not created or empty: {right_video_output_path}")
                    
                except subprocess.CalledProcessError as e:
                    print(f"    ❌ Failed to create separate videos: {e}")
                    print(f"    Command output: {e.stderr.decode()}")
                except FileNotFoundError:
                    print(f"    ❌ ffmpeg not found. Please install ffmpeg to create videos.")
                    print(f"    Rendered frames saved in: {current_video_temp_dir}")
                
                # Step 3: Clean up current video frames (keep only overlapping frames for next video)
                print(f"  [CLEANUP] Cleaning up processed frames...")
                frames_to_keep = []
                if video_idx < len(video_start_frames) - 1:  # Not the last video
                    next_video_start = video_start_frames[video_idx + 1]
                    # Keep frames that will be needed for the next video (overlap)
                    for frame_num in range(next_video_start, video_end_frame + 1, frame_skip):
                        frame_idx_in_current = (frame_num - start_frame_num) // frame_skip
                        if 0 <= frame_idx_in_current < len(frames_for_video):
                            frames_to_keep.append(frame_idx_in_current)
                
                # Remove frames that won't be needed (use 1-based indexing for ffmpeg compatibility)
                frames_removed = 0
                for frame_idx in range(len(frames_for_video)):
                    if frame_idx not in frames_to_keep:
                        left_frame_path = current_video_temp_dir / f"left_frame_{frame_idx + 1:04d}.png"
                        right_frame_path = current_video_temp_dir / f"right_frame_{frame_idx + 1:04d}.png"
                        if left_frame_path.exists():
                            left_frame_path.unlink()
                            frames_removed += 1
                        if right_frame_path.exists():
                            right_frame_path.unlink()
                            frames_removed += 1
                
                if frames_removed > 0:
                    print(f"    Removed {frames_removed} frame files, kept {len(frames_to_keep)} for next video")
                
                videos_completed += 2
                
                # Progress update
                total_elapsed = time.time() - start_time
                if videos_completed > 2:
                    avg_video_time = total_elapsed / (videos_completed // 2)
                    remaining_videos = len(video_start_frames) - (videos_completed // 2)
                    eta = remaining_videos * avg_video_time
                    print(f"  📊 Progress: {videos_completed // 2}/{len(video_start_frames)} video pairs")
                    print(f"  ⏱️  Avg: {avg_video_time:.1f}s per video pair")
                    print(f"  🎯 ETA: {eta/60:.1f} min | Elapsed: {total_elapsed/60:.1f} min")
            
            # Final cleanup
            print(f"\n🧹 Final cleanup of temp directory: {current_video_temp_dir}")
            import shutil
            shutil.rmtree(current_video_temp_dir, ignore_errors=True)
            
            print(f"\n[SEPARATE HAND VIDEOS] Completed: {videos_completed // 2} video pairs ({videos_completed} total videos)")
            print(f"Total frames rendered: {frames_rendered_total}, Total frames skipped: {frames_skipped_total}")
        else:
            # Normal mode: progressive rendering - render frames and create videos incrementally
            print(f"\n--- PROGRESSIVE NORMAL MODE: Render frames and create videos incrementally ---")
            
            # Create temp directory for current video frames
            current_video_temp_dir = temp_dir / "current_video_frames"
            current_video_temp_dir.mkdir(exist_ok=True)
            
            frames_rendered_total = 0
            frames_skipped_total = 0
            
            for video_idx, start_frame_num in enumerate(video_start_frames):
                video_end_frame = start_frame_num + (clip_length - 1) * frame_skip
                frames_for_video = list(range(start_frame_num, video_end_frame + 1, frame_skip))

                # Define output path first
                video_output_path = os.path.join(videos_output_path, f"{video_idx:05d}.mp4")
                
                # Check if video already exists
                video_exists, needs_video_rendering = check_video_exists(video_idx, videos_output_path)
                
                # Force re-rendering if --no-skip-existing is used
                if not args.skip_existing:
                    needs_video_rendering = True
                    print(f"    [FORCE] --no-skip-existing: Will re-render video")
                    
                    # Delete existing video file to ensure clean re-rendering
                    if video_exists:
                        try:
                            os.remove(video_output_path)
                            print(f"    [DELETE] Removed existing video: {video_output_path}")
                        except OSError as e:
                            print(f"    [WARNING] Could not remove video: {e}")
            
                if args.skip_existing and not needs_video_rendering:
                    print(f"[VIDEO {video_idx + 1}] SKIPPED: Video already exists")
                    videos_completed += 1
                    continue

                print(f"\n[VIDEO {video_idx + 1}] Processing frames {start_frame_num}..{video_end_frame} ({len(frames_for_video)} frames)")
                print(f"  Frame sequence: {frames_for_video[:5]}{'...' if len(frames_for_video) > 5 else ''} (showing first 5)")
                
                # Step 1: Render frames for this video
                frames_rendered = 0
                frames_skipped = 0
                video_frames = []
                
                for frame_idx, frame_num in enumerate(frames_for_video):
                    scene.frame_set(frame_num)
                    
                    # Check if frame already exists
                    # Use 1-based indexing for ffmpeg compatibility
                    frame_path = current_video_temp_dir / f"frame_{frame_idx + 1:04d}.png"
                    
                    if args.skip_existing and frame_path.exists():
                        frames_skipped += 1
                        video_frames.append(frame_path)
                        continue
                    
                    # Render with PyTorch3D
                    frame_render_start = time.time()
                    if frame_idx % 10 == 0 or frame_idx == len(frames_for_video) - 1:
                        print(f"  [RENDER] Frame {frame_num} ({frame_idx + 1}/{len(frames_for_video)})...")
                    
                    # Render hands using PyTorch3D
                    image, depth = render_hands_pytorch3d(camera_obj, hand_objects, render_shape=(args.height, args.width), grayscale=args.grayscale)
                    
                    # Save image to current video temp directory
                    import cv2
                    cv2.imwrite(str(frame_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                    video_frames.append(frame_path)
                    
                    frame_time = time.time() - frame_render_start
                    total_render_time += frame_time
                    frames_rendered += 1
                    
                    if frame_idx % 10 == 0 or frame_idx == len(frames_for_video) - 1:
                        progress = (frame_idx + 1) / len(frames_for_video) * 100.0
                        print(f"    Frame {frame_idx + 1}/{len(frames_for_video)} ({progress:.1f}%) - {frame_time:.1f}s")
                
                frames_rendered_total += frames_rendered
                frames_skipped_total += frames_skipped
                print(f"  [RENDER COMPLETE] Rendered: {frames_rendered} frames, Skipped: {frames_skipped} frames")
                
                # Step 2: Create video from rendered frames
                print(f"  [VIDEO CREATION] Creating video...")
                
                try:
                    import subprocess
                    
                    # Create RGB video from temp images
                    rgb_cmd = [
                        'ffmpeg', '-y', '-framerate', str(fps),
                        '-i', str(current_video_temp_dir / 'frame_%04d.png'),
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                        '-crf', '18', video_output_path
                    ]
                    print(f"    [FFMPEG] Creating video with {len(video_frames)} frames")
                    print(f"    [FFMPEG] Command: {' '.join(rgb_cmd)}")
                    result = subprocess.run(rgb_cmd, check=True, capture_output=True, text=True)
                    if os.path.exists(video_output_path) and os.path.getsize(video_output_path) > 0:
                        print(f"    ✅ Created video: {video_output_path}")
                    else:
                        print(f"    ❌ Video file not created or empty: {video_output_path}")
                    
                except subprocess.CalledProcessError as e:
                    print(f"    ❌ Failed to create video: {e}")
                    print(f"    Command output: {e.stderr.decode()}")
                except FileNotFoundError:
                    print(f"    ❌ ffmpeg not found. Please install ffmpeg to create videos.")
                    print(f"    Rendered frames saved in: {current_video_temp_dir}")
                
                # Step 3: Clean up current video frames (keep only overlapping frames for next video)
                print(f"  [CLEANUP] Cleaning up processed frames...")
                frames_to_keep = []
                if video_idx < len(video_start_frames) - 1:  # Not the last video
                    next_video_start = video_start_frames[video_idx + 1]
                    # Keep frames that will be needed for the next video (overlap)
                    for frame_num in range(next_video_start, video_end_frame + 1, frame_skip):
                        frame_idx_in_current = (frame_num - start_frame_num) // frame_skip
                        if 0 <= frame_idx_in_current < len(frames_for_video):
                            frames_to_keep.append(frame_idx_in_current)
                
                # Remove frames that won't be needed (use 1-based indexing for ffmpeg compatibility)
                frames_removed = 0
                for frame_idx in range(len(frames_for_video)):
                    if frame_idx not in frames_to_keep:
                        frame_path = current_video_temp_dir / f"frame_{frame_idx + 1:04d}.png"
                        if frame_path.exists():
                            frame_path.unlink()
                            frames_removed += 1
                
                if frames_removed > 0:
                    print(f"    Removed {frames_removed} frame files, kept {len(frames_to_keep)} for next video")
                
                videos_completed += 1
                
                # Progress update
                total_elapsed = time.time() - start_time
                if videos_completed > 1:
                    avg_video_time = total_elapsed / videos_completed
                    remaining_videos = len(video_start_frames) - videos_completed
                    eta = remaining_videos * avg_video_time
                    print(f"  📊 Progress: {videos_completed}/{len(video_start_frames)} videos")
                    print(f"  ⏱️  Avg: {avg_video_time:.1f}s per video")
                    print(f"  🎯 ETA: {eta/60:.1f} min | Elapsed: {total_elapsed/60:.1f} min")
            
            # Final cleanup
            print(f"\n🧹 Final cleanup of temp directory: {current_video_temp_dir}")
            import shutil
            shutil.rmtree(current_video_temp_dir, ignore_errors=True)
            
            print(f"Total frames rendered: {frames_rendered_total}, Total frames skipped: {frames_skipped_total}")

        total_time = time.time() - start_time
        avg_fps = (videos_completed * clip_length) / total_time if total_time > 0 else 0

        print("\n" + "="*50)
        print(f"COMPLETED: Animation {animation_index} ({animation_name}) - VIDEO MODE")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Videos created: {videos_completed}/{len(video_start_frames)}")
        print(f"Frame step: {frame_skip} | Stride: {stride} | Effective stride: {effective_stride}")
        print(f"Average throughput: {avg_fps:.2f} fps")
        print(f"Skip existing: {args.skip_existing}")
        print("="*50)
        
        # Final cleanup of temporary directory
        if temp_dir is not None and temp_dir.exists():
            print(f"🧹 Final cleanup of temporary directory: {temp_dir}")
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

# ---------------------------
# Setup hand rendering
# ---------------------------
print("Setting up simple hand rendering...")
if not setup_simple_hand_rendering():
    print("Failed to setup hand rendering. Exiting.")
    sys.exit(1)

# Hide all non-hand objects
hide_non_hand_objects()

setup_lighting_for_hands()

# Test render to verify PyTorch3D setup works
print("Testing PyTorch3D render setup...")

# Check if there are any visible hand objects
hand_objects = []
for obj_name in ["CC_Hand_L", "CC_Hand_R"]:
    obj = bpy.data.objects.get(obj_name)
    if obj and obj.visible_get():
        hand_objects.append(obj)

if not hand_objects:
    print("⚠️  No visible hand objects found!")

try:
    camera_obj = bpy.context.scene.camera
    if camera_obj and hand_objects:
        image, depth = render_hands_pytorch3d(camera_obj, hand_objects, render_shape=(args.height, args.width), grayscale=args.grayscale)
        mode = "grayscale" if args.grayscale else "color"
        print(f"✓ PyTorch3D setup verified - ready for {mode} rendering")
    else:
        print("⚠️  Cannot test render - missing camera or hand objects")
except Exception as e:
    print(f"✗ PyTorch3D test render failed: {e}")

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
# rendered frames count with step (video-based)
total_frames_rendered = 0
for idx, _name in animations_to_render:
    if idx not in [f[0] for f in failed_animations]:
        scene = bpy.context.scene
        # Calculate frames per video sequence
        clip_length = 49
        stride = args.stride
        frame_skip = args.frame_skip
        effective_stride = stride * frame_skip
        
        render_start = scene.frame_start if start_frame is None else start_frame
        render_end = scene.frame_end if end_frame is None else end_frame
        
        # Count video sequences
        video_count = 0
        current_frame = render_start
        while current_frame + (clip_length - 1) * frame_skip <= render_end:
            video_count += 1
            current_frame += effective_stride
        
        frames_per_video = clip_length
        total_frames_rendered += video_count * frames_per_video
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
if args.save_images:
    if args.separate:
        print("    ├── images_hands_left/     # Left hand-only images (PNG) - Individual frames")
        print("    └── images_hands_right/    # Right hand-only images (PNG) - Individual frames")
    else:
        print("    └── images_hands/         # Hand-only images (PNG) - Individual frames")
else:
    print("    └── sequences/")
    if args.separate:
        print("        ├── videos_hands_gray_left/   # Left hand-only video sequences (MP4) - Grayscale mode")
        print("        └── videos_hands_gray_right/  # Right hand-only video sequences (MP4) - Grayscale mode")
    else:
        print("        ├── videos_hands/         # Hand-only video sequences (MP4) - Color mode")
        print("        └── videos_hands_gray/    # Hand-only video sequences (MP4) - Grayscale mode")
if failed_animations:
    print(f"\nFAILED ANIMATIONS ({len(failed_animations)}):")
    for anim_idx, anim_name, error_type in failed_animations:
        print(f"  Animation {anim_idx} ({anim_name}): {error_type}")
    print(f"Check error log for details: {error_log_file}")
else:
    print("\nAll animations completed successfully!")
print("="*60) 