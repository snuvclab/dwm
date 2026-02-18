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
parser.add_argument("--animation_name", type=str, default=None, help="Specific animation name (e.g., '2023-01-14@22-06-10.pkl')")
parser.add_argument("--samples", type=int, default=32, help="Cycles samples")
parser.add_argument("--save-path", type=str, default="/home/byungjun/workspace/trumans_ego/ego_render_new",
                    help="Root output dir")
parser.add_argument("--skip-existing", action="store_true", default=True, help="Skip frames that already exist")
parser.add_argument("--no-skip-existing", action="store_true", help="Disable skipping existing frames")
parser.add_argument("--frame-skip", type=int, default=3, help="Render every Nth frame")
parser.add_argument("--stride", type=int, default=25, help="Stride for video sequences (default: 25)")
parser.add_argument("--clip-length", type=int, default=49, help="Frames per video clip (default: 49)")
parser.add_argument("--fps", type=float, default=8.0, help="FPS for output videos (default: 8.0)")
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

    # Persistent data caching (Blender 4.x) - improves performance for animation rendering
    if hasattr(scene.render, "use_persistent_data"):
        scene.render.use_persistent_data = True

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


@torch.no_grad()
def render_hands_pytorch3d(camera_obj, hand_objects, render_shape=None, grayscale=False):
    """
    Render CC_Hand_L/CC_Hand_R with simplified lighting for stable conditioning.
    Uses backup version's approach with improved color stability.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Camera intrinsics ---
    if render_shape is None:
        H, W = 480, 720
    else:
        H, W = render_shape
    fov_rad = camera_obj.data.angle
    fx = fy = (W / 2.0) / math.tan(fov_rad / 2.0)
    cx, cy = W / 2.0, H / 2.0
    focal = torch.tensor([[fx, fy]], device=device, dtype=torch.float32)
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
        faces_per_pixel=1,  # Use backup version's simpler setting
        perspective_correct=True,
    )
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings).to(device)

    # Simplified lighting (backup version approach)
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

        # Hand colors (restored from 2025-08/09: vibrant green / deep red)
        if "CC_Hand_L" in obj.name:
            col = (0.5, 0.8, 0.5)   # vibrant green with slight blue tint
        else:
            col = (0.8, 0.4, 0.4)   # deep, rich red
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

    # --- Grayscale option ---
    if grayscale:
        gray = (0.299*final_rgb[...,0] + 0.587*final_rgb[...,1] + 0.114*final_rgb[...,2]).unsqueeze(-1)
        gray = torch.clamp((gray - 0.3)*1.5 + 0.3, 0, 1)
        final_rgb = gray.expand_as(final_rgb)

    out = (final_rgb.clamp(0,1)*255.0).byte().cpu().numpy()
    depth = final_z.cpu().numpy()
    return out, depth


# @torch.no_grad()
# def render_hands_pytorch3d(camera_obj, hand_objects, render_shape=None, grayscale=False):
#     """
#     Render both CC_Hand_L and CC_Hand_R together (occlusion-correct),
#     using depth-normal based pseudo shading with hue-separated tinting.
#     """
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     # --- Camera intrinsics ---
#     if render_shape is None:
#         H, W = 480, 720
#     else:
#         H, W = render_shape
#     fov_rad = camera_obj.data.angle
#     fx = fy = (W / 2.0) / math.tan(fov_rad / 2.0)
#     cx, cy = W / 2.0, H / 2.0
#     focal = torch.tensor([[fx, fy]], device=device)
#     princpt = torch.tensor([[cx, cy]], device=device)
#     image_size = torch.tensor([[H, W]], device=device)

#     # --- Extrinsics (Blender → PyTorch3D coords) ---
#     M_wc = np.array(camera_obj.matrix_world.inverted(), dtype=np.float32)
#     R_wc, t_wc = M_wc[:3, :3], M_wc[:3, 3]
#     C = np.diag([-1.0, 1.0, -1.0]).astype(np.float32)
#     R_p3d = C @ R_wc
#     t_p3d = C @ t_wc
#     R_p3d_t, t_p3d_t = torch.from_numpy(R_p3d).to(device), torch.from_numpy(t_p3d).to(device)

#     # Camera setup
#     cameras = PerspectiveCameras(
#         R=torch.eye(3, device=device).unsqueeze(0),
#         T=torch.zeros(1, 3, device=device),
#         focal_length=focal,
#         principal_point=princpt,
#         in_ndc=False,
#         image_size=image_size,
#         device=device,
#     )
#     raster_settings = RasterizationSettings(
#         image_size=(H, W), blur_radius=0.0, faces_per_pixel=1, perspective_correct=True
#     )
#     rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings).to(device)
#     lights = DirectionalLights(device=device, direction=[[0.0, 0.0, -1.0]])
#     shader = SoftPhongShader(device=device, cameras=cameras, lights=lights)
#     renderer = MeshRendererWithFragments(rasterizer=rasterizer, shader=shader)

#     # --- Collect both hands ---
#     all_verts, all_faces, all_colors = [], [], []
#     offset = 0
#     left_col, right_col = torch.tensor([0.6, 0.8, 1.0], device=device), torch.tensor([1.0, 0.7, 0.8], device=device)

#     for obj in hand_objects:
#         if not obj.visible_get():
#             continue

#         V_np, F_np = get_hand_mesh_data(obj)
#         if V_np.size == 0 or F_np.size == 0:
#             continue

#         Vw = torch.from_numpy(V_np).to(device)
#         Vc = (Vw @ R_p3d_t.t()) + t_p3d_t
#         F = torch.from_numpy(F_np).to(device).long() + offset
#         offset += Vc.shape[0]

#         base_color = left_col if "CC_Hand_L" in obj.name else right_col
#         Cverts = base_color.view(1, 3).expand(Vc.shape[0], 3)
#         all_verts.append(Vc)
#         all_faces.append(F)
#         all_colors.append(Cverts)

#     if not all_verts:
#         return np.zeros((H, W, 3), np.uint8), np.full((H, W), np.inf, np.float32)

#     V_all = torch.cat(all_verts, 0).unsqueeze(0)
#     F_all = torch.cat(all_faces, 0).unsqueeze(0)
#     C_all = torch.cat(all_colors, 0).unsqueeze(0)

#     mesh = Meshes(verts=V_all, faces=F_all, textures=TexturesVertex(C_all))

#     # --- Render combined ---
#     img, frags = renderer(mesh)
#     rgb = img[0, :, :, :3]
#     zbuf = frags.zbuf[0, :, :, 0].float()
#     valid = frags.pix_to_face[0, :, :, 0] >= 0

#     # --- Depth-normal shading ---
#     mesh_tmp = Meshes(verts=V_all, faces=F_all, textures=TexturesVertex(torch.ones_like(C_all)))
#     n = mesh_tmp.verts_normals_packed()
#     mesh_norm = Meshes(verts=V_all, faces=F_all, textures=TexturesVertex(n.unsqueeze(0)))
#     n_img, _ = renderer(mesh_norm)
#     normals = n_img[0, :, :, :3]

#     zbuf[~valid] = float('inf')
#     z_clamped = torch.clamp(zbuf, 0.2, 1.2)
#     depth_inv = 1 - (z_clamped - 0.2)
#     view = torch.tensor([0, 0, 1.0], device=device)
#     shade = (normals * view).sum(-1, keepdim=True).clamp(0, 1)
#     shaded = 0.5 * depth_inv.unsqueeze(-1) + 0.5 * shade
#     rgb_out = rgb * shaded

#     # --- Grayscale option ---
#     if grayscale:
#         gray = (0.299*rgb_out[...,0] + 0.587*rgb_out[...,1] + 0.114*rgb_out[...,2]).unsqueeze(-1)
#         gray = torch.clamp((gray - 0.3)*1.5 + 0.3, 0, 1)
#         rgb_out = gray.expand_as(rgb_out)

#     out = (rgb_out.clamp(0,1)*255).byte().cpu().numpy()
#     depth = zbuf.cpu().numpy()
#     return out, depth



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
        # Video mode: create temp directory using scene name and animation info
        scene_code = os.path.basename(output_folder)
        
        # Simple temp directory naming
        base_temp_dir = Path.cwd() / "temp_hand_images"
        base_temp_dir.mkdir(exist_ok=True)
        
        temp_dir = base_temp_dir / f"{scene_code}_{animation_name}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        print(f"Temporary directory for images: {temp_dir}")
        print(f"  Scene code: {scene_code}, Animation: {animation_name}")
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
        sequences_folder = os.path.join(anim_output_folder, "processed2")
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
                
                # Render hands using PyTorch3D (both hands in one pass)
                image, depth = render_hands_pytorch3d(camera_obj, hand_objects, render_shape=(args.height, args.width), grayscale=args.grayscale)
                
                # For separate mode, we need to render each hand separately
                # This is a simplified approach - in practice you might want to implement proper channel separation
                left_image = image.copy()
                right_image = image.copy()
                
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
        clip_length = args.clip_length
        stride = args.stride
        frame_skip = args.frame_skip
        fps = args.fps
        
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
            # Separate mode: stable rendering - render all frames first, then create videos
            print(f"\n--- STABLE SEPARATE MODE: Render all frames first, then create videos ---")
            
            # Create temp directories for left and right hand frames (separate folders)
            left_frames_temp_dir = temp_dir / "left_hand_frames"
            right_frames_temp_dir = temp_dir / "right_hand_frames"
            left_frames_temp_dir.mkdir(exist_ok=True)
            right_frames_temp_dir.mkdir(exist_ok=True)
            
            # Step 1: Smart frame rendering - only render frames needed for incomplete videos
            print(f"\n=== STEP 1: SMART FRAME RENDERING ===")
            
            frames_rendered_total = 0
            frames_skipped_total = 0
            
            # Check which videos are already complete
            incomplete_videos = []
            for video_idx, video_start in enumerate(video_start_frames):
                left_video_path = os.path.join(videos_output_path_left, f"{video_idx:05d}.mp4")
                right_video_path = os.path.join(videos_output_path_right, f"{video_idx:05d}.mp4")
                
                left_exists = os.path.exists(left_video_path) and os.path.getsize(left_video_path) > 0
                right_exists = os.path.exists(right_video_path) and os.path.getsize(right_video_path) > 0
                
                if not (left_exists and right_exists):
                    incomplete_videos.append((video_idx, video_start))
            
            print(f"Total videos: {len(video_start_frames)}")
            print(f"Complete videos: {len(video_start_frames) - len(incomplete_videos)}")
            print(f"Incomplete videos: {len(incomplete_videos)}")
            
            if not incomplete_videos:
                print("All videos are already complete! Skipping frame rendering.")
                frames_rendered_total = 0
                frames_skipped_total = 0
            else:
                # Calculate frames needed only for incomplete videos
                all_frames_needed = set()
                for video_idx, video_start in incomplete_videos:
                    video_end = video_start + (clip_length - 1) * frame_skip
                    for frame_num in range(video_start, video_end + 1, frame_skip):
                        all_frames_needed.add(frame_num)
                
                all_frames_sorted = sorted(list(all_frames_needed))
                print(f"Frames needed for incomplete videos: {len(all_frames_sorted)}")
                print(f"Frame range: {all_frames_sorted[0]} to {all_frames_sorted[-1]}")
                
                # Render only needed frames
                for frame_idx, frame_num in enumerate(all_frames_sorted):
                    scene.frame_set(frame_num)
                
                    # Frame paths using frame_num as identifier (stable naming)
                    left_frame_path = left_frames_temp_dir / f"frame_{frame_num:06d}.png"
                    right_frame_path = right_frames_temp_dir / f"frame_{frame_num:06d}.png"
                    
                    left_exists = left_frame_path.exists()
                    right_exists = right_frame_path.exists()
                    
                    if args.skip_existing and left_exists and right_exists:
                        frames_skipped_total += 1
                        continue
                    
                    # Render left and right hands SEPARATELY for better quality
                    frame_render_start = time.time()
                    if frame_idx % 10 == 0 or frame_idx == len(all_frames_sorted) - 1:
                        print(f"  [RENDER] Frame {frame_num} ({frame_idx + 1}/{len(all_frames_sorted)})...")
                    
                    import cv2
                    
                    # Render left hand separately
                    if not left_exists:
                        left_image, left_depth = render_hands_pytorch3d(
                            camera_obj, hand_objects,
                            render_shape=(args.height, args.width),
                            grayscale=args.grayscale,
                            separate_hands=True,
                            target_hand="left"
                        )
                        cv2.imwrite(str(left_frame_path), cv2.cvtColor(left_image, cv2.COLOR_RGB2BGR))
                    
                    # Render right hand separately
                    if not right_exists:
                        right_image, right_depth = render_hands_pytorch3d(
                            camera_obj, hand_objects,
                            render_shape=(args.height, args.width),
                            grayscale=args.grayscale,
                            separate_hands=True,
                            target_hand="right"
                        )
                        cv2.imwrite(str(right_frame_path), cv2.cvtColor(right_image, cv2.COLOR_RGB2BGR))
                    
                    frame_time = time.time() - frame_render_start
                    total_render_time += frame_time
                    frames_rendered_total += 1
                    
                    if frame_idx % 10 == 0 or frame_idx == len(all_frames_sorted) - 1:
                        progress = (frame_idx + 1) / len(all_frames_sorted) * 100.0
                        avg_time = total_render_time / frames_rendered_total if frames_rendered_total > 0 else 0
                        print(f"    Progress: {frame_idx + 1}/{len(all_frames_sorted)} ({progress:.1f}%) - {frame_time:.1f}s (avg: {avg_time:.1f}s)")
            
            print(f"\n✅ FRAME RENDERING COMPLETE: {frames_rendered_total} rendered, {frames_skipped_total} skipped")
            
            # Step 2: Create videos from rendered frames (only for incomplete videos)
            print(f"\n=== STEP 2: CREATING VIDEOS FROM FRAMES ===")
            
            if not incomplete_videos:
                print("All videos are already complete! Skipping video creation.")
                videos_completed = len(video_start_frames) * 2  # Both left and right
            else:
                for video_idx, start_frame_num in incomplete_videos:
                    video_end_frame = start_frame_num + (clip_length - 1) * frame_skip
                    frames_for_video = list(range(start_frame_num, video_end_frame + 1, frame_skip))

                    # Define output paths first
                    left_video_output_path = os.path.join(videos_output_path_left, f"{video_idx:05d}.mp4")
                    right_video_output_path = os.path.join(videos_output_path_right, f"{video_idx:05d}.mp4")
                    
                    # Delete existing videos first if in force mode
                    if not args.skip_existing:
                        for video_path in [left_video_output_path, right_video_output_path]:
                            if os.path.exists(video_path):
                                try:
                                    os.remove(video_path)
                                    print(f"    [DELETE] Removed existing: {video_path}")
                                except OSError as e:
                                    print(f"    [WARNING] Could not remove: {e}")
                    
                    # Check if videos need to be created (after deletion)
                    if args.skip_existing:
                        left_video_exists, left_needs_video_rendering = check_video_exists(video_idx, videos_output_path_left)
                        right_video_exists, right_needs_video_rendering = check_video_exists(video_idx, videos_output_path_right)
                    else:
                        # Force re-rendering mode: always render
                        left_needs_video_rendering = True
                        right_needs_video_rendering = True
                
                        if args.skip_existing and not left_needs_video_rendering and not right_needs_video_rendering:
                            print(f"[VIDEO {video_idx + 1}/{len(video_start_frames)}] SKIPPED: Both videos already exist")
                            videos_completed += 2
                            continue

                    print(f"\n[VIDEO {video_idx + 1}/{len(video_start_frames)}] Creating from frames {start_frame_num}..{video_end_frame}")
                    print(f"  Frames: {len(frames_for_video)} (expected: {clip_length})")
                    
                    # Sanity check: ensure we have exactly clip_length frames
                    if len(frames_for_video) != clip_length:
                        print(f"  ⚠️  WARNING: Expected {clip_length} frames but got {len(frames_for_video)}!")
                    
                    # Create symlinks with sequential numbering for ffmpeg (no separate video folders)
                    video_temp_left = temp_dir / f"left_hand_frames"
                    video_temp_right = temp_dir / f"right_hand_frames"
                    video_temp_left.mkdir(exist_ok=True)
                    video_temp_right.mkdir(exist_ok=True)
                
                # Create symlinks for this video's frames with sequential numbering
                print(f"  [SYMLINK] Creating {clip_length} sequential frame links...")
                for seq_idx, frame_num in enumerate(frames_for_video):
                    # Source frames (using frame_num as identifier)
                    left_source = left_frames_temp_dir / f"frame_{frame_num:06d}.png"
                    right_source = right_frames_temp_dir / f"frame_{frame_num:06d}.png"
                    
                    # Target symlinks (sequential numbering for ffmpeg)
                    left_link = video_temp_left / f"frame_{seq_idx + 1:04d}.png"
                    right_link = video_temp_right / f"frame_{seq_idx + 1:04d}.png"
                    
                    # Create symlinks (or copy if symlink fails)
                    if left_needs_video_rendering and left_source.exists():
                        try:
                            if left_link.exists():
                                left_link.unlink()
                            left_link.symlink_to(left_source.absolute())
                        except OSError:
                            # Fallback to copy if symlink not supported
                            import shutil
                            shutil.copy2(left_source, left_link)
                    
                    if right_needs_video_rendering and right_source.exists():
                        try:
                            if right_link.exists():
                                right_link.unlink()
                            right_link.symlink_to(right_source.absolute())
                        except OSError:
                            # Fallback to copy if symlink not supported
                            import shutil
                            shutil.copy2(right_source, right_link)
                
                print(f"  [VIDEO CREATION] Creating videos with {clip_length} frames each...")
                
                try:
                    import subprocess
                    
                    # Create left hand video
                    if left_needs_video_rendering:
                        left_rgb_cmd = [
                            'ffmpeg', '-y', 
                            '-start_number', '1',  # Start from frame_0001.png
                            '-framerate', str(fps),
                            '-i', str(video_temp_left / 'frame_%04d.png'),
                            '-frames:v', str(clip_length),  # MUST be exactly clip_length frames
                            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                            '-crf', '18', left_video_output_path
                        ]
                        print(f"    [FFMPEG] Creating left hand video: {clip_length} frames @ {fps} fps")
                        result = subprocess.run(left_rgb_cmd, check=True, capture_output=True, text=True)
                        if os.path.exists(left_video_output_path) and os.path.getsize(left_video_output_path) > 0:
                            print(f"    ✅ Created left hand video: {left_video_output_path}")
                        else:
                            print(f"    ❌ Left hand video file not created or empty: {left_video_output_path}")
                    
                    # Create right hand video
                    if right_needs_video_rendering:
                        right_rgb_cmd = [
                            'ffmpeg', '-y',
                            '-start_number', '1',  # Start from frame_0001.png
                            '-framerate', str(fps),
                            '-i', str(video_temp_right / 'frame_%04d.png'),
                            '-frames:v', str(clip_length),  # MUST be exactly clip_length frames
                            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                            '-crf', '18', right_video_output_path
                        ]
                        print(f"    [FFMPEG] Creating right hand video: {clip_length} frames @ {fps} fps")
                        result = subprocess.run(right_rgb_cmd, check=True, capture_output=True, text=True)
                        if os.path.exists(right_video_output_path) and os.path.getsize(right_video_output_path) > 0:
                            print(f"    ✅ Created right hand video: {right_video_output_path}")
                        else:
                            print(f"    ❌ Right hand video file not created or empty: {right_video_output_path}")
                    
                except subprocess.CalledProcessError as e:
                    print(f"    ❌ Failed to create videos: {e}")
                    stderr_output = e.stderr if isinstance(e.stderr, str) else e.stderr.decode() if e.stderr else "No error output"
                    print(f"    Error output: {stderr_output}")
                    print(f"    Frames preserved in: {video_temp_left} and {video_temp_right}")
                except FileNotFoundError:
                    print(f"    ❌ ffmpeg not found. Please install ffmpeg to create videos.")
                
                # Clean up symlinks for this video (not the original frames!)
                import shutil
                try:
                    shutil.rmtree(video_temp_left, ignore_errors=True)
                    shutil.rmtree(video_temp_right, ignore_errors=True)
                except:
                    pass
                
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
            
            # Keep frame directories for manual cleanup later
            print(f"\n📁 Frame directories preserved for manual cleanup:")
            print(f"   Left frames: {left_frames_temp_dir}")
            print(f"   Right frames: {right_frames_temp_dir}")
            print(f"   You can manually delete these directories when no longer needed.")
            
            print(f"\n[SEPARATE HAND VIDEOS] Completed: {videos_completed // 2} video pairs ({videos_completed} total videos)")
            print(f"Total frames rendered: {frames_rendered_total}, Total frames skipped: {frames_skipped_total}")
        else:
            # Simple mode: smart frame rendering - only render frames needed for incomplete videos
            print(f"\n--- SIMPLE MODE: Smart frame rendering ---")
            
            # Check which videos are already complete
            incomplete_videos = []
            for video_idx, video_start in enumerate(video_start_frames):
                video_output_path = os.path.join(videos_output_path, f"{video_idx:05d}.mp4")
                video_exists = os.path.exists(video_output_path) and os.path.getsize(video_output_path) > 0
                
                if not video_exists:
                    incomplete_videos.append((video_idx, video_start))
            
            print(f"Total videos: {len(video_start_frames)}")
            print(f"Complete videos: {len(video_start_frames) - len(incomplete_videos)}")
            print(f"Incomplete videos: {len(incomplete_videos)}")
            
            if not incomplete_videos:
                print("All videos are already complete! Skipping frame rendering.")
                frames_rendered_total = 0
                frames_skipped_total = 0
            else:
                # Calculate frames needed only for incomplete videos
                all_frames_needed = set()
                for video_idx, video_start in incomplete_videos:
                    video_end = video_start + (clip_length - 1) * frame_skip
                    for frame_num in range(video_start, video_end + 1, frame_skip):
                        all_frames_needed.add(frame_num)
                
                all_frames_sorted = sorted(list(all_frames_needed))
                print(f"Frames needed for incomplete videos: {len(all_frames_sorted)}")
                print(f"Frame range: {all_frames_sorted[0]} to {all_frames_sorted[-1]}")
                
                # Create temp directory for frames
                frames_temp_dir = temp_dir / "all_frames"
                frames_temp_dir.mkdir(exist_ok=True)
                
                frames_rendered_total = 0
                frames_skipped_total = 0
                
                for frame_idx, frame_num in enumerate(all_frames_sorted):
                    scene.frame_set(frame_num)
                    
                    # Check if frame already exists
                    frame_path = frames_temp_dir / f"frame_{frame_num:06d}.png"
                    
                    if args.skip_existing and frame_path.exists():
                        frames_skipped_total += 1
                        if frame_idx % 50 == 0:
                            print(f"  Skipped frame {frame_num} ({frame_idx + 1}/{len(all_frames_sorted)})")
                        continue
                
                    # Render frame
                    try:
                        image, depth = render_hands_pytorch3d(camera_obj, hand_objects, render_shape=(args.height, args.width), grayscale=args.grayscale)
                        
                        if image is None or image.size == 0:
                            print(f"    ⚠️  Warning: Empty image for frame {frame_num}")
                            continue
                        
                        import cv2
                        success = cv2.imwrite(str(frame_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                        if not success:
                            print(f"    ❌ Failed to save frame {frame_num}")
                            continue
                            
                        frames_rendered_total += 1
                        if frame_idx % 50 == 0:
                            print(f"  Rendered frame {frame_num} ({frame_idx + 1}/{len(all_frames_sorted)})")
                            
                    except Exception as e:
                        print(f"    ❌ Error rendering frame {frame_num}: {e}")
                        continue
            
                print(f"Step 1 Complete: Rendered {frames_rendered_total} frames, Skipped {frames_skipped_total} frames")
            
            # Step 2: Create videos from rendered frames (only for incomplete videos)
            print(f"Step 2: Creating videos from frames")
            
            if not incomplete_videos:
                print("All videos are already complete! Skipping video creation.")
            else:
                for video_idx, start_frame_num in incomplete_videos:
                    video_end_frame = start_frame_num + (clip_length - 1) * frame_skip
                    frames_for_video = list(range(start_frame_num, video_end_frame + 1, frame_skip))
                    
                    # Define output path
                    video_output_path = os.path.join(videos_output_path, f"{video_idx:05d}.mp4")
                    
                    # Check if video already exists
                    if args.skip_existing and os.path.exists(video_output_path):
                        print(f"  [VIDEO {video_idx + 1}] SKIPPED: Video already exists")
                        continue
                
                    print(f"  [VIDEO {video_idx + 1}] Creating video from frames {start_frame_num}..{video_end_frame}")
                    
                    # Collect frame files for this video
                    video_frame_files = []
                    missing_frames = []
                    for frame_num in frames_for_video:
                        frame_path = frames_temp_dir / f"frame_{frame_num:06d}.png"
                        if frame_path.exists():
                            video_frame_files.append(frame_path)
                        else:
                            missing_frames.append(frame_num)
                            print(f"    ⚠️  Missing frame: {frame_path}")
                    
                    if not video_frame_files:
                        print(f"    ❌ No frame files found for video {video_idx}")
                        continue
                
                    # Check if we have the expected number of frames
                    expected_frames = len(frames_for_video)
                    actual_frames = len(video_frame_files)
                    if actual_frames != expected_frames:
                        print(f"    ⚠️  WARNING: Expected {expected_frames} frames but got {actual_frames}")
                        print(f"    Missing frames: {missing_frames}")
                        if actual_frames < 40:  # Too few frames, skip this video
                            print(f"    ❌ Too few frames ({actual_frames}), skipping video {video_idx}")
                            continue
                    
                    # Create video from frame files
                    try:
                        import subprocess
                        
                        # Create video using direct frame input (more reliable than concat)
                        # Copy frames to sequential naming for ffmpeg
                        video_temp_dir = frames_temp_dir / f"video_{video_idx}_temp"
                        video_temp_dir.mkdir(exist_ok=True)
                        
                        # Copy frames with sequential naming
                        for seq_idx, frame_file in enumerate(video_frame_files):
                            if frame_file.exists():
                                import shutil
                                target_file = video_temp_dir / f"frame_{seq_idx + 1:04d}.png"
                                shutil.copy2(frame_file, target_file)
                        
                        rgb_cmd = [
                            'ffmpeg', '-y',
                            '-start_number', '1',
                            '-framerate', str(fps),
                            '-i', str(video_temp_dir / 'frame_%04d.png'),
                            '-frames:v', str(len(video_frame_files)),  # Exact frame count
                            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                            '-crf', '18', '-preset', 'medium',
                            video_output_path
                        ]
                        
                        print(f"    [FFMPEG] Creating video with {len(video_frame_files)} frames")
                        result = subprocess.run(rgb_cmd, check=True, capture_output=True, text=True, timeout=300)
                        
                        # Verify output file
                        if os.path.exists(video_output_path):
                            file_size = os.path.getsize(video_output_path)
                            if file_size > 0:
                                print(f"    ✅ Created video: {video_output_path} ({file_size:,} bytes)")
                            else:
                                print(f"    ❌ Video file is empty: {video_output_path}")
                        else:
                            print(f"    ❌ Video file not created: {video_output_path}")
                        
                        # Clean up temp directory
                        import shutil
                        shutil.rmtree(video_temp_dir, ignore_errors=True)
                        
                    except subprocess.TimeoutExpired:
                        print(f"    ❌ FFmpeg timed out after 5 minutes")
                    except subprocess.CalledProcessError as e:
                        print(f"    ❌ Failed to create video: {e}")
                        stderr_output = e.stderr if isinstance(e.stderr, str) else e.stderr.decode() if e.stderr else "No error output"
                        print(f"    Command output: {stderr_output}")
                    except FileNotFoundError:
                        print(f"    ❌ ffmpeg not found. Please install ffmpeg to create videos.")
                    except Exception as e:
                        print(f"    ❌ Unexpected error during video creation: {e}")
            
            print(f"Step 2 Complete: Created {len(video_start_frames)} videos")
            
            # Keep frame directory for manual cleanup later
            print(f"Step 3: Frame directory preserved for manual cleanup:")
            print(f"  Frame directory: {frames_temp_dir}")
            print(f"  You can manually delete this directory when no longer needed.")
            
            print(f"Simple mode completed successfully!")
            
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
        
        # Keep temporary directory for manual cleanup later
        if temp_dir is not None and temp_dir.exists():
            print(f"📁 Temporary directory preserved for manual cleanup: {temp_dir}")
            print(f"   You can manually delete this directory when no longer needed.")

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
    print(f"Rendering specific animation by index: {animation_index}")
elif args.animation_name is not None:
    # Find animation index by name
    animation_names_list = list(animation_sets.keys())
    
    # Try exact match first
    if args.animation_name in animation_names_list:
        animation_index = animation_names_list.index(args.animation_name)
        animations_to_render = [(animation_index, args.animation_name)]
        print(f"Rendering specific animation by name: {args.animation_name} (index: {animation_index})")
    else:
        # Try partial match (without .pkl extension)
        animation_name_no_ext = args.animation_name.replace('.pkl', '')
        found = False
        for i, name in enumerate(animation_names_list):
            if animation_name_no_ext in name or name.replace('.pkl', '') == animation_name_no_ext:
                animation_index = i
                animations_to_render = [(i, name)]
                print(f"Rendering specific animation by name: {name} (index: {i})")
                print(f"  (matched with input: {args.animation_name})")
                found = True
                break
        
        if not found:
            print(f"Error: Animation name '{args.animation_name}' not found!")
            print(f"Available animations:")
            for i, name in enumerate(animation_names_list):
                print(f"  {i}: {name}")
            sys.exit(1)
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
        clip_length = args.clip_length
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