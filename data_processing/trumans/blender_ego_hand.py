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


def render_hands_pytorch3d(camera_obj, hand_objects, render_shape=(480, 720)):
    """
    Render CC_Hand_L/CC_Hand_R with Soft Phong shading via PyTorch3D.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Intrinsics (pixels) ---
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

    # Gentle lighting so it never goes black
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

        # Per-vertex color (vibrant green=L, deep red=R) - matching the reference image
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

    out = (final_rgb.clamp(0, 1) * 255.0).byte().cpu().numpy()
    return out, final_z.cpu().numpy()



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
    sequences_folder = os.path.join(anim_output_folder, "sequences")
    videos_output_path = os.path.join(sequences_folder, "videos_hands")
    os.makedirs(videos_output_path, exist_ok=True)

    print(f"Rendering animation {animation_index}: {animation_name}")
    print(f"  Videos: {videos_output_path}")

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

    # Frame range and video sequence parameters
    scene = bpy.context.scene
    render_start_frame = scene.frame_start if start_frame is None else start_frame
    render_end_frame   = scene.frame_end   if end_frame   is None else end_frame
    
    # Video sequence parameters (same as static.py)
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
    videos_completed = 0
    total_render_time = 0.0

    for video_idx, start_frame_num in enumerate(video_start_frames):
        video_end_frame = start_frame_num + (clip_length - 1) * frame_skip
        frames_to_render = list(range(start_frame_num, video_end_frame + 1, frame_skip))

        print(f"\n========== VIDEO {video_idx + 1}/{len(video_start_frames)} ==========")
        print(f"Frames: {start_frame_num}..{video_end_frame} (step {frame_skip}) -> {len(frames_to_render)} frames")

        # Create temp directory for this video
        video_temp_dir = os.path.join(videos_output_path, f"temp_{video_idx:05d}")
        os.makedirs(video_temp_dir, exist_ok=True)
        
        # Render all frames for this video
        video_start_time = time.time()
        frames_rendered = 0
        
        for frame_idx, frame_num in enumerate(frames_to_render):
            scene.frame_set(frame_num)

            # Render with PyTorch3D
            frame_render_start = time.time()
            print(f"[VIDEO {video_idx + 1}] Rendering frame {frame_num} ({frame_idx + 1}/{len(frames_to_render)})...")
            
            # Render hands using PyTorch3D
            image, depth = render_hands_pytorch3d(camera_obj, hand_objects, render_shape=(480, 720))
            
            # Save image to temp directory
            image_path = os.path.join(video_temp_dir, f"{frame_idx:04d}.png")
            import cv2
            cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            
            frame_time = time.time() - frame_render_start
            total_render_time += frame_time
            frames_rendered += 1

            if frame_idx % 10 == 0 or frame_idx == len(frames_to_render) - 1:
                progress = (frame_idx + 1) / len(frames_to_render) * 100.0
                avg_frame_time = total_render_time / (videos_completed * clip_length + frames_rendered)
                print(f"  Frame {frame_idx + 1}/{len(frames_to_render)} ({progress:.1f}%) - {frame_time:.1f}s")
        
        # Convert frames to video using ffmpeg
        video_output_path = os.path.join(videos_output_path, f"{video_idx:05d}.mp4")
        
        try:
            import subprocess
            
            # Create RGB video
            rgb_cmd = [
                'ffmpeg', '-y', '-framerate', str(fps),
                '-pattern_type', 'glob', '-i', os.path.join(video_temp_dir, '*.png'),
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                '-crf', '18', video_output_path
            ]
            subprocess.run(rgb_cmd, check=True, capture_output=True)
            
            print(f"  ✅ Created video: {os.path.basename(video_output_path)}")
            
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Failed to create video: {e}")
            print(f"  Command output: {e.stderr.decode()}")
        except FileNotFoundError:
            print(f"  ❌ ffmpeg not found. Please install ffmpeg to create videos.")
            print(f"  Rendered frames saved in: {video_temp_dir}")
        
        # Clean up temporary files (optional - uncomment if you want to save disk space)
        # try:
        #     import shutil
        #     shutil.rmtree(video_temp_dir)
        # except Exception as e:
        #     print(f"  Warning: Could not clean up temp files: {e}")
        
        videos_completed += 1
        
        # Overall progress
        total_elapsed = time.time() - start_time
        if videos_completed > 1:
            avg_video_time = total_elapsed / videos_completed
            remaining_videos = len(video_start_frames) - videos_completed
            eta = remaining_videos * avg_video_time
            print(f"  📊 Progress: {videos_completed}/{len(video_start_frames)} videos")
            print(f"  ⏱️  Video time: {time.time() - video_start_time:.1f}s | Avg: {avg_video_time:.1f}s")
            print(f"  🎯 ETA: {eta/60:.1f} min | Elapsed: {total_elapsed/60:.1f} min")

    total_time = time.time() - start_time
    avg_fps = (videos_completed * clip_length) / total_time if total_time > 0 else 0

    print("\n" + "="*50)
    print(f"COMPLETED: Animation {animation_index} ({animation_name})")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Videos created: {videos_completed}/{len(video_start_frames)}")
    print(f"Frame step: {frame_skip} | Stride: {stride} | Effective stride: {effective_stride}")
    print(f"Average throughput: {avg_fps:.2f} fps")
    print("="*50)

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
        image, depth = render_hands_pytorch3d(camera_obj, hand_objects, render_shape=(480, 720))
        print("✓ PyTorch3D setup verified - ready for rendering")
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
print("    └── sequences/")
print("        └── videos_hands/         # Hand-only video sequences (MP4)")
if failed_animations:
    print(f"\nFAILED ANIMATIONS ({len(failed_animations)}):")
    for anim_idx, anim_name, error_type in failed_animations:
        print(f"  Animation {anim_idx} ({anim_name}): {error_type}")
    print(f"Check error log for details: {error_log_file}")
else:
    print("\nAll animations completed successfully!")
print("="*60) 