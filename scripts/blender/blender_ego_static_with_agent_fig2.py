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
if not argv:
    # Default: use SEQ1_FRAMES from blender_teaser.py
    SEQ0_INDEX = 0
    SEQ0_FRAMES = [18, 54, 102, 117]
    frames_str = ",".join(map(str, SEQ0_FRAMES))
    argv = [
        "--frames", frames_str,
        "--save-path", "/home/byungjunkim/dwm_teaser/output",
        "--animation_index", SEQ0_INDEX,
    ]

parser = argparse.ArgumentParser()
parser.add_argument("--start_frame", type=int, default=None)
parser.add_argument("--end_frame", type=int, default=None)
parser.add_argument("--animation_index", type=int, default=None, help="Specific animation index (else: all)")
parser.add_argument("--samples", type=int, default=128, help="Cycles samples")
parser.add_argument("--save-path", type=str, default="/home/byungjun/workspace/trumans_ego/ego_render_new",
                    help="Root output dir")
parser.add_argument("--skip-existing", action="store_true", default=True, help="Skip frames that already exist")
parser.add_argument("--no-skip-existing", action="store_true", help="Disable skipping existing frames")
parser.add_argument("--frame-skip", type=int, default=3, help="Render every Nth frame")
parser.add_argument("--stride", type=int, default=25, help="Stride for video sequences (default: 25)")
parser.add_argument("--frames", type=str, help="Comma-separated frame numbers to render (e.g., '2110,2130,2150,2170,2190')")
parser.add_argument("--fov", type=float, default=90.0, help="Camera FOV in degrees (perspective)")
parser.add_argument("--width", type=int, default=1500, help="Render width in pixels (default: 720)")
parser.add_argument("--height", type=int, default=1500, help="Render height in pixels (default: 480)")
args = parser.parse_args(argv)
if args.no_skip_existing:
    args.skip_existing = False

# Parse frame list if provided
frames_list = []
if args.frames:
    try:
        frames_list = [int(token.strip()) for token in args.frames.split(",") if token.strip()]
        frames_list = sorted(set(frames_list))  # Remove duplicates and sort
    except ValueError:
        print(f"Error: Invalid frame list supplied to --frames: '{args.frames}'")
        sys.exit(1)
    if not frames_list:
        print("Error: --frames was provided but no valid frame numbers were parsed.")
        sys.exit(1)
args.frames_list = frames_list

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
PHONE_OBJECT_NAME = "rigid_03_root_phone_01"
PHONE_REFERENCE_FRAME = 850

# Sequence frame definitions (from blender_teaser.py)
SEQ0_INDEX = 0
SEQ1_INDEX = 1
SEQ0_FRAMES = [600, 630, 660, 690]
SEQ1_FRAMES = [850, 868, 901, 919]

# Color ramp definitions (from blender_teaser.py)
WARM_START = (1.0, 0.75, 0.5, 0.15)
WARM_END = (0.65, 0.2, 0.05, 0.95)
COOL_START = (0.62, 0.88, 0.92, 0.35)
COOL_END = (0.08, 0.20, 0.65, 1.0)

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

def find_object_by_name(name):
    """Find object by name, with fallback to partial match."""
    obj = bpy.data.objects.get(name)
    if obj:
        return obj
    candidates = [o for o in bpy.data.objects if name in o.name]
    if candidates:
        print(f"Warning: Exact object '{name}' not found. Using '{candidates[0].name}' instead.")
        return candidates[0]
    return None

def capture_object_world_matrix(obj):
    """Capture world matrix of an object."""
    if obj is None:
        return None
    return obj.matrix_world.copy()

def apply_world_matrix_to_object(obj, matrix, keyframe=None, constant=False, clear_animation=False):
    """Apply world matrix to an object."""
    if obj is None or matrix is None:
        return
    if clear_animation and obj.animation_data:
        obj.animation_data_clear()
    parent = obj.parent
    if parent:
        obj.matrix_basis = parent.matrix_world.inverted() @ matrix
    else:
        obj.matrix_world = matrix
    if keyframe is not None:
        obj.keyframe_insert(data_path="location", frame=keyframe)
        obj.keyframe_insert(data_path="rotation_euler", frame=keyframe)
        obj.keyframe_insert(data_path="scale", frame=keyframe)
        if constant and obj.animation_data and obj.animation_data.action:
            for fcurve in obj.animation_data.action.fcurves:
                if fcurve.data_path in {"location", "rotation_euler", "scale"}:
                    for kp in fcurve.keyframe_points:
                        if kp.co.x == keyframe:
                            kp.interpolation = 'CONSTANT'

def rgb_to_hsv(r, g, b):
    """Convert RGB to HSV."""
    max_val = max(r, g, b)
    min_val = min(r, g, b)
    delta = max_val - min_val
    
    if delta == 0:
        h = 0
    elif max_val == r:
        h = 60 * (((g - b) / delta) % 6)
    elif max_val == g:
        h = 60 * ((b - r) / delta + 2)
    else:  # max_val == b
        h = 60 * ((r - g) / delta + 4)
    
    s = delta / max_val if max_val > 0 else 0
    v = max_val
    
    return (h, s, v)

def hsv_to_rgb(h, s, v):
    """Convert HSV to RGB."""
    c = v * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = v - c
    
    if 0 <= h < 60:
        r, g, b = c, x, 0
    elif 60 <= h < 120:
        r, g, b = x, c, 0
    elif 120 <= h < 180:
        r, g, b = 0, c, x
    elif 180 <= h < 240:
        r, g, b = 0, x, c
    elif 240 <= h < 300:
        r, g, b = x, 0, c
    else:  # 300 <= h < 360
        r, g, b = c, 0, x
    
    return (r + m, g + m, b + m)

def build_color_ramp(start_rgba, end_rgba, total):
    """Build a color ramp from start to end RGBA values using HSV interpolation for better visual distinction."""
    if total <= 1:
        return [end_rgba]
    
    # For small palettes (4 frames), use discrete colors for maximum distinction
    if total == 4:
        # Determine if warm or cool palette based on start/end colors
        start_hsv = rgb_to_hsv(start_rgba[0], start_rgba[1], start_rgba[2])
        end_hsv = rgb_to_hsv(end_rgba[0], end_rgba[1], end_rgba[2])
        
        # Check if warm (hue around 0-60 or 300-360) or cool (hue around 180-240)
        start_hue = start_hsv[0]
        end_hue = end_hsv[0]
        
        # Normalize hue to 0-360
        if start_hue < 0:
            start_hue += 360
        if end_hue < 0:
            end_hue += 360
        
        ramp = []
        for i in range(total):
            t = i / (total - 1)
            
            # Interpolate in HSV space for smoother hue transitions
            start_h, start_s, start_v = rgb_to_hsv(start_rgba[0], start_rgba[1], start_rgba[2])
            end_h, end_s, end_v = rgb_to_hsv(end_rgba[0], end_rgba[1], end_rgba[2])
            
            # Handle hue wrap-around (e.g., 350 to 10 degrees)
            if abs(end_h - start_h) > 180:
                if end_h > start_h:
                    start_h += 360
                else:
                    end_h += 360
            
            h = start_h + (end_h - start_h) * t
            s = start_s + (end_s - start_s) * t
            v = start_v + (end_v - start_v) * t
            
            # Normalize hue back to 0-360
            h = h % 360
            
            # Interpolate alpha linearly
            alpha = start_rgba[3] + (end_rgba[3] - start_rgba[3]) * t
            
            # Convert back to RGB
            r, g, b = hsv_to_rgb(h, s, v)
            ramp.append((r, g, b, alpha))
        
        return ramp
    
    # For larger palettes, use HSV interpolation
    ramp = []
    for i in range(total):
        t = i / (total - 1)
        
        # Interpolate in HSV space for better visual distinction
        start_h, start_s, start_v = rgb_to_hsv(start_rgba[0], start_rgba[1], start_rgba[2])
        end_h, end_s, end_v = rgb_to_hsv(end_rgba[0], end_rgba[1], end_rgba[2])
        
        # Handle hue wrap-around
        if abs(end_h - start_h) > 180:
            if end_h > start_h:
                start_h += 360
            else:
                end_h += 360
        
        h = start_h + (end_h - start_h) * t
        s = start_s + (end_s - start_s) * t
        v = start_v + (end_v - start_v) * t
        
        # Normalize hue back to 0-360
        h = h % 360
        
        # Interpolate alpha linearly
        alpha = start_rgba[3] + (end_rgba[3] - start_rgba[3]) * t
        
        # Convert back to RGB
        r, g, b = hsv_to_rgb(h, s, v)
        ramp.append((r, g, b, alpha))
    
    return ramp

def slugify_name(name):
    """Convert name to slug format."""
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name) if name else "default"

POSE_OBJECT_PREFIX = "PoseSnapshot"
POSE_MATERIAL_PREFIX = "PoseSnapshotMaterial"

def ensure_pose_collection(collection_name):
    """Ensure pose collection exists (from blender_teaser.py)."""
    collection = bpy.data.collections.get(collection_name)
    scene_collection = bpy.context.scene.collection
    if collection is None:
        collection = bpy.data.collections.new(collection_name)
        scene_collection.children.link(collection)
    elif collection not in scene_collection.children:
        scene_collection.children.link(collection)
    return collection

def clear_pose_snapshots(collection):
    """Clear all pose snapshots from collection (from blender_teaser.py)."""
    if collection is None:
        return
    for obj in list(collection.objects):
        mesh_data = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if mesh_data and mesh_data.users == 0:
            bpy.data.meshes.remove(mesh_data)
    for material in list(bpy.data.materials):
        if material.name.startswith(POSE_MATERIAL_PREFIX) and material.users == 0:
            bpy.data.materials.remove(material)

def create_pose_material(index, total, frame_number, animation_tag=None, color_rgba=None):
    """Create material for pose snapshot (from blender_teaser.py)."""
    suffix = f"{slugify_name(animation_tag)}_{frame_number}" if animation_tag else str(frame_number)
    material_name = f"{POSE_MATERIAL_PREFIX}_{suffix}"
    material = bpy.data.materials.get(material_name)
    if material is None:
        material = bpy.data.materials.new(material_name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    for node in list(nodes):
        nodes.remove(node)
    output_node = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    links.new(bsdf.outputs["BSDF"], output_node.inputs["Surface"])

    if color_rgba is not None:
        base_color = (color_rgba[0], color_rgba[1], color_rgba[2], 1.0)
    else:
        t = index / max(1, total - 1)
        color_strength = 0.9 - 0.4 * t
        base_color = (1.0, color_strength, color_strength, 1.0)

    # Alpha increases over time with smoother, more balanced progression
    # All hands stay visible, with gradual increase in opacity
    start_alpha = 0.4
    end_alpha = 0.9
    t = index / max(1, total - 1)
    # Apply mild easing (t^1.2) for smoother perception
    alpha = start_alpha + (end_alpha - start_alpha) * (t ** 1.2)

    bsdf.inputs["Base Color"].default_value = base_color
    bsdf.inputs["Alpha"].default_value = alpha
    bsdf.inputs["Roughness"].default_value = 0.6

    # Enable transparency blending
    material.blend_method = 'HASHED'
    material.shadow_method = 'HASHED'
    material.use_backface_culling = False
    return material

def create_body_transparent_gray_material(index, total, frame_number, animation_tag=None):
    """Create transparent gray material for body snapshot."""
    suffix = f"{slugify_name(animation_tag)}_{frame_number}" if animation_tag else str(frame_number)
    material_name = f"{POSE_MATERIAL_PREFIX}_Body_Gray_{suffix}"
    material = bpy.data.materials.get(material_name)
    if material is None:
        material = bpy.data.materials.new(material_name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    for node in list(nodes):
        nodes.remove(node)
    output_node = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    links.new(bsdf.outputs["BSDF"], output_node.inputs["Surface"])

    # Transparent gray color
    base_color = (0.5, 0.5, 0.5, 1.0)  # Gray
    alpha = 0.6  # Semi-transparent

    bsdf.inputs["Base Color"].default_value = base_color
    bsdf.inputs["Alpha"].default_value = alpha
    bsdf.inputs["Roughness"].default_value = 0.6

    # Enable transparency blending
    material.blend_method = 'HASHED'
    material.shadow_method = 'HASHED'
    material.use_backface_culling = False
    return material

def create_frustum_material_with_color(index, total, frame_number, color_rgba, animation_tag=None):
    """Create frustum material with color gradation from color ramp."""
    suffix = f"{slugify_name(animation_tag)}_{frame_number}" if animation_tag else str(frame_number)
    material_name = f"{POSE_MATERIAL_PREFIX}_Frustum_{suffix}"
    material = bpy.data.materials.get(material_name)
    if material is None:
        material = bpy.data.materials.new(material_name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    
    # Remove existing nodes
    for n in list(nodes):
        nodes.remove(n)
    
    out_node = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    color_node = nodes.new("ShaderNodeRGB")
    
    # Use color from color ramp almost as-is, with minimal processing to preserve distinction
    if color_rgba is not None:
        r, g, b = color_rgba[0], color_rgba[1], color_rgba[2]
        
        # Small brightness scaling for emission visibility (keep colors distinct)
        brightness_scale = 1.1  # Slight boost for visibility, but preserves color differences
        r = min(1.0, r * brightness_scale)
        g = min(1.0, g * brightness_scale)
        b = min(1.0, b * brightness_scale)
        
        frustum_color = (r, g, b, 1.0)
    else:
        # Default cyan/blue if no color ramp
        frustum_color = (0.2, 0.8, 1.0, 1.0)
    
    color_node.outputs[0].default_value = frustum_color
    # Keep emission strength constant to avoid hiding color differences
    emission.inputs["Strength"].default_value = 3.0
    
    links.new(color_node.outputs[0], emission.inputs["Color"])
    links.new(emission.outputs[0], out_node.inputs["Surface"])
    
    return material

def create_hand_material_with_alpha(base_material, index, total, frame_num, hand_name):
    """Create hand material with alpha that increases over time (transparent => opaque)."""
    material_name = f"{POSE_MATERIAL_PREFIX}_Hand_{hand_name}_{frame_num}"
    material = bpy.data.materials.get(material_name)
    if material is None:
        material = bpy.data.materials.new(material_name)
    
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    
    # Clear existing nodes
    for node in list(nodes):
        nodes.remove(node)
    
    output_node = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    links.new(bsdf.outputs["BSDF"], output_node.inputs["Surface"])
    
    # Get base color from original hand material
    base_color = (0.5, 0.8, 0.5, 1.0)  # Default green for left hand
    if hand_name == "Hand_R":
        base_color = (0.8, 0.4, 0.4, 1.0)  # Default red for right hand
    
    if base_material and base_material.use_nodes:
        base_bsdf = base_material.node_tree.nodes.get("Principled BSDF")
        if base_bsdf and "Base Color" in base_bsdf.inputs:
            base_color_value = base_bsdf.inputs["Base Color"].default_value
            base_color = (base_color_value[0], base_color_value[1], base_color_value[2], 1.0)
    
    # Alpha increases over time with smoother, more balanced progression
    # All hands stay visible, with gradual increase in opacity
    start_alpha = 0.4
    end_alpha = 0.9
    t = index / max(1, total - 1)
    # Apply mild easing (t^1.2) for smoother perception
    alpha = start_alpha + (end_alpha - start_alpha) * (t ** 1.2)
    
    bsdf.inputs["Base Color"].default_value = base_color
    bsdf.inputs["Alpha"].default_value = alpha
    bsdf.inputs["Metallic"].default_value = 0.0
    bsdf.inputs["Roughness"].default_value = 0.3
    if "Specular" in bsdf.inputs:
        bsdf.inputs["Specular"].default_value = 0.8
    elif "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.8
    if "IOR" in bsdf.inputs:
        bsdf.inputs["IOR"].default_value = 1.45
    
    # Enable transparency blending
    material.blend_method = 'HASHED'
    material.shadow_method = 'HASHED'
    material.use_backface_culling = False
    
    return material

def create_pose_snapshots(base_mesh_obj, frames, collection_name, animation_tag=None, color_ramp=None):
    """Create pose snapshots for multiple frames (from blender_teaser.py)."""
    if base_mesh_obj is None:
        print("Pose snapshot skipped: No base mesh object provided.")
        return False
    if not frames:
        print("Pose snapshot skipped: No frames provided.")
        return False

    scene = bpy.context.scene
    original_frame = scene.frame_current
    collection = ensure_pose_collection(collection_name)
    clear_pose_snapshots(collection)

    total = len(frames)
    for idx, frame in enumerate(frames):
        scene.frame_set(frame)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        eval_obj = base_mesh_obj.evaluated_get(depsgraph)
        try:
            mesh_data = bpy.data.meshes.new_from_object(
                eval_obj,
                preserve_all_data_layers=True,
                depsgraph=depsgraph
            )
        except RuntimeError as err:
            print(f"Failed to create snapshot mesh at frame {frame}: {err}")
            continue

        mesh_data.name = f"{POSE_OBJECT_PREFIX}_Mesh_{frame}"
        name_suffix = f"{slugify_name(animation_tag)}_{frame}" if animation_tag else str(frame)
        snapshot_obj = bpy.data.objects.new(f"{POSE_OBJECT_PREFIX}_{name_suffix}", mesh_data)
        snapshot_obj.matrix_world = base_mesh_obj.matrix_world.copy()
        collection.objects.link(snapshot_obj)

        color_override = None
        if color_ramp and idx < len(color_ramp):
            color_override = color_ramp[idx]
        material = create_pose_material(idx, total, frame, animation_tag, color_override)
        snapshot_obj.data.materials.clear()
        snapshot_obj.data.materials.append(material)
        snapshot_obj.hide_render = False
        snapshot_obj.hide_viewport = False

        print(f"Created pose snapshot for frame {frame} -> {snapshot_obj.name} (material: {material.name})")

    scene.frame_set(original_frame)
    print(f"Pose snapshots stored in collection '{collection.name}'.")
    return True

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

def show_actor_in_rendering(hide_clothes=True):
    """Show the actor in rendering."""
    if armature_obj:
        # Show the armature itself
        armature_obj.hide_render = False
        print(f"Shown armature '{armature_name}' in rendering")
        
        # Show all children of the armature (except body if hide_clothes, but keep hand objects visible)
        for child in armature_obj.children:
            if hide_clothes and child.name.startswith("CC_Base_Body"):
                continue
            # Keep hand objects visible for rendering
            if child.name in ("CC_Hand_L", "CC_Hand_R"):
                child.hide_render = False
                child.hide_viewport = False
                print(f"Shown hand object '{child.name}' in rendering")
            else:
                child.hide_render = True
                print(f"Hidden child '{child.name}' in rendering")

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


def disable_animations_except_camera(camera_obj, armature_obj):
    """
    1) Save camera parenting and clear it (KEEP transform).
    2) Save each object's current action, then detach (freeze) for all objects except the camera and armature.
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

    # --- 2) Detach actions of all non-camera objects except armature ---
    for obj in bpy.data.objects:
        if obj == camera_obj or obj == armature_obj:
            continue
        if obj.animation_data:
            # record current action (could be None)
            _original_animation_state["actions"][obj.name] = obj.animation_data.action
            # keep the action from being purged
            if obj.animation_data.action:
                obj.animation_data.action.use_fake_user = True
            # detach to freeze at current pose
            obj.animation_data.action = None

    # Depsgraph refresh helps ensure "frozen" state is used during rendering
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


def adjust_head_bone_rotation(frames, delta_pitch=0.2, delta_yaw=0.1):
    """Adjust CC_Base_Head quaternion components across specified frames."""
    if armature_obj is None:
        return
    head_bone = armature_obj.pose.bones.get("CC_Base_Head")
    if head_bone is None:
        print("Warning: 'CC_Base_Head' bone not found; skipping head adjustment.")
        return
    scene = bpy.context.scene
    previous_frame = scene.frame_current
    original_mode = head_bone.rotation_mode
    head_bone.rotation_mode = 'QUATERNION'
    for frame in frames:
        head_bone.keyframe_delete(data_path="rotation_quaternion", frame=frame)
    for frame in frames:
        scene.frame_set(frame)
        adjusted_quat = head_bone.rotation_quaternion.copy()
        adjusted_quat[1] -= delta_pitch
        adjusted_quat[2] -= delta_yaw
        head_bone.rotation_quaternion = adjusted_quat
        head_bone.keyframe_insert(data_path="rotation_quaternion", frame=frame)
    if original_mode != 'QUATERNION':
        head_bone.rotation_mode = original_mode
    scene.frame_set(previous_frame)

def create_camera_frustum_mesh(camera_obj, frustum_length=0.8, bevel_depth=0.005, name=None):
    """Create a pyramid-shaped camera frustum as a Curve (renderable lines, no diagonal)."""
    cam = camera_obj.data
    scene = bpy.context.scene
    
    # Get camera parameters
    fov_rad = cam.angle
    width = scene.render.resolution_x
    height = scene.render.resolution_y
    aspect = width / height
    
    # Calculate frustum dimensions
    near = 0.01  # Very small near plane
    far = frustum_length
    
    # Calculate frustum corners at far plane only (pyramid shape)
    far_height = 2.0 * math.tan(fov_rad / 2.0) * far
    far_width = far_height * aspect
    
    # In Blender camera space: -Z is forward, Y is up
    # Camera position (apex of pyramid)
    apex = Vector((0, 0, 0))
    
    # Far plane corners (base of pyramid)
    ftl = Vector((-far_width/2, far_height/2, -far))
    ftr = Vector((far_width/2, far_height/2, -far))
    fbr = Vector((far_width/2, -far_height/2, -far))
    fbl = Vector((-far_width/2, -far_height/2, -far))
    
    # Transform to world space
    mw = camera_obj.matrix_world
    apex_w = mw @ apex
    ftl_w = mw @ ftl
    ftr_w = mw @ ftr
    fbr_w = mw @ fbr
    fbl_w = mw @ fbl
    
    curve_name = name if name else "CameraFrustum"
    
    # Clean up existing objects/data only if using default name
    if not name:
        if curve_name in bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects[curve_name], do_unlink=True)
        if curve_name in bpy.data.curves:
            bpy.data.curves.remove(bpy.data.curves[curve_name])
    
    # === Create Curve data ===
    curve_data = bpy.data.curves.new(curve_name, type='CURVE')
    curve_data.dimensions = '3D'
    # fill_mode is not needed for 3D curves (only for 2D curves)
    
    def add_polyline(points):
        """Add a polyline spline with given points."""
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
    
    # Base square loop (no diagonal)
    base_spline = add_polyline([ftl_w, ftr_w, fbr_w, fbl_w])
    base_spline.use_cyclic_u = True  # Closed square
    
    # Line thickness (tube) settings for rendering
    curve_data.bevel_depth = bevel_depth
    curve_data.bevel_resolution = 2
    curve_data.resolution_u = 8
    
    frustum_obj = bpy.data.objects.new(curve_name, curve_data)
    scene.collection.objects.link(frustum_obj)
    
    # === Material: Emission for clear lines ===
    mat_name = "FrustumMaterial"
    if mat_name in bpy.data.materials:
        mat = bpy.data.materials[mat_name]
    else:
        mat = bpy.data.materials.new(mat_name)
    
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    # Remove existing nodes
    for n in list(nodes):
        nodes.remove(n)
    
    out_node = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    color_node = nodes.new("ShaderNodeRGB")
    
    # Color and brightness (cyan/blue for camera visualization)
    color_node.outputs[0].default_value = (0.2, 0.8, 1.0, 1.0)  # Cyan/blue color (camera-like)
    emission.inputs["Strength"].default_value = 1.0  # Brighter for visibility
    
    links.new(color_node.outputs[0], emission.inputs["Color"])
    links.new(emission.outputs[0], out_node.inputs["Surface"])
    
    frustum_obj.data.materials.append(mat)
    
    return frustum_obj

def setup_hand_rendering_with_colors():
    """Setup hand rendering with different colors (from blender_ego_hand.py)."""
    print("\n=== Setting up hand rendering with colors ===")
    body = body_mesh_obj
    if not body:
        print(f"ERROR: Body mesh not found (body_mesh_obj is None)")
        print(f"  Available objects: {[obj.name for obj in bpy.data.objects if 'Body' in obj.name or 'CC' in obj.name][:10]}")
        return False
    
    print(f"  Found body mesh: {body.name}")
    
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
    hand_L = duplicate_object(body, "CC_Hand_L")
    hand_R = duplicate_object(body, "CC_Hand_R")
    print(f"  Created hand objects: {hand_L.name}, {hand_R.name}")
    
    # Ensure armature modifier is properly set for both hands
    for hand_obj in [hand_L, hand_R]:
        for mod in hand_obj.modifiers:
            if mod.type == 'ARMATURE':
                mod.object = armature_obj
                mod.use_deform_preserve_volume = True
                print(f"  Set armature modifier for {hand_obj.name}: {mod.object.name if mod.object else 'None'}")
    
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
    
    # Verify modifiers are applied
    print(f"  Left hand modifiers: {[m.name for m in hand_L.modifiers]}")
    print(f"  Right hand modifiers: {[m.name for m in hand_R.modifiers]}")
    
    # Update depsgraph to evaluate modifiers
    bpy.context.view_layer.update()
    
    # Check if hand objects have vertices after modifiers
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_L = hand_L.evaluated_get(depsgraph)
    eval_R = hand_R.evaluated_get(depsgraph)
    mesh_L = eval_L.to_mesh(preserve_all_data_layers=False, depsgraph=depsgraph)
    mesh_R = eval_R.to_mesh(preserve_all_data_layers=False, depsgraph=depsgraph)
    print(f"  Left hand vertices after modifiers: {len(mesh_L.vertices)}")
    print(f"  Right hand vertices after modifiers: {len(mesh_R.vertices)}")
    eval_L.to_mesh_clear()
    eval_R.to_mesh_clear()
    
    # Create materials for hands (Phong-like, matching blender_ego_hand.py)
    print("  Creating hand materials...")
    def create_phong_material(name, color):
        """Create Phong-like material (from blender_ego_hand.py)."""
        # Remove existing material if it exists to ensure fresh creation
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
    
    # Use colors from blender_ego_hand_backup.py (HaWoR demo colors)
    print("  Creating left hand material (vibrant green)...")
    mat_L = create_phong_material("LeftHandMaterial", (0.5, 0.8, 0.5))  # Vibrant green
    print("  Creating right hand material (deep red)...")
    mat_R = create_phong_material("RightHandMaterial", (0.8, 0.4, 0.4))  # Deep red
    
    # Verify material colors
    print("  Verifying material colors...")
    bsdf_L = mat_L.node_tree.nodes.get("Principled BSDF")
    bsdf_R = mat_R.node_tree.nodes.get("Principled BSDF")
    if bsdf_L:
        color_L = bsdf_L.inputs["Base Color"].default_value
        print(f"  Left hand material color: ({color_L[0]:.2f}, {color_L[1]:.2f}, {color_L[2]:.2f})")
    if bsdf_R:
        color_R = bsdf_R.inputs["Base Color"].default_value
        print(f"  Right hand material color: ({color_R[0]:.2f}, {color_R[1]:.2f}, {color_R[2]:.2f})")
    
    print("  Assigning materials to hand objects...")
    # Clear all materials and assign new ones (like blender_ego_hand.py)
    for obj, mat in ((hand_L, mat_L), (hand_R, mat_R)):
        obj.data.materials.clear()
        obj.data.materials.append(mat)
        # Ensure object is visible
        obj.hide_render = False
        obj.hide_viewport = False
        # Apply shade smooth
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.shade_smooth()
    
    # Verify materials are assigned
    print(f"  Left hand material: {hand_L.data.materials[0].name if hand_L.data.materials else 'NONE'}")
    print(f"  Right hand material: {hand_R.data.materials[0].name if hand_R.data.materials else 'NONE'}")
    
    # Ensure objects are in the correct collection and visible
    for obj in [hand_L, hand_R]:
        # Make sure object is in scene collection
        if obj.name not in bpy.context.scene.collection.objects:
            bpy.context.scene.collection.objects.link(obj)
        # Ensure visibility - CRITICAL for rendering
        obj.hide_render = False
        obj.hide_viewport = False
        obj.hide_set(False)
        # Ensure object is selectable and visible in all view layers
        obj.select_set(True)
        # Ensure modifiers are enabled and evaluated
        for mod in obj.modifiers:
            mod.show_viewport = True
            mod.show_render = True
        # Ensure armature modifier is properly set
        for mod in obj.modifiers:
            if mod.type == 'ARMATURE' and mod.object:
                mod.object = armature_obj
                mod.use_deform_preserve_volume = True
    
    # Force update of all modifiers
    bpy.context.view_layer.update()
    
    # Verify hand objects are visible in render
    print(f"  Left hand hide_render: {hand_L.hide_render}, hide_viewport: {hand_L.hide_viewport}")
    print(f"  Right hand hide_render: {hand_R.hide_render}, hide_viewport: {hand_R.hide_viewport}")
    
    print("✓ Built hand objects: CC_Hand_L (vibrant green) / CC_Hand_R (deep red)")
    return True

def setup_body_transparent_gray():
    """Setup body mesh with transparent gray material, hiding hand parts."""
    if not body_mesh_obj:
        return False
    
    # Get hand objects
    hand_L_obj = bpy.data.objects.get("CC_Hand_L")
    hand_R_obj = bpy.data.objects.get("CC_Hand_R")
    
    # Create hand vertex groups in body if they don't exist (for masking)
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
    
    # Create union vertex groups in body for masking
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
    
    # Build hand union groups in body (for masking)
    if hand_L_obj:
        build_union_via_modifiers(body_mesh_obj, "Hand_L_All", left_hand_groups)
        # Add mask modifier to hide left hand in body
        mask_L = body_mesh_obj.modifiers.new(name="Mask_Hand_L", type='MASK')
        mask_L.vertex_group = "Hand_L_All"
        mask_L.invert_vertex_group = True  # Hide hand part
        mask_L.show_viewport = True
        mask_L.show_render = True
    
    if hand_R_obj:
        build_union_via_modifiers(body_mesh_obj, "Hand_R_All", right_hand_groups)
        # Add mask modifier to hide right hand in body
        mask_R = body_mesh_obj.modifiers.new(name="Mask_Hand_R", type='MASK')
        mask_R.vertex_group = "Hand_R_All"
        mask_R.invert_vertex_group = True  # Hide hand part
        mask_R.show_viewport = True
        mask_R.show_render = True
    
    mat_name = "BodyTransparentGray"
    mat = bpy.data.materials.get(mat_name)
    if mat is None:
        mat = bpy.data.materials.new(mat_name)
    
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for node in list(nodes):
        nodes.remove(node)
    
    output = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (0.5, 0.5, 0.5, 1.0)  # Gray
    bsdf.inputs["Alpha"].default_value = 0.6  # Less transparent (was 0.3)
    bsdf.inputs["Roughness"].default_value = 0.6
    
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    
    mat.blend_method = 'BLEND'
    mat.shadow_method = 'HASHED'
    mat.use_backface_culling = False
    
    # Only modify body_mesh_obj, not hand objects
    body_mesh_obj.data.materials.clear()
    body_mesh_obj.data.materials.append(mat)
    
    # Ensure hand objects keep their materials (if they exist)
    if hand_L_obj:
        if hand_L_obj.data.materials:
            print(f"  Preserving left hand material: {hand_L_obj.data.materials[0].name}")
            # Verify hand material is still correct
            if hand_L_obj.active_material:
                bsdf = hand_L_obj.active_material.node_tree.nodes.get("Principled BSDF")
                if bsdf:
                    color = bsdf.inputs["Base Color"].default_value
                    print(f"    Left hand color after body setup: ({color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f})")
        else:
            print(f"  WARNING: Left hand has no materials!")
    
    if hand_R_obj:
        if hand_R_obj.data.materials:
            print(f"  Preserving right hand material: {hand_R_obj.data.materials[0].name}")
            # Verify hand material is still correct
            if hand_R_obj.active_material:
                bsdf = hand_R_obj.active_material.node_tree.nodes.get("Principled BSDF")
                if bsdf:
                    color = bsdf.inputs["Base Color"].default_value
                    print(f"    Right hand color after body setup: ({color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f})")
        else:
            print(f"  WARNING: Right hand has no materials!")
    
    print("✓ Set body mesh to transparent gray")
    return True


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
body_mesh_name = "CC_Base_Body"
body_mesh_obj = bpy.data.objects.get(body_mesh_name)
if not armature_obj:
    msg = f"Armature '{armature_name}' not found"
    log_error("MISSING_ARMATURE", msg); print(f"Error: {msg}"); sys.exit(1)
if not eye_mesh_obj:
    msg = f"Mesh '{eye_mesh_name}' not found"
    log_error("MISSING_EYE_MESH", msg); print(f"Error: {msg}"); sys.exit(1)
if not body_mesh_obj:
    print(f"Warning: Body mesh '{body_mesh_name}' not found. Warm color will not be applied.")

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

# Find "Camera" object and set it as render camera
render_camera_obj = bpy.data.objects.get("Camera")
if render_camera_obj:
    bpy.context.scene.camera = render_camera_obj
    print(f"Set 'Camera' as render camera")
else:
    print(f"Warning: 'Camera' object not found. Using POV_Camera instead.")
    bpy.context.scene.camera = camera_obj

# Create POV Camera frustum visualization
print("Creating POV Camera frustum...")
frustum_obj = create_camera_frustum_mesh(camera_obj, frustum_length=0.5)
print(f"✓ Created frustum mesh: {frustum_obj.name}")

# Setup hand rendering with different colors FIRST (before body material)
print("\n=== Setting up hand rendering ===")
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

# Setup body as transparent gray AFTER hand setup (so it doesn't override hand materials)
print("\n=== Setting up body material ===")
setup_body_transparent_gray()

# ---------------------------
# Render sequence
# ---------------------------
def render_animation_sequence(animation_index, animation_name):
    global frustum_obj  # Access global frustum_obj
    
    anim_output_folder = os.path.join(output_folder, f"{animation_name}")
    
    # If rendering specific frames, use images directory
    if args.frames_list:
        images_output_path = os.path.join(anim_output_folder, "images")
        os.makedirs(images_output_path, exist_ok=True)
    
    print(f"Rendering animation {animation_index}: {animation_name}")
    if args.frames_list:
        print(f"  Images: {images_output_path}")

    # Frame range and video sequence parameters
    scene = bpy.context.scene
    
    # If specific frames list is provided, use those frames directly
    if args.frames_list:
        # Render specific frames only as images (like blender_teaser.py)
        frames_to_render = args.frames_list
        print(f"Rendering specific frames: {frames_to_render}")
        print(f"Total frames to render: {len(frames_to_render)}")
        print(f"Images output: {images_output_path}")
        
        # === Setup color ramp based on animation_index (SEQ0 or SEQ1) ===
        # Determine if this is SEQ0 or SEQ1 based on animation_index
        is_seq0 = (animation_index == SEQ0_INDEX)
        is_seq1 = (animation_index == SEQ1_INDEX)
        
        if is_seq0:
            color_start = COOL_START
            color_end = COOL_END
            color_name = "cool"
        elif is_seq1:
            color_start = WARM_START
            color_end = WARM_END
            color_name = "warm"
        else:
            # Default to warm if not SEQ0 or SEQ1
            color_start = WARM_START
            color_end = WARM_END
            color_name = "warm"
        
        color_ramp = build_color_ramp(color_start, color_end, len(frames_to_render))
        print(f"Using {color_name} color ramp: {color_start} -> {color_end}")
        
        # Set render format to PNG for images
        original_file_format = render.image_settings.file_format
        render.image_settings.file_format = 'PNG'
        render.image_settings.color_mode = 'RGBA'
        
        # === Setup: Apply animation ===
        restore_animations(camera_obj)
        show_actor_in_rendering()
        
        # Apply animation
        if not apply_animation_set(animation_index):
            print(f"Failed to apply animation set")
            return
        
        # Adjust head bone rotation for all frames
        if is_seq1:
            adjust_head_bone_rotation(frames_to_render, delta_pitch=0.2, delta_yaw=0.1)
        
        # === (B) Sample camera poses for all frames (while camera is still parented) ===
        cam_locs, cam_rots = sample_camera_world_transforms(camera_obj, frames_to_render)
        
        # === (C) Capture phone position at reference frame, then freeze non-character objects ===
        # Find and capture phone position at reference frame (like blender_teaser.py)
        phone_obj = find_object_by_name(PHONE_OBJECT_NAME)
        phone_world_matrix = None
        if phone_obj:
            scene.frame_set(PHONE_REFERENCE_FRAME)
            bpy.context.view_layer.update()
            phone_world_matrix = capture_object_world_matrix(phone_obj)
            print(f"Captured phone '{phone_obj.name}' position at reference frame ({PHONE_REFERENCE_FRAME})")
        else:
            print(f"Warning: Phone object '{PHONE_OBJECT_NAME}' not found. Phone position will not be fixed.")
        
        # Go to first frame and freeze non-character objects (like blender_ego_static.py)
        first_frame = frames_to_render[0]
        scene.frame_set(first_frame)
        bpy.context.view_layer.update()  # Ensure agent pose is correct
        
        disable_animations_except_camera(camera_obj, armature_obj)
        print(f"Frozen non-character objects at first frame ({first_frame}) state")
        
        # === Create pose snapshots for all frames (like blender_teaser.py) ===
        print("\n--- Creating pose snapshots for all frames ---")
        snapshot_collection_name = f"PoseSnapshots_{animation_name}"
        start_time = time.time()
        
        # Get hand objects
        hand_L_obj = bpy.data.objects.get("CC_Hand_L")
        hand_R_obj = bpy.data.objects.get("CC_Hand_R")
        
        # Create collection for snapshots
        collection = ensure_pose_collection(snapshot_collection_name)
        clear_pose_snapshots(collection)
        
        # Create snapshots for each frame
        original_frame = scene.frame_current
        total = len(frames_to_render)
        
        for idx, frame_num in enumerate(frames_to_render):
            scene.frame_set(frame_num)
            depsgraph = bpy.context.evaluated_depsgraph_get()
            
            # Bake camera for this frame
            bake_camera_keys(camera_obj, [frame_num], [cam_locs[idx]], [cam_rots[idx]])
            bpy.context.view_layer.update()
            
            # === Body snapshot (only for first frame) ===
            if idx == 0:
                eval_body = body_mesh_obj.evaluated_get(depsgraph)
                try:
                    body_mesh_data = bpy.data.meshes.new_from_object(
                        eval_body,
                        preserve_all_data_layers=True,
                        depsgraph=depsgraph
                    )
                    body_mesh_data.name = f"{POSE_OBJECT_PREFIX}_Body_Mesh_{frame_num}"
                    name_suffix = f"{slugify_name(animation_name)}_{frame_num}"
                    body_snapshot = bpy.data.objects.new(f"{POSE_OBJECT_PREFIX}_Body_{name_suffix}", body_mesh_data)
                    body_snapshot.matrix_world = body_mesh_obj.matrix_world.copy()
                    collection.objects.link(body_snapshot)
                    
                    # Use transparent gray material for body
                    material = create_body_transparent_gray_material(idx, total, frame_num, animation_name)
                    body_snapshot.data.materials.clear()
                    body_snapshot.data.materials.append(material)
                    body_snapshot.hide_render = False
                    body_snapshot.hide_viewport = False
                    print(f"Created body snapshot for frame {frame_num} (transparent gray) - first frame only")
                except RuntimeError as err:
                    print(f"Failed to create body snapshot at frame {frame_num}: {err}")
            
            # === Hand snapshots ===
            for hand_obj, hand_name in [(hand_L_obj, "Hand_L"), (hand_R_obj, "Hand_R")]:
                if hand_obj:
                    eval_hand = hand_obj.evaluated_get(depsgraph)
                    try:
                        hand_mesh_data = bpy.data.meshes.new_from_object(
                            eval_hand,
                            preserve_all_data_layers=True,
                            depsgraph=depsgraph
                        )
                        hand_mesh_data.name = f"{POSE_OBJECT_PREFIX}_{hand_name}_Mesh_{frame_num}"
                        hand_snapshot = bpy.data.objects.new(f"{POSE_OBJECT_PREFIX}_{hand_name}_{frame_num}", hand_mesh_data)
                        hand_snapshot.matrix_world = hand_obj.matrix_world.copy()
                        collection.objects.link(hand_snapshot)
                        
                        # Create material with alpha that increases over time
                        base_material = None
                        if hand_obj.data.materials:
                            base_material = hand_obj.data.materials[0]
                        hand_material = create_hand_material_with_alpha(
                            base_material, idx, total, frame_num, hand_name
                        )
                        hand_snapshot.data.materials.clear()
                        hand_snapshot.data.materials.append(hand_material)
                        hand_snapshot.hide_render = False
                        hand_snapshot.hide_viewport = False
                        print(f"Created {hand_name} snapshot for frame {frame_num} (alpha: {hand_material.node_tree.nodes.get('Principled BSDF').inputs['Alpha'].default_value:.2f})")
                    except RuntimeError as err:
                        print(f"Failed to create {hand_name} snapshot at frame {frame_num}: {err}")
            
            # === Frustum snapshot ===
            if camera_obj:
                try:
                    # Create frustum with unique name
                    frustum_snapshot_name = f"{POSE_OBJECT_PREFIX}_Frustum_{frame_num}"
                    frustum_snapshot = create_camera_frustum_mesh(
                        camera_obj, 
                        frustum_length=0.1, 
                        name=frustum_snapshot_name
                    )
                    
                    # Apply color gradation from color ramp
                    color_override = None
                    if color_ramp and idx < len(color_ramp):
                        color_override = color_ramp[idx]
                    frustum_material = create_frustum_material_with_color(
                        idx, total, frame_num, color_override, animation_name
                    )
                    frustum_snapshot.data.materials.clear()
                    frustum_snapshot.data.materials.append(frustum_material)
                    
                    # Unlink from scene.collection and link to snapshot collection
                    if frustum_snapshot.name in scene.collection.objects:
                        scene.collection.objects.unlink(frustum_snapshot)
                    collection.objects.link(frustum_snapshot)
                    print(f"Created frustum snapshot for frame {frame_num} (color: {color_override if color_override else 'default'})")
                except Exception as err:
                    print(f"Failed to create frustum snapshot at frame {frame_num}: {err}")
                    import traceback
                    traceback.print_exc()
            
            # Clear camera keys for next frame
            clear_camera_keys(camera_obj)
        
        scene.frame_set(original_frame)
        print(f"Pose snapshots stored in collection '{collection.name}'.")
        
        # Hide original objects
        original_body_hide_render = body_mesh_obj.hide_render
        original_body_hide_viewport = body_mesh_obj.hide_viewport
        body_mesh_obj.hide_render = True
        body_mesh_obj.hide_viewport = True
        
        if hand_L_obj:
            hand_L_obj.hide_render = True
            hand_L_obj.hide_viewport = True
        if hand_R_obj:
            hand_R_obj.hide_render = True
            hand_R_obj.hide_viewport = True
        if frustum_obj:
            frustum_obj.hide_render = True
            frustum_obj.hide_viewport = True
        
        # Set phone position to reference frame state
        if phone_obj and phone_world_matrix is not None:
            apply_world_matrix_to_object(
                phone_obj,
                phone_world_matrix,
                keyframe=first_frame,
                constant=True,
                clear_animation=True
            )
            print(f"Phone '{phone_obj.name}' positioned using stored matrix from frame {PHONE_REFERENCE_FRAME}")
        
        # Render single composite image with all pose snapshots
        print(f"\n--- Rendering composite image with {len(frames_to_render)} pose snapshots ---")
        render.filepath = os.path.join(images_output_path, "composite")
        
        render_start_time = time.time()
        bpy.ops.render.render(write_still=True)
        render_time = time.time() - render_start_time
        
        print(f"✅ Created composite image: composite.png ({render_time:.1f}s)")
        
        # Restore body visibility
        body_mesh_obj.hide_render = original_body_hide_render
        body_mesh_obj.hide_viewport = original_body_hide_viewport
        
        # Cleanup
        clear_camera_keys(camera_obj)
        frames_completed = len(frames_to_render)
        
        # Restore animations after rendering
        restore_animations(camera_obj)
        show_actor_in_rendering()
        
        # Restore original file format
        render.image_settings.file_format = original_file_format
        
        total_time = time.time() - start_time
        print("\n" + "="*50)
        print(f"COMPLETED: Animation {animation_index} ({animation_name})")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Pose snapshots created: {len(frames_to_render)} frames")
        print(f"{color_name.capitalize()} color applied: {color_start} -> {color_end}")
        print(f"Composite image saved to: {images_output_path}/composite.png")
        print(f"Snapshot collection: '{snapshot_collection_name}'")
        print("="*50)
        return
    
    # If no frames list provided, exit (video sequence rendering removed)
    print("Error: --frames argument is required. Please specify frames to render.")
    print("Example: --frames '850,868,886,904'")
    return

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
