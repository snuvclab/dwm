import bpy
import math
import sys
import os
import json
import numpy as np
from mathutils import Matrix
import argparse
import time
import traceback
from datetime import datetime

# ---------------------------
# CLI args
# ---------------------------
argv = sys.argv
if "--" in argv:
    argv = argv[argv.index("--") + 1:]
else:
    argv = []

if not argv:
    argv = [
        "--composite-pose",
        "--save-path", "/home/byungjunkim/dwm_teaser/output",
        "--pose-collection", "PoseSnapshots",
        "--body-mesh", "CC_Base_Body",
    ]

parser = argparse.ArgumentParser()
parser.add_argument("--start_frame", type=int, default=None)
parser.add_argument("--end_frame", type=int, default=None)
parser.add_argument("--animation_index", type=int, default=None, help="Specific animation index (else: all)")
parser.add_argument("--samples", type=int, default=1024, help="Cycles samples")
parser.add_argument("--save-path", type=str, default="/home/byungjunkim/workspace/trumans_ego/ego_render_new",
                    help="Root output dir")
parser.add_argument("--skip-existing", action="store_true", default=True, help="Skip frames that already exist")
parser.add_argument("--no-skip-existing", action="store_true", help="Disable skipping existing frames")
parser.add_argument("--frame-skip", type=int, default=3, help="Render every Nth frame")
parser.add_argument("--fov", type=float, default=90.0, help="Camera FOV in degrees (perspective)")
parser.add_argument("--width", type=int, default=1500, help="Render width in pixels (default: 1500)")
parser.add_argument("--height", type=int, default=1200, help="Render height in pixels (default: 1500)")
parser.add_argument("--pose-frames", type=str, help="Comma-separated frame numbers to freeze as pose snapshots")
parser.add_argument("--pose-collection", type=str, default="PoseSnapshots",
                    help="Collection name used to store pose snapshot objects")
parser.add_argument("--body-mesh", type=str, default="CC_Base_Body",
                    help="Name of the animated body mesh to duplicate for pose snapshots")
parser.add_argument("--composite-pose", action="store_true",
                    help="Generate a composite pose figure using predefined frame sets from animations 0 and 1")
args = parser.parse_args(argv)
if args.no_skip_existing:
    args.skip_existing = False
args.skip_existing = False

pose_frames_list = []
if args.pose_frames:
    try:
        pose_frames_list = [int(token.strip()) for token in args.pose_frames.split(",") if token.strip()]
    except ValueError:
        print(f"Error: Invalid frame list supplied to --pose-frames: '{args.pose_frames}'")
        sys.exit(1)
    if not pose_frames_list:
        print("Error: --pose-frames was provided but no valid frame numbers were parsed.")
        sys.exit(1)
args.pose_frames = pose_frames_list

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
#start_frame = args.start_frame
#end_frame = args.end_frame
#animation_index = args.animation_index
start_frame = 1250
end_frame = 1250
animation_index = 0

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

POSE_OBJECT_PREFIX = "PoseSnapshot"
POSE_MATERIAL_PREFIX = "PoseSnapshotMaterial"

def slugify_name(name):
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name) if name else "default"

def find_object_by_name(name):
    obj = bpy.data.objects.get(name)
    if obj:
        return obj
    candidates = [o for o in bpy.data.objects if name in o.name]
    if candidates:
        print(f"Warning: Exact object '{name}' not found. Using '{candidates[0].name}' instead.")
        return candidates[0]
    return None

def build_color_ramp(start_rgba, end_rgba, total):
    if total <= 1:
        return [end_rgba]
    ramp = []
    for i in range(total):
        t = i / (total - 1)
        ramp.append(tuple(start_rgba[j] + (end_rgba[j] - start_rgba[j]) * t for j in range(4)))
    return ramp

def capture_object_world_matrix(obj):
    if obj is None:
        return None
    return obj.matrix_world.copy()

def apply_world_matrix_to_object(obj, matrix, keyframe=None, constant=False, clear_animation=False):
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

def ensure_pose_collection(collection_name):
    collection = bpy.data.collections.get(collection_name)
    scene_collection = bpy.context.scene.collection
    if collection is None:
        collection = bpy.data.collections.new(collection_name)
        scene_collection.children.link(collection)
    elif collection not in scene_collection.children:
        scene_collection.children.link(collection)
    return collection

def clear_pose_snapshots(collection):
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
        alpha = color_rgba[3]
    else:
        t = index / max(1, total - 1)
        color_strength = 0.9 - 0.4 * t
        base_color = (1.0, color_strength, color_strength, 1.0)
        alpha = max(0.3, 1.0 - 0.18 * index)

    bsdf.inputs["Base Color"].default_value = base_color
    bsdf.inputs["Alpha"].default_value = alpha
    bsdf.inputs["Roughness"].default_value = 0.6

    material.blend_method = 'HASHED'
    material.shadow_method = 'HASHED'
    material.use_backface_culling = False
    return material

def create_pose_snapshots(base_mesh_obj, frames, collection_name, animation_tag=None, color_ramp=None):
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

def run_composite_pose_workflow(animation_sets, body_mesh_obj, rgb_output_node):
    SEQ0_INDEX = 0
    SEQ1_INDEX = 1
    # SEQ0_FRAMES = [600, 630, 660, 690]
    SEQ0_FRAMES = [600, 630, 672, 690]
    # SEQ1_FRAMES = [850, 868, 886, 904]
    SEQ1_FRAMES = [850, 868, 901, 919]
    CHAIR_REFERENCE_FRAME = 850
    CHAIR_OBJECT_NAMES = [
        "movable_chair_seat_root_movable_chair_seat_01",
        "movable_chair_base_root_movable_chair_base_01"
    ]
    PHONE_OBJECT_NAME = "rigid_03_root_phone_01"

    animation_names = list(animation_sets.keys())
    if max(SEQ0_INDEX, SEQ1_INDEX) >= len(animation_names):
        print("Error: Required animation indices 0 and 1 are not available in the scene.")
        sys.exit(1)

    if body_mesh_obj is None:
        print("Error: Body mesh required for composite pose generation was not found.")
        sys.exit(1)

    chair_objects = {}
    for name in CHAIR_OBJECT_NAMES:
        obj = find_object_by_name(name)
        if obj is None:
            print(f"Error: Chair object '{name}' not found. Composite workflow aborted.")
            sys.exit(1)
        chair_objects[name] = obj

    phone_obj = find_object_by_name(PHONE_OBJECT_NAME)
    if phone_obj is None:
        print(f"Error: Phone object '{PHONE_OBJECT_NAME}' not found. Composite workflow aborted.")
        sys.exit(1)

    scene = bpy.context.scene
    original_frame = scene.frame_current

    print("\n--- Capturing chair and phone pose from animation 1 ---")
    if not apply_animation_set(SEQ1_INDEX):
        print("Error: Unable to apply animation 1 for composite workflow.")
        sys.exit(1)
    # Adjust head bone rotation (using adjust_head_bone_rotation function)
    adjust_head_bone_rotation(SEQ1_FRAMES, delta_pitch=0.3, delta_yaw=0.0)
    scene.frame_set(CHAIR_REFERENCE_FRAME)
    chair_world_matrices = {}
    for name, obj in chair_objects.items():
        chair_world_matrices[name] = capture_object_world_matrix(obj)
        print(f"Stored chair matrix for '{obj.name}' at frame {CHAIR_REFERENCE_FRAME}")
    phone_world_matrix = capture_object_world_matrix(phone_obj)
    print(f"Stored phone matrix for '{phone_obj.name}' at frame {CHAIR_REFERENCE_FRAME}")

    print("\n--- Creating pose snapshots for animation 1 ---")
    warm_start = (1.0, 0.75, 0.5, 0.15)
    warm_end = (0.65, 0.2, 0.05, 0.95)
    warm_ramp = build_color_ramp(warm_start, warm_end, len(SEQ1_FRAMES))
    seq1_collection = f"{args.pose_collection}_anim1"
    create_pose_snapshots(
        body_mesh_obj,
        SEQ1_FRAMES,
        seq1_collection,
        f"{animation_names[SEQ1_INDEX]}_anim1",
        color_ramp=warm_ramp
    )

    print("\n--- Creating pose snapshots for animation 0 ---")
    if not apply_animation_set(SEQ0_INDEX):
        print("Error: Unable to apply animation 0 for composite workflow.")
        sys.exit(1)

    cool_start = (0.62, 0.88, 0.92, 0.35)
    cool_end = (0.08, 0.20, 0.65, 1.0)
    cool_ramp = build_color_ramp(cool_start, cool_end, len(SEQ0_FRAMES))
    seq0_collection = f"{args.pose_collection}_anim0"
    create_pose_snapshots(
        body_mesh_obj,
        SEQ0_FRAMES,
        seq0_collection,
        f"{animation_names[SEQ0_INDEX]}_anim0",
        color_ramp=cool_ramp
    )

    render_frame = SEQ0_FRAMES[0]
    scene.frame_set(render_frame)
    for obj_name, matrix in chair_world_matrices.items():
        apply_world_matrix_to_object(
            chair_objects[obj_name],
            matrix,
            keyframe=render_frame,
            constant=True,
            clear_animation=True
        )
        print(f"Chair '{chair_objects[obj_name].name}' positioned using stored matrix from animation 1 frame {CHAIR_REFERENCE_FRAME}")
    apply_world_matrix_to_object(
        phone_obj,
        phone_world_matrix,
        keyframe=render_frame,
        constant=True,
        clear_animation=True
    )
    print(f"Phone '{phone_obj.name}' positioned using stored matrix from animation 1 frame {CHAIR_REFERENCE_FRAME}")

    original_hide_render = body_mesh_obj.hide_render
    original_hide_viewport = body_mesh_obj.hide_viewport
    body_mesh_obj.hide_render = True
    body_mesh_obj.hide_viewport = True

    rgb_output_node.base_path = output_folder
    rgb_output_node.file_slots[0].path = "composite"

    print("\n--- Rendering composite pose ---")
    bpy.ops.render.render(write_still=True)
    print("Composite render complete.")

    body_mesh_obj.hide_render = original_hide_render
    body_mesh_obj.hide_viewport = original_hide_viewport
    scene.frame_set(original_frame)

    print("\nComposite pose workflow finished successfully.")
    print(f"Snapshot collections created: '{seq0_collection}', '{seq1_collection}'")
    print(f"Chair and phone pose sourced from animation 1 frame {CHAIR_REFERENCE_FRAME}.")
    sys.exit(0)

def check_frame_exists(frame_num, images_output_path, cam_params_path):
    """
    Check if frame files exist and are complete.
    Uses filesize thresholds to detect incomplete files from interrupted renders.
    """
    def file_ok(path, min_size=1024):
        """File must exist, have reasonable size, and not be too recent (avoid race conditions)"""
        if not os.path.isfile(path):
            return False
        size = os.path.getsize(path)
        if size < min_size:  # Too small = likely incomplete
            return False
        # Check if file was modified very recently (< 2 seconds ago)
        # This helps avoid race conditions during concurrent rendering
        import time
        mtime = os.path.getmtime(path)
        if time.time() - mtime < 2.0:
            return False  # File too fresh, might still be writing
        return True
    
    image_path = os.path.join(images_output_path, f"{frame_num:04d}.png")
    cam_param_path = os.path.join(cam_params_path, f"cam_{frame_num:04d}.npy")
    
    # PNG images should be at least ~10KB for 720x480
    # NPY cam params should be at least ~200 bytes
    rgb_exists = file_ok(image_path, min_size=10240)  # 10KB
    cam_param_exists = file_ok(cam_param_path, min_size=200)  # 200 bytes
    
    needs_rendering = not rgb_exists
    needs_cam_param = not cam_param_exists
    return rgb_exists, cam_param_exists, needs_rendering, needs_cam_param

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
# Compositor nodes
# ---------------------------
scene.use_nodes = True
tree = scene.node_tree
for node in list(tree.nodes):
    tree.nodes.remove(node)

# Default ViewLayer (RGB only)
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

# ---------------------------
# Locate objects & camera setup
# ---------------------------
armature_obj = bpy.data.objects.get(armature_name)
if not armature_obj:
    msg = f"Armature '{armature_name}' not found"
    log_error("MISSING_ARMATURE", msg); print(f"Error: {msg}"); sys.exit(1)

body_mesh_name = args.body_mesh
body_mesh_obj = bpy.data.objects.get(body_mesh_name)
if args.pose_frames and body_mesh_obj is None:
    msg = f"Body mesh '{body_mesh_name}' not found"
    log_error("MISSING_BODY_MESH", msg); print(f"Error: {msg}"); sys.exit(1)
if body_mesh_obj is None:
    print(f"Warning: Body mesh '{body_mesh_name}' not found. Pose snapshots will be unavailable.")

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

camera_obj = bpy.data.objects.get("Camera")
if camera_obj is None:
    raise ValueError("There is no 'Camera' Object!")

scene.camera = camera_obj
print(f"Using existing camera: {camera_obj.name}")

# ---------------------------
# Render sequence
# ---------------------------
def render_animation_sequence(animation_index, animation_name):
    anim_output_folder = os.path.join(output_folder, f"{animation_name}")
    images_output_path = os.path.join(anim_output_folder, "images")
    cam_params_path    = os.path.join(anim_output_folder, "cam_params")
    
    # Create directories for render outputs
    os.makedirs(images_output_path, exist_ok=True)
    os.makedirs(cam_params_path, exist_ok=True)

    print(f"Rendering animation {animation_index}: {animation_name}")
    print(f"  Images: {images_output_path}")
    print(f"  Cam   : {cam_params_path}")

    # Set base paths per animation
    rgb_output_node.base_path   = images_output_path
    rgb_output_node.file_slots[0].path   = "####"

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
    frames_to_render = [2110, 2130, 2150, 2170, 2190]
    for frame_num in frames_to_render:
        rgb_exists, cam_param_exists, needs_rendering, needs_cam_param = \
            check_frame_exists(frame_num, images_output_path, cam_params_path)

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
            if cam_param_exists: status.append("Cam")
            print(f"[ANIM {animation_index}] Frame {frame_num}: SKIPPED ({', '.join(status)})")
            continue

        # Render (compositor writes RGB directly to animation folders)
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

if args.composite_pose:
    run_composite_pose_workflow(animation_sets, body_mesh_obj, rgb_output_node)

if animation_index is not None:
    if animation_index >= len(animation_sets):
        print(f"Error: Animation index {animation_index} out of range."); sys.exit(1)
    animations_to_render = [(animation_index, list(animation_sets.keys())[animation_index])]
    print(f"Rendering specific animation: {animation_index}")
else:
    animations_to_render = [(i, name) for i, name in enumerate(animation_sets.keys())]
    print(f"Rendering all {len(animations_to_render)} animations")

pose_mode = bool(args.pose_frames)
if pose_mode:
    print(f"Starting pose snapshot generation for frames: {args.pose_frames}")
else:
    print("Starting animation rendering loop...")
total_start_time = time.time()
failed_animations = []
pose_collections_created = []
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
        animation_label = anim_name[:-4] if anim_name.lower().endswith(".fbx") else anim_name
        if pose_mode:
            snapshot_collection_name = args.pose_collection
            if animation_label:
                snapshot_collection_name = f"{args.pose_collection}_{slugify_name(animation_label)}"
            success = create_pose_snapshots(body_mesh_obj, args.pose_frames, snapshot_collection_name, animation_label)
            if not success:
                failed_animations.append((anim_idx, anim_name, "POSE_SNAPSHOT_FAILED"))
                print(f"Pose snapshot generation failed for animation {anim_idx}.")
            else:
                pose_collections_created.append(snapshot_collection_name)
                print(f"Pose snapshots generated for animation {anim_idx}: {anim_name} (collection: {snapshot_collection_name})")
            continue
        render_animation_sequence(anim_idx, animation_label)
        print(f"Completed animation {anim_idx}: {anim_name}")
    except Exception as e:
        msg = f"Unexpected error during animation {anim_idx}: {str(e)}"
        log_error("RENDERING_ERROR", msg, animation_name=anim_name)
        print(f"Error during animation {anim_idx}: {str(e)}")
        failed_animations.append((anim_idx, anim_name, "RENDERING_ERROR"))
        continue

total_time = time.time() - total_start_time

if pose_mode:
    print("\n" + "="*60)
    print("POSE SNAPSHOT GENERATION COMPLETE!")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Frames captured: {args.pose_frames}")
    if pose_collections_created:
        unique_collections = list(dict.fromkeys(pose_collections_created))
        print("Snapshot collections:")
        for name in unique_collections:
            print(f"  - {name}")
    else:
        print(f"Snapshot collection: '{args.pose_collection}'")
    if failed_animations:
        print(f"\nFAILED SNAPSHOTS ({len(failed_animations)}):")
        for anim_idx, anim_name, error_type in failed_animations:
            print(f"  Animation {anim_idx} ({anim_name}): {error_type}")
        print(f"Check error log for details: {error_log_file}")
    else:
        print("\nAll pose snapshots generated successfully!")
    print("="*60)

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