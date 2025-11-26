# First set keyframes of "Camera" object
# This is for 1a1e205b-3f54-49fb-8154-e2d61c3682ae.blend
import bpy
import math
import os
import signal
import sys

# ------------------------------------------------------------------
# 설정: 출력 폴더 / 프레임 범위 / 포맷
# ------------------------------------------------------------------
output_dir = "/home/byungjunkim/dwm_teaser/overview_video"  # 원하는 경로로 바꿔줘
os.makedirs(output_dir, exist_ok=True)

scene = bpy.context.scene
render = scene.render

# Frame range
scene.frame_start = 0
scene.frame_end = 89

# Ensure frame step is 1 (render every frame, not skipping)
if hasattr(scene, "frame_step"):
    scene.frame_step = 1
print(f"Frame step: {getattr(scene, 'frame_step', 1)}")

# ------------------------------------------------------------------
# Hide character (armature and its children) from rendering
# ------------------------------------------------------------------
print("\n" + "="*60)
print("HIDING CHARACTER FROM RENDERING")
print("="*60)

# Find armature (from blender_ego_static.py)
armature_name = None
armature_obj = None
try:
    armature_name = bpy.context.scene.hsi_properties.name_armature_CC
    if not armature_name:
        for obj in bpy.data.objects:
            if obj.type == 'ARMATURE' and 'CC_Base_Hip' in obj.pose.bones:
                armature_name = obj.name
                break
except Exception as e:
    print(f"Could not access HSI addon properties: {e}")
    # Fallback: search for armature manually
    for obj in bpy.data.objects:
        if obj.type == 'ARMATURE' and 'CC_Base_Hip' in obj.pose.bones:
            armature_name = obj.name
            break

if armature_name:
    armature_obj = bpy.data.objects.get(armature_name)
    if armature_obj:
        # Hide the armature itself
        armature_obj.hide_render = True
        print(f"✓ Hidden armature '{armature_name}' from rendering")
        
        # Hide all children of the armature
        for child in armature_obj.children:
            child.hide_render = True
            print(f"✓ Hidden child '{child.name}' from rendering")
        
        # Also hide any objects that are parented to bones
        for obj in bpy.data.objects:
            if obj.parent_bone and obj.parent == armature_obj:
                obj.hide_render = True
                print(f"✓ Hidden bone-child '{obj.name}' from rendering")
    else:
        print(f"⚠️  Warning: Armature '{armature_name}' not found")
else:
    print("⚠️  Warning: No armature found - character may still be visible")

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

# Camera FOV: 90 degrees
render_camera_obj = bpy.data.objects.get("Camera")
if render_camera_obj and render_camera_obj.data:
    render_camera_obj.data.lens_unit = 'FOV'
    render_camera_obj.data.angle = math.radians(90.0)
    print(f"✓ Set Camera FOV to 90°")
else:
    print("⚠️  Warning: 'Camera' object not found or has no data")

# Cycles settings: 128 samples
cycles_samples = 128
scene.cycles.samples = cycles_samples
scene.cycles.device = 'GPU'

# Persistent data caching (Blender 4.x)
if hasattr(scene.render, "use_persistent_data"):
    scene.render.use_persistent_data = True

# # Denoising ON (OptiX GPU based)
# scene.cycles.use_denoising = True
# if hasattr(scene.cycles, "denoiser"):
#     keys = scene.cycles.bl_rna.properties['denoiser'].enum_items.keys()
#     if "OPTIX" in keys:
#         scene.cycles.denoiser = "OPTIX"
#         scene.cycles.use_preview_denoising = True
#     elif "OPENIMAGEDENOISE" in keys:
#         scene.cycles.denoiser = "OPENIMAGEDENOISE"  # CPU fallback
# scene.cycles.preview_denoising = True

# Video output settings (from blender_ego_static.py)
render.image_settings.file_format = 'FFMPEG'
render.ffmpeg.format = 'MPEG4'
render.ffmpeg.codec = 'H264'
render.ffmpeg.constant_rate_factor = 'MEDIUM'   # Quality/speed balance: 18~23 range
render.ffmpeg.ffmpeg_preset = 'REALTIME'       # Speed up in light I/O environments
render.fps = 30  # 30fps video

# Output file path
video_output_path = os.path.join(output_dir, "static_scene")
render.filepath = video_output_path

print(f"\n" + "="*60)
print(f"RENDERING STATIC SCENE VIDEO")
print("="*60)
print(f"Resolution: {render.resolution_x}x{render.resolution_y}")
print(f"FOV: 90°")
print(f"Cycles samples: {cycles_samples}")
print(f"Frame range: {scene.frame_start}..{scene.frame_end}")
print(f"FPS: {render.fps}")
print(f"Output: {video_output_path}.mp4")
print("\n⚠️  To stop rendering: Press Ctrl+C in the terminal (not in Blender GUI)")
print("   Or close Blender window to abort\n")

# Signal handler for graceful shutdown
interrupted = False

def signal_handler(sig, frame):
    global interrupted
    interrupted = True
    print("\n\n⚠️  INTERRUPTED: Rendering stopped by user (Ctrl+C)")
    print("   Partial video may be saved. Check output directory.")
    sys.exit(0)

# Register signal handler (works when run from terminal)
signal.signal(signal.SIGINT, signal_handler)

try:
    # Render video
    bpy.ops.render.render(animation=True)
    
    if not interrupted:
        print(f"✅ Video rendering complete: {video_output_path}.mp4")
except KeyboardInterrupt:
    print("\n\n⚠️  INTERRUPTED: Rendering stopped by user")
    print("   Partial video may be saved. Check output directory.")
except Exception as e:
    print(f"\n❌ ERROR during rendering: {e}")
    raise
