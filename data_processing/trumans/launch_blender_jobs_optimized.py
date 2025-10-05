import os
import subprocess
import time
import json
import signal
import sys
from collections import deque
from pathlib import Path
import argparse
from check_rendering_status import get_animation_frame_count
from check_video_rendering_status import get_expected_video_count, check_video_sequence_completeness

# === User Configuration ===
DEFAULT_RECORDINGS_PATH = "./data/trumans/Recordings_blend"
# DEFAULT_RECORDINGS_PATH = "./Recordings_blend"
DEFAULT_SAVE_PATH = "./data/trumans/ego_render_fov90"
SCRIPT_PATH = "data_processing/trumans/blender_ego_rgb_depth_optimized.py"
# SCRIPT_PATH = "data_processing/trumans/blender_ego_static.py"
# SCRIPT_PATH = "data_processing/trumans/blender_ego_hand.py"
NUM_GPUS = 8
# ===========================

# Global variable to track running processes
running_processes = {}  # GPU_ID -> list of processes
failed_jobs = []  # Track failed jobs for reporting

def signal_handler(signum, frame):
    """Handle Ctrl-C and other termination signals."""
    print(f"\n🛑 Received signal {signum}. Terminating all Blender processes...")
    
    # Terminate all running processes
    for gpu_id, processes in running_processes.items():
        for proc in processes:
            if proc.poll() is None:  # Process is still running
                print(f"  Terminating process on GPU {gpu_id} (PID: {proc.pid})")
                try:
                    proc.terminate()  # Send SIGTERM first
                except:
                    pass
    
    # Wait a bit for graceful termination
    time.sleep(2)
    
    # Force kill any remaining processes
    for gpu_id, processes in running_processes.items():
        for proc in processes:
            if proc.poll() is None:  # Process is still running
                print(f"  Force killing process on GPU {gpu_id} (PID: {proc.pid})")
                try:
                    proc.kill()  # Send SIGKILL
                except:
                    pass
    
    print("✅ All Blender processes terminated.")
    sys.exit(1)



def filter_scenes(blend_jobs, scene_names=None, scene_pattern=None):
    """Filter blend jobs based on scene names or pattern."""
    if not scene_names and not scene_pattern:
        return blend_jobs
    
    # Get all available scene names for validation
    all_scene_names = set(os.path.basename(os.path.dirname(blend_file)) for blend_file in blend_jobs)
    
    filtered_jobs = []
    matched_scenes = set()
    
    for blend_file in blend_jobs:
        directory_name = os.path.basename(os.path.dirname(blend_file))
        
        # Check if scene name matches any specified names
        if scene_names and directory_name in scene_names:
            filtered_jobs.append(blend_file)
            matched_scenes.add(directory_name)
            continue
        
        # Check if scene name matches pattern
        if scene_pattern:
            import fnmatch
            if fnmatch.fnmatch(directory_name, scene_pattern):
                filtered_jobs.append(blend_file)
                matched_scenes.add(directory_name)
                continue
    
    # Validate specified scene names
    if scene_names:
        missing_scenes = set(scene_names) - matched_scenes
        if missing_scenes:
            print(f"⚠️  Warning: The following specified scenes were not found: {', '.join(missing_scenes)}")
            print(f"Available scenes: {', '.join(sorted(all_scene_names))}")
    
    return filtered_jobs

def load_rendering_status_report(report_path=None, script_path=None):
    """Load the rendering status report from check_rendering_status.py."""
    if report_path is None:
        # Determine report file based on script type
        script_name = os.path.basename(script_path) if script_path else ""
        if script_name in ['blender_ego_static.py']:
            report_file = "rendering_status_report_static.json"
        elif script_name in ['blender_ego_hand.py']:
            report_file = "rendering_status_report_hands.json"
        else:
            report_file = "rendering_status_report.json"  # Default for frame-based rendering
    else:
        report_file = report_path
    
    if os.path.exists(report_file):
        try:
            with open(report_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load rendering status report from {report_file}: {e}")
    else:
        if report_path is not None:
            print(f"Warning: Rendering status report not found at {report_file}")
        else:
            print(f"Warning: Rendering status report not found at {report_file}")
            print(f"  Expected report for script: {script_name}")
            print(f"  You can generate it using: python check_rendering_status.py --video-type {'static' if 'static' in script_name else 'hands' if 'hand' in script_name else 'frames'}")
    return None

def get_animation_names(blend_file):
    """Get the list of all animation names from a blend file."""
    try:
        import subprocess
        import tempfile
        
        # Create a temporary Python script to avoid command-line parsing issues
        script_content = '''import bpy
import json
import sys

try:
    # Try to access HSI addon properties
    animation_sets = json.loads(bpy.context.scene.hsi_properties.animation_sets)
    if animation_sets:
        # Get armature name from HSI addon properties
        armature_name = bpy.context.scene.hsi_properties.name_armature_CC
        if not armature_name:
            # Fallback: try to find any armature with CC bones
            for obj in bpy.data.objects:
                if obj.type == 'ARMATURE' and 'CC_Base_Hip' in obj.pose.bones:
                    armature_name = obj.name
                    break
        
        # Only return animation names that have the armature
        valid_animation_names = []
        for anim_key, anim_data in animation_sets.items():
            if anim_data and armature_name and armature_name in anim_data:
                valid_animation_names.append(anim_key)
        
        sys.stdout.write(json.dumps(valid_animation_names) + "\\n")
    else:
        sys.stdout.write("[]\\n")
except Exception as e:
    # Fallback to counting actions
    if bpy.data.actions:
        action_names = set()
        for action in bpy.data.actions:
            if action.name:
                base_name = action.name.split('.')[0]
                action_names.add(base_name)
        sys.stdout.write(json.dumps(list(action_names) if action_names else []) + "\\n")
    else:
        sys.stdout.write("[]\\n")
'''
        
        # Write temporary script
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script_content)
            temp_script = f.name
        
        cmd = ["blender", "--background", blend_file, "--python", temp_script]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        # Clean up temp file
        try:
            os.unlink(temp_script)
        except:
            pass
            
        if result.returncode == 0:
            # Filter out Blender warnings and get only the last line (our script output)
            lines = result.stdout.strip().split('\n')
            for line in reversed(lines):
                line = line.strip()
                if line and not line.startswith('WARN') and not line.startswith('Traceback'):
                    try:
                        animation_names = json.loads(line)
                        return animation_names
                    except (ValueError, json.JSONDecodeError):
                        continue
            # If no valid output found, return empty list
            print(f"Warning: Could not parse animation names for {os.path.basename(blend_file)}")
            return []
        else:
            print(f"Blender command failed for {os.path.basename(blend_file)}: {result.stderr}")
            return []
    except Exception as e:
        print(f"Exception getting animation names for {os.path.basename(blend_file)}: {e}")
        return []

def check_already_rendered(blend_file, save_path, status_report=None, script_path=None, grayscale=False, separate=False):
    """Check if a scene has already been rendered using status report if available.
    Returns: (is_complete, rendered_animations, missing_animations, all_animations)"""
    directory_name = os.path.basename(os.path.dirname(blend_file))
    blend_name = Path(blend_file).stem
    
    # Determine output format based on script path
    is_video_output = False
    video_type = None
    if script_path:
        script_name = os.path.basename(script_path)
        if script_name in ['blender_ego_static.py']:
            is_video_output = True
            video_type = "static"
        elif script_name in ['blender_ego_hand.py']:
            is_video_output = True
            video_type = "hands"
    
    # Use status report if available (preferred method - no Blender overhead)
    if status_report and "rendered_scenes_details" in status_report:
        # Use directory_name as the key (consistent with check_rendering_status.py)
        scene_key = directory_name
        
        # Try to find the scene in the status report using directory_name
        if scene_key in status_report["rendered_scenes_details"]:
            scene_details = status_report["rendered_scenes_details"][scene_key]
            complete_anims = scene_details["complete_animations"]
            incomplete_anims = scene_details.get("incomplete_animations", [])
            not_started_anims = scene_details.get("not_started_animations", [])
            expected_animations = scene_details["expected_animations"]
            
            # Handle both frame-based and video-based formats
            if incomplete_anims and isinstance(incomplete_anims[0], dict):
                # Video-based format: incomplete_anims contains dicts with "name" key
                incomplete_anim_names = [anim["name"] for anim in incomplete_anims]
                not_started_anim_names = not_started_anims if (not_started_anims and isinstance(not_started_anims[0], str)) else [anim["name"] for anim in not_started_anims]
            else:
                # Frame-based format: incomplete_anims contains strings
                incomplete_anim_names = incomplete_anims
                not_started_anim_names = not_started_anims
            
            # Get all animation names from the status report
            all_animations = complete_anims + incomplete_anim_names + not_started_anim_names
            
            # Calculate missing animations (incomplete + not started)
            missing_anims = incomplete_anim_names + not_started_anim_names
            
            # Check if scene is fully complete
            is_fully_complete = len(incomplete_anims) == 0 and len(not_started_anims) == 0
            
            if is_fully_complete:
                return True, complete_anims, missing_anims, all_animations
            else:
                return False, complete_anims, missing_anims, all_animations
        else:
            # Scene not found in rendered_scenes_details, check if it's in incomplete_animations_by_scene
            if "incomplete_animations_by_scene" in status_report and scene_key in status_report["incomplete_animations_by_scene"]:
                incomplete_anims = status_report["incomplete_animations_by_scene"][scene_key]
                if incomplete_anims and isinstance(incomplete_anims[0], dict):
                    # Video-based format
                    incomplete_anim_names = [anim["name"] for anim in incomplete_anims]
                else:
                    # Frame-based format
                    incomplete_anim_names = incomplete_anims
                return False, [], incomplete_anim_names, incomplete_anim_names
            # Check if it's in not_started_animations_by_scene
            elif "not_started_animations_by_scene" in status_report and scene_key in status_report["not_started_animations_by_scene"]:
                not_started_anims = status_report["not_started_animations_by_scene"][scene_key]
                if not_started_anims and isinstance(not_started_anims[0], dict):
                    # Video-based format
                    not_started_anim_names = [anim["name"] for anim in not_started_anims]
                else:
                    # Frame-based format
                    not_started_anim_names = not_started_anims
                return False, [], not_started_anim_names, not_started_anim_names
            # Check if it's in scenes with no animations
            elif "scenes_with_no_animations_list" in status_report and scene_key in status_report["scenes_with_no_animations_list"]:
                return False, [], [], []  # No animations
            else:
                # Scene not found in status report at all - fall back to Blender query
                print(f"Scene {scene_key} ({blend_name}) not found in status report, falling back to Blender query")
                # Debug: Show what scenes are available in the status report
                if "not_started_animations_by_scene" in status_report:
                    not_started_scenes = list(status_report["not_started_animations_by_scene"].keys())
                    print(f"DEBUG: Available not_started scenes: {not_started_scenes}")
                if "incomplete_animations_by_scene" in status_report:
                    incomplete_scenes = list(status_report["incomplete_animations_by_scene"].keys())
                    print(f"DEBUG: Available incomplete scenes: {incomplete_scenes}")
    
    # Only query Blender if we don't have a status report or scene wasn't found
    if not status_report:
        print(f"Querying Blender for animation data: {os.path.basename(blend_file)}")
        all_animations = get_animation_names(blend_file)
        print(all_animations)
        if not all_animations:
            print(f"Warning: No animations found in {os.path.basename(blend_file)}")
            return False, [], [], []
        expected_animations = len(all_animations)
        
        # Fallback to basic file system check
        output_base = os.path.join(save_path, directory_name)
        
        if not os.path.exists(output_base):
            return False, [], all_animations, all_animations
        
        # Check for animation folders
        rendered_animations = []
        for item in os.listdir(output_base):
            item_path = os.path.join(output_base, item)
            if os.path.isdir(item_path):
                if is_video_output:
                    # For video output scripts, check for video sequences
                    sequences_path = os.path.join(item_path, "sequences")
                    if not os.path.exists(sequences_path):
                        continue
                    
                    # Check for video files based on script type
                    if script_path and 'blender_ego_static.py' in script_path:
                        videos_path = os.path.join(sequences_path, "videos_static")
                    elif script_path and 'blender_ego_hand.py' in script_path:
                        # Use different path based on grayscale and separate options
                        if grayscale and separate:
                            # Separate mode: check both left and right video folders
                            videos_path_left = os.path.join(sequences_path, "videos_hands_gray_left")
                            videos_path_right = os.path.join(sequences_path, "videos_hands_gray_right")
                            videos_path = videos_path_left  # Use left as primary for existence check
                        elif grayscale:
                            videos_path = os.path.join(sequences_path, "videos_hands_gray")
                        else:
                            videos_path = os.path.join(sequences_path, "videos_hands")
                    else:
                        videos_path = sequences_path
                    
                    if os.path.exists(videos_path):
                        # Count video files based on mode
                        if grayscale and separate and script_path and 'blender_ego_hand.py' in script_path:
                            # Separate mode: count videos in both left and right folders
                            left_videos = 0
                            right_videos = 0
                            if os.path.exists(videos_path_left):
                                left_videos = len([f for f in os.listdir(videos_path_left) if f.endswith('.mp4')])
                            if os.path.exists(videos_path_right):
                                right_videos = len([f for f in os.listdir(videos_path_right) if f.endswith('.mp4')])
                            video_files = min(left_videos, right_videos)  # Complete only if both have same count
                        else:
                            # Normal mode: count videos in single folder
                            video_files = len([f for f in os.listdir(videos_path) if f.endswith('.mp4')])
                        
                        # Get expected video count for this specific animation
                        if script_path and 'blender_ego_static.py' in script_path:
                            video_type = "static"
                        elif script_path and 'blender_ego_hand.py' in script_path:
                            video_type = "hands"
                        else:
                            video_type = "static"  # Default
                        
                        expected_videos = get_expected_video_count(blend_file, item, frame_skip=3, stride=25, clip_length=49)
                        if expected_videos is None:
                            expected_videos = 0
                        
                        # Consider complete if we have the expected number of videos
                        if video_files >= expected_videos:
                            rendered_animations.append(item)
                else:
                    # For frame-based output (blender_ego_rgb_depth_optimized.py)
                    # Check if this animation folder has complete output
                    rgb_path = os.path.join(item_path, "images")  # Updated to match Blender script
                    depth_path = os.path.join(item_path, "depth")
                    cam_params_path = os.path.join(item_path, "cam_params")
                    
                    # Check if all required folders exist
                    if not all(os.path.exists(p) for p in [rgb_path, depth_path, cam_params_path]):
                        continue
                    
                    # Count files in each folder
                    rgb_files = len([f for f in os.listdir(rgb_path) if f.endswith('.png')])
                    depth_files = len([f for f in os.listdir(depth_path) if f.endswith('.exr')])
                    cam_files = len([f for f in os.listdir(cam_params_path) if f.startswith('cam')])
                    
                    # Get expected frame count for this animation
                    expected_frames = get_animation_frame_count(blend_file, item)
                    if expected_frames is None:
                        expected_frames = 0
                    
                    # Consider complete only if all file counts match expected frames
                    if (rgb_files == expected_frames and 
                        depth_files == expected_frames and 
                        cam_files == expected_frames):
                        rendered_animations.append(item)
        
        # Calculate missing animations
        missing_animations = [anim for anim in all_animations if anim not in rendered_animations]
        
        # Only consider "already rendered" if ALL expected animations are complete
        is_complete = len(rendered_animations) >= expected_animations
        return is_complete, rendered_animations, missing_animations, all_animations
    else:
        # We have a status report but this scene wasn't found in it
        print(f"Scene {blend_name} not found in status report, treating as not rendered")
        return False, [], [], []



def main():
    # Set up signal handlers for graceful termination
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl-C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal

    try:
        devnull = open(os.devnull, 'w')
    except Exception as e:
        print(f"Error opening os.devnull: {e}. All blender output will be visible.")
        devnull = None
    
    parser = argparse.ArgumentParser(description="Launch optimized Blender rendering jobs")
    parser.add_argument("--data-path", type=str, default=DEFAULT_RECORDINGS_PATH, 
                       help=f"Path to directory containing .blend files (default: {DEFAULT_RECORDINGS_PATH})")
    parser.add_argument("--save-path", type=str, default=DEFAULT_SAVE_PATH,
                       help=f"Path to save rendered outputs (default: {DEFAULT_SAVE_PATH})")
    parser.add_argument("--script-path", type=str, default=SCRIPT_PATH,
                       help=f"Path to Blender rendering script (default: {SCRIPT_PATH})")
    parser.add_argument("--check-only", action="store_true", help="Only check what's already rendered")
    parser.add_argument("--force", action="store_true", help="Force re-render even if already done")
    parser.add_argument("--max-gpus", type=int, default=NUM_GPUS, help="Maximum number of GPUs to use")
    parser.add_argument("--gpu-indices", type=str, help="Comma-separated list of GPU indices to use (e.g., '0,2,4,6'). Overrides --max-gpus")
    parser.add_argument("--use-status-report", action="store_true", help="Use rendering status report from check_rendering_status.py")
    parser.add_argument("--status-report-path", type=str, help="Custom path to rendering status report JSON file")
    parser.add_argument("--partial-render", action="store_true", default=True, help="Enable partial re-rendering of failed animations (default: True)")
    parser.add_argument("--full-scene", action="store_true", help="Force full scene re-rendering instead of partial re-rendering")
    parser.add_argument("--status-only", action="store_true", help="Only show current rendering status and exit")
    parser.add_argument("--show-incomplete", action="store_true", help="Show detailed information about incomplete animations")
    parser.add_argument("--show-rendered", action="store_true", help="Show detailed information about already rendered scenes")
    parser.add_argument("--frame-skip", type=int, default=3, help="Render every Nth frame (default: 1, i.e., all frames)")
    parser.add_argument("--fov", type=float, default=90.0, help="Camera field of view in degrees (default: 90)")
    parser.add_argument("--width", type=int, default=720, help="Render width in pixels (default: 720)")
    parser.add_argument("--height", type=int, default=480, help="Render height in pixels (default: 480)")
    parser.add_argument("--samples", type=int, default=32, help="Cycles samples for rendering (default: 16) - only applies to blender_ego_rgb_depth_optimized.py")
    parser.add_argument("--no-depth", action="store_true", help="Skip depth rendering (RGB only) - only applies to blender_ego_rgb_depth_optimized.py")
    parser.add_argument("--grayscale", action="store_true", help="Render hands in grayscale (black & white) - only applies to blender_ego_hand.py")
    parser.add_argument("--save-images", action="store_true", help="Save individual images instead of creating videos - only applies to blender_ego_hand.py")
    parser.add_argument("--separate", action="store_true", help="Create separate videos for left and right hands (only works with --grayscale) - only applies to blender_ego_hand.py")
    parser.add_argument("--no-skip-existing", action="store_true", help="Force re-render even if files already exist - only applies to blender_ego_hand.py")
    parser.add_argument("--parallel-animations", action="store_true", help="Launch separate jobs for each animation (faster for single scenes, more blend file overhead)")
    parser.add_argument("--scenes", type=str, nargs='+', help="Specific scene names (directory names) to render. If not specified, renders all scenes.")
    parser.add_argument("--scene-pattern", type=str, help="Pattern to match scene names (e.g., 'scene_*' or '*_walk')")

    args = parser.parse_args()
    
    # If status-only mode, just show current status
    if args.status_only:
        print("=" * 80)
        print("🎬 CURRENT RENDERING STATUS")
        print("=" * 80)
        
        # Check if there are any Blender processes running
        try:
            result = subprocess.run(['pgrep', '-f', 'blender.*background'], capture_output=True, text=True)
            if result.returncode == 0:
                pids = result.stdout.strip().split('\n')
                print(f"Found {len(pids)} active Blender processes:")
                for pid in pids:
                    if pid:
                        try:
                            # Get process info
                            ps_result = subprocess.run(['ps', '-p', pid, '-o', 'pid,ppid,cmd'], capture_output=True, text=True)
                            if ps_result.returncode == 0:
                                lines = ps_result.stdout.strip().split('\n')
                                if len(lines) > 1:
                                    print(f"  PID {pid}: {lines[1]}")
                        except:
                            print(f"  PID {pid}: (process info unavailable)")
            else:
                print("No active Blender processes found.")
        except Exception as e:
            print(f"Error checking processes: {e}")
        
        print("=" * 80)
        return
    
    # Discover all .blend files recursively
    blend_jobs = []
    for root, dirs, files in os.walk(args.data_path):
        for file in files:
            if file.endswith(".blend"):
                blend_jobs.append(os.path.join(root, file))
    
    print(f"Data path: {args.data_path}")
    print(f"Save path: {args.save_path}")
    print(f"Script path: {args.script_path}")
    print(f"Frame skip: {args.frame_skip} (rendering every {args.frame_skip}th frame)")
    print(f"Camera FOV: {args.fov} degrees")
    print(f"Resolution: {args.width}x{args.height} pixels")
    if args.samples and 'blender_ego_rgb_depth_optimized.py' in args.script_path:
        print(f"Cycles samples: {args.samples}")
    if args.grayscale and 'blender_ego_hand.py' in args.script_path:
        print(f"Grayscale mode: Enabled (hands will be rendered in black & white)")
    if args.save_images and 'blender_ego_hand.py' in args.script_path:
        print(f"Image mode: Enabled (individual images instead of videos)")
    if args.separate and 'blender_ego_hand.py' in args.script_path:
        print(f"Separate mode: Enabled (separate videos for left and right hands)")
    if args.no_skip_existing and 'blender_ego_hand.py' in args.script_path:
        print(f"No-skip-existing: Enabled (will force re-render existing files)")
    if 'blender_ego_hand.py' in args.script_path:
        print(f"Hand rendering resolution: {args.width}x{args.height} pixels")
    if args.no_depth and 'blender_ego_rgb_depth_optimized.py' in args.script_path:
        print(f"No-depth mode: Enabled (RGB only, no depth rendering)")
    print(f"Found {len(blend_jobs)} .blend files.")
    
    # Apply scene filtering if specified
    original_count = len(blend_jobs)
    blend_jobs = filter_scenes(blend_jobs, args.scenes, args.scene_pattern)
    
    if args.scenes:
        print(f"Filtered to {len(blend_jobs)} scenes matching specified names: {', '.join(args.scenes)}")
    elif args.scene_pattern:
        print(f"Filtered to {len(blend_jobs)} scenes matching pattern: {args.scene_pattern}")
    
    if len(blend_jobs) == 0:
        print("No scenes match the specified criteria!")
        return
    
    # Load status report if requested
    status_report = None
    if args.use_status_report or args.status_report_path:
        status_report = load_rendering_status_report(args.status_report_path, args.script_path)
        if status_report:
            script_name = os.path.basename(args.script_path)
            if script_name in ['blender_ego_static.py', 'blender_ego_hand.py']:
                report_source = args.status_report_path if args.status_report_path else "check_video_rendering_status.py"
            else:
                report_source = args.status_report_path if args.status_report_path else "check_rendering_status.py"
            
            print(f"✓ Using rendering status report from {report_source}")
            print(f"  Status report contains data for {len(status_report.get('rendered_scenes_details', {}))} scenes")
            
            # Show summary from status report (handle both frame-based and video-based formats)
            if "incomplete_animations_by_scene" in status_report:
                incomplete_by_scene = status_report["incomplete_animations_by_scene"]
                total_incomplete = sum(len(anims) for anims in incomplete_by_scene.values())
                if total_incomplete > 0:
                    print(f"  Found {total_incomplete} incomplete animations across {len(incomplete_by_scene)} scenes")
                    print("  Incomplete animations by scene:")
                    for scene, anims in incomplete_by_scene.items():
                        if anims and isinstance(anims[0], dict) and "name" in anims[0]:
                            # Video-based format
                            anim_names = [anim["name"] for anim in anims]
                        else:
                            # Frame-based format
                            anim_names = anims
                        print(f"    {scene}: {', '.join(anim_names[:5])}{'...' if len(anim_names) > 5 else ''}")
                    
                    # Show detailed information if requested
                    if args.show_incomplete:
                        print("\n  Detailed incomplete animation information:")
                        for scene, anims in incomplete_by_scene.items():
                            print(f"    {scene}:")
                            for anim in anims:
                                if isinstance(anim, dict) and "videos" in anim:
                                    # Video-based format
                                    print(f"      {anim['name']}: Videos {anim['videos']}/{anim['expected']}")
                                elif isinstance(anim, dict) and "rgb" in anim:
                                    # Frame-based format
                                    print(f"      {anim['name']}: RGB {anim['rgb']}/{anim['expected']}, "
                                          f"Depth {anim['depth']}/{anim['expected']}, Cam {anim['cam']}/{anim['expected']}")
            
            # Show not started animations summary
            if "not_started_animations_by_scene" in status_report:
                not_started_by_scene = status_report["not_started_animations_by_scene"]
                total_not_started = sum(len(anims) for anims in not_started_by_scene.values())
                if total_not_started > 0:
                    print(f"  Found {total_not_started} not started animations across {len(not_started_by_scene)} scenes")
            
            # Show scenes with no animations summary
            if "scenes_with_no_animations_list" in status_report:
                scenes_with_no_anims = status_report["scenes_with_no_animations_list"]
                if scenes_with_no_anims:
                    print(f"  Found {len(scenes_with_no_anims)} scenes with no animations")
            
            print(f"  This will significantly reduce Blender query overhead")
        else:
            if args.status_report_path:
                print(f"⚠️  Status report not found at {args.status_report_path}, falling back to basic file system check")
            else:
                print("⚠️  Status report not found, falling back to basic file system check")
                # Provide helpful guidance based on script type
                script_name = os.path.basename(args.script_path)
                if script_name in ['blender_ego_static.py']:
                    print(f"💡 To generate a status report for {script_name}, run:")
                    print(f"   python check_video_rendering_status.py --video-type static --stride 25 --clip-length 49")
                elif script_name in ['blender_ego_hand.py']:
                    print(f"💡 To generate a status report for {script_name}, run:")
                    print(f"   python check_video_rendering_status.py --video-type hands --stride 25 --clip-length 49")
                else:
                    print(f"💡 To generate a status report for {script_name}, run:")
                    print(f"   python check_rendering_status.py")
    else:
        print("ℹ️  No status report specified. Use --use-status-report or --status-report-path for faster checking.")
        
        # Provide helpful guidance based on script type
        script_name = os.path.basename(args.script_path)
        if script_name in ['blender_ego_static.py']:
            print(f"💡 To generate a status report for {script_name}, run:")
            print(f"   python check_video_rendering_status.py --video-type static --stride 25 --clip-length 49")
        elif script_name in ['blender_ego_hand.py']:
            print(f"💡 To generate a status report for {script_name}, run:")
            print(f"   python check_video_rendering_status.py --video-type hands --stride 25 --clip-length 49")
        else:
            print(f"💡 To generate a status report for {script_name}, run:")
            print(f"   python check_rendering_status.py")
    
    # Check what's already rendered
    rendered_scenes = {}
    pending_jobs = []
    
    for blend_file in blend_jobs:
        blend_name = Path(blend_file).stem
        directory_name = os.path.basename(os.path.dirname(blend_file))
        # Use directory_name as scene key for consistency with status report and actual output folders
        scene_name = directory_name
        display_name = directory_name  # For user-friendly display (same as scene_name in this case)
        is_rendered, rendered_anims, missing_anims, all_anims = check_already_rendered(blend_file, args.save_path, status_report, args.script_path, args.grayscale, args.separate)
        
        if is_rendered:
            rendered_scenes[scene_name] = rendered_anims
            if not args.force:
                print(f"✓ {display_name} ({scene_name}): Already rendered ({len(rendered_anims)} animations)")
                if len(rendered_anims) <= 10:  # Show animation names if not too many
                    for anim in rendered_anims:
                        print(f"    ✅ {anim}")
                else:
                    print(f"    ✅ {', '.join(rendered_anims[:5])}... and {len(rendered_anims)-5} more")
                continue
        
        if args.check_only:
            if missing_anims:
                print(f"⚠️  {display_name} ({scene_name}): Partially rendered ({len(rendered_anims)}/{len(all_anims)} animations)")
                print(f"    Missing: {', '.join(missing_anims)}")
            continue
        
        # Determine what needs to be rendered
        if not all_anims:
            print(f"⚠️  {display_name} ({scene_name}): No animations found, skipping")
            continue
            
        if args.full_scene or not args.partial_render:
            # Force full scene re-rendering
            animations_to_render = all_anims
            if missing_anims:
                print(f"🔄 {display_name} ({scene_name}): Re-rendering all {len(all_anims)} animations (forced full scene)")
            else:
                print(f"⏳ {display_name} ({scene_name}): Rendering all {len(all_anims)} animations")
        else:
            # Use partial re-rendering
            animations_to_render = missing_anims if missing_anims else all_anims
            
            if missing_anims:
                print(f"🔄 {display_name} ({scene_name}): Re-rendering {len(missing_anims)} missing animations: {', '.join(missing_anims)}")
            else:
                print(f"⏳ {display_name} ({scene_name}): Rendering all {len(all_anims)} animations")
        
        if args.parallel_animations:
            # Create jobs for individual animations (parallel processing)
            for anim_name in animations_to_render:
                # Find animation index
                try:
                    anim_index = all_anims.index(anim_name)
                except ValueError:
                    print(f"⚠️  Warning: Animation '{anim_name}' not found in blend file, skipping")
                    continue
                
                pending_jobs.append({
                    'file': blend_file,
                    'name': scene_name,
                    'display_name': display_name,
                    'animation_name': anim_name,
                    'animation_index': anim_index,
                    'job_type': 'animation'
                })
        else:
            # Create job for the entire scene (all animations)
            pending_jobs.append({
                'file': blend_file,
                'name': scene_name,
                'display_name': display_name,
                'animations_to_render': animations_to_render,
                'all_animations': all_anims,
                'job_type': 'scene'
            })
    
    if args.check_only:
        print(f"\n=== RENDERING STATUS ===")
        print(f"Total scenes: {len(blend_jobs)}")
        
        # If we have a status report, use its data directly for consistency
        if status_report:
            print("📊 Using data from status report for consistency")
            
            # Get counts from status report
            fully_rendered = len(status_report.get("rendered_scenes_details", {}))
            total_expected_animations = status_report.get("total_expected_animations", 0)
            complete_animations = status_report.get("complete_animations", 0)
            incomplete_animations = status_report.get("incomplete_animations", 0)
            not_started_animations = status_report.get("not_started_animations", 0)
            scenes_with_no_animations_count = status_report.get("scenes_with_no_animations", 0)
            
            print(f"Fully rendered: {fully_rendered}")
            print(f"Partially rendered: {len(status_report.get('incomplete_animations_by_scene', {}))}")
            print(f"Not rendered (has animations): {len(status_report.get('not_started_animations_by_scene', {}))}")
            print(f"No animations: {scenes_with_no_animations_count}")
            
            # Verify the math
            total_counted = fully_rendered + len(status_report.get('incomplete_animations_by_scene', {})) + len(status_report.get('not_started_animations_by_scene', {})) + scenes_with_no_animations_count
            if total_counted != len(blend_jobs):
                print(f"⚠️  WARNING: Count mismatch! {fully_rendered} + {len(status_report.get('incomplete_animations_by_scene', {}))} + {len(status_report.get('not_started_animations_by_scene', {}))} + {scenes_with_no_animations_count} = {total_counted} != {len(blend_jobs)}")
            
            # Show animation summary from status report
            print(f"\n📊 ANIMATION SUMMARY (from status report):")
            print(f"Total expected animations: {total_expected_animations}")
            print(f"Complete animations: {complete_animations}")
            print(f"Incomplete animations: {incomplete_animations}")
            print(f"Not started animations: {not_started_animations}")
            print(f"Verification: {complete_animations} + {incomplete_animations} + {not_started_animations} = {complete_animations + incomplete_animations + not_started_animations} (expected: {total_expected_animations})")
            
        else:
            # Fallback to manual counting when no status report
            print(f"Fully rendered: {len(rendered_scenes)}")
            
            # Count partially rendered, not rendered, and no animation scenes
            partially_rendered = 0
            not_rendered = 0
            no_animations = 0
            
            # Track scenes we've already processed to avoid double counting
            processed_scenes = set()
            
            for blend_file in blend_jobs:
                blend_name = Path(blend_file).stem
                directory_name = os.path.basename(os.path.dirname(blend_file))
                output_path = os.path.join(args.save_path, directory_name)
                
                if directory_name not in rendered_scenes and directory_name not in processed_scenes:
                    processed_scenes.add(directory_name)
                    
                    # Check if this scene has animations
                    _, _, _, all_anims = check_already_rendered(blend_file, args.save_path, status_report, args.script_path, args.grayscale, args.separate)
                    if not all_anims:
                        no_animations += 1
                    else:
                        # Scene has animations, check if it's partially rendered or not rendered
                        if os.path.exists(output_path):
                            # Check if there are any animation folders
                            has_any_animations = False
                            try:
                                for item in os.listdir(output_path):
                                    item_path = os.path.join(output_path, item)
                                    if os.path.isdir(item_path):
                                        # Check if this looks like an animation folder
                                        rgb_path = os.path.join(item_path, "rgb")
                                        depth_path = os.path.join(item_path, "depth")
                                        if os.path.exists(rgb_path) or os.path.exists(depth_path):
                                            has_any_animations = True
                                            break
                            except:
                                pass
                            
                            if has_any_animations:
                                partially_rendered += 1
                            else:
                                not_rendered += 1
                        else:
                            not_rendered += 1
            
            # Verify the math
            total_counted = len(rendered_scenes) + partially_rendered + not_rendered + no_animations
            if total_counted != len(blend_jobs):
                print(f"⚠️  WARNING: Count mismatch! {len(rendered_scenes)} + {partially_rendered} + {not_rendered} + {no_animations} = {total_counted} != {len(blend_jobs)}")
                print(f"   This suggests some scenes were counted multiple times or missed")
                
                # Debug: show which scenes are in rendered_scenes
                print(f"   Rendered scenes: {sorted(list(rendered_scenes.keys()))}")
                
                # Debug: show which scenes were processed
                print(f"   Processed scenes: {sorted(list(processed_scenes))}")
                
                # Find scenes that might be missing
                all_scene_names = set(os.path.basename(os.path.dirname(f)) for f in blend_jobs)
                counted_scenes = set(rendered_scenes.keys()) | processed_scenes
                missing_scenes = all_scene_names - counted_scenes
                if missing_scenes:
                    print(f"   Missing scenes: {sorted(list(missing_scenes))}")
            
            print(f"Partially rendered: {partially_rendered}")
            print(f"Not rendered (has animations): {not_rendered}")
            print(f"No animations: {no_animations}")
        
        print(f"\n=== FULLY RENDERED SCENES ===")
        for scene, animations in rendered_scenes.items():
            print(f"{scene}: {len(animations)} animations")
            if len(animations) <= 15:  # Show all animation names if not too many
                for anim in animations:
                    print(f"  ✅ {anim}")
            else:
                # Show first few and last few
                for anim in animations[:7]:
                    print(f"  ✅ {anim}")
                print(f"  ... ({len(animations)-14} more animations) ...")
                for anim in animations[-7:]:
                    print(f"  ✅ {anim}")
        
        # Show summary of what's already rendered
        if status_report:
            # Use status report data for consistency
            total_rendered_animations = status_report.get("complete_animations", 0)
            total_expected = status_report.get("total_expected_animations", 0)
            incomplete_animations = status_report.get("incomplete_animations", 0)
            not_started_animations = status_report.get("not_started_animations", 0)
            
            print(f"\n📊 SUMMARY: {total_rendered_animations} animations already rendered across {len(status_report.get('rendered_scenes_details', {}))} scenes")
            
            if total_expected > 0:
                progress_percent = (total_rendered_animations / total_expected) * 100
                print(f"📈 PROGRESS: {progress_percent:.1f}% complete ({total_rendered_animations}/{total_expected} animations)")
                print(f"📊 BREAKDOWN: {len(status_report.get('rendered_scenes_details', {}))} fully rendered scenes, {len(status_report.get('incomplete_animations_by_scene', {}))} scenes with incomplete animations, {len(status_report.get('not_started_animations_by_scene', {}))} scenes not started")
                print(f"💡 Missing animations: {incomplete_animations + not_started_animations} (incomplete: {incomplete_animations}, not started: {not_started_animations})")
        else:
            # Fallback to manual calculation when no status report
            total_rendered_animations = sum(len(anims) for anims in rendered_scenes.values())
            print(f"\n📊 SUMMARY: {total_rendered_animations} animations already rendered across {len(rendered_scenes)} scenes")
            
            # Calculate total expected animations for progress percentage
            total_expected = 0
            scenes_with_estimates = 0
            scenes_with_status = 0
            scenes_with_default = 0
            
            # First pass: count rendered animations
            for blend_file in blend_jobs:
                directory_name = os.path.basename(os.path.dirname(blend_file))
                if directory_name in rendered_scenes:
                    total_expected += len(rendered_scenes[directory_name])
            
            # Second pass: count actual missing animations from scenes that have animations
            actual_missing_total = 0
            for blend_file in blend_jobs:
                directory_name = os.path.basename(os.path.dirname(blend_file))
                if directory_name not in rendered_scenes:
                    # Force real-time Blender query instead of using status report for accuracy
                    _, _, missing_anims, all_anims = check_already_rendered(blend_file, args.save_path, None, args.script_path, args.grayscale, args.separate)  # Pass None to force Blender query
                    if all_anims:  # Scene has animations
                        actual_missing_total += len(missing_anims)
                        total_expected += len(all_anims)  # Add all animations (rendered + missing)
                        if status_report and "rendered_scenes_details" in status_report and directory_name in status_report["rendered_scenes_details"]:
                            scenes_with_status += 1
                        else:
                            scenes_with_default += 1
                    else:
                        scenes_with_estimates += 1  # Scene has no animations
            
            # Verify the calculation is correct
            calculated_total = total_rendered_animations + actual_missing_total
            if calculated_total != total_expected:
                print(f"⚠️  WARNING: Calculation mismatch! Using corrected total: {calculated_total}")
                total_expected = calculated_total
            
            if total_expected > 0:
                progress_percent = (total_rendered_animations / total_expected) * 100
                print(f"📈 PROGRESS: {progress_percent:.1f}% complete ({total_rendered_animations}/{total_expected} animations)")
                print(f"📊 BREAKDOWN: {len(rendered_scenes)} fully rendered scenes, {scenes_with_status} scenes with animations (not rendered), {scenes_with_estimates} scenes with no animations")
                print(f"💡 Actual missing animations: {actual_missing_total}")
        
        # Show scenes with no animations
        print(f"\n=== SCENES WITH NO ANIMATIONS ===")
        if status_report and "scenes_with_no_animations_list" in status_report:
            scenes_with_no_anims = status_report["scenes_with_no_animations_list"]
            if scenes_with_no_anims:
                print(f"Found {len(scenes_with_no_anims)} scenes with no animations:")
                for scene in scenes_with_no_anims:
                    print(f"  ❌ {scene}")
            else:
                print("No scenes found without animations.")
        else:
            # Fallback to manual detection
            scenes_with_no_anims = []
            for blend_file in blend_jobs:
                directory_name = os.path.basename(os.path.dirname(blend_file))
                if directory_name not in rendered_scenes:
                    # Force real-time Blender query for accuracy
                    _, _, _, all_anims = check_already_rendered(blend_file, args.save_path, None, args.script_path, args.grayscale, args.separate)  # Pass None to force Blender query
                    if not all_anims:
                        scenes_with_no_anims.append(directory_name)
            
            if scenes_with_no_anims:
                print(f"Found {len(scenes_with_no_anims)} scenes with no animations:")
                for scene in scenes_with_no_anims:
                    print(f"  ❌ {scene}")
            else:
                print("No scenes found without animations.")
        
        # Show missing animations from scenes that have animations but aren't rendered
        print(f"\n=== MISSING ANIMATIONS (from scenes with animations) ===")
        if status_report:
            # Use status report data
            incomplete_by_scene = status_report.get("incomplete_animations_by_scene", {})
            not_started_by_scene = status_report.get("not_started_animations_by_scene", {})
            
            missing_animations_total = 0
            scenes_with_missing_anims = 0
            
            # Show incomplete animations
            for scene, anims in incomplete_by_scene.items():
                scenes_with_missing_anims += 1
                missing_animations_total += len(anims)
                print(f"  {scene}: {len(anims)} incomplete animations")
                if len(anims) <= 3:  # Show animation names if not too many
                    for anim in anims:
                        if isinstance(anim, dict) and "videos" in anim:
                            # Video-based format
                            print(f"    ⚠️  {anim['name']}: Videos {anim['videos']}/{anim['expected']}")
                        elif isinstance(anim, dict) and "rgb" in anim:
                            # Frame-based format
                            print(f"    ⚠️  {anim['name']}: RGB {anim['rgb']}/{anim['expected']}, Depth {anim['depth']}/{anim['expected']}, Cam {anim['cam']}/{anim['expected']}")
                        else:
                            # String format
                            print(f"    ⚠️  {anim}")
            
            # Show not started animations
            for scene, anims in not_started_by_scene.items():
                scenes_with_missing_anims += 1
                missing_animations_total += len(anims)
                print(f"  {scene}: {len(anims)} not started animations")
                if len(anims) <= 3:  # Show animation names if not too many
                    for anim in anims:
                        print(f"    ❌ {anim}")
            
            if missing_animations_total > 0:
                print(f"\n📊 Total missing animations: {missing_animations_total} across {scenes_with_missing_anims} scenes")
            else:
                print("No missing animations found.")
        else:
            # Fallback to manual detection
            missing_animations_total = 0
            scenes_with_missing_anims = 0
            
            for blend_file in blend_jobs:
                directory_name = os.path.basename(os.path.dirname(blend_file))
                if directory_name not in rendered_scenes:
                    # Force real-time Blender query for accuracy
                    _, _, missing_anims, all_anims = check_already_rendered(blend_file, args.save_path, None, args.script_path, args.grayscale, args.separate)  # Pass None to force Blender query
                    if all_anims and missing_anims:  # Scene has animations but some are missing
                        scenes_with_missing_anims += 1
                        missing_animations_total += len(missing_anims)
                        print(f"  {directory_name}: {len(missing_anims)} missing animations")
                        if len(missing_anims) <= 5:  # Show animation names if not too many
                            for anim in missing_anims:
                                print(f"    ⚠️  {anim}")
            
            if missing_animations_total > 0:
                print(f"\n📊 Total missing animations: {missing_animations_total} across {scenes_with_missing_anims} scenes")
            else:
                print("No missing animations found.")
        
        return
    
    # Show summary of what's already rendered
    if rendered_scenes:
        total_rendered_animations = sum(len(anims) for anims in rendered_scenes.values())
        print(f"\n📊 ALREADY RENDERED: {total_rendered_animations} animations across {len(rendered_scenes)} scenes")
        
        if args.show_rendered or len(rendered_scenes) <= 5:
            print("\n=== ALREADY RENDERED SCENES ===")
            for scene, animations in rendered_scenes.items():
                print(f"✓ {scene}: {len(animations)} animations")
                if args.show_rendered:
                    if len(animations) <= 20:
                        for anim in animations:
                            print(f"  ✅ {anim}")
                    else:
                        # Show first few and last few
                        for anim in animations[:10]:
                            print(f"  ✅ {anim}")
                        print(f"  ... ({len(animations)-20} more animations) ...")
                        for anim in animations[-10:]:
                            print(f"  ✅ {anim}")
                elif len(animations) <= 10:
                    for anim in animations:
                        print(f"  ✅ {anim}")
    
    if not pending_jobs:
        print("\n🎉 All scenes are already rendered!")
        return
    
    # Count total animations for summary
    total_animations = 0
    scene_jobs = 0
    animation_jobs = 0
    
    for job in pending_jobs:
        job_type = job.get('job_type', 'scene')
        if job_type == 'animation':
            animation_jobs += 1
            total_animations += 1
        else:
            scene_jobs += 1
            total_animations += len(job.get('animations_to_render', []))
    
    if args.parallel_animations:
        print(f"\n🚀 Starting rendering for {len(pending_jobs)} animation jobs ({total_animations} total animations)...")
        print(f"📊 Job breakdown: {animation_jobs} animation jobs, {scene_jobs} scene jobs")
    else:
        print(f"\n🚀 Starting rendering for {len(pending_jobs)} scene jobs ({total_animations} total animations)...")
        print(f"📊 Job breakdown: {scene_jobs} scene jobs, {animation_jobs} animation jobs")
    
    # Sort jobs by scene name for consistent ordering
    pending_jobs.sort(key=lambda x: x['name'])
    
    # Determine which GPUs to use
    if args.gpu_indices:
        # Parse comma-separated GPU indices
        try:
            gpu_indices = [int(idx.strip()) for idx in args.gpu_indices.split(',')]
            print(f"🎯 Using specified GPU indices: {gpu_indices}")
        except ValueError as e:
            print(f"❌ Error parsing GPU indices '{args.gpu_indices}': {e}")
            print("Please use comma-separated integers (e.g., '0,2,4,6')")
            return
        available_gpus = deque(gpu_indices)
        max_gpus = len(gpu_indices)
    else:
        # Use sequential GPUs from 0 to max_gpus-1
        available_gpus = deque(range(args.max_gpus))
        max_gpus = args.max_gpus
    
    # Track running processes with GPU assignment (using global variable)
    global running_processes
    running_processes = {}  # GPU_ID -> list of (process, job_info) tuples
    
    if args.parallel_animations:
        print(f"🚀 GPU Configuration: {max_gpus} GPUs, 1 animation per GPU (max parallelization)")
        print(f"📊 Total concurrent animations: {max_gpus}")
    else:
        print(f"🚀 GPU Configuration: {max_gpus} GPUs, 1 scene per GPU (optimal for Blender)")
        print(f"📊 Total concurrent scenes: {max_gpus}")
    
    # Dispatch jobs with proper GPU distribution
    for job in pending_jobs:
        # Wait for a GPU to become available
        while len(available_gpus) == 0:
            # Check if any processes have finished
            finished_gpus = []
            for gpu_id, process_jobs in running_processes.items():
                # Remove finished processes
                process_jobs[:] = [(p, job_info) for p, job_info in process_jobs if p.poll() is None]
                # If no processes left on this GPU, mark it as available
                if len(process_jobs) == 0:
                    finished_gpus.append(gpu_id)
            
            # Add finished GPUs back to available queue
            for gpu_id in finished_gpus:
                available_gpus.append(gpu_id)
                del running_processes[gpu_id]
            
            if len(available_gpus) == 0:
                print("All GPUs busy, waiting...")
                time.sleep(10)  # Wait before checking again
        
        # Get next available GPU
        gpu_id = available_gpus.popleft()
        
        cmd = [
            "blender",
            "--background",
            job['file'],
            "--python",
            args.script_path,
            "--",
            "--save-path", args.save_path,
            "--frame-skip", str(args.frame_skip),
            "--fov", str(args.fov),
            "--width", str(args.width),
            "--height", str(args.height)
        ]
        
        # Add grayscale option if specified and using blender_ego_hand.py
        if args.grayscale and 'blender_ego_hand.py' in args.script_path:
            cmd.append("--grayscale")
        
        # Add save-images option if specified and using blender_ego_hand.py
        if args.save_images and 'blender_ego_hand.py' in args.script_path:
            cmd.append("--save-images")
        
        # Add separate option if specified and using blender_ego_hand.py
        if args.separate and 'blender_ego_hand.py' in args.script_path:
            cmd.append("--separate")
        
        # Add no-skip-existing option if specified and using blender_ego_hand.py
        if args.no_skip_existing and 'blender_ego_hand.py' in args.script_path:
            cmd.append("--no-skip-existing")
        
        # Add width and height options for blender_ego_hand.py
        if 'blender_ego_hand.py' in args.script_path:
            cmd.extend(["--width", str(args.width)])
            cmd.extend(["--height", str(args.height)])
        
        # Add samples option if specified and using blender_ego_rgb_depth_optimized.py
        if 'blender_ego_rgb_depth_optimized.py' in args.script_path:
            cmd.extend(["--samples", str(args.samples)])
        
        # Add no-depth option if specified and using blender_ego_rgb_depth_optimized.py
        if args.no_depth and 'blender_ego_rgb_depth_optimized.py' in args.script_path:
            cmd.append("--no-depth")
        
        # Add animation index for animation-based jobs
        if job.get('job_type') == 'animation':
            cmd.extend(["--animation_index", str(job['animation_index'])])
        
        # Assign environment variable to restrict GPU usage
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        display_name = job.get('display_name', job['name'])
        job_type = job.get('job_type', 'scene')
        
        if job_type == 'animation':
            anim_info = f" - Animation: {job['animation_name']}"
        else:
            animations_count = len(job.get('animations_to_render', []))
            anim_info = f" - {animations_count} animations" if animations_count > 0 else ""
        
        frame_info = f" (frame skip: {args.frame_skip}, FOV: {args.fov}°)" if args.frame_skip > 1 or args.fov != 90.0 else ""
        print(f"[GPU {gpu_id}] Launching Blender for: {display_name} ({job['name']}){anim_info}{frame_info}")
        
        # Print the command being executed
        print(f"[GPU {gpu_id}] Command: {' '.join(cmd)}")
        
        # proc = subprocess.Popen(cmd, env=env)
        proc = subprocess.Popen(cmd, env=env, stdout=devnull, stderr=devnull)

        
        # Check if process started successfully
        time.sleep(1)  # Give it a moment to start
        if proc.poll() is not None:
            print(f"⚠️  GPU {gpu_id}: Process failed to start or completed immediately (return code: {proc.returncode})")
            anim_info = f" - Animation: {job['animation_name']}" if 'animation_name' in job else ""
            failed_jobs.append({
                'name': f"{display_name} ({job['name']})" + anim_info,
                'file': job['file'],
                'gpu': gpu_id,
                'error': f"Process failed to start (return code: {proc.returncode})"
            })
        else:
            print(f"✓ GPU {gpu_id}: Process started successfully (PID: {proc.pid})")
        
        # Track this process on the assigned GPU
        if gpu_id not in running_processes:
            running_processes[gpu_id] = []
        # Store process and job info together
        job_info = {
            'name': display_name,
            'file': job['file'],
            'animation_name': job.get('animation_name', 'N/A'),
            'job_type': job.get('job_type', 'scene')
        }
        running_processes[gpu_id].append((proc, job_info))
    
    # Wait for all remaining processes to finish
    print("Waiting for all processes to complete...")
    print("Press Ctrl-C to terminate all jobs gracefully.")
    
    try:
        while running_processes:
            finished_gpus = []
            for gpu_id, process_jobs in running_processes.items():
                # Check for failed processes and add them to failed_jobs
                for proc, job_info in process_jobs[:]:  # Copy list to avoid modification during iteration
                    if proc.poll() is not None:  # Process has finished
                        if proc.returncode != 0:  # Process failed
                            # Capture output from failed process
                            try:
                                stdout, stderr = proc.communicate(timeout=5)  # Get output with timeout
                                # Handle None values safely
                                stderr_str = stderr.strip() if stderr else ""
                                stdout_str = stdout.strip() if stdout else ""
                                error_output = stderr_str if stderr_str else stdout_str
                                if not error_output:
                                    error_output = f"Process failed with return code: {proc.returncode} (no output captured)"
                            except subprocess.TimeoutExpired:
                                proc.kill()  # Force kill if it hangs
                                error_output = f"Process failed with return code: {proc.returncode} (timeout getting output)"
                            
                            # Add to failed jobs with proper job info
                            failed_jobs.append({
                                'name': f"{job_info['name']} ({job_info['animation_name']})" if job_info['job_type'] == 'animation' else job_info['name'],
                                'file': job_info['file'],
                                'gpu': gpu_id,
                                'error': error_output
                            })
                            
                            # Print immediate error for debugging
                            job_display = f"{job_info['name']} - {job_info['animation_name']}" if job_info['job_type'] == 'animation' else job_info['name']
                            print(f"\n❌ GPU {gpu_id}: Job failed - {job_display}")
                            print(f"Return code: {proc.returncode}")
                            print(f"Error output:")
                            print(f"{'='*60}")
                            print(error_output)
                            print(f"{'='*60}")
                        process_jobs.remove((proc, job_info))
                
                # If no processes left on this GPU, mark it as finished
                if len(process_jobs) == 0:
                    finished_gpus.append(gpu_id)
                    print(f"GPU {gpu_id} completed all jobs")
            
            # Remove finished GPUs
            for gpu_id in finished_gpus:
                del running_processes[gpu_id]
            
            if running_processes:
                active_count = sum(len(process_jobs) for process_jobs in running_processes.values())
                print(f"Still waiting for {active_count} processes on {len(running_processes)} GPUs...")
                time.sleep(30)
        
        if args.parallel_animations:
            print("All Blender animation render jobs completed.")
        else:
            print("All Blender scene render jobs completed.")
        
        # Report failed jobs
        if failed_jobs:
            print(f"\n{'='*60}")
            print(f"FAILED JOBS ({len(failed_jobs)}):")
            print(f"{'='*60}")
            for job in failed_jobs:
                print(f"Scene: {job['name']}")
                print(f"File: {job['file']}")
                print(f"GPU: {job['gpu']}")
                print(f"Error: {job['error']}")
                print("-" * 40)
            print(f"\nCheck the error log for detailed Blender errors: {os.path.join(args.save_path, 'rendering_errors.log')}")
        else:
            print("\nAll jobs completed successfully!")
            
    except KeyboardInterrupt:
        # This will be handled by the signal handler
        pass

if __name__ == "__main__":
    main() 
