#!/usr/bin/env python3
"""
Check video rendering status for Trumans dataset.
Shows what scenes and animations are already rendered for video-based scripts.
"""

import os
import json
from pathlib import Path

def get_expected_video_count(blend_file, animation_name, frame_skip=3, stride=25, clip_length=49):
    """Get the expected number of video sequences from a blend file for a specific animation."""
    try:
        import subprocess
        import tempfile
        
        # Create a temporary Python script to avoid command-line parsing issues
        script_content = f'''import bpy
import json
import sys

try:
    # Try to access HSI addon properties
    animation_sets = json.loads(bpy.context.scene.hsi_properties.animation_sets)
    sys.stdout.write("DEBUG: Available animation names: " + str(list(animation_sets.keys())) + "\\n")
    sys.stdout.write("DEBUG: Looking for: " + "{animation_name}" + "\\n")
    
    if animation_sets and "{animation_name}" in animation_sets:
        # Get the specific animation data
        anim_data = animation_sets["{animation_name}"]
        if anim_data:
            # Get armature name from HSI addon properties
            armature_name = bpy.context.scene.hsi_properties.name_armature_CC
            if not armature_name:
                # Fallback: try to find any armature with CC bones
                for obj in bpy.data.objects:
                    if obj.type == 'ARMATURE' and 'CC_Base_Hip' in obj.pose.bones:
                        armature_name = obj.name
                        break
            
            if armature_name and armature_name in anim_data:
                action_name = anim_data[armature_name]
                if action_name in bpy.data.actions:
                    action = bpy.data.actions[action_name]
                    frame_range = action.frame_range
                    total_frames = int(frame_range[1] - frame_range[0] + 1)
                else:
                    total_frames = bpy.context.scene.frame_end - bpy.context.scene.frame_start + 1
            else:
                total_frames = bpy.context.scene.frame_end - bpy.context.scene.frame_start + 1
        else:
            total_frames = bpy.context.scene.frame_end - bpy.context.scene.frame_start + 1
    else:
        # Try to find a matching animation name (case-insensitive or partial match)
        found_match = False
        for anim_key in animation_sets.keys():
            if anim_key.lower() == "{animation_name}".lower() or "{animation_name}".lower() in anim_key.lower():
                anim_data = animation_sets[anim_key]
                if anim_data:
                    # Get armature name from HSI addon properties
                    armature_name = bpy.context.scene.hsi_properties.name_armature_CC
                    if not armature_name:
                        # Fallback: try to find any armature with CC bones
                        for obj in bpy.data.objects:
                            if obj.type == 'ARMATURE' and 'CC_Base_Hip' in obj.pose.bones:
                                armature_name = obj.name
                                break
                    
                    if armature_name and armature_name in anim_data:
                        action_name = anim_data[armature_name]
                        if action_name in bpy.data.actions:
                            action = bpy.data.actions[action_name]
                            frame_range = action.frame_range
                            total_frames = int(frame_range[1] - frame_range[0] + 1)
                            found_match = True
                            break
                    else:
                        total_frames = bpy.context.scene.frame_end - bpy.context.scene.frame_start + 1
                        found_match = True
                        break
        
        if not found_match:
            total_frames = bpy.context.scene.frame_end - bpy.context.scene.frame_start + 1
    
    # Calculate video sequences based on the same logic as the rendering scripts
    render_start_frame = 0  # Assuming frame_start is 0
    render_end_frame = total_frames - 1
    
    # Video sequence parameters (same as in blender_ego_static.py and blender_ego_hand.py)
    effective_stride = {stride} * {frame_skip}  # Actual frame stride
    video_start_frames = []
    current_frame = render_start_frame
    while current_frame + ({clip_length} - 1) * {frame_skip} <= render_end_frame:
        video_start_frames.append(current_frame)
        current_frame += effective_stride
    
    expected_videos = len(video_start_frames)
    sys.stdout.write(str(expected_videos) + "\\n")
except Exception as e:
    # Fallback to a reasonable default
    sys.stdout.write("10\\n")
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
                        return int(line)
                    except ValueError:
                        continue
            # If no valid output found, use default
            return 10
        else:
            print(f"Blender command failed for {os.path.basename(blend_file)}: {result.stderr}")
            return 10  # Default fallback
    except Exception as e:
        print(f"Exception getting video count for {os.path.basename(blend_file)}: {e}")
        return 10  # Default fallback

def check_video_sequence_completeness(anim_path, expected_videos, video_type="static"):
    """
    Check if an animation folder has all expected video sequences rendered.
    Uses filesize threshold to detect incomplete files from interrupted renders.
    """
    def is_valid_video(video_path, min_size=10240):
        """Video must exist, have reasonable size, and not be too recent"""
        if not os.path.isfile(video_path):
            return False
        size = os.path.getsize(video_path)
        if size < min_size:  # Too small = likely incomplete
            return False
        # Check if file was modified very recently (< 2 seconds ago)
        import time
        mtime = os.path.getmtime(video_path)
        if time.time() - mtime < 2.0:
            return False  # File too fresh, might still be writing
        return True
    
    sequences_path = os.path.join(anim_path, "sequences")
    videos_path = os.path.join(sequences_path, f"videos_{video_type}")
    
    # Check if videos folder exists
    if not os.path.exists(videos_path):
        return False, 0
    
    # Count valid video files (with size check)
    video_files = [f for f in os.listdir(videos_path) if f.endswith('.mp4')]
    valid_video_count = 0
    for video_file in video_files:
        video_full_path = os.path.join(videos_path, video_file)
        if is_valid_video(video_full_path, min_size=10240):  # 10KB minimum
            valid_video_count += 1
    
    # Check if we have the expected number of valid videos
    is_complete = valid_video_count >= expected_videos
    
    return is_complete, valid_video_count

def get_animation_names(blend_file):
    """Get the names of animations from a blend file."""
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

def check_video_rendering_status(frame_skip=3, stride=25, clip_length=49, video_type="static"):
    """Check what's already rendered in the Trumans dataset for video-based rendering scripts."""
    
    # Configuration
    recordings_path = "data/trumans/Recordings_blend"
    output_base = "data/trumans/ego_render_fov90_480p_s32"
    
    print(f"Checking video rendering status for {video_type} videos")
    print(f"Using frame skip: {frame_skip}, stride: {stride}, clip length: {clip_length}")
    
    # Load existing status report if available
    existing_status_report = None
    status_report_path = f"rendering_status_report_{video_type}.json"
    if os.path.exists(status_report_path):
        try:
            with open(status_report_path, 'r') as f:
                existing_status_report = json.load(f)
            print(f"✓ Loaded existing status report: {status_report_path}")
            print(f"  Contains data for {len(existing_status_report.get('rendered_scenes_details', {}))} scenes")
        except Exception as e:
            print(f"⚠️  Could not load existing status report: {e}")
    
    # Create a set of scenes that are already fully rendered according to the status report
    fully_rendered_scenes = set()
    if existing_status_report and "rendered_scenes_details" in existing_status_report:
        for scene_key, scene_details in existing_status_report["rendered_scenes_details"].items():
            # Check if scene is fully complete (no incomplete or not started animations)
            incomplete_count = len(scene_details.get("incomplete_animations", []))
            not_started_count = len(scene_details.get("not_started_animations", []))
            if incomplete_count == 0 and not_started_count == 0:
                fully_rendered_scenes.add(scene_key)
        
        print(f"  Found {len(fully_rendered_scenes)} fully rendered scenes in status report")
        print(f"  Will skip these scenes to speed up checking")
    
    # Find all blend files
    blend_files = []
    for root, dirs, files in os.walk(recordings_path):
        for file in files:
            if file.endswith(".blend"):
                blend_files.append(os.path.join(root, file))
    
    print(f"Found {len(blend_files)} .blend files")
    print("=" * 80)
    
    # Check each scene
    rendered_scenes = {}
    total_animations = 0
    rendered_animations = 0
    incomplete_animations_by_scene = {}  # Organize by scene
    not_started_animations_by_scene = {}  # Track animations not started
    scenes_with_no_animations = []  # Track scenes with no animations
    
    for blend_file in blend_files:
        blend_name = Path(blend_file).stem
        # Get the directory name from the blend file path
        directory_name = os.path.basename(os.path.dirname(blend_file))
        # Create the expected output folder name using directory name only
        scene_output = os.path.join(output_base, directory_name)
        
        # Use directory_name as the key for consistency with actual output folders
        scene_key = directory_name
        
        # Check if this scene is already fully rendered according to the status report
        if scene_key in fully_rendered_scenes:
            # Use existing data from status report
            scene_details = existing_status_report["rendered_scenes_details"][scene_key]
            rendered_scenes[scene_key] = scene_details
            rendered_animations += len(scene_details["complete_animations"])
            total_animations += scene_details["expected_animations"]
            print(f"✓ {scene_key} ({blend_name}): Already fully rendered (using status report data)")
            continue
        
        # Check if this scene has existing data in the status report (but not fully rendered)
        existing_scene_data = None
        if existing_status_report and "rendered_scenes_details" in existing_status_report and scene_key in existing_status_report["rendered_scenes_details"]:
            existing_scene_data = existing_status_report["rendered_scenes_details"][scene_key]
            print(f"📊 {scene_key} ({blend_name}): Using existing status report data (not fully rendered)")
        else:
            # Get expected animation count (only for new scenes or scenes not in status report)
            animation_names = get_animation_names(blend_file)
            expected_animations = len(animation_names) if animation_names else 0
        
        if existing_scene_data is None:
            # New scene or scene not in status report - need to check everything
            if expected_animations == 0:
                print(f"⚠️  {scene_key} ({blend_name}): No animations found")
                scenes_with_no_animations.append(scene_key)
                continue
            
            print(f"📊 {scene_key} ({blend_name}): Expected {expected_animations} animations")
        else:
            # Use existing data from status report
            expected_animations = existing_scene_data["expected_animations"]
            print(f"📊 {scene_key} ({blend_name}): Expected {expected_animations} animations (from status report)")
        
        if not os.path.exists(scene_output):
            print(f"❌ {scene_key} ({blend_name}): Not rendered")
            # For scenes not rendered at all, check if they have animations
            if existing_scene_data is None:
                # Need to check blend file for animations
                animation_names = get_animation_names(blend_file)
                if not animation_names:
                    print(f"  ⚠️  No animations found in this scene, skipping")
                    scenes_with_no_animations.append(scene_key)
                    continue
                # Clean animation names (remove .pkl extension if present)
                clean_animation_names = []
                for anim_name in animation_names:
                    clean_name = anim_name[:-4] if anim_name.endswith('.pkl') else anim_name
                    clean_animation_names.append(clean_name)
                not_started_animations_by_scene[scene_key] = clean_animation_names
                total_animations += len(clean_animation_names)
            else:
                # Use existing data from status report
                not_started_anims = existing_scene_data.get("not_started_animations", [])
                not_started_animations_by_scene[scene_key] = not_started_anims
                total_animations += len(not_started_anims)
                print(f"  Using status report data: {len(not_started_anims)} not started animations")
            continue
        
        # Check for animation folders
        scene_animations = []
        scene_incomplete = []
        scene_not_started = []
        
        if existing_scene_data is None:
            # Need to check blend file for animations
            animation_names = get_animation_names(blend_file)
            if not animation_names:
                print(f"  ⚠️  No animations found in this scene, skipping")
                scenes_with_no_animations.append(scene_key)
                continue
        else:
            # Use existing data from status report
            complete_anims = existing_scene_data.get("complete_animations", [])
            incomplete_anims = existing_scene_data.get("incomplete_animations", [])
            not_started_anims = existing_scene_data.get("not_started_animations", [])
            
            # Use existing data for complete and incomplete animations
            scene_animations = complete_anims.copy()
            scene_incomplete = incomplete_anims.copy()
            scene_not_started = not_started_anims.copy()
            
            # Only re-check incomplete animations to see if they're now complete
            animation_names = [anim["name"] for anim in incomplete_anims] + not_started_anims
            print(f"  Using status report data: {len(complete_anims)} complete, {len(incomplete_anims)} incomplete, {len(not_started_anims)} not started")
            print(f"  Will re-check {len(animation_names)} incomplete/not started animations")
        
        # Check each expected animation
        for anim_name in animation_names:
            # Handle both cases: raw animation names from blend file and clean names from status report
            if existing_scene_data is None:
                # Remove .pkl extension if present (to match folder naming in Blender script)
                clean_anim_name = anim_name[:-4] if anim_name.endswith('.pkl') else anim_name
            else:
                # Animation name is already clean from status report
                clean_anim_name = anim_name
            
            anim_path = os.path.join(scene_output, clean_anim_name)
            
            if not os.path.exists(anim_path):
                # Animation not started
                if clean_anim_name not in scene_not_started:
                    scene_not_started.append(clean_anim_name)
                print(f"  ❌ {clean_anim_name}: Not started")
            else:
                # Animation has been started, check completeness
                expected_videos = get_expected_video_count(blend_file, anim_name, frame_skip, stride, clip_length)
                is_complete, video_count = check_video_sequence_completeness(
                    anim_path, expected_videos, video_type
                )
                
                if is_complete:
                    # Remove from incomplete/not_started lists and add to complete
                    scene_not_started = [anim for anim in scene_not_started if anim != clean_anim_name]
                    scene_incomplete = [anim for anim in scene_incomplete if anim.get('name') != clean_anim_name]
                    if clean_anim_name not in scene_animations:
                        scene_animations.append(clean_anim_name)
                    print(f"  ✅ {clean_anim_name}: Complete ({video_count} videos)")
                else:
                    # Update incomplete animation data
                    existing_incomplete = next((anim for anim in scene_incomplete if anim.get('name') == clean_anim_name), None)
                    if existing_incomplete:
                        # Update existing entry
                        existing_incomplete.update({
                            'videos': video_count,
                            'expected': expected_videos
                        })
                    else:
                        # Add new incomplete entry
                        scene_incomplete.append({
                            'name': clean_anim_name,
                            'videos': video_count,
                            'expected': expected_videos
                        })
                    print(f"  ⚠️  {clean_anim_name}: Incomplete (Videos: {video_count}/{expected_videos})")
        
        # Store scene results
        rendered_scenes[scene_key] = {
            'complete_animations': scene_animations,
            'incomplete_animations': scene_incomplete,
            'not_started_animations': scene_not_started,
            'expected_animations': expected_animations
        }
        rendered_animations += len(scene_animations)
        total_animations += expected_animations
        
        # Store incomplete animations by scene
        if scene_incomplete:
            incomplete_animations_by_scene[scene_key] = scene_incomplete
        
        # Store not started animations by scene
        if scene_not_started:
            not_started_animations_by_scene[scene_key] = scene_not_started
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total scenes: {len(blend_files)}")
    print(f"Scenes with output: {len(rendered_scenes)}")
    print(f"Total expected animations: {total_animations}")
    total_incomplete = sum(len(anims) for anims in incomplete_animations_by_scene.values())
    total_not_started = sum(len(anims) for anims in not_started_animations_by_scene.values())
    print(f"Complete animations: {rendered_animations}")
    print(f"Incomplete animations (partially rendered): {total_incomplete}")
    print(f"Not started animations: {total_not_started}")
    print(f"Scenes with no animations: {len(scenes_with_no_animations)}")
    print(f"Remaining scenes: {len(blend_files) - len(rendered_scenes)}")
    
    # Verify math
    print(f"\nVerification: {rendered_animations} + {total_incomplete} + {total_not_started} = {rendered_animations + total_incomplete + total_not_started} (expected: {total_animations})")
    
    if scenes_with_no_animations:
        print(f"\nSCENES WITH NO ANIMATIONS ({len(scenes_with_no_animations)}):")
        print("-" * 40)
        for scene in scenes_with_no_animations:
            print(f"  ❌ {scene}")
    
    if incomplete_animations_by_scene:
        print(f"\nINCOMPLETE ANIMATIONS BY SCENE (partially rendered):")
        print("-" * 50)
        for scene, anims in incomplete_animations_by_scene.items():
            print(f"{scene}:")
            for anim in anims:
                print(f"  {anim['name']}: Videos {anim['videos']}/{anim['expected']}")
    
    if not_started_animations_by_scene:
        print(f"\nNOT STARTED ANIMATIONS BY SCENE:")
        print("-" * 40)
        for scene, anims in not_started_animations_by_scene.items():
            print(f"{scene}: {len(anims)} animations not started")
            if len(anims) <= 5:  # Show details if not too many
                for anim in anims:
                    print(f"  ❌ {anim}")
            else:
                print(f"  ❌ {anims[0]}, {anims[1]}, ... ({len(anims)} total)")
    
    # Detailed breakdown
    if rendered_scenes:
        print(f"\nRENDERED SCENES DETAILS:")
        print("-" * 40)
        for scene, details in rendered_scenes.items():
            print(f"{scene}:")
            print(f"  Expected: {details['expected_animations']} animations")
            print(f"  Complete: {len(details['complete_animations'])} animations")
            print(f"  Incomplete (partially rendered): {len(details['incomplete_animations'])} animations")
            print(f"  Not started: {len(details['not_started_animations'])} animations")
            for anim in details['complete_animations']:
                print(f"    ✅ {anim}")
    
    # Save detailed report
    report_file = f"rendering_status_report_{video_type}.json"
    report = {
        "total_scenes": len(blend_files),
        "scenes_with_output": len(rendered_scenes),
        "total_expected_animations": total_animations,
        "complete_animations": rendered_animations,
        "incomplete_animations": total_incomplete,
        "not_started_animations": total_not_started,
        "scenes_with_no_animations": len(scenes_with_no_animations),
        "remaining_scenes": len(blend_files) - len(rendered_scenes),
        "rendered_scenes_details": rendered_scenes,
        "incomplete_animations_by_scene": incomplete_animations_by_scene,
        "not_started_animations_by_scene": not_started_animations_by_scene,
        "scenes_with_no_animations_list": scenes_with_no_animations,
        "all_blend_files": [Path(f).stem for f in blend_files],
        "scene_status": {},  # Add scene status for easy lookup
        "frame_skip": frame_skip,  # Store the frame skip value used
        "stride": stride,  # Store the stride value used
        "clip_length": clip_length,  # Store the clip length used
        "video_type": video_type  # Store the video type
    }
    
    # Create a simple scene status lookup
    for blend_file in blend_files:
        blend_name = Path(blend_file).stem
        directory_name = os.path.basename(os.path.dirname(blend_file))
        scene_key = directory_name # Use directory_name as the key for consistency
        if scene_key in rendered_scenes:
            scene_details = rendered_scenes[scene_key]
            report["scene_status"][scene_key] = {
                "is_rendered": True,
                "complete_animations": len(scene_details['complete_animations']),
                "incomplete_animations": len(scene_details['incomplete_animations']),
                "not_started_animations": len(scene_details['not_started_animations']),
                "expected_animations": scene_details['expected_animations'],
                "is_fully_complete": len(scene_details['incomplete_animations']) == 0 and len(scene_details['not_started_animations']) == 0
            }
        else:
            not_started_count = len(not_started_animations_by_scene.get(scene_key, []))
            report["scene_status"][scene_key] = {
                "is_rendered": False,
                "complete_animations": 0,
                "incomplete_animations": 0,
                "not_started_animations": not_started_count,
                "expected_animations": not_started_count,
                "is_fully_complete": False
            }
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nDetailed report saved to: {report_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Check video rendering status for Trumans dataset")
    parser.add_argument("--frame-skip", type=int, default=3, 
                       help="Frame skip value used during rendering (default: 3)")
    parser.add_argument("--stride", type=int, default=25,
                       help="Stride value used during video rendering (default: 25)")
    parser.add_argument("--clip-length", type=int, default=49,
                       help="Clip length used during video rendering (default: 49)")
    parser.add_argument("--video-type", type=str, choices=["static", "hands"], default="static",
                       help="Type of rendering to check: 'static' for blender_ego_static.py, 'hands' for blender_ego_hand.py (default: static)")
    
    args = parser.parse_args()
    
    check_video_rendering_status(
        frame_skip=args.frame_skip,
        stride=args.stride,
        clip_length=args.clip_length,
        video_type=args.video_type
    )
