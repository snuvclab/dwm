#!/usr/bin/env python3
"""
Check rendering status for Trumans dataset.
Shows what scenes and animations are already rendered.
"""

import os
import json
from pathlib import Path

def get_expected_frame_count(blend_file):
    """Get the expected number of frames from a blend file."""
    try:
        import subprocess
        # Create a temporary Python script to avoid command-line parsing issues
        script_content = '''import bpy
import json
import sys

try:
    # Try to access HSI addon properties
    animation_sets = json.loads(bpy.context.scene.hsi_properties.animation_sets)
    if animation_sets:
        # Get the first animation to determine frame count
        first_anim_key = list(animation_sets.keys())[0]
        first_anim_data = animation_sets[first_anim_key]
        if first_anim_data:
            # Get armature name from HSI addon properties
            armature_name = bpy.context.scene.hsi_properties.name_armature_CC
            if not armature_name:
                # Fallback: try to find any armature with CC bones
                for obj in bpy.data.objects:
                    if obj.type == 'ARMATURE' and 'CC_Base_Hip' in obj.pose.bones:
                        armature_name = obj.name
                        break
            
            if armature_name and armature_name in first_anim_data:
                action_name = first_anim_data[armature_name]
                if action_name in bpy.data.actions:
                    action = bpy.data.actions[action_name]
                    frame_range = action.frame_range
                    sys.stdout.write(str(int(frame_range[1] - frame_range[0] + 1)) + "\\n")
                else:
                    sys.stdout.write(str(bpy.context.scene.frame_end - bpy.context.scene.frame_start + 1) + "\\n")
            else:
                sys.stdout.write(str(bpy.context.scene.frame_end - bpy.context.scene.frame_start + 1) + "\\n")
        else:
            sys.stdout.write(str(bpy.context.scene.frame_end - bpy.context.scene.frame_start + 1) + "\\n")
    else:
        sys.stdout.write(str(bpy.context.scene.frame_end - bpy.context.scene.frame_start + 1) + "\\n")
except Exception as e:
    # Fallback to scene frame range or default
    sys.stdout.write(str(bpy.context.scene.frame_end - bpy.context.scene.frame_start + 1) + "\\n")
'''
        
        # Write temporary script
        import tempfile
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
            return 2400
        else:
            print(f"Blender command failed for {os.path.basename(blend_file)}: {result.stderr}")
            return 2400  # Default fallback
    except Exception as e:
        print(f"Exception getting frame count for {os.path.basename(blend_file)}: {e}")
        return 2400  # Default fallback
    return 2400  # Default fallback

def get_animation_count(blend_file):
    """Get the number of animations in a blend file."""
    try:
        import subprocess
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
        
        # Count animations that have the armature
        valid_animations = 0
        for anim_key, anim_data in animation_sets.items():
            if anim_data and armature_name and armature_name in anim_data:
                valid_animations += 1
        
        sys.stdout.write(str(valid_animations) + "\\n")
    else:
        sys.stdout.write("0\\n")
except Exception as e:
    # Fallback to counting actions
    if bpy.data.actions:
        action_names = set()
        for action in bpy.data.actions:
            if action.name:
                base_name = action.name.split('.')[0]
                action_names.add(base_name)
        sys.stdout.write(str(len(action_names) if action_names else 0) + "\\n")
    else:
        sys.stdout.write("0\\n")
'''
        
        # Write temporary script
        import tempfile
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
        print(f"Exception getting animation count for {os.path.basename(blend_file)}: {e}")
        return 10  # Default fallback
    return 10  # Default fallback

def get_animation_frame_count(blend_file, animation_name):
    """Get the frame count for a specific animation in a blend file."""
    try:
        import subprocess
        import tempfile
        
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
                    sys.stdout.write(str(int(frame_range[1] - frame_range[0] + 1)) + "\\n")
                else:
                    sys.stdout.write(str(bpy.context.scene.frame_end - bpy.context.scene.frame_start + 1) + "\\n")
            else:
                sys.stdout.write(str(bpy.context.scene.frame_end - bpy.context.scene.frame_start + 1) + "\\n")
        else:
            sys.stdout.write(str(bpy.context.scene.frame_end - bpy.context.scene.frame_start + 1) + "\\n")
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
                            sys.stdout.write(str(int(frame_range[1] - frame_range[0] + 1)) + "\\n")
                            found_match = True
                            break
        
        if not found_match:
            sys.stdout.write(str(bpy.context.scene.frame_end - bpy.context.scene.frame_start + 1) + "\\n")
except Exception as e:
    # Fallback to scene frame range or default
    sys.stdout.write(str(bpy.context.scene.frame_end - bpy.context.scene.frame_start + 1) + "\\n")
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script_content)
            temp_script = f.name
        
        cmd = ["blender", "--background", blend_file, "--python", temp_script]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
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
        return 2400  # Default fallback
    except Exception as e:
        return 2400  # Default fallback

def check_animation_completeness(anim_path, expected_frames):
    """Check if an animation folder has all expected frames rendered."""
    rgb_path = os.path.join(anim_path, "images")
    depth_path = os.path.join(anim_path, "depth")
    cam_params_path = os.path.join(anim_path, "cam_params")
    
    # Check if all required folders exist
    if not all(os.path.exists(p) for p in [rgb_path, depth_path, cam_params_path]):
        return False, 0, 0, 0
    
    # Count files in each folder
    rgb_files = len([f for f in os.listdir(rgb_path) if f.endswith('.png')])
    depth_files = len([f for f in os.listdir(depth_path) if f.endswith('.exr')])
    cam_files = len([f for f in os.listdir(cam_params_path) if f.startswith('cam')])
    
    # Check if we have the expected number of files
    is_complete = (rgb_files == expected_frames and 
                   depth_files == expected_frames and 
                   cam_files == expected_frames)
    
    return is_complete, rgb_files, depth_files, cam_files

def check_rendering_status():
    """Check what's already rendered in the Trumans dataset."""
    
    # Configuration
    recordings_path = "../nas1/public_dataset/trumans/Recordings_blend"
    output_base = "/home/byungjun/workspace/trumans_ego/ego_render_new"
    
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
        
        # Get expected animation count
        expected_animations = get_animation_count(blend_file)
        
        if expected_animations is None or expected_animations == 0:
            print(f"⚠️  {scene_key} ({blend_name}): No animations found")
            scenes_with_no_animations.append(scene_key)
            continue
        
        print(f"📊 {scene_key} ({blend_name}): Expected {expected_animations} animations")
        
        if not os.path.exists(scene_output):
            print(f"❌ {scene_key} ({blend_name}): Not rendered")
            # For scenes not rendered at all, check if they have animations
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
            continue
        
        # Check for animation folders
        scene_animations = []
        scene_incomplete = []
        scene_not_started = []
        
        # Get all animation names from the blend file
        animation_names = get_animation_names(blend_file)
        if not animation_names:
            print(f"  ⚠️  No animations found in this scene, skipping")
            scenes_with_no_animations.append(scene_key)
            continue
        
        # Check each expected animation
        for anim_name in animation_names:
            # Remove .pkl extension if present (to match folder naming in Blender script)
            clean_anim_name = anim_name[:-4] if anim_name.endswith('.pkl') else anim_name
            anim_path = os.path.join(scene_output, clean_anim_name)
            
            if not os.path.exists(anim_path):
                # Animation not started
                scene_not_started.append(clean_anim_name)
                print(f"  ❌ {clean_anim_name}: Not started")
            else:
                # Animation has been started, check completeness
                expected_frames = get_animation_frame_count(blend_file, anim_name)
                is_complete, rgb_count, depth_count, cam_count = check_animation_completeness(
                    anim_path, expected_frames
                )
                
                if is_complete:
                    scene_animations.append(clean_anim_name)
                    print(f"  ✅ {clean_anim_name}: Complete ({rgb_count} RGB, {depth_count} depth, {cam_count} cam)")
                else:
                    scene_incomplete.append({
                        'name': clean_anim_name,
                        'rgb': rgb_count,
                        'depth': depth_count,
                        'cam': cam_count,
                        'expected': expected_frames
                    })
                    print(f"  ⚠️  {clean_anim_name}: Incomplete (RGB: {rgb_count}/{expected_frames}, "
                          f"Depth: {depth_count}/{expected_frames}, Cam: {cam_count}/{expected_frames})")
        
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
                print(f"  {anim['name']}: RGB {anim['rgb']}/{anim['expected']}, "
                      f"Depth {anim['depth']}/{anim['expected']}, Cam {anim['cam']}/{anim['expected']}")
    
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
    report_file = "rendering_status_report.json"
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
        "scene_status": {}  # Add scene status for easy lookup
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

if __name__ == "__main__":
    check_rendering_status() 



