#!/usr/bin/env python3
"""
Script to run camera_pose_to_raymap.py for all actions in all scenes under 250712_sample.
"""

import os
import subprocess
import argparse
from pathlib import Path
from tqdm import tqdm

def find_all_actions(data_root):
    """Find all action directories in all scenes."""
    actions = []
    
    # Find all scene directories, excluding "processed"
    scene_dirs = [d for d in os.listdir(data_root) 
                  if os.path.isdir(os.path.join(data_root, d)) and d != "processed"]
    
    for scene in scene_dirs:
        scene_path = os.path.join(data_root, scene)
        # Find all action directories (timestamp-like names)
        action_dirs = [d for d in os.listdir(scene_path) 
                      if os.path.isdir(os.path.join(scene_path, d)) and '@' in d]
        
        for action in action_dirs:
            action_path = os.path.join(scene_path, action)
            actions.append({
                'scene': scene,
                'action': action,
                'path': action_path
            })
    
    return actions

def check_action_requirements(action_path):
    """Check if an action directory has all required files for raymap generation."""
    required_paths = [
        os.path.join(action_path, "sequences", "trajectory"),
        os.path.join(action_path, "cam_params", "intrinsics.npy"),
    ]
    
    # Check for either disparity or disparity_video
    disp_paths = [
        os.path.join(action_path, "sequences", "disparity"),
        os.path.join(action_path, "sequences", "disparity_video"),
    ]
    
    # Check if at least one disparity directory exists
    has_disparity = any(os.path.exists(p) for p in disp_paths)
    
    # Check if all required paths exist
    all_required = all(os.path.exists(p) for p in required_paths) and has_disparity
    
    return all_required, required_paths, disp_paths, has_disparity

def check_raymaps_exist(action_path, disparity_format="auto"):
    """Check if all expected raymaps already exist for this action."""
    raymaps_path = os.path.join(action_path, "sequences", "raymaps")
    trajectory_path = os.path.join(action_path, "sequences", "trajectory")
    
    # Check if raymaps directory exists
    if not os.path.exists(raymaps_path):
        return False
    
    # Check if trajectory directory exists
    if not os.path.exists(trajectory_path):
        return False
    
    # Get all trajectory files (these are the expected raymaps)
    trajectory_files = list(Path(trajectory_path).glob("*.npy"))
    if not trajectory_files:
        return False
    
    # Get all existing raymap files
    raymap_files = list(Path(raymaps_path).glob("*.npy"))
    
    # Check if we have the same number of raymaps as trajectories
    if len(raymap_files) != len(trajectory_files):
        return False
    
    # Check if each trajectory has a corresponding raymap
    trajectory_names = {f.stem for f in trajectory_files}
    raymap_names = {f.stem for f in raymap_files}
    
    # All trajectory names should have corresponding raymap names
    missing_raymaps = trajectory_names - raymap_names
    if missing_raymaps:
        return False
    
    return True

def run_raymap_generation(action_path, disparity_format="auto", debug=False):
    """Run camera_pose_to_raymap.py for a single action."""
    cmd = [
        "python", "training/aether/utils/camera_pose_to_raymap.py",
        "--data_root", action_path,
        "--disparity_format", disparity_format
    ]
    
    if debug:
        cmd.append("--debug")
    
    try:
        # Run with real-time output to show progress
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                 text=True, bufsize=1, universal_newlines=True)
        
        # Collect output in real-time
        output_lines = []
        while process.stdout is not None:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                output_lines.append(output.strip())
                # Print progress lines that contain clip information
                if any(keyword in output for keyword in ["Generated raymap for", "Loaded", "Frame shape", "Max disparity"]):
                    print(f"    {output.strip()}")
        
        # Wait for process to complete
        return_code = process.wait()
        return return_code == 0, "\n".join(output_lines), ""
    except Exception as e:
        return False, "", str(e)

def main():
    parser = argparse.ArgumentParser(description="Run raymap generation for all actions in all scenes")
    parser.add_argument("--data_root", type=str, default="data/trumans/250712_sample", 
                       help="Root directory containing scene data")
    parser.add_argument("--disparity_format", type=str, choices=["video", "npy", "auto"], 
                       default="auto", help="Format of disparity data")
    parser.add_argument("--debug", action="store_true", help="Process only one action for debugging")
    parser.add_argument("--dry_run", action="store_true", help="Show what would be processed without running")
    parser.add_argument("--skip_existing", action="store_true", help="Skip actions that already have raymaps")
    args = parser.parse_args()
    
    print(f"Scanning for actions in: {args.data_root}")
    
    # Find all actions
    actions = find_all_actions(args.data_root)
    
    if not actions:
        print("No actions found!")
        return
    
    print(f"Found {len(actions)} actions across {len(set(a['scene'] for a in actions))} scenes")
    
    # Filter actions that meet requirements
    valid_actions = []
    skipped_actions = []
    
    for action in actions:
        is_valid, required_paths, disp_paths, has_disparity = check_action_requirements(action['path'])
        
        if is_valid:
            # Check if raymaps already exist
            if args.skip_existing and check_raymaps_exist(action['path'], args.disparity_format):
                skipped_actions.append(action)
                print(f"⏭️  Skipping {action['scene']}/{action['action']}: All raymaps already exist")
            else:
                # Check what's missing for better reporting
                raymaps_path = os.path.join(action['path'], "sequences", "raymaps")
                trajectory_path = os.path.join(action['path'], "sequences", "trajectory")
                
                if os.path.exists(trajectory_path):
                    trajectory_files = list(Path(trajectory_path).glob("*.npy"))
                    if os.path.exists(raymaps_path):
                        raymap_files = list(Path(raymaps_path).glob("*.npy"))
                        missing_count = len(trajectory_files) - len(raymap_files)
                        if missing_count > 0:
                            print(f"🔄 Processing {action['scene']}/{action['action']}: {missing_count} raymaps missing ({len(raymap_files)}/{len(trajectory_files)})")
                        else:
                            print(f"🔄 Processing {action['scene']}/{action['action']}: Regenerating all raymaps")
                    else:
                        print(f"🔄 Processing {action['scene']}/{action['action']}: No raymaps directory found")
                else:
                    print(f"🔄 Processing {action['scene']}/{action['action']}: No trajectory directory found")
                
                valid_actions.append(action)
        else:
            # Detailed reporting of missing files
            print(f"⚠️  Skipping {action['scene']}/{action['action']}: Missing required files")
            print(f"   📁 Action path: {action['path']}")
            
            # Check each required path
            for path in required_paths:
                if os.path.exists(path):
                    print(f"   ✅ {path}")
                else:
                    print(f"   ❌ {path} (MISSING)")
            
            # Check disparity paths
            print(f"   📊 Disparity check:")
            for path in disp_paths:
                if os.path.exists(path):
                    print(f"   ✅ {path}")
                else:
                    print(f"   ❌ {path} (MISSING)")
            
            if not has_disparity:
                print(f"   ❌ No disparity data found (need either 'disparity' or 'disparity_video' directory)")
            
            print()  # Empty line for readability
    
    print(f"Found {len(valid_actions)} valid actions to process")
    if args.skip_existing:
        print(f"Skipped {len(skipped_actions)} actions with existing raymaps")
    
    if args.dry_run:
        print("\nActions that would be processed:")
        for action in valid_actions:
            print(f"  {action['scene']}/{action['action']}")
        return
    
    if args.debug and valid_actions:
        print(f"\nDebug mode: Processing only first action: {valid_actions[0]['scene']}/{valid_actions[0]['action']}")
        valid_actions = valid_actions[:1]
    
    # Process actions
    successful = 0
    failed = 0
    
    print(f"\nProcessing {len(valid_actions)} actions...")
    
    for i, action in enumerate(valid_actions, 1):
        scene_action = f"{action['scene']}/{action['action']}"
        print(f"\n[{i}/{len(valid_actions)}] 🔄 Processing: {scene_action}")
        print(f"   📁 Action path: {action['path']}")
        
        success, stdout, stderr = run_raymap_generation(
            action['path'], 
            args.disparity_format, 
            args.debug
        )
        
        if success:
            print(f"   ✅ Success: {scene_action}")
            successful += 1
        else:
            print(f"   ❌ Failed: {scene_action}")
            if stderr.strip():
                print(f"   Error: {stderr}")
            failed += 1
        
        # Show progress summary
        print(f"   📊 Progress: {successful} successful, {failed} failed ({i}/{len(valid_actions)} completed)")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total actions found: {len(actions)}")
    print(f"Valid actions: {len(valid_actions)}")
    if args.skip_existing:
        print(f"Skipped (existing): {len(skipped_actions)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {successful/(successful+failed)*100:.1f}%" if (successful+failed) > 0 else "N/A")

if __name__ == "__main__":
    main() 