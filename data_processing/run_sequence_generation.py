#!/usr/bin/env python3
"""
Script to run make_sequences.py for all actions in all scenes under 250712_sample.
Now includes direct depth-to-disparity conversion with per-clip normalization.
"""

import os
import subprocess
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import imageio.v3 as iio
import json
import re

# Add the training/aether/utils directory to the path to import postprocess_utils
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'training', 'aether', 'utils'))

try:
    from postprocess_utils import depth_to_disparity, colorize_depth
except ImportError:
    print("Warning: Could not import postprocess_utils. Using local implementation.")
    
    def depth_to_disparity(depth, sqrt_disparity=True):
        """Convert depth to disparity.
        
        Args:
            depth: (N, H, W) depth map
            sqrt_disparity (bool, optional): Whether to take the square root of the disparity.
                Defaults to True.
        Returns:
            (N, H, W) disparity map, dmax
        """
        import torch
        
        is_numpy = isinstance(depth, np.ndarray)
        if is_numpy:
            depth = torch.from_numpy(depth).float()
        
        disparity = 1.0 / depth
        valid_disparity = disparity[depth > 1e-6]
        dmax = valid_disparity.max()
        disparity = torch.clamp(disparity / dmax, min=0.0, max=1.0)

        if sqrt_disparity:
            disparity = torch.sqrt(disparity)

        if is_numpy:
            disparity = disparity.cpu().numpy()
        return disparity, dmax
    
    def colorize_depth(depth, cmap="Spectral"):
        """Colorize depth map for visualization."""
        import matplotlib.cm as cmaps
        
        min_d, max_d = (depth[depth > 0]).min(), (depth[depth > 0]).max()
        depth = (max_d - depth) / (max_d - min_d)

        cm = cmaps.get_cmap(cmap)
        depth = depth.clip(0, 1)
        depth = cm(depth, bytes=False)[..., 0:3]
        return depth

def find_all_actions(data_root, scene_start=0, scene_end=-1):
    """Find all action directories in all scenes, optionally filtered by scene index range."""
    actions = []
    
    # Find all scene directories and sort them naturally
    scene_dirs = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
    scene_dirs = sorted(scene_dirs, key=lambda x: [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', x)])
    
    # Apply scene index filtering
    if scene_end == -1:
        scene_end = len(scene_dirs)
    
    filtered_scene_dirs = scene_dirs[scene_start:scene_end]
    
    for scene in filtered_scene_dirs:
        scene_path = os.path.join(data_root, scene)
        # Find all action directories (timestamp-like names) and sort them naturally
        action_dirs = [d for d in os.listdir(scene_path) 
                      if os.path.isdir(os.path.join(scene_path, d)) and '@' in d]
        action_dirs = sorted(action_dirs, key=lambda x: [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', x)])
        
        for action in action_dirs:
            action_path = os.path.join(scene_path, action)
            actions.append({
                'scene': scene,
                'action': action,
                'path': action_path
            })
    
    return actions


def check_action_requirements(action_path):
    """Check if an action directory has required files for sequence generation."""
    # At minimum, we need images folder
    required_paths = [
        os.path.join(action_path, "images"),
    ]
    
    # Check if images folder exists and has PNG files
    images_path = os.path.join(action_path, "images")
    if not os.path.exists(images_path):
        return False
    
    # Check if there are any PNG files
    png_files = list(Path(images_path).glob("*.png"))
    if not png_files:
        return False
    
    return True


def get_sequence_parameters():
    # Default values (fallback) - MUST match make_sequences.py defaults
    clip_length = 49
    stride = 25
    sample_every_nth = 3
    
    try:
        # Read parameters from make_sequences.py without executing argument parsing
        make_sequences_path = "data_processing/make_sequences.py"
        if os.path.exists(make_sequences_path):
            with open(make_sequences_path, 'r') as f:
                content = f.read()
                
            # Extract parameters using regex (before args.parse_args())
            import re
            
            # Find clip_length = 49
            clip_match = re.search(r'clip_length\s*=\s*(\d+)', content)
            if clip_match:
                clip_length = int(clip_match.group(1))
                
            # Find stride = 25 (updated to match make_sequences.py default)
            stride_match = re.search(r'stride\s*=\s*(\d+)', content)
            if stride_match:
                stride = int(stride_match.group(1))
                
            # Find sample_every_nth = 3
            sample_match = re.search(r'sample_every_nth\s*=\s*(\d+)', content)
            if sample_match:
                sample_every_nth = int(sample_match.group(1))
                
    except Exception as e:
        print(f"Warning: Could not read parameters from make_sequences.py: {e}")
        print(f"Using default values: clip_length={clip_length}, stride={stride}, sample_every_nth={sample_every_nth}")
    
    return clip_length, stride, sample_every_nth


def check_sequences_exist(action_path, disparity_format="image", save_root=None, scene_name=None, action_name=None, save_colorized_disparity=False):
    """Check if all expected sequences already exist for this action."""
    if save_root:
        # Use provided scene_name and action_name if available, otherwise extract from path
        if scene_name and action_name:
            sequences_path = os.path.join(save_root, scene_name, action_name, "sequences")
        else:
            # Extract scene and action from action_path to construct save path
            action_path_parts = Path(action_path).parts
            if len(action_path_parts) >= 2:
                scene_name = action_path_parts[-2]
                action_name = action_path_parts[-1]
                sequences_path = os.path.join(save_root, scene_name, action_name, "sequences")
            else:
                sequences_path = os.path.join(action_path, "sequences")
    else:
        sequences_path = os.path.join(action_path, "sequences")
    images_path = os.path.join(action_path, "images")
    
    # Check if sequences directory exists
    if not os.path.exists(sequences_path):
        return False
    
    # Check if images directory exists (to determine expected count)
    if not os.path.exists(images_path):
        return False
    
    # Get all image files to determine expected sequence count
    image_files = sorted(list(Path(images_path).glob("*.png")))
    if not image_files:
        return False
    
    # Get parameters from make_sequences.py
    clip_length, stride, sample_every_nth = get_sequence_parameters()
    
    # Calculate expected number of clips
    total_frames = len(image_files)
    expected_clips = max(1, (total_frames - clip_length * sample_every_nth) // (stride * sample_every_nth) + 1)
    
    # Check required sequence directories
    required_dirs = ["videos", "trajectory"]
    if disparity_format == "image":
        required_dirs.append("disparity_video")
    else:
        required_dirs.append("disparity")
    
    # Add colorized disparity directory if requested
    if save_colorized_disparity:
        required_dirs.append("disparity_colorized")
    
    # Check if all required directories exist
    for dir_name in required_dirs:
        dir_path = os.path.join(sequences_path, dir_name)
        if not os.path.exists(dir_path):
            return False
    
    # Check if we have the expected number of files in each directory
    for dir_name in required_dirs:
        dir_path = os.path.join(sequences_path, dir_name)
        if dir_name == "videos":
            # Check for MP4 files
            files = list(Path(dir_path).glob("*.mp4"))
        elif dir_name == "disparity_video":
            # Check for MP4 files
            files = list(Path(dir_path).glob("*.mp4"))
        elif dir_name == "disparity":
            # Check for NPY or NPZ files based on format
            if disparity_format == "npy":
                files = list(Path(dir_path).glob("*.npy"))
            elif disparity_format == "npz":
                files = list(Path(dir_path).glob("*.npz"))
            else:
                files = []
        elif dir_name == "disparity_colorized":
            # Check for MP4 files (colorized disparity videos)
            files = list(Path(dir_path).glob("*.mp4"))
        elif dir_name == "trajectory":
            # Check for NPY files (both relative and absolute trajectories)
            files = list(Path(dir_path).glob("*.npy"))
            # Count only unique clips (ignore _abs files for counting)
            unique_clips = len(set(f.stem.replace("_abs", "") for f in files))
            if unique_clips != expected_clips:
                return False
            continue
        else:
            continue
        
        if len(files) != expected_clips:
            return False
    
    # Check for cam_params/intrinsics.npy if using save_root
    if save_root:
        # Determine the path for intrinsics.npy in the save location
        if scene_name and action_name:
            save_action_path = os.path.join(save_root, scene_name, action_name)
        else:
            # Extract scene and action from action_path to construct save path
            action_path_parts = Path(action_path).parts
            if len(action_path_parts) >= 2:
                scene_name = action_path_parts[-2]
                action_name = action_path_parts[-1]
                save_action_path = os.path.join(save_root, scene_name, action_name)
            else:
                save_action_path = action_path
        
        intrinsics_path = os.path.join(save_action_path, "cam_params", "intrinsics.npy")
        if not os.path.exists(intrinsics_path):
            return False
    
    return True


def check_optional_data(action_path):
    """Check what optional data is available for an action."""
    optional_data = {}
    
    # Check for depth data (new primary source)
    depth_path = os.path.join(action_path, "depth")
    if os.path.exists(depth_path):
        exr_files = list(Path(depth_path).glob("*.exr"))
        if exr_files:
            optional_data['depth'] = len(exr_files)
    
    # Check for disparity data (legacy support)
    disparity_path = os.path.join(action_path, "disparity")
    if os.path.exists(disparity_path):
        png_files = list(Path(disparity_path).glob("*.png"))
        if png_files:
            optional_data['disparity'] = len(png_files)
    
    # Check for camera parameters
    cam_params_path = os.path.join(action_path, "cam_params")
    if os.path.exists(cam_params_path):
        npy_files = list(Path(cam_params_path).glob("*.npy"))
        if npy_files:
            optional_data['cam_params'] = len(npy_files)
    
    # Check for egoallo human poses (ARIA datasets)
    egoallo_path = os.path.join(action_path, "egoallo.npz")
    if os.path.exists(egoallo_path):
        optional_data['egoallo'] = True
    
    # Check for SMPLX data (Trumans datasets)
    action_name = Path(action_path).stem
    smplx_path = Path("./data/trumans/smplx_result") / f"{action_name}_smplx_results.pkl"
    if smplx_path.exists():
        optional_data['smplx'] = True
    
    return optional_data


def run_sequence_generation(action_path, disparity_format="image", start_idx=0, end_idx=-1, debug=False, sqrt_disparity=True, skip_existing_clips=False, dataset_type=None, smplx_base_path=None, save_root=None, scene_name=None, action_name=None, verbose=False, force_depth_reprocessing=False, force_cam_pose_reprocessing=False, save_colorized_disparity=False):
    """Run make_sequences.py for a single action with optional depth-to-disparity conversion."""
    
    # Check if we should use existing disparity or convert from depth
    depth_path = os.path.join(action_path, "depth")
    disparity_path = os.path.join(action_path, "disparity")
    
    # Build the command
    cmd = [
        "python", "data_processing/make_sequences.py",
        "--data_root", action_path,
        "--disparity_format", disparity_format,
        "--start_idx", str(start_idx),
        "--end_idx", str(end_idx)
    ]
    
    # Add save_root if provided
    if save_root:
        # Use provided scene_name and action_name if available, otherwise extract from path
        if scene_name and action_name:
            save_path = os.path.join(save_root, scene_name, action_name)
        else:
            # Extract scene and action from action_path to construct save path
            action_path_parts = Path(action_path).parts
            # Find the scene and action parts (they should be the last two parts)
            if len(action_path_parts) >= 2:
                scene_name = action_path_parts[-2]
                action_name = action_path_parts[-1]
                save_path = os.path.join(save_root, scene_name, action_name)
            else:
                print(f"Warning: Could not extract scene/action from path: {action_path}")
                print(f"Using default save location: {action_path}")
                save_path = action_path
        
        # Create the save directory structure
        try:
            os.makedirs(save_path, exist_ok=True)
            print(f"Created save directory: {save_path}")
        except Exception as e:
            print(f"Warning: Could not create directory {save_path}: {e}")
        

        
        cmd.extend(["--save_root", save_path])
        print(f"Saving sequences to: {save_path}")
    
    if skip_existing_clips:
        cmd.append("--skip_existing_clips")
    
    if not sqrt_disparity:
        cmd.append("--no_sqrt_disparity")
    
    if dataset_type:
        cmd.extend(["--dataset_type", dataset_type])
    
    if smplx_base_path:
        cmd.extend(["--smplx_base_path", smplx_base_path])
    
    # Add force reprocessing flags if specified
    if force_depth_reprocessing:
        cmd.append("--force_depth_reprocessing")
    if force_cam_pose_reprocessing:
        cmd.append("--force_cam_pose_reprocessing")
    
    # Add colorized disparity flag if specified
    if save_colorized_disparity:
        cmd.append("--save_colorized_disparity")
    
    # Check if depth files exist (for Trumans datasets)
    if os.path.exists(depth_path):
        exr_files = list(Path(depth_path).glob("*.exr"))
        if exr_files:
            print(f"Converting depth to disparity from: {depth_path}")
            if force_depth_reprocessing:
                print(f"🔄 Force reprocessing depth files (OpenEXR required)")
        else:
            print(f"Depth folder exists but no .exr files found in: {depth_path}")
    elif os.path.exists(disparity_path):
        # Use existing disparity files
        print(f"Using existing disparity files from: {disparity_path}")
    else:
        # No depth or disparity data available
        return False, "", "No depth or disparity data found"
    
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
                # Print progress lines that contain frame/clip information
                if verbose or any(keyword in output for keyword in ["Processing frame", "Generated clip", "Converting depth", "Saved", "frames processed", "Error", "Exception", "Failed"]):
                    print(f"    {output.strip()}")
        
        # Wait for process to complete
        return_code = process.wait()
        
        # If the process failed, print the full output for debugging
        if return_code != 0:
            print(f"    ❌ Process failed with return code: {return_code}")
            print(f"    📋 Full output:")
            for line in output_lines[-20:]:  # Show last 20 lines
                print(f"        {line}")
        
        return return_code == 0, "\n".join(output_lines), ""
    except Exception as e:
        print(f"    ❌ Exception occurred: {e}")
        return False, "", str(e)


def load_rendering_status_report(status_report_path="rendering_status_report.json"):
    """Load the rendering status report from check_rendering_status.py."""
    if not os.path.exists(status_report_path):
        print(f"⚠️  Status report not found: {status_report_path}")
        return None
    
    try:
        with open(status_report_path, 'r') as f:
            status_report = json.load(f)
        print(f"✓ Loaded rendering status report: {status_report_path}")
        return status_report
    except Exception as e:
        print(f"❌ Error loading status report: {e}")
        return None


def get_fully_rendered_scenes(status_report):
    """Get list of fully rendered scenes from the status report."""
    if not status_report or "rendered_scenes_details" not in status_report:
        return []
    
    fully_rendered = []
    for scene_key, scene_details in status_report["rendered_scenes_details"].items():
        # Check if scene is fully complete (no incomplete or not started animations)
        incomplete_count = len(scene_details.get("incomplete_animations", []))
        not_started_count = len(scene_details.get("not_started_animations", []))
        if incomplete_count == 0 and not_started_count == 0:
            fully_rendered.append(scene_key)
    
    return fully_rendered


def filter_actions_by_rendering_status(actions, status_report, only_fully_rendered=False):
    """Filter actions based on rendering status."""
    if not status_report or not only_fully_rendered:
        return actions
    
    fully_rendered_scenes = get_fully_rendered_scenes(status_report)
    print(f"📊 Found {len(fully_rendered_scenes)} fully rendered scenes in status report")
    
    if not fully_rendered_scenes:
        print("⚠️  No fully rendered scenes found in status report")
        return []
    
    # Filter actions to only include those from fully rendered scenes
    filtered_actions = []
    for action in actions:
        if action['scene'] in fully_rendered_scenes:
            filtered_actions.append(action)
    
    print(f"📊 Filtered to {len(filtered_actions)} actions from fully rendered scenes")
    print(f"📊 Excluded {len(actions) - len(filtered_actions)} actions from incomplete scenes")
    
    return filtered_actions


def main():
    parser = argparse.ArgumentParser(description="Run sequence generation for all actions in all scenes")
    parser.add_argument("--data_root", type=str, default="data/trumans/250712_sample", 
                       help="Root directory containing scene data")
    parser.add_argument("--disparity_format", type=str, choices=["image", "npy", "npz"], 
                       default="npz", help="Format for disparity output: 'image' for MP4 video, 'npy' for NPY files, 'npz' for compressed NPZ files")
    parser.add_argument("--start_idx", type=int, default=0, help="Starting index for image sequence")
    parser.add_argument("--end_idx", type=int, default=-1, help="Ending index for image sequence")
    parser.add_argument("--debug", action="store_true", help="Process only one action for debugging")
    parser.add_argument("--dry_run", action="store_true", help="Show what would be processed without running")
    parser.add_argument("--skip_existing", action="store_true", help="Skip actions that already have sequences folder")
    parser.add_argument("--skip_existing_clips", action="store_true", help="Skip individual clips that already exist within actions")
    parser.add_argument("--no_sqrt_disparity", action="store_true", help="Disable square root of disparity (default: enabled)")
    parser.add_argument("--dataset_type", type=str, choices=["aria", "trumans"], default="trumans",
                       help="Dataset type: 'aria' for egoallo data, 'trumans' for SMPLX data. Auto-detected if not specified.")
    parser.add_argument("--smplx_base_path", type=str, default=None, 
                       help="Base path for SMPLX result files (default: {data_root}/../smplx_result)")
    parser.add_argument("--only_fully_rendered", action="store_true", 
                       help="Process only actions from fully rendered scenes (requires rendering_status_report.json)")
    parser.add_argument("--status_report_path", type=str, default="rendering_status_report.json",
                       help="Path to the rendering status report JSON file")
    parser.add_argument("--save_root", type=str, default=None,
                       help="Root directory for saving sequences. If None, uses data_root. Preserves scene/action structure.")
    parser.add_argument("--scene_start", type=int, default=0,
                       help="Starting scene index (0-based, for manual parallel processing)")
    parser.add_argument("--scene_end", type=int, default=-1,
                       help="Ending scene index (exclusive, -1 for all scenes)")
    parser.add_argument("--verbose", action="store_true",
                       help="Show verbose output including all subprocess output")
    parser.add_argument("--force_depth_reprocessing", action="store_true",
                       help="Force reprocessing of depth files even if disparity files already exist")
    parser.add_argument("--force_cam_pose_reprocessing", action="store_true",
                       help="Force reprocessing of camera poses even if trajectory files already exist")
    parser.add_argument("--save_colorized_disparity", action="store_true",
                       help="Additionally save colorized disparity videos (MP4 format)")

    
    args = parser.parse_args()
    
    # Set sqrt_disparity based on argument
    sqrt_disparity = not args.no_sqrt_disparity
    print(f"Using sqrt_disparity: {sqrt_disparity}")
    
    print(f"Scanning for actions in: {args.data_root}")
    
    # Find all actions with scene filtering
    actions = find_all_actions(args.data_root, args.scene_start, args.scene_end)
    
    # Show scene range being processed
    if args.scene_end == -1:
        print(f"Processing scenes {args.scene_start} to end")
    else:
        print(f"Processing scenes {args.scene_start} to {args.scene_end-1} (exclusive)")
    
    if not actions:
        print("No actions found!")
        return
    
    print(f"Found {len(actions)} actions across {len(set(a['scene'] for a in actions))} scenes")
    
    # Load rendering status report if needed
    status_report = None
    if args.only_fully_rendered:
        status_report = load_rendering_status_report(args.status_report_path)
        if not status_report:
            print("❌ Cannot proceed with --only_fully_rendered without a valid status report")
            return
    
    # Filter actions based on rendering status if requested
    if args.only_fully_rendered and status_report:
        actions = filter_actions_by_rendering_status(actions, status_report, only_fully_rendered=True)
        if not actions:
            print("❌ No actions found after filtering for fully rendered scenes")
            return
        
        # Show which scenes are being processed
        scenes_to_process = sorted(set(a['scene'] for a in actions), 
                                 key=lambda x: [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', x)])
        print(f"\n🎯 Processing actions from {len(scenes_to_process)} fully rendered scenes:")
        for scene in scenes_to_process[:10]:  # Show first 10
            print(f"  ✓ {scene}")
        if len(scenes_to_process) > 10:
            print(f"  ... and {len(scenes_to_process) - 10} more scenes")
    
    # Filter actions that meet requirements
    valid_actions = []
    skipped_actions = []
    
    for action in actions:
        if check_action_requirements(action['path']):
            # Check if sequences already exist
            if args.skip_existing and check_sequences_exist(action['path'], args.disparity_format, args.save_root, action['scene'], action['action'], args.save_colorized_disparity):
                skipped_actions.append(action)
                print(f"⏭️  Skipping {action['scene']}/{action['action']}: All sequences already exist")
            else:
                # Check what's missing for better reporting
                sequences_path = os.path.join(action['path'], "sequences")
                images_path = os.path.join(action['path'], "images")
                
                if os.path.exists(images_path):
                    image_files = sorted(list(Path(images_path).glob("*.png")))
                    total_frames = len(image_files)
                    
                    # Get parameters from make_sequences.py
                    clip_length, stride, sample_every_nth = get_sequence_parameters()
                    expected_clips = max(1, (total_frames - clip_length * sample_every_nth) // (stride * sample_every_nth) + 1)
                    
                    # Determine the correct sequences path
                    if args.save_root:
                        # Use scene and action names directly
                        sequences_path = os.path.join(args.save_root, action['scene'], action['action'], "sequences")
                    else:
                        sequences_path = os.path.join(action['path'], "sequences")
                    
                    if os.path.exists(sequences_path):
                        # Check existing files
                        videos_path = os.path.join(sequences_path, "videos")
                        trajectory_path = os.path.join(sequences_path, "trajectory")
                        
                        existing_videos = len(list(Path(videos_path).glob("*.mp4"))) if os.path.exists(videos_path) else 0
                        existing_trajectories = len(list(Path(trajectory_path).glob("*_abs.npy"))) if os.path.exists(trajectory_path) else 0
                        
                        if args.disparity_format == "image":
                            disparity_path = os.path.join(sequences_path, "disparity_video")
                            existing_disparities = len(list(Path(disparity_path).glob("*.mp4"))) if os.path.exists(disparity_path) else 0
                        elif args.disparity_format == "npy":
                            disparity_path = os.path.join(sequences_path, "disparity")
                            existing_disparities = len(list(Path(disparity_path).glob("*.npy"))) if os.path.exists(disparity_path) else 0
                        else:  # npz format
                            disparity_path = os.path.join(sequences_path, "disparity")
                            existing_disparities = len(list(Path(disparity_path).glob("*.npz"))) if os.path.exists(disparity_path) else 0
                        
                        # Check colorized disparity files if requested
                        existing_colorized_disparities = 0
                        if args.save_colorized_disparity:
                            colorized_disparity_path = os.path.join(sequences_path, "disparity_colorized")
                            existing_colorized_disparities = len(list(Path(colorized_disparity_path).glob("*.mp4"))) if os.path.exists(colorized_disparity_path) else 0
                        
                        missing_videos = expected_clips - existing_videos
                        missing_trajectories = expected_clips - existing_trajectories
                        missing_disparities = expected_clips - existing_disparities
                        missing_colorized_disparities = expected_clips - existing_colorized_disparities if args.save_colorized_disparity else 0
                        print(f"existing_videos: {existing_videos}, expected_clips: {expected_clips}")
                        print(f"existing_trajectories: {existing_trajectories}, expected_clips: {expected_clips}")
                        print(f"existing_disparities: {existing_disparities}, expected_clips: {expected_clips}")
                        if missing_videos > 0 or missing_trajectories > 0 or missing_disparities > 0 or missing_colorized_disparities > 0:
                            colorized_info = f", {missing_colorized_disparities} colorized disparities" if args.save_colorized_disparity else ""
                            print(f"🔄 Processing {action['scene']}/{action['action']}: {missing_videos} videos, {missing_trajectories} trajectories, {missing_disparities} disparities{colorized_info} missing ({existing_videos}/{expected_clips} videos)")
                        else:
                            # All sequences are complete, skip this action
                            if args.skip_existing:
                                skipped_actions.append(action)
                                print(f"⏭️  Skipping {action['scene']}/{action['action']}: All sequences already exist")
                                continue
                            else:
                                print(f"🔄 Processing {action['scene']}/{action['action']}: Regenerating all sequences (--skip_existing not used)")
                    else:
                        print(f"🔄 Processing {action['scene']}/{action['action']}: No sequences directory found")
                else:
                    print(f"🔄 Processing {action['scene']}/{action['action']}: No images directory found")
                
                valid_actions.append(action)
        else:
            print(f"⚠️  Skipping {action['scene']}/{action['action']}: Missing required files (images folder with PNG files)")
    
    print(f"Found {len(valid_actions)} valid actions to process")
    if args.skip_existing:
        print(f"Skipped {len(skipped_actions)} actions with existing sequences")
    
    # Show data availability for valid actions
    print(f"\nData availability for valid actions:")
    for action in valid_actions[:5]:  # Show first 5 as examples
        optional_data = check_optional_data(action['path'])
        data_info = []
        if 'depth' in optional_data:
            data_info.append(f"depth: {optional_data['depth']} files (will convert to disparity)")
        if 'disparity' in optional_data:
            data_info.append(f"disparity: {optional_data['disparity']} files")
        if 'cam_params' in optional_data:
            data_info.append(f"cam_params: {optional_data['cam_params']} files")
        if 'egoallo' in optional_data:
            data_info.append(f"egoallo: available (ARIA dataset)")
        if 'smplx' in optional_data:
            data_info.append(f"smplx: available (Trumans dataset)")
        
        data_str = ", ".join(data_info) if data_info else "images only"
        print(f"  {action['scene']}/{action['action']}: {data_str}")
    
    if len(valid_actions) > 5:
        print(f"  ... and {len(valid_actions) - 5} more actions")
    
    if args.dry_run:
        print(f"\nActions that would be processed:")
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
    print(f"Disparity format: {args.disparity_format}")
    print(f"Frame range: {args.start_idx} to {args.end_idx if args.end_idx != -1 else 'end'}")
    print(f"Depth conversion: {'Converting depth to disparity' if args.disparity_format != 'image' else 'Using existing disparity'}")
    
    for i, action in enumerate(valid_actions, 1):
        scene_action = f"{action['scene']}/{action['action']}"
        print(f"\n[{i}/{len(valid_actions)}] 🔄 Processing: {scene_action}")
        print(f"   📁 Action path: {action['path']}")
        
        # Show expected frame count
        images_path = os.path.join(action['path'], "images")
        if os.path.exists(images_path):
            png_files = list(Path(images_path).glob("*.png"))
            print(f"   📸 Found {len(png_files)} image frames")
        
        success, stdout, stderr = run_sequence_generation(
            action['path'], 
            args.disparity_format,
            args.start_idx,
            args.end_idx,
            args.debug,
            sqrt_disparity,
            args.skip_existing_clips,
            args.dataset_type,
            args.smplx_base_path,
            args.save_root,
            action['scene'],
            action['action'],
            args.verbose,
            args.force_depth_reprocessing,
            args.force_cam_pose_reprocessing,
            args.save_colorized_disparity
        )
        
        if success:
            print(f"   ✅ Success: {scene_action}")
            successful += 1
        else:
            print(f"   ❌ Failed: {scene_action}")
            if stderr:
                print(f"   Error: {stderr}")
            failed += 1
            
            # Suggest debugging if this is the first failure
            if failed == 1:
                print(f"   💡 To debug this issue, run with --debug --verbose flags:")
                print(f"      python data_processing/run_sequence_generation.py --debug --verbose --data_root {args.data_root} --save_root {args.save_root}")
        
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
    
    # Show output structure
    if successful > 0:
        print(f"\nOutput structure for successful actions:")
        print(f"  {args.data_root}/<scene>/<action>/sequences/")
        print(f"    ├── videos/                    # RGB video sequences")
        if args.disparity_format == "image":
            print(f"    ├── disparity_video/           # Disparity video sequences (MP4)")
        elif args.disparity_format == "npy":
            print(f"    ├── disparity/                 # Disparity sequences (NPY)")
        else:  # npz format
            print(f"    ├── disparity/                 # Disparity sequences (compressed NPZ)")
        if args.save_colorized_disparity:
            print(f"    ├── disparity_colorized/       # Colorized disparity videos (MP4)")
        print(f"    ├── trajectory/                  # Camera trajectory sequences")
        print(f"    └── human_motions/               # Human pose sequences (if available)")

if __name__ == "__main__":
    main() 