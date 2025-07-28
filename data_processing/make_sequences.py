import cv2
import imageio.v3 as iio
import os
from glob import glob
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
import pickle
import json

# Add the training/aether/utils directory to the path to import postprocess_utils
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'training', 'aether', 'utils'))

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
            # Ensure the array is writable before converting to tensor
            depth = np.array(depth, copy=True)
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

def load_exr_depth(file_path):
    """Load depth data from .exr file."""
    try:
        # Try using OpenEXR library first (more reliable for EXR files)
        try:
            import OpenEXR
            import Imath
            import array
            
            # Open the EXR file
            exr_file = OpenEXR.InputFile(str(file_path))
            
            # Get the data window
            dw = exr_file.header()['dataWindow']
            width = dw.max.x - dw.min.x + 1
            height = dw.max.y - dw.min.y + 1
            
            # Read the first channel (usually 'R' for depth)
            channels = exr_file.header()['channels'].keys()
            first_channel = list(channels)[0]
            
            # Read the channel data
            channel_data = exr_file.channel(first_channel, Imath.PixelType(Imath.PixelType.FLOAT))
            depth = np.frombuffer(channel_data, dtype=np.float32)
            depth = depth.reshape(height, width)
            # Make the array writable to avoid PyTorch warnings
            depth = np.array(depth, copy=True)
            
            exr_file.close()
            return depth
            
        except ImportError:
            # Fallback to imageio with OpenEXR support
            try:
                depth_data = iio.imread(file_path)
                
                # EXR files can have multiple channels, we want the first one for depth
                if depth_data.ndim == 3 and depth_data.shape[2] > 1:
                    # If it's a multi-channel EXR, take the first channel
                    depth = depth_data[:, :, 0]
                else:
                    depth = depth_data
                
                # Make the array writable to avoid PyTorch warnings
                depth = np.array(depth.astype(np.float32), copy=True)
                return depth
                
            except Exception as e2:
                print(f"imageio failed: {e2}")
                raise e2
    
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        print("Try installing OpenEXR support:")
        print("  pip install OpenEXR")
        print("  or")
        print("  conda install -c conda-forge openexr-python")
        return None

def load_smplx_data(file_path):
    """Load SMPLX data from pickle file."""
    try:
        with open(file_path, 'rb') as f:
            smplx_data = pickle.load(f)
        return smplx_data
    except Exception as e:
        print(f"Error loading SMPLX data from {file_path}: {e}")
        return None

def extract_smplx_body_pose(smplx_data):
    """Extract body pose parameters from SMPLX data."""
    if smplx_data is None:
        return None
    
    # Extract body pose parameters (shape: (num_frames, 63))
    body_pose = smplx_data.get('body_pose', None)
    if body_pose is None:
        print("Warning: No body_pose found in SMPLX data")
        return None
    
    return body_pose

parser = argparse.ArgumentParser(description="Generate video and trajectory sequences from images and camera parameters.")
parser.add_argument("--data_root", type=str, required=True, help="Root directory containing folders of image and camera parameter data.")
parser.add_argument("--start_idx", type=int, default=1, help="Starting index for image sequence.")
parser.add_argument("--end_idx", type=int, default=-1, help="Ending index for image sequence.")
parser.add_argument("--stride", type=int, default=10, help="Stride for selecting frames in the sequence.")
parser.add_argument("--disparity_format", type=str, choices=["image", "npy"], default="image", 
                   help="Format for disparity output: 'image' for MP4 video, 'npy' for concatenated frames")
parser.add_argument("--use_depth", action="store_true", help="Convert depth to disparity with per-clip normalization")
parser.add_argument("--no_sqrt_disparity", action="store_true", help="Disable square root of disparity (default: enabled)")
parser.add_argument("--skip_existing_clips", action="store_true", help="Skip clips that already exist")
parser.add_argument("--smplx_base_path", type=str, default=None, 
                   help="Base path for SMPLX result files (default: {data_root}/../smplx_result)")
parser.add_argument("--skip_smplx", action="store_true", help="Skip SMPLX processing")
parser.add_argument("--dry_run", action="store_true", help="Show what would be processed without actually processing")
args = parser.parse_args()


# input folders
image_folder = Path(args.data_root) / "images"
disparity_folder = Path(args.data_root) / "disparity"
depth_folder = Path(args.data_root) / "depth"
cam_param_folder = Path(args.data_root) / "cam_params"
egoallo_human_pose_file = Path(args.data_root) / "egoallo.npz"
human_pose_folder = Path(args.data_root) / "human_poses"  # For SMPLX human pose files

# output folders
output_folder = Path(args.data_root) / "sequences"
video_output_folder = Path(output_folder) / "videos"

# Set disparity output folder based on format
if args.disparity_format == "image":
    disparity_output_folder = Path(output_folder) / "disparity_video"  # Compatible with camera_pose_to_raymap.py
else:  # npy format
    disparity_output_folder = Path(output_folder) / "disparity"

trajectory_output_folder = Path(output_folder) / "trajectory"
human_pose_output_folder = Path(output_folder) / "human_motions"

# parameters
clip_length = 49
stride = args.stride
fps = 8
start_image_idx = args.start_idx  # Start from
end_image_idx = args.end_idx  # End at
output_size = None  # Set this if you want to resize (e.g., (720, 480))
sqrt_disparity = not args.no_sqrt_disparity

# Create output directory
if not image_folder.exists():
    raise FileNotFoundError(f"Image folder {image_folder} does not exist.")
if image_folder.exists(): os.makedirs(video_output_folder, exist_ok=True)
if disparity_folder.exists(): os.makedirs(disparity_output_folder, exist_ok=True)
if depth_folder.exists(): os.makedirs(disparity_output_folder, exist_ok=True)
if cam_param_folder.exists(): os.makedirs(trajectory_output_folder, exist_ok=True)
if egoallo_human_pose_file.exists(): os.makedirs(human_pose_output_folder, exist_ok=True)

# save arguments to a JSON file
with open(f"{str(output_folder)}/args.json", "w") as f:
    args_dict = {
        "clip_length": clip_length,
        "stride": stride,
        "fps": fps,
        "start_image_idx": start_image_idx,
        "end_image_idx": end_image_idx,
    }
    json.dump(args_dict, f, indent=4)

# Get sorted list of paths
image_paths = sorted(image_folder.glob("*.png"))
image_paths = image_paths[start_image_idx:end_image_idx]

if disparity_folder.exists():
    disparity_paths = sorted(disparity_folder.glob("*.png"))
    disparity_paths = disparity_paths[start_image_idx:end_image_idx]

if depth_folder.exists():
    depth_paths = sorted(depth_folder.glob("*.exr"))
    depth_paths = depth_paths[start_image_idx:end_image_idx]

if cam_param_folder.exists():
    cam_param_paths = sorted(cam_param_folder.glob("*.npy"))
    cam_param_paths = cam_param_paths[start_image_idx:end_image_idx]

# Load egoallo human pose data if available
smplh_data = None
if egoallo_human_pose_file.exists():
    smplh_data = np.load(egoallo_human_pose_file)
    smplh_data = {key: smplh_data[key].astype(np.float32) for key in smplh_data.files} 
    for key in smplh_data:
        if key in ["Ts_world_root", "body_quats", "left_hand_quats", "right_hand_quats", "betas"]:
            smplh_data[key] = smplh_data[key][:, start_image_idx-1:end_image_idx-1] # egoallo predicts from the second frame

# Load SMPLX human pose data if available
human_pose_paths = None
if human_pose_folder.exists():
    human_pose_paths = sorted(human_pose_folder.glob("cam*.npy"))
    human_pose_paths = human_pose_paths[start_image_idx:end_image_idx]

print(f"Total images: {len(image_paths)}")
if egoallo_human_pose_file.exists() and smplh_data is not None:
    print(f"Total egoallo poses: {smplh_data['Ts_world_root'].shape[1]}")
if human_pose_folder.exists() and human_pose_paths is not None:
    print(f"Total SMPLX poses: {len(human_pose_paths)}")

# Find corresponding SMPLX file based on the current action folder
smplx_data = None
if not args.skip_smplx:
    # Extract timestamp from the current data root path
    # Expected structure: .../action_folder_timestamp/
    data_root_name = Path(args.data_root).name
    print(f"Looking for SMPLX data for action folder: {data_root_name}")
    
    # First, check if there's a SMPLX file in the current action directory
    local_smplx_file = Path(args.data_root) / f"{data_root_name}_smplx_results.pkl"
    if local_smplx_file.exists():
        print(f"Found local SMPLX file: {local_smplx_file}")
        smplx_data = load_smplx_data(local_smplx_file)
    else:
        # Determine SMPLX base path for external directory
        if args.smplx_base_path:
            smplx_base_path = Path(args.smplx_base_path)
        else:
            smplx_base_path = Path(args.data_root).parent.parent / "smplx_result"
        
        if smplx_base_path.exists():
            # Try to find SMPLX file with the same timestamp as the action folder
            smplx_file_pattern = f"{data_root_name}_smplx_results.pkl"
            smplx_file_path = smplx_base_path / smplx_file_pattern
            
            if smplx_file_path.exists():
                print(f"Found matching SMPLX file: {smplx_file_path}")
                smplx_data = load_smplx_data(smplx_file_path)
            else:
                print(f"SMPLX file not found: {smplx_file_path}")
                # List available SMPLX files for debugging
                available_smplx_files = list(smplx_base_path.glob("*_smplx_results.pkl"))
                print(f"Available SMPLX files: {[f.name for f in available_smplx_files[:5]]}")
        else:
            print(f"SMPLX base directory not found: {smplx_base_path}")
    
    # Process SMPLX data if found
    if smplx_data is not None:
        body_pose = extract_smplx_body_pose(smplx_data)
        if body_pose is not None:
            print(f"SMPLX body pose shape: {body_pose.shape}")
            print(f"SMPLX data keys: {list(smplx_data.keys())}")
        else:
            print("Warning: Could not extract body pose from SMPLX data")
            smplx_data = None
    else:
        print("No SMPLX data found")
elif args.skip_smplx:
    print("Skipping SMPLX processing as requested")

# Create SMPLX output folder if we have SMPLX data
if smplx_data is not None:
    os.makedirs(human_pose_output_folder, exist_ok=True)

num_images = len(image_paths)

# Generate clips
clip_idx = 0
for start_idx in tqdm(range(0, num_images - clip_length + 1, stride)):
    # Check if this clip already exists (if skip_existing_clips is enabled)
    if args.skip_existing_clips:
        video_exists = os.path.exists(os.path.join(video_output_folder, f"{clip_idx:05}.mp4"))
        trajectory_exists = os.path.exists(os.path.join(trajectory_output_folder, f"{clip_idx:05}.npy"))
        
        # Check disparity existence based on format
        if args.disparity_format == "image":
            disparity_exists = os.path.exists(os.path.join(disparity_output_folder, f"{clip_idx:05}.mp4"))
        else:
            disparity_exists = os.path.exists(os.path.join(disparity_output_folder, f"{clip_idx:05}.npy"))
        
        # Check SMPLX existence if processing SMPLX
        smplx_exists = True  # Default to True if not processing SMPLX
        if smplx_data is not None:
            smplx_body_exists = os.path.exists(os.path.join(human_pose_output_folder, f"{clip_idx:05}_smplx_body.npy"))
            smplx_full_exists = os.path.exists(os.path.join(human_pose_output_folder, f"{clip_idx:05}_smplx_full.npy"))
            smplx_exists = smplx_body_exists and smplx_full_exists
        
        # Skip if all required files exist
        if video_exists and trajectory_exists and disparity_exists and smplx_exists:
            print(f"Skipping clip {clip_idx}: all files already exist")
            clip_idx += 1
            continue
    
    clip_frames = image_paths[start_idx : start_idx + clip_length]
    if disparity_folder.exists():
        clip_disparity = disparity_paths[start_idx : start_idx + clip_length]
    if depth_folder.exists() and args.use_depth:
        clip_depth = depth_paths[start_idx : start_idx + clip_length]
    if cam_param_folder.exists():
        clip_cam_params = cam_param_paths[start_idx : start_idx + clip_length]
    
    # Prepare egoallo human pose data for this clip
    clip_smplh = None
    if egoallo_human_pose_file.exists() and smplh_data is not None:
        clip_smplh = {}
        for key in smplh_data:
            if key == 'Ts_world_root':
                clip_smplh["global_orient_quat"] = smplh_data[key][0, start_idx : start_idx + clip_length, :4] # global orientation in quaternion format Fx4
                clip_smplh["transl"] = smplh_data[key][0, start_idx : start_idx + clip_length, 4:] # translation Fx3
            elif key == 'body_quats':
                clip_smplh["body_pose_quat"] = smplh_data[key][0, start_idx : start_idx + clip_length, :] # body pose in quaternion format FxN_jx4
            elif key == 'left_hand_quats':
                clip_smplh["left_hand_pose_quat"] = smplh_data[key][0, start_idx : start_idx + clip_length, :] # left hand pose in quaternion format FxN_jx4
            elif key == 'right_hand_quats':
                clip_smplh["right_hand_pose_quat"] = smplh_data[key][0, start_idx : start_idx + clip_length, :] # right hand pose in quaternion format FxN_jx4
            elif key == 'betas':
                clip_smplh["betas"] = smplh_data[key][0, start_idx : start_idx + clip_length, :] # shape parameters FxN_betas_smplh
            else:
                continue # Skip any other keys
    
    # Prepare SMPLX human pose data for this clip
    clip_human_poses = None
    if human_pose_folder.exists() and human_pose_paths is not None:
        clip_human_poses = human_pose_paths[start_idx : start_idx + clip_length]

    # RGB video output
    out_path = os.path.join(video_output_folder, f"{clip_idx:05}.mp4")
    if not args.dry_run:
        frames = []
        for frame_path in clip_frames:
            frame = iio.imread(frame_path)
            if output_size:
                frame = cv2.resize(frame, output_size)
            frames.append(frame)
        iio.imwrite(
            out_path,
            np.stack(frames),
            fps=fps,
            codec='libx264'
        )
    else:
        print(f"Would create video: {out_path}")

    # Disparity output
    if disparity_folder.exists():
        if args.disparity_format == "image":
            # Save as MP4 video
            out_path = os.path.join(disparity_output_folder, f"{clip_idx:05}.mp4")
            if not args.dry_run:
                disparity_frames = []
                for disparity_path in clip_disparity:
                    disparity_frame = iio.imread(disparity_path)
                    if output_size:
                        disparity_frame = cv2.resize(disparity_frame, output_size)
                    disparity_frames.append(disparity_frame)
                iio.imwrite(
                    out_path,
                    np.stack(disparity_frames),
                    fps=fps,
                    codec='libx264'
                )
            else:
                print(f"Would create disparity video: {out_path}")
        else:  # npy format
            # Save as concatenated NPY file
            out_path = os.path.join(disparity_output_folder, f"{clip_idx:05}.npy")
            if not args.dry_run:
                disparity_frames = []
                for disparity_path in clip_disparity:
                    disparity_frame = iio.imread(disparity_path)
                    if output_size:
                        disparity_frame = cv2.resize(disparity_frame, output_size)
                    # Convert to grayscale if RGB
                    if disparity_frame.ndim == 3 and disparity_frame.shape[-1] == 3:
                        disparity_frame = np.mean(disparity_frame, axis=-1)
                    disparity_frames.append(disparity_frame.astype(np.float32))
                # Stack frames and save as NPY
                disparity_sequence = np.stack(disparity_frames, axis=0)
                np.save(out_path, disparity_sequence)
            else:
                print(f"Would create disparity npy: {out_path}")
    
    elif depth_folder.exists() and args.use_depth:
        # Convert depth to disparity with per-clip normalization
        print(f"Converting depth to disparity for clip {clip_idx} with per-clip normalization")
        
        # Load depth data for this clip
        clip_depths = []
        for depth_path in clip_depth:
            depth = load_exr_depth(depth_path)
            if depth is not None:
                clip_depths.append(depth)
            else:
                clip_depths.append(None)
        
        # Calculate dmax for this specific clip
        valid_depths = [d for d in clip_depths if d is not None]
        if not valid_depths:
            print(f"Warning: No valid depth data for clip {clip_idx}")
            continue
        
        # Stack depths for batch processing
        depth_stack = np.stack(valid_depths)
        disparity_stack, clip_dmax = depth_to_disparity(depth_stack, sqrt_disparity=sqrt_disparity)
        
        print(f"Clip {clip_idx} dmax: {clip_dmax:.6f}")
        
        if args.disparity_format == "image":
            # Save as MP4 video
            out_path = os.path.join(disparity_output_folder, f"{clip_idx:05}.mp4")
            disparity_frames = []
            for i, disparity in enumerate(disparity_stack):
                # Convert to 8-bit grayscale (0-255)
                disparity_8bit = (disparity * 255).astype(np.uint8)
                if output_size:
                    disparity_8bit = cv2.resize(disparity_8bit, output_size)
                disparity_frames.append(disparity_8bit)
            iio.imwrite(
                out_path,
                np.stack(disparity_frames),
                fps=fps,
                codec='libx264'
            )
        else:  # npy format
            # Save as concatenated NPY file
            out_path = os.path.join(disparity_output_folder, f"{clip_idx:05}.npy")
            # Stack frames and save as NPY
            np.save(out_path, disparity_stack.astype(np.float32))

    # Camera trajectory output
    if cam_param_folder.exists():
        if not args.dry_run:
            cam_params = [np.load(cam_param_path) for cam_param_path in clip_cam_params]
            T0_inv = np.linalg.inv(cam_params[0])
            relative_trajectory = np.array([T0_inv @ T for T in cam_params]) # relative to the first frame
            absolute_trajectory = np.array(cam_params) # absolute trajectory in world coordinates

            np.save(os.path.join(trajectory_output_folder, f"{clip_idx:05}.npy"), relative_trajectory)
            np.save(os.path.join(trajectory_output_folder, f"{clip_idx:05}_abs.npy"), absolute_trajectory)
        else:
            print(f"Would create trajectory files: {clip_idx:05}.npy, {clip_idx:05}_abs.npy")

    # Human pose output - egoallo format
    if egoallo_human_pose_file.exists() and clip_smplh is not None:
        if not args.dry_run:
            np.savez_compressed(os.path.join(human_pose_output_folder, f"{clip_idx:05}.npz"), **clip_smplh)
        else:
            print(f"Would create egoallo pose file: {clip_idx:05}.npz")

    # Human pose output - SMPLX format
    if human_pose_folder.exists() and clip_human_poses is not None:
        if not args.dry_run:
            human_poses = [np.load(human_pose_path) for human_pose_path in clip_human_poses]
            np.save(os.path.join(human_pose_output_folder, f"{clip_idx:05}.npy"), human_poses)
        else:
            print(f"Would create SMPLX pose file: {clip_idx:05}.npy")

    # SMPLX body pose output
    if smplx_data is not None:
        body_pose = extract_smplx_body_pose(smplx_data)
        if body_pose is not None:
            # Extract the corresponding frames for this clip
            # Note: This assumes SMPLX data has the same number of frames as images
            # You might need to adjust this based on your specific data structure
            clip_start = start_idx
            clip_end = start_idx + clip_length
            
            # Ensure we don't go out of bounds
            if clip_end <= body_pose.shape[0]:
                if not args.dry_run:
                    clip_body_pose = body_pose[clip_start:clip_end]
                    
                    # Save the body pose clip
                    out_path = os.path.join(human_pose_output_folder, f"{clip_idx:05}_smplx_body.npy")
                    np.save(out_path, clip_body_pose)
                    
                    # Also save additional SMPLX parameters if needed
                    smplx_clip_data = {}
                    for key, value in smplx_data.items():
                        if isinstance(value, np.ndarray) and value.shape[0] == body_pose.shape[0]:
                            # Only include parameters that have the same number of frames
                            smplx_clip_data[key] = value[clip_start:clip_end]
                    
                    # Save full SMPLX clip data
                    full_out_path = os.path.join(human_pose_output_folder, f"{clip_idx:05}_smplx_full.npy")
                    np.save(full_out_path, smplx_clip_data, allow_pickle=True)
                else:
                    print(f"Would create SMPLX files: {clip_idx:05}_smplx_body.npy, {clip_idx:05}_smplx_full.npy")
            else:
                print(f"Warning: SMPLX data doesn't have enough frames for clip {clip_idx} (need {clip_end}, have {body_pose.shape[0]})")
        else:
            print(f"Warning: Could not extract body pose for clip {clip_idx}")

    clip_idx += 1

# Create video list files
with open(f"{output_folder}/videos.txt", "w") as f:
    for i in range(clip_idx):
        f.write(f"videos/{i:05}.mp4\n")
with open(f"{output_folder}/prompts.txt", "w") as f:
    for i in range(clip_idx):
        f.write(f"\n")

print(f"Generated {clip_idx} clips")
if smplx_data is not None:
    print(f"SMPLX body pose data processed and saved to {human_pose_output_folder}")
    print(f"SMPLX files created: {clip_idx * 2} files ({clip_idx} body pose + {clip_idx} full SMPLX)")
else:
    print("No SMPLX data was processed")
if egoallo_human_pose_file.exists():
    print(f"Egoallo human pose data processed and saved to {human_pose_output_folder}")
if human_pose_folder.exists():
    print(f"SMPLX human pose data processed and saved to {human_pose_output_folder}")
