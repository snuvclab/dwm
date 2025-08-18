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
    
    # min_d, max_d = (depth[depth > 0]).min(), (depth[depth > 0]).max()
    # depth = (max_d - depth) / (max_d - min_d)
    depth = 1 - depth  # assume depth is in range [0, 1]

    cm = cmaps.get_cmap(cmap)
    depth = depth.clip(0, 1)
    depth = cm(depth, bytes=False)[..., 0:3]
    return depth

def load_exr_depth(file_path):
    """Load depth data from .exr file."""
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
        
    except Exception as e:
        raise ImportError("OpenEXR library required for depth processing but not available")

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

def blender_to_opencv_transform():
    """
    Create transformation matrix to convert from Blender to OpenCV coordinates.
    
    According to OpenCV/COLMAP convention:
    - OpenCV uses: Forward: +Z, Up: -Y, Right: +X
    - Blender uses: Forward: -Z, Up: +Y, Right: +X
    
    The conversion is a 180° rotation around the local X-axis.
    
    Returns:
        np.ndarray: 4x4 transformation matrix
    """
    # 180° rotation around X-axis to convert from Blender to OpenCV
    # This flips Y and Z axes
    rotation = np.array([
        [1,  0,  0],  # X stays the same
        [0,  -1, 0],  # Y becomes -Y
        [0,  0, -1]   # Z becomes -Z
    ])
    
    # Create 4x4 transformation matrix
    transform = np.eye(4)
    transform[:3, :3] = rotation
    
    return transform

def convert_blender_to_opencv_poses(poses):
    """
    Convert camera poses from Blender coordinate system to OpenCV coordinate system.
    
    Args:
        poses: (N, 4, 4) camera poses in Blender coordinates (Y-up)
        
    Returns:
        poses_converted: (N, 4, 4) camera poses in OpenCV coordinates (Z-up)
    """
    transform = blender_to_opencv_transform()
    poses_converted = np.zeros_like(poses)
    
    for i in range(poses.shape[0]):
        # Apply the coordinate transformation
        poses_converted[i] = poses[i] @ transform.T
    
    return poses_converted

parser = argparse.ArgumentParser(description="Generate video and trajectory sequences from images and camera parameters.")
parser.add_argument("--data_root", type=str, required=True, help="Root directory containing folders of image and camera parameter data.")
parser.add_argument("--start_idx", type=int, default=0, help="Starting index for image sequence.")
parser.add_argument("--end_idx", type=int, default=-1, help="Ending index for image sequence.")
parser.add_argument("--stride", type=int, default=25, help="Stride for selecting frames in the sequence.")
parser.add_argument("--sample_every_nth", type=int, default=3, help="Sample every nth frame for the sequence.")
parser.add_argument("--disparity_format", type=str, choices=["npy", "npz"], default="npz", 
                   help="Format for disparity output: 'npy' for concatenated frames, 'npz' for compressed frames")
parser.add_argument("--save_colorized_disparity", action="store_true", 
                   help="Additionally save colorized disparity videos (MP4 format)")

parser.add_argument("--sqrt_disparity", action="store_true", help="Enable square root of disparity (default: disabled)")
parser.add_argument("--skip_existing_clips", action="store_true", help="Skip clips that already exist")
parser.add_argument("--smplx_base_path", type=str, default=None, 
                   help="Base path for SMPLX result files (default: {data_root}/../smplx_result)")

parser.add_argument("--dataset_type", type=str, choices=["aria", "trumans"], default="trumans",
                   help="Dataset type: 'aria' for egoallo data, 'trumans' for SMPLX data. Auto-detected if not specified.")
parser.add_argument("--dry_run", action="store_true", help="Show what would be processed without actually processing")
parser.add_argument("--save_root", type=str, default=None, help="Root directory for saving sequences. If None, uses data_root.")
parser.add_argument("--force_depth_reprocessing", action="store_true", 
                   help="Force reprocessing of depth files even if disparity files already exist")
parser.add_argument("--force_cam_pose_reprocessing", action="store_true", 
                   help="Force reprocessing of camera poses even if trajectory files already exist")
args = parser.parse_args()


# input folders
image_folder = Path(args.data_root) / "images"
disparity_folder = Path(args.data_root) / "disparity"
depth_folder = Path(args.data_root) / "depth"
cam_param_folder = Path(args.data_root) / "cam_params"
egoallo_human_pose_file = Path(args.data_root) / "egoallo.npz"

# output folders
if args.save_root:
    output_folder = Path(args.save_root) / "sequences"
else:
    output_folder = Path(args.data_root) / "sequences"
video_output_folder = Path(output_folder) / "videos"

# Set disparity output folder
disparity_output_folder = Path(output_folder) / "disparity"
colorized_disparity_output_folder = Path(output_folder) / "disparity_colorized" if args.save_colorized_disparity else None

trajectory_output_folder = Path(output_folder) / "trajectory"
human_pose_output_folder = Path(output_folder) / "human_motions"

# parameters
clip_length = 49
stride = args.stride
sample_every_nth = args.sample_every_nth
fps = 8
start_image_idx = args.start_idx  # Start from
end_image_idx = args.end_idx  # End at
output_size = None  # Set this if you want to resize (e.g., (720, 480))
sqrt_disparity = args.sqrt_disparity

# Create output directory
if not image_folder.exists():
    raise FileNotFoundError(f"Image folder {image_folder} does not exist.")

# Create output directories
if image_folder.exists(): os.makedirs(video_output_folder, exist_ok=True)
if disparity_folder.exists(): os.makedirs(disparity_output_folder, exist_ok=True)
if depth_folder.exists(): os.makedirs(disparity_output_folder, exist_ok=True)
if cam_param_folder.exists(): os.makedirs(trajectory_output_folder, exist_ok=True)
if args.save_colorized_disparity: os.makedirs(colorized_disparity_output_folder, exist_ok=True)

# Copy cam_params/intrinsics.npy to save_root if using save_root
if args.save_root:
    source_intrinsics_path = cam_param_folder / "intrinsics.npy"
    dest_cam_params_path = Path(args.save_root) / "cam_params"
    dest_intrinsics_path = dest_cam_params_path / "intrinsics.npy"
    
    if source_intrinsics_path.exists():
        if not dest_intrinsics_path.exists():
            try:
                # Create cam_params directory in destination
                os.makedirs(dest_cam_params_path, exist_ok=True)
                # Copy intrinsics.npy
                import shutil
                shutil.copy2(source_intrinsics_path, dest_intrinsics_path)
                print(f"📋 Copied cam_params/intrinsics.npy to save location: {dest_intrinsics_path}")
            except Exception as e:
                print(f"Warning: Could not copy intrinsics.npy: {e}")
        else:
            print(f"📋 cam_params/intrinsics.npy already exists in save location: {dest_intrinsics_path}")
    else:
        print(f"📋 No cam_params/intrinsics.npy found in source: {source_intrinsics_path}")


# save arguments to a JSON file
with open(f"{str(output_folder)}/args.json", "w") as f:
    args_dict = {
        "clip_length": clip_length,
        "stride": stride,
        "sample_every_nth": sample_every_nth,
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

# Auto-detect dataset type if not specified
if args.dataset_type is None:
    if egoallo_human_pose_file.exists():
        args.dataset_type = "aria"
        print(f"🔍 Auto-detected dataset type: ARIA (found egoallo.npz)")
    else:
        args.dataset_type = "trumans"
        print(f"🔍 Auto-detected dataset type: Trumans (SMPLX processing enabled)")

print(f"📊 Dataset type: {args.dataset_type.upper()}")

# Load egoallo human pose data if available (ARIA datasets)
smplh_data = None
if args.dataset_type == "aria" and egoallo_human_pose_file.exists():
    print(f"📁 Loading egoallo data from: {egoallo_human_pose_file}")
    smplh_data = np.load(egoallo_human_pose_file)
    smplh_data = {key: smplh_data[key].astype(np.float32) for key in smplh_data.files} 
    for key in smplh_data:
        if key in ["Ts_world_root", "body_quats", "left_hand_quats", "right_hand_quats", "betas"]:
            smplh_data[key] = smplh_data[key][:, start_image_idx-1:end_image_idx-1] # egoallo predicts from the second frame
    print(f"✅ Loaded egoallo data with keys: {list(smplh_data.keys())}")
elif args.dataset_type == "aria" and not egoallo_human_pose_file.exists():
    print(f"⚠️  Warning: ARIA dataset specified but egoallo.npz not found at {egoallo_human_pose_file}")



# Load SMPLX pose data from pickle file (Trumans datasets)
smplx_data = None
if args.dataset_type == "trumans":
    # Extract action name from data_root stem
    action_name = Path(args.data_root).stem
    print(f"🔍 Looking for SMPLX pose data for action: {action_name}")
    
    # Determine SMPLX pose data path
    if args.smplx_base_path:
        smplx_pose_path = Path(args.smplx_base_path) / f"{action_name}_smplx_results.pkl"
    else:
        smplx_pose_path = Path("./data/trumans/smplx_result") / f"{action_name}_smplx_results.pkl"
    
    if smplx_pose_path.exists():
        print(f"📁 Loading SMPLX pose data from: {smplx_pose_path}")
        try:
            smplx_data = load_smplx_data(smplx_pose_path)
            if smplx_data is not None:
                print(f"✅ Loaded SMPLX pose data with keys: {list(smplx_data.keys())}")
                # Check if we have body_pose data
                if 'body_pose' in smplx_data:
                    print(f"📊 SMPLX body pose shape: {smplx_data['body_pose'].shape}")
                else:
                    print(f"⚠️  Warning: No 'body_pose' key found in SMPLX data")
            else:
                print(f"❌ Failed to load SMPLX pose data from {smplx_pose_path}")
        except Exception as e:
            print(f"❌ Error loading SMPLX pose data: {e}")
    else:
        print(f"ℹ️  SMPLX pose file not found: {smplx_pose_path}")
        # List available SMPLX files for debugging
        smplx_dir = smplx_pose_path.parent
        if smplx_dir.exists():
            available_files = list(smplx_dir.glob("*.pkl"))
            print(f"📁 Available SMPLX files: {[f.stem for f in available_files[:5]]}")
        else:
            print(f"❌ SMPLX directory not found: {smplx_dir}")

# Check for depth files (Trumans datasets have .exr depth files)
if args.dataset_type == "trumans" and depth_folder.exists():
    print(f"📁 Found depth folder: {depth_folder}")
    print(f"ℹ️  Trumans dataset - will convert .exr depth to disparity")
    
    # Check OpenEXR availability - MANDATORY for depth processing
    try:
        import OpenEXR
        print(f"✅ OpenEXR library available for depth processing")
    except ImportError:
        print(f"❌ ERROR: OpenEXR library is required for depth processing but not available")
        print(f"💡 Install OpenEXR with: pip install OpenEXR")
        print(f"💡 Or: conda install -c conda-forge openexr-python")
        raise ImportError("OpenEXR library required for depth processing but not available")
elif args.dataset_type == "aria" and depth_folder.exists():
    print(f"⚠️  Warning: ARIA dataset but depth folder found at {depth_folder}")
    print(f"ℹ️  ARIA datasets typically don't have depth files")
elif args.dataset_type == "aria" and not depth_folder.exists():
    print(f"ℹ️  ARIA dataset - no depth files (expected)")
elif args.dataset_type == "trumans" and not depth_folder.exists():
    print(f"⚠️  Warning: Trumans dataset but no depth folder found at {depth_folder}")
    print(f"ℹ️  Trumans datasets typically have .exr depth files")

print(f"📊 Total images: {len(image_paths)}")
if args.dataset_type == "aria" and smplh_data is not None:
    print(f"📊 Total egoallo poses: {smplh_data['Ts_world_root'].shape[1]}")
elif args.dataset_type == "trumans" and smplx_data is not None:
    if 'body_pose' in smplx_data:
        print(f"📊 Total SMPLX poses: {smplx_data['body_pose'].shape[0]}")

# Process SMPLX pose data for Trumans datasets
if args.dataset_type == "trumans" and smplx_data is not None:
    # Extract body pose from the loaded SMPLX pose data
    body_pose = extract_smplx_body_pose(smplx_data)
    if body_pose is not None:
        print(f"✅ SMPLX body pose extracted, shape: {body_pose.shape}")
    else:
        print("⚠️  Warning: Could not extract body pose from SMPLX pose data")
        smplx_data = None
elif args.dataset_type == "aria":
    print("ℹ️  ARIA dataset - SMPLX processing not applicable")
elif args.dataset_type == "trumans" and smplx_data is None:
    print("ℹ️  No SMPLX pose data available for processing")

# Create human pose output folder if we have any human pose data
if smplx_data is not None or smplh_data is not None:
    os.makedirs(human_pose_output_folder, exist_ok=True)

num_images = len(image_paths)

# Generate clips
clip_idx = 0
for start_idx in tqdm(range(0, num_images - clip_length * sample_every_nth + 1, stride * sample_every_nth)):
    # Check if this clip already exists (if skip_existing_clips is enabled)
    if args.skip_existing_clips:
        video_exists = os.path.exists(os.path.join(video_output_folder, f"{clip_idx:05}.mp4"))
        trajectory_exists = os.path.exists(os.path.join(trajectory_output_folder, f"{clip_idx:05}.npy"))
        
        # Check disparity existence based on format
        if args.disparity_format == "image":
            disparity_exists = os.path.exists(os.path.join(disparity_output_folder, f"{clip_idx:05}.mp4"))
        elif args.disparity_format == "npy":
            disparity_exists = os.path.exists(os.path.join(disparity_output_folder, f"{clip_idx:05}.npy"))
        else:  # npz format
            disparity_exists = os.path.exists(os.path.join(disparity_output_folder, f"{clip_idx:05}.npz"))
        
        # Check SMPLX existence if processing SMPLX
        smplx_exists = True  # Default to True if not processing SMPLX
        if smplx_data is not None:
            smplx_exists = os.path.exists(os.path.join(human_pose_output_folder, f"{clip_idx:05}.npz"))
        
        # Skip if all required files exist (respect force reprocessing flags)
        skip_video = video_exists
        skip_trajectory = trajectory_exists and not args.force_cam_pose_reprocessing
        skip_disparity = disparity_exists and not args.force_depth_reprocessing
        skip_smplx = smplx_exists
        
        if skip_video and skip_trajectory and skip_disparity and skip_smplx:
            print(f"Skipping clip {clip_idx}: all files already exist")
            clip_idx += 1
            continue
    
    clip_frames = image_paths[start_idx : start_idx + clip_length * sample_every_nth : sample_every_nth]
    if disparity_folder.exists():
        clip_disparity = disparity_paths[start_idx : start_idx + clip_length * sample_every_nth : sample_every_nth]
    if depth_folder.exists():
        clip_depth = depth_paths[start_idx : start_idx + clip_length * sample_every_nth : sample_every_nth]
    if cam_param_folder.exists():
        clip_cam_params = cam_param_paths[start_idx : start_idx + clip_length * sample_every_nth : sample_every_nth]
    
    # Prepare egoallo human pose data for this clip (ARIA datasets)
    clip_smplh = None
    if args.dataset_type == "aria" and smplh_data is not None:
        clip_smplh = {}
        for key in smplh_data:
            if key == 'Ts_world_root':
                clip_smplh["global_orient_quat"] = smplh_data[key][0, start_idx : start_idx + clip_length * sample_every_nth : sample_every_nth, :4] # global orientation in quaternion format Fx4
                clip_smplh["transl"] = smplh_data[key][0, start_idx : start_idx + clip_length * sample_every_nth : sample_every_nth, 4:] # translation Fx3
            elif key == 'body_quats':
                clip_smplh["body_pose_quat"] = smplh_data[key][0, start_idx : start_idx + clip_length * sample_every_nth : sample_every_nth, :] # body pose in quaternion format FxN_jx4
            elif key == 'left_hand_quats':
                clip_smplh["left_hand_pose_quat"] = smplh_data[key][0, start_idx : start_idx + clip_length * sample_every_nth : sample_every_nth, :] # left hand pose in quaternion format FxN_jx4
            elif key == 'right_hand_quats':
                clip_smplh["right_hand_pose_quat"] = smplh_data[key][0, start_idx : start_idx + clip_length * sample_every_nth : sample_every_nth, :] # right hand pose in quaternion format FxN_jx4
            elif key == 'betas':
                clip_smplh["betas"] = smplh_data[key][0, start_idx : start_idx + clip_length * sample_every_nth : sample_every_nth, :] # shape parameters FxN_betas_smplh
            else:
                continue # Skip any other keys
    


    # RGB video output
    out_path = os.path.join(video_output_folder, f"{clip_idx:05}.mp4")
    if not args.dry_run:
        if os.path.exists(out_path) and args.skip_existing_clips:
            print(f"Skipping existing video: {out_path}")
        else:
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

    # Disparity output - handle existing disparity files (if any)
    if disparity_folder.exists():
        if args.disparity_format == "npy":
            # Save as concatenated NPY file
            out_path = os.path.join(disparity_output_folder, f"{clip_idx:05}.npy")
            if not args.dry_run:
                if os.path.exists(out_path) and args.skip_existing_clips:
                    print(f"Skipping existing disparity npy: {out_path}")
                else:
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
        else:  # npz format
            # Save as compressed NPZ file
            out_path = os.path.join(disparity_output_folder, f"{clip_idx:05}.npz")
            if not args.dry_run:
                if os.path.exists(out_path) and args.skip_existing_clips:
                    print(f"Skipping existing disparity npz: {out_path}")
                else:
                    disparity_frames = []
                    for disparity_path in clip_disparity:
                        disparity_frame = iio.imread(disparity_path)
                        if output_size:
                            disparity_frame = cv2.resize(disparity_frame, output_size)
                        # Convert to grayscale if RGB
                        if disparity_frame.ndim == 3 and disparity_frame.shape[-1] == 3:
                            disparity_frame = np.mean(disparity_frame, axis=-1)
                        disparity_frames.append(disparity_frame.astype(np.float16))
                    # Stack frames and save as compressed NPZ
                    disparity_sequence = np.stack(disparity_frames, axis=0)
                    np.savez_compressed(out_path, disparity=disparity_sequence)
            else:
                print(f"Would create disparity npz: {out_path}")
        
        # Save colorized disparity video if requested
        if args.save_colorized_disparity:
            out_path = os.path.join(colorized_disparity_output_folder, f"{clip_idx:05}.mp4")
            if not args.dry_run:
                if os.path.exists(out_path) and args.skip_existing_clips:
                    print(f"Skipping existing colorized disparity video: {out_path}")
                else:
                    disparity_frames = []
                    for disparity_path in clip_disparity:
                        disparity_frame = iio.imread(disparity_path)
                        if output_size:
                            disparity_frame = cv2.resize(disparity_frame, output_size)
                        
                        # Convert to grayscale if RGB
                        if disparity_frame.ndim == 3 and disparity_frame.shape[-1] == 3:
                            disparity_frame = np.mean(disparity_frame, axis=-1)
                        
                        # Normalize to 0-1 range for colorize_depth function
                        disparity_frame = disparity_frame.astype(np.float32) / 255.0
                        # Apply colorization
                        disparity_frame = colorize_depth(disparity_frame)
                        # Convert back to 0-255 range for video
                        disparity_frame = (disparity_frame * 255).astype(np.uint8)
                        
                        disparity_frames.append(disparity_frame)
                    iio.imwrite(
                        out_path,
                        np.stack(disparity_frames),
                        fps=fps,
                        codec='libx264'
                    )
            else:
                print(f"Would create colorized disparity video: {out_path}")
    
    # Depth to disparity conversion (Trumans datasets)
    elif depth_folder.exists():
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
        
        if args.disparity_format == "npy":
            # Save as concatenated NPY file
            out_path = os.path.join(disparity_output_folder, f"{clip_idx:05}.npy")
            if not args.dry_run:
                if os.path.exists(out_path) and args.skip_existing_clips and not args.force_depth_reprocessing:
                    print(f"Skipping existing disparity npy: {out_path}")
                else:
                    # Stack frames and save as NPY
                    np.save(out_path, disparity_stack.astype(np.float32))
            else:
                print(f"Would create disparity npy: {out_path}")
        else:  # npz format
            # Save as compressed NPZ file
            out_path = os.path.join(disparity_output_folder, f"{clip_idx:05}.npz")
            if not args.dry_run:
                if os.path.exists(out_path) and args.skip_existing_clips and not args.force_depth_reprocessing:
                    print(f"Skipping existing disparity npz: {out_path}")
                else:
                    # Stack frames and save as compressed NPZ
                    np.savez_compressed(out_path, disparity=disparity_stack.astype(np.float16))
            else:
                print(f"Would create disparity npz: {out_path}")
        
        # Save colorized disparity video if requested
        if args.save_colorized_disparity:
            out_path = os.path.join(colorized_disparity_output_folder, f"{clip_idx:05}.mp4")
            if not args.dry_run:
                if os.path.exists(out_path) and args.skip_existing_clips and not args.force_depth_reprocessing:
                    print(f"Skipping existing colorized disparity video: {out_path}")
                else:
                    disparity_frames = []
                    for i, disparity in enumerate(disparity_stack):
                        # Apply colorization
                        disparity_colorized = colorize_depth(disparity)
                        # Convert to 8-bit RGB (0-255)
                        disparity_8bit = (disparity_colorized * 255).astype(np.uint8)
                        
                        if output_size:
                            disparity_8bit = cv2.resize(disparity_8bit, output_size)
                        disparity_frames.append(disparity_8bit)
                    iio.imwrite(
                        out_path,
                        np.stack(disparity_frames),
                        fps=fps,
                        codec='libx264'
                    )
            else:
                print(f"Would create colorized disparity video: {out_path}")

    # Camera trajectory output
    if cam_param_folder.exists():
        if not args.dry_run:
            relative_path = os.path.join(trajectory_output_folder, f"{clip_idx:05}.npy")
            absolute_path = os.path.join(trajectory_output_folder, f"{clip_idx:05}_abs.npy")
            
            # Check if both trajectory files exist
            if os.path.exists(relative_path) and os.path.exists(absolute_path) and args.skip_existing_clips and not args.force_cam_pose_reprocessing:
                print(f"Skipping existing trajectory files: {relative_path}, {absolute_path}")
            else:
                cam_params = [np.load(cam_param_path) for cam_param_path in clip_cam_params]
                
                # Apply Blender-to-OpenCV coordinate transformation for Trumans datasets
                if args.dataset_type == "trumans":
                    print(f"🔄 Applying Blender-to-OpenCV coordinate transformation for clip {clip_idx}")
                    # Transform the original camera parameters first
                    cam_params = [convert_blender_to_opencv_poses(cam_param.reshape(1, 4, 4)).squeeze(0) for cam_param in cam_params]
                
                # Compute relative and absolute trajectories from transformed camera parameters
                T0_inv = np.linalg.inv(cam_params[0])
                relative_trajectory = np.array([T0_inv @ T for T in cam_params]) # relative to the first frame
                absolute_trajectory = np.array(cam_params) # absolute trajectory in world coordinates

                np.save(relative_path, relative_trajectory)
                np.save(absolute_path, absolute_trajectory)
        else:
            print(f"Would create trajectory files: {clip_idx:05}.npy, {clip_idx:05}_abs.npy")

    # Human pose output - egoallo format (ARIA datasets)
    if args.dataset_type == "aria" and clip_smplh is not None:
        if not args.dry_run:
            pose_path = os.path.join(human_pose_output_folder, f"{clip_idx:05}.npz")
            if os.path.exists(pose_path) and args.skip_existing_clips:
                print(f"Skipping existing egoallo pose file: {pose_path}")
            else:
                np.savez_compressed(pose_path, **clip_smplh)
        else:
            print(f"Would create egoallo pose file: {clip_idx:05}.npz")

    # SMPLX pose output (Trumans datasets) - same format as ARIA
    if args.dataset_type == "trumans" and smplx_data is not None:
        body_pose = extract_smplx_body_pose(smplx_data)
        if body_pose is not None:
            # Extract the corresponding frames for this clip
            # Note: This assumes SMPLX data has the same number of frames as images
            clip_start = start_idx
            clip_end = start_idx + clip_length * sample_every_nth
            
            # Ensure we don't go out of bounds
            if clip_end <= body_pose.shape[0]:
                if not args.dry_run:
                    out_path = os.path.join(human_pose_output_folder, f"{clip_idx:05}.npz")
                    if os.path.exists(out_path) and args.skip_existing_clips:
                        print(f"Skipping existing SMPLX pose file: {out_path}")
                    else:
                        clip_body_pose = body_pose[clip_start:clip_end:sample_every_nth]
                        
                        # Prepare SMPLX data in the same format as ARIA egoallo data
                        smplx_clip_data = {}
                        
                        # Add all SMPLX parameters (following ARIA case pattern)
                        for key, value in smplx_data.items():
                            # Skip non-numerical data (NPZ can only store numpy arrays)
                            if not isinstance(value, np.ndarray):
                                print(f"⚠️  Skipping SMPLX key '{key}' - not a numpy array (type: {type(value)})")
                                continue
                                
                            if value.shape[0] == body_pose.shape[0]:
                                # Only include parameters that have the same number of frames
                                smplx_clip_data[key] = value[clip_start:clip_end:sample_every_nth].astype(np.float32)
                            elif len(value.shape) == 1:
                                # Handle scalar parameters (like betas) that don't have frame dimension
                                smplx_clip_data[key] = value.astype(np.float32)
                            else:
                                # Skip arrays with different frame count
                                print(f"⚠️  Skipping SMPLX key '{key}' with different frame count: {value.shape[0]} vs {body_pose.shape[0]}")
                        
                        # Save as compressed NPZ (same format as ARIA)
                        np.savez_compressed(out_path, **smplx_clip_data)
                else:
                    print(f"Would create SMPLX pose file: {clip_idx:05}.npz")
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

# Summary based on dataset type
if args.dataset_type == "aria":
    if smplh_data is not None:
        print(f"✅ Egoallo human pose data processed and saved to {human_pose_output_folder}")
    else:
        print("ℹ️  No egoallo data was processed")
elif args.dataset_type == "trumans":
    if smplx_data is not None:
        print(f"✅ SMPLX pose data processed and saved to {human_pose_output_folder}")
        print(f"📊 SMPLX files created: {clip_idx} files (compressed NPZ format)")
    else:
        print("ℹ️  No SMPLX pose data was processed")
    

