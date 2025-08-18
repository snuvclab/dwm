import argparse
import numpy as np
import os
from tqdm import tqdm
from postprocess_utils import camera_pose_to_raymap
from pathlib import Path
import imageio.v3 as iio

def load_disparity_video(video_path, reverse_sqrt=True):
    """
    Load disparity video and extract frames.
    
    Args:
        video_path: Path to MP4 disparity video
        reverse_sqrt: Whether to square the values to reverse sqrt operation from exr_to_disparity.py
        
    Returns:
        frames: List of disparity frames as numpy arrays
        dmax: Maximum disparity value across all frames
    """
    try:
        # Load the video
        video = iio.imread(video_path)
        
        # Convert to grayscale if it's RGB
        if video.ndim == 4 and video.shape[-1] == 3:
            # Convert RGB to grayscale
            video = np.mean(video, axis=-1).astype(np.float32)
        elif video.ndim == 4 and video.shape[-1] == 1:
            # Remove the channel dimension
            video = video.squeeze(-1).astype(np.float32)
        else:
            video = video.astype(np.float32)
        
        # Normalize to [0, 1] range (assuming 8-bit input)
        if video.max() > 1.0:
            video = video / 255.0
        
        # Square the values to reverse the sqrt operation applied in exr_to_disparity.py
        if reverse_sqrt:
            video = video ** 2
        
        # Extract frames
        frames = [video[i] for i in range(video.shape[0])]
        
        # Find maximum disparity value across all frames
        dmax = max(frame.max() for frame in frames if frame.size > 0)
        
        print(f"Loaded {len(frames)} frames from {video_path}")
        print(f"Frame shape: {frames[0].shape}")
        if reverse_sqrt:
            print(f"Max disparity value (after squaring): {dmax:.6f}")
        else:
            print(f"Max disparity value: {dmax:.6f}")
        
        return frames, dmax
        
    except Exception as e:
        print(f"Error loading video {video_path}: {e}")
        return None, None

def main(args):
    ext_dir = Path(args.data_root) / "sequences" / "trajectory"  # Load Fx4x4 extrinsics
    int_path = Path(args.data_root) / "cam_params" / "intrinsics.npy"  # Load 3x3 intrinsic parameters
    
    # Check if disparity is in video format or npy format
    disp_dir = Path(args.data_root) / "sequences" / "disparity"
    disp_video_dir = Path(args.data_root) / "sequences" / "disparity_video"  # New directory for video format
    
    # Determine which disparity directory to use
    if args.disparity_format == "video":
        if disp_video_dir.exists():
            disp_dir = disp_video_dir
        else:
            print(f"Warning: {disp_video_dir} does not exist, trying {disp_dir}")
    elif args.disparity_format == "npy":
        if not disp_dir.exists():
            print(f"Error: {disp_dir} does not exist")
            return
    else:  # auto-detect
        if disp_video_dir.exists():
            disp_dir = disp_video_dir
            args.disparity_format = "video"
            print(f"Auto-detected video format in {disp_video_dir}")
        elif disp_dir.exists():
            # Check if npz files exist first, then npy
            npz_files = list(disp_dir.glob("*.npz"))
            npy_files = list(disp_dir.glob("*.npy"))
            if npz_files:
                args.disparity_format = "npz"
                print(f"Auto-detected npz format in {disp_dir}")
            elif npy_files:
                args.disparity_format = "npy"
                print(f"Auto-detected npy format in {disp_dir}")
            else:
                print(f"Error: No disparity files found in {disp_dir}")
                return
        else:
            print(f"Error: Neither {disp_video_dir} nor {disp_dir} exist")
            return

    out_dir = Path(args.data_root) / "sequences" / "raymaps"
    out_dir.mkdir(parents=True, exist_ok=True)

    K = np.load(int_path)

    for traj in tqdm(sorted(ext_dir.glob("*.npy"))):
        Rt = np.load(traj)
        name = traj.stem.replace("_abs", "")
        
        if args.disparity_format == "video":
            # Handle MP4 video format
            video_path = disp_dir / f"{name}.mp4"
            if not video_path.exists():
                print(f"Warning: Video file {video_path} not found, skipping...")
                continue
                
            # Load disparity video and extract frames
            disparity_frames, dmax = load_disparity_video(video_path, reverse_sqrt=args.reverse_sqrt)
            if disparity_frames is None:
                print(f"Failed to load disparity video {video_path}, skipping...")
                continue
                
        else:
            # Handle NPY/NPZ format
            if args.disparity_format == "npz":
                disparity_path = disp_dir / f"{name}.npz"
                if not disparity_path.exists():
                    print(f"Warning: Disparity file {disparity_path} not found, skipping...")
                    continue
                disparity = np.load(disparity_path)['disparity']
            else:  # npy format
                disparity_path = disp_dir / f"{name}.npy"
                if not disparity_path.exists():
                    print(f"Warning: Disparity file {disparity_path} not found, skipping...")
                    continue
                disparity = np.load(disparity_path)
            
            # Apply reverse sqrt operation for NPY/NPZ formats (same as MP4 videos)
            if args.reverse_sqrt:
                disparity = disparity ** 2
                print(f"Applied reverse sqrt (squared) to {args.disparity_format} disparity data for {args.dataset_type} dataset")
            
            dmax = disparity.max()

            # If disparity is 3D (T, H, W), extract frames
            if disparity.ndim == 3:
                disparity_frames = [disparity[i] for i in range(disparity.shape[0])]
            else:
                # If it's 2D, treat as single frame
                disparity_frames = [disparity]

        # Handle dmax value appropriately
        # For Trumans datasets: reverse sqrt operation is applied (sqrt was applied in exr_to_disparity.py/make_sequences.py)
        # For ARIA datasets: no reverse sqrt operation (DepthAnyVideo doesn't apply sqrt)
        # - For EXR-derived NPY/NPZ: dmax represents pre-normalization maximum (data is normalized to [0,1])
        # - For MP4 videos: dmax represents the actual maximum (typically ≈ 1.0)
        
        print(f"Using dmax={dmax:.6f} for {args.disparity_format} format")

        # Check if output file already exists
        output_path = out_dir / f"{traj.stem}.npz"
        if args.skip_existing and output_path.exists():
            print(f"Skipping {name} - output file already exists: {output_path}")
            continue

        # Generate raymap using the maximum disparity value
        raymap = camera_pose_to_raymap(Rt, np.tile(K, (len(Rt), 1, 1)), ray_o_scale_factor=10.0, dmax=dmax)
        np.savez_compressed(output_path, raymap=raymap)

        print(f"Generated raymap for {name} with dmax={dmax:.6f}")

        if args.debug:
            break
        
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert camera pose to raymap.")
    parser.add_argument("--data_root", type=str, required=True, help="Data root dir.")
    parser.add_argument("--disparity_format", type=str, choices=["video", "npy", "npz", "auto"], 
                       default="auto", help="Format of disparity data: video (MP4), npy, or npz files")
    parser.add_argument("--dataset_type", type=str, choices=["aria", "trumans"], 
                       default="trumans", help="Dataset type: 'aria' for DepthAnyVideo results, 'trumans' for Blender rendering (default: trumans)")
    parser.add_argument("--reverse_sqrt", type=bool, default=True,
                       help="Whether to square disparity values to reverse sqrt operation from exr_to_disparity.py/make_sequences.py (default: True)")
    parser.add_argument("--no_reverse_sqrt", action="store_true", default=False,
                       help="Disable reverse sqrt operation (use with --no_sqrt_disparity in exr_to_disparity.py)")
    parser.add_argument("--debug", action="store_true", help="Process only one trajectory for debugging")
    parser.add_argument("--skip_existing", action="store_true", help="Skip processing if output file already exists")
    args = parser.parse_args()
    
    print(f"📊 Dataset type: {args.dataset_type.upper()}")
    
    # Set reverse_sqrt based on dataset type and arguments
    # Trumans datasets need reverse sqrt (sqrt applied in exr_to_disparity.py/make_sequences.py)
    # ARIA datasets don't need reverse sqrt (DepthAnyVideo doesn't apply sqrt)
    if args.no_reverse_sqrt or args.dataset_type == "aria":
        args.reverse_sqrt = False
        if args.dataset_type == "aria":
            print(f"ℹ️  ARIA dataset - disabling reverse sqrt (DepthAnyVideo doesn't apply sqrt)")
        else:
            print(f"ℹ️  Manual override - disabling reverse sqrt")
    else:
        args.reverse_sqrt = True
        print(f"ℹ️  Trumans dataset - enabling reverse sqrt (to reverse sqrt from exr_to_disparity.py/make_sequences.py)")
    
    print(f"Using reverse_sqrt: {args.reverse_sqrt}")

    main(args)