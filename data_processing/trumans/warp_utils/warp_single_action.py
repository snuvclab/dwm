#!/usr/bin/env python3
"""
Create colored depth warped video using first frame image/depth and camera parameters.
Load depth exr file from data/trumans/ego_render_fov90/{scene_name}/{action_name}/depth/*.exr
Each exr file is a sequence of 49 depth maps (hxw=720x480). Each frame was sampled with frame-skip 3.
So for example, 00000.exr corresponds to cam_params/cam_0000.npy, cam_0003.npy, ... cam_0144.npy
and each exr has half overlap - so the 0001.exr starts from cam_0075.npy (frame skip 3 * 25 = 75). 
And intrinsic is given as cam_params/intrinsics.npy
I want to make a colored depth warped video only using the image and depth map of the first frame. 
Camera parameters are camera to world.
The first frame will be images/0000.png for the first video, 0075.png for the second video, ....
"""

import os
import sys
import numpy as np
import torch
import cv2
from pathlib import Path
import argparse
from tqdm import tqdm
import imageio.v3 as iio

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))
from utils import Warper, save_video, colorize_depth

# Reduce CPU over-subscription when running multiple processes
try:
    import torch
    torch.set_num_threads(1)
except Exception:
    pass
try:
    cv2.setNumThreads(0)
except Exception:
    pass

def load_exr_depth(file_path):
    """Load depth data from .exr file."""
    try:
        import OpenEXR
        import Imath
        
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

def is_valid_file(path: Path, min_size: int = 1024) -> bool:
    try:
        return path.is_file() and path.stat().st_size >= min_size
    except Exception:
        return False


def create_colored_depth_warped_video(
    data_root,
    output_path,
    device='cuda',
    skip_existing=True,
    save_warped_depth=False,
    batch_size: int = 4,
):
    """
    Create colored depth warped videos for each video in data_root/videos.
    Each video corresponds to a specific frame number, and we warp that frame's image/depth.
    
    Args:
        data_root: Path to data directory containing images/, depth/, cam_params/, videos/
        output_path: Path to save the output videos
        device: Device to use for processing
    """
    
    # Setup paths
    data_root = Path(data_root)
    # Updated folder structure: use sequences/{images_static, depth_static, videos_static}
    images_folder = data_root / "sequences" / "images_static"
    depth_folder = data_root / "sequences" / "depth_static"
    cam_param_folder = data_root / "cam_params"
    
    # Check if required folders exist
    if not images_folder.exists():
        raise FileNotFoundError(f"Images folder not found: {images_folder}")
    if not depth_folder.exists():
        raise FileNotFoundError(f"Depth folder not found: {depth_folder}")
    if not cam_param_folder.exists():
        raise FileNotFoundError(f"Camera parameters folder not found: {cam_param_folder}")
    # videos_folder is no longer required; we derive video indices from images/depth
    
    # Derive video names from images_static (only those that have matching depth)
    image_files_all = sorted(list(images_folder.glob("*.png")))
    video_names_all = [p.stem for p in image_files_all if (depth_folder / f"{p.stem}.exr").exists()]
    if not video_names_all:
        raise FileNotFoundError(f"No matching image/depth pairs found in {images_folder} and {depth_folder}")
    total_videos = len(video_names_all)
    print(f"Found {total_videos} image/depth pairs to process")
    
    # Load intrinsics
    intrinsics_path = cam_param_folder / "intrinsics.npy"
    if not intrinsics_path.exists():
        raise FileNotFoundError(f"Intrinsics file not found: {intrinsics_path}")
    
    K = np.load(intrinsics_path)  # (3, 3) intrinsic matrix
    
    # Determine dimensions from the first available image in images_static
    first_image_files = sorted(list(images_folder.glob("*.png")))
    if not first_image_files:
        raise FileNotFoundError(f"No images found in {images_folder}")
    first_image = iio.imread(first_image_files[0])  # (H, W, 3)
    H, W = first_image.shape[:2]
    
    # Load sequence parameters from args.json (created by make_sequences.py)
    args_json_path = data_root / "sequences" / "args.json"
    if args_json_path.exists():
        import json
        with open(args_json_path, 'r') as f:
            sequence_args = json.load(f)
        stride = sequence_args.get('stride', 25)
        sample_every_nth = sequence_args.get('sample_every_nth', 3)
        print(f"📊 Loaded sequence parameters: stride={stride}, sample_every_nth={sample_every_nth}")
    else:
        # Default values if args.json not found
        stride = 25
        sample_every_nth = 3
        print(f"⚠️  args.json not found, using default values: stride={stride}, sample_every_nth={sample_every_nth}")
    
    def get_video_start_frame(video_name):
        """Calculate the start frame index for camera parameters only (image/depth come from images_static/depth_static)."""
        try:
            video_idx = int(video_name)
            return video_idx * stride
        except ValueError:
            print(f"⚠️  Could not parse video name: {video_name}")
            return 0
    
    # Setup warper
    print("Setting up warper...")
    warper = Warper(resolution=(H, W), device=device)
    
    # Convert intrinsics to torch tensor
    device_torch = torch.device(device)
    K_tensor = torch.from_numpy(K).unsqueeze(0).float().to(device_torch)
    
    # Create output directories
    output_dir = data_root / output_path if output_path else data_root / "warped_depth"
    warped_videos_dir = output_dir / "warped_videos"
    warped_depth_videos_dir = output_dir / "warped_depth_videos"
    warped_mask_videos_dir = output_dir / "warped_mask_videos"
    
    warped_videos_dir.mkdir(parents=True, exist_ok=True)
    warped_depth_videos_dir.mkdir(parents=True, exist_ok=True)
    warped_mask_videos_dir.mkdir(parents=True, exist_ok=True)
    
    # Process videos in batches for better GPU utilization
    # batch_size is now provided via CLI (default 4)
    print(f"Processing videos in batches of {batch_size}")
    processed_videos = 0
    skipped_videos = 0
    
    # Camera parameters will be loaded per video since each video has different start frame
    # We'll load them in the video processing loop
    num_frames = 49  # Each video has 49 frames
    
    # Process videos in batches
    for batch_idx in tqdm(range(0, len(video_names_all), batch_size), desc="Processing batches"):
        batch_video_names_all = video_names_all[batch_idx:batch_idx + batch_size]
        
        # Load images and depths for this batch
        batch_images = []
        batch_depths = []
        batch_video_names = []
        
        for video_name in batch_video_names_all:
            # Skip if all outputs already exist (and skip_existing is True)
            if skip_existing:
                out_rgb = warped_videos_dir / f"{video_name}.mp4"
                out_mask = warped_mask_videos_dir / f"{video_name}.mp4"
                has_rgb = is_valid_file(out_rgb)
                has_mask = is_valid_file(out_mask)
                has_depth = is_valid_file(warped_depth_videos_dir / f"{video_name}.mp4") if save_warped_depth else True
                if has_rgb and has_mask and has_depth:
                    print(f"  ⏭️  [{processed_videos + skipped_videos + 1}/{total_videos}] Skipping {video_name}: outputs already exist")
                    skipped_videos += 1
                    continue
            
            # Image/depth now map directly by video index name
            image_file = images_folder / f"{video_name}.png"
            depth_file = depth_folder / f"{video_name}.exr"
            
            print(f"  Video {video_name}: using first-frame files -> {image_file.name}, {depth_file.name}")
            
            if not image_file.exists():
                print(f"  ⚠️  Image file not found: {image_file}, skipping {video_name}")
                continue
                
            if not depth_file.exists():
                print(f"  ⚠️  Depth file not found: {depth_file}, skipping {video_name}")
                continue
            
            image = iio.imread(image_file)  # (H, W, 3)
            depth = load_exr_depth(depth_file)  # (H, W)
            
            batch_images.append(image)
            batch_depths.append(depth)
            batch_video_names.append(video_name)
        
        if not batch_images:
            continue
        
        # Convert to batch tensors
        batch_images_tensor = torch.stack([
            torch.from_numpy(img).permute(2, 0, 1).float()
            for img in batch_images
        ])  # (batch_size, 3, H, W)
        batch_images_tensor = (batch_images_tensor / 255.0) * 2.0 - 1.0  # [0, 255] -> [-1, 1]
        batch_images_tensor = batch_images_tensor.to(device_torch)
        
        batch_depths_tensor = torch.stack([
            torch.from_numpy(depth).unsqueeze(0).float()
            for depth in batch_depths
        ])  # (batch_size, 1, H, W)
        batch_depths_tensor = batch_depths_tensor.to(device_torch)
        
        # Process each video in batch (frames are processed sequentially)
        for video_idx, video_name in enumerate(batch_video_names):
            print(f"\n[{processed_videos + skipped_videos + 1}/{total_videos}] Processing video: {video_name} (batch {batch_idx//batch_size + 1})")
            
            # Calculate the start frame only for camera parameters (images/depth come from images_static/depth_static)
            start_frame = get_video_start_frame(video_name)
            
            # Load all frames' camera params for this video at once
            print(f"  Loading all {num_frames} frames for video {video_name}...")
            video_images = []
            video_depths = []
            
            for i in range(num_frames):
                # For warping, we always use the first-frame image/depth of this video
                video_images.append(batch_images[video_idx])
                video_depths.append(batch_depths[video_idx])
            
            if not video_images:
                print(f"  ❌ No valid frames found for video {video_name}")
                continue
            
            # Convert to batch tensors
            video_images_tensor = torch.stack([
                torch.from_numpy(img).permute(2, 0, 1).float()
                for img in video_images
            ])  # (num_frames, 3, H, W)
            video_images_tensor = (video_images_tensor / 255.0) * 2.0 - 1.0  # [0, 255] -> [-1, 1]
            video_images_tensor = video_images_tensor.to(device_torch)
            
            video_depths_tensor = torch.stack([
                torch.from_numpy(depth).unsqueeze(0).float()
                for depth in video_depths
            ])  # (num_frames, 1, H, W)
            video_depths_tensor = video_depths_tensor.to(device_torch)
            
            # Load camera parameters for this video's sequence
            cam_param_files = []
            for i in range(num_frames):
                frame_idx = start_frame + i
                cam_file_idx = frame_idx * sample_every_nth
                cam_file = cam_param_folder / f"cam_{cam_file_idx:04d}.npy"
                if cam_file.exists():
                    cam_param_files.append(cam_file)
                else:
                    print(f"  Warning: Camera parameter file not found: {cam_file}")
                    break
            
            if len(cam_param_files) != num_frames:
                print(f"  Warning: Expected {num_frames} camera files, found {len(cam_param_files)}")
                num_frames_actual = len(cam_param_files)
            else:
                num_frames_actual = num_frames
            
            if num_frames_actual == 0:
                print(f"  ❌ No camera parameters found for {video_name}")
                continue
            
            print(f"  Loading {num_frames_actual} camera parameters...")
            cam_params = []
            for cam_file in cam_param_files:
                cam_param = np.load(cam_file)  # (4, 4) camera-to-world matrix
                cam_params.append(cam_param)
            
            cam_params = np.array(cam_params)  # (num_frames, 4, 4)
            print(f"  ✅ Loaded {len(cam_params)} camera parameters for {video_name}")
            
            # Convert camera parameters from Blender to OpenCV coordinates
            cam_params_opencv = convert_blender_to_opencv_poses(cam_params)
            
            # Convert to world-to-camera (inverse of camera-to-world)
            w2c_params = []
            for cam_param in cam_params_opencv:
                w2c_param = np.linalg.inv(cam_param)
                w2c_params.append(w2c_param)
            w2c_params = np.array(w2c_params)  # (num_frames, 4, 4)
            
            # Convert to torch tensor
            w2c_tensor = torch.from_numpy(w2c_params).float().to(device_torch)
            
            # Create warped frames for this video
            warped_frames = []
            colored_depth_frames = []
            mask_frames = []
            
            # Get single video's first frame image and depth (already loaded in batch)
            image_tensor = batch_images_tensor[video_idx:video_idx+1]  # (1, 3, H, W)
            depth_tensor = batch_depths_tensor[video_idx:video_idx+1]  # (1, 1, H, W)
            
            # First frame pose (source pose for warping)
            pose_first = w2c_tensor[0:1]  # (1, 4, 4)
            
            for i in tqdm(range(num_frames_actual), total=num_frames_actual, leave=False, desc=f"{video_name} frames"):
                # Target pose (each frame's camera pose)
                pose_t = w2c_tensor[i:i+1]  # (1, 4, 4)
                
                # Warp first frame to current frame's pose
                warped_frame, mask, warped_depth, _ = warper.forward_warp(
                    image_tensor[:,:3],
                    None,  # mask
                    depth_tensor,
                    pose_first,  # Source pose (first frame)
                    pose_t,     # Target pose (current frame)
                    K_tensor,
                    None,  # intrinsic2
                    mask=False,
                    twice=False
                )
                
                # Convert back to numpy and normalize to [0, 1]
                warped_frame_np = warped_frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
                warped_frame_np = (warped_frame_np + 1.0) / 2.0  # [-1, 1] -> [0, 1]
                warped_frame_np = np.clip(warped_frame_np, 0, 1)
                warped_frames.append(warped_frame_np)
                
                # Create colored depth version using warped depth
                mask_np = mask.squeeze(0).squeeze(0).cpu().numpy()  # (H, W)
                warped_depth_np = warped_depth.squeeze(0).squeeze(0).cpu().numpy()  # (H, W)
                
                # Store mask frame for mask video (convert to 3-channel for video)
                mask_frame = np.stack([mask_np] * 3, axis=-1)  # (H, W, 3)
                mask_frames.append(mask_frame)
                
                # Normalize warped depth for coloring
                warped_depth_normalized = warped_depth_np / warped_depth_np.max() if warped_depth_np.max() > 0 else warped_depth_np
                
                # Create colored depth visualization using warped depth
                depth_colored = colorize_depth(warped_depth_normalized)
                depth_colored = (depth_colored * 255).astype(np.uint8)
                
                # Apply mask to colored depth
                mask_3d = np.stack([mask_np] * 3, axis=-1)
                colored_depth_masked = depth_colored * mask_3d
                colored_depth_frames.append(colored_depth_masked)
            
            # Save videos for this video
            warped_video_path = warped_videos_dir / f"{video_name}.mp4"
            print(f"  Saving warped video: {warped_video_path}")
            warped_video = np.stack(warped_frames, axis=0)  # (num_frames, H, W, 3)
            save_video(warped_video, warped_video_path, fps=8)
            
            if save_warped_depth:
                warped_depth_video_path = warped_depth_videos_dir / f"{video_name}.mp4"
                print(f"  Saving warped depth video: {warped_depth_video_path}")
                warped_depth_video = np.stack(colored_depth_frames, axis=0)  # (num_frames, H, W, 3)
                save_video(warped_depth_video, warped_depth_video_path, fps=8)
            
            warped_mask_video_path = warped_mask_videos_dir / f"{video_name}.mp4"
            print(f"  Saving warped mask video: {warped_mask_video_path}")
            warped_mask_video = np.stack(mask_frames, axis=0)  # (num_frames, H, W, 3)
            save_video(warped_mask_video, warped_mask_video_path, fps=8)

            processed_videos += 1
            print(f"  ✅ Completed {video_name}  | Progress: {processed_videos}/{total_videos} (skipped: {skipped_videos})")
        
        # Clear GPU memory after each batch
        if device_torch.type == 'cuda':
            del batch_images_tensor, batch_depths_tensor
            torch.cuda.empty_cache()
    
    print("\n✅ All warped videos creation complete!")
    print(f"📊 Summary: processed={processed_videos}, skipped={skipped_videos}, total={total_videos}")
    print(f"📁 Warped videos: {warped_videos_dir}")
    if save_warped_depth:
        print(f"📁 Warped depth videos: {warped_depth_videos_dir}")
    print(f"📁 Warped mask videos: {warped_mask_videos_dir}")

def main():
    parser = argparse.ArgumentParser(description="Create colored depth warped video from first frame")
    parser.add_argument("--data_root", type=str, required=True, 
                       help="Path to data directory containing images/, depth/, cam_params/")
    parser.add_argument("--output", type=str, default="",
                       help="Output video path")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--no-skip-existing", action="store_true", help="Recreate outputs even if they exist")
    parser.add_argument("--save-warped-depth", action="store_true", help="Also save warped depth videos")
    parser.add_argument("--batch-size", type=int, default=4, help="How many videos to load/process per batch (default: 4)")
    
    args = parser.parse_args()
    
    print("🎬 Colored Depth Warped Video Creator")
    print(f"📁 Data root: {args.data_root}")
    print(f"📁 Output: {args.output}")
    print(f"🖥️  Device: {args.device}")
    print(f"📦 Batch size: {args.batch_size}")
    print("=" * 60)
    
    try:
        create_colored_depth_warped_video(
            args.data_root,
            args.output,
            args.device,
            skip_existing=not args.no_skip_existing,
            save_warped_depth=args.save_warped_depth,
            batch_size=args.batch_size,
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
