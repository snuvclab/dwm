#!/usr/bin/env python3
"""
HaWoR Video Rendering Script - Camera Space Only
Directly uses camera space predictions from hawor_motion_estimation.
No SLAM or infiller required.

Renders hand meshes overlaid on video and saves outputs:
1. Mesh overlaid on original video
2. Mesh only on black background
3. Hand mask
"""

import argparse
import sys
import os
import torch
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import cv2
import subprocess
import tempfile
import shutil
import json
from glob import glob
from natsort import natsorted
import imageio

from scripts.scripts_test_video.detect_track_video import detect_track_video
from scripts.scripts_test_video.hawor_video import hawor_motion_estimation
from hawor.utils.process import get_mano_faces, run_mano, run_mano_left
from hawor.utils.rotation import rotation_matrix_to_angle_axis
from lib.pipeline.tools import detect_track, parse_chunks
from lib.datasets.track_dataset import TrackDatasetEval

# PyTorch3D imports
try:
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import (
        PointLights,
        PerspectiveCameras,
        SoftPhongShader,
        RasterizationSettings,
        MeshRendererWithFragments,
        MeshRasterizer,
        TexturesVertex
    )
    PYTORCH3D_AVAILABLE = True
except ImportError:
    print("Error: PyTorch3D is required for this script")
    print("Install with: pip install pytorch3d")
    sys.exit(1)


def render_hands_pytorch3d_cam_only(left_verts, right_verts, left_faces, right_faces, 
                                     focal_length, img_width, img_height, pred_valid=None):
    """
    Render hands using PyTorch3D directly in camera space.
    Vertices are already in camera space (OpenCV coordinate system).
    
    Args:
        left_verts: (T, V, 3) left hand vertices in camera space (OpenCV)
        right_verts: (T, V, 3) right hand vertices in camera space (OpenCV)
        left_faces: (F, 3) left hand faces
        right_faces: (F, 3) right hand faces
        focal_length: camera focal length
        img_width, img_height: image dimensions
        pred_valid: (2, T) validity mask, where [0, :] is left hand, [1, :] is right hand
    
    Returns:
        rendered_frames: List of rendered mesh images (RGB, 0-255)
        mask_frames: List of binary masks (0 or 255)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    num_frames = len(left_verts)
    rendered_frames = []
    mask_frames = []
    
    # Setup camera intrinsics
    fx = fy = focal_length
    cx, cy = img_width / 2.0, img_height / 2.0
    focal_tensor = torch.tensor([[fx, fy]], device=device, dtype=torch.float32)
    princpt = torch.tensor([[cx, cy]], device=device, dtype=torch.float32)
    image_size = torch.tensor([[img_height, img_width]], device=device, dtype=torch.int64)
    
    # Setup rasterizer
    raster_settings = RasterizationSettings(
        image_size=(img_height, img_width),
        blur_radius=0.0,
        faces_per_pixel=1,
        perspective_correct=True,
    )
    
    # Setup lighting
    lights = PointLights(
        device=device,
        location=[[0.0, 0.0, 0.0]],
        ambient_color=((0.7, 0.7, 0.7),),
        diffuse_color=((0.6, 0.6, 0.6),),
        specular_color=((0.0, 0.0, 0.0),),
    )
    
    # Convert faces to torch - ensure they are (F, 3) shape
    if isinstance(left_faces, np.ndarray):
        left_faces_np = left_faces
    else:
        left_faces_np = np.array(left_faces)
    
    if isinstance(right_faces, np.ndarray):
        right_faces_np = right_faces
    else:
        right_faces_np = np.array(right_faces)
    
    # Ensure shape is (F, 3)
    if left_faces_np.ndim == 1:
        left_faces_np = left_faces_np.reshape(-1, 3)
    if right_faces_np.ndim == 1:
        right_faces_np = right_faces_np.reshape(-1, 3)
    
    left_faces_t = torch.from_numpy(left_faces_np).to(device).long()
    right_faces_t = torch.from_numpy(right_faces_np).to(device).long()
    
    # Hand colors
    left_color = (0.5, 0.8, 0.5)   # Vibrant green
    right_color = (0.8, 0.4, 0.4)  # Deep red
    
    # Coordinate conversion: OpenCV (X-right, Y-down, Z-out) -> PyTorch3D (X-left, Y-up, Z-out)
    coord_convert = np.diag([-1.0, -1.0, 1.0]).astype(np.float32)
    R_p3d = torch.from_numpy(coord_convert).to(device).float()
    
    # Check validity if provided
    left_hand_valid = None
    right_hand_valid = None
    if pred_valid is not None:
        # pred_valid shape: (2, T) where [0, :] is left, [1, :] is right
        left_hand_valid = pred_valid[0].cpu().numpy()  # (T,)
        right_hand_valid = pred_valid[1].cpu().numpy()  # (T,)
    
    for frame_idx in range(num_frames):
        if frame_idx % 10 == 0 or frame_idx == num_frames - 1:
            print(f"  Rendering frame {frame_idx + 1}/{num_frames} ({(frame_idx + 1) / num_frames * 100:.1f}%)...")
        
        # Check if hands are valid for this frame
        left_valid_frame = left_hand_valid is None or left_hand_valid[frame_idx] > 0
        right_valid_frame = right_hand_valid is None or right_hand_valid[frame_idx] > 0
        
        # Setup camera (identity since vertices are already in camera space)
        cameras = PerspectiveCameras(
            R=torch.eye(3, device=device).unsqueeze(0),
            T=torch.zeros(1, 3, device=device),
            focal_length=focal_tensor,
            principal_point=princpt,
            in_ndc=False,
            image_size=image_size,
            device=device,
        )
        
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings).to(device)
        shader = SoftPhongShader(device=device, cameras=cameras, lights=lights)
        renderer = MeshRendererWithFragments(rasterizer=rasterizer, shader=shader)
        
        # Render both hands
        final_rgb = torch.zeros((img_height, img_width, 3), device=device, dtype=torch.float32)
        final_z = torch.full((img_height, img_width), float('inf'), device=device, dtype=torch.float32)
        
        # Render left hand (only if valid)
        if left_valid_frame:
            # Vertices are in camera space (OpenCV coordinate system)
            # Convert to PyTorch3D coordinate system
            left_v_cam_t = left_verts[frame_idx].to(device).float()
            left_v_p3d = left_v_cam_t @ R_p3d.t()
            left_v_tensor = left_v_p3d.unsqueeze(0)
            
            left_colors = torch.tensor(left_color, device=device).view(1, 1, 3).expand(1, left_v_tensor.shape[1], 3)
            left_faces_batch = left_faces_t.unsqueeze(0) if left_faces_t.ndim == 2 else left_faces_t
            left_mesh = Meshes(verts=left_v_tensor, faces=left_faces_batch, textures=TexturesVertex(left_colors))
            
            with torch.no_grad():
                left_images, left_frags = renderer(left_mesh)
            
            left_rgb = left_images[0, :, :, :3]
            left_zbuf = left_frags.zbuf[0, :, :, 0].float()
            left_valid_mask = left_frags.pix_to_face[0, :, :, 0] >= 0
            
            closer = left_valid_mask & (left_zbuf < final_z)
            closer3 = closer.unsqueeze(-1)
            final_rgb = torch.where(closer3, left_rgb, final_rgb)
            final_z = torch.where(closer, left_zbuf, final_z)
        
        # Render right hand (only if valid)
        if right_valid_frame:
            # Vertices are in camera space (OpenCV coordinate system)
            # Convert to PyTorch3D coordinate system
            right_v_cam_t = right_verts[frame_idx].to(device).float()
            right_v_p3d = right_v_cam_t @ R_p3d.t()
            right_v_tensor = right_v_p3d.unsqueeze(0)
            
            right_colors = torch.tensor(right_color, device=device).view(1, 1, 3).expand(1, right_v_tensor.shape[1], 3)
            right_faces_batch = right_faces_t.unsqueeze(0) if right_faces_t.ndim == 2 else right_faces_t
            right_mesh = Meshes(verts=right_v_tensor, faces=right_faces_batch, textures=TexturesVertex(right_colors))
            
            with torch.no_grad():
                right_images, right_frags = renderer(right_mesh)
            
            right_rgb = right_images[0, :, :, :3]
            right_zbuf = right_frags.zbuf[0, :, :, 0].float()
            right_valid_mask = right_frags.pix_to_face[0, :, :, 0] >= 0
            
            closer = right_valid_mask & (right_zbuf < final_z)
            closer3 = closer.unsqueeze(-1)
            final_rgb = torch.where(closer3, right_rgb, final_rgb)
            final_z = torch.where(closer, right_zbuf, final_z)
        
        # Convert to numpy
        rendered_img = (final_rgb.clamp(0, 1) * 255.0).byte().cpu().numpy()
        
        # Create binary mask from depth (0 or 255)
        mask = (final_z < float('inf')).cpu().numpy().astype(np.uint8) * 255
        
        # Store rendered mesh (RGB) and mask
        rendered_frames.append(cv2.cvtColor(rendered_img, cv2.COLOR_RGB2BGR))
        mask_frames.append(mask)
    
    return rendered_frames, mask_frames


def create_video_from_frames(frames, output_path, fps=30):
    """Create MP4 video from list of frames using ffmpeg."""
    temp_dir = tempfile.mkdtemp()
    try:
        # Save frames to temp directory
        for i, frame in enumerate(frames):
            frame_path = os.path.join(temp_dir, f'frame_{i:06d}.jpg')
            cv2.imwrite(frame_path, frame)
        
        # Create video with ffmpeg
        ffmpeg_cmd = [
            '/usr/bin/ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', os.path.join(temp_dir, 'frame_%06d.jpg'),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '18',
            output_path
        ]
        
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error creating video: {e}")
        if e.stderr:
            print(f"stderr: {e.stderr.decode()}")
        return False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def load_camera_space_predictions(seq_folder, frame_chunks_all, num_frames):
    """
    Load camera space predictions directly from cam_space folder.
    
    Args:
        seq_folder: sequence folder path
        frame_chunks_all: list of frame chunks for each hand
        num_frames: total number of frames
    
    Returns:
        pred_trans: (2, T, 3) camera space translations
        pred_rot: (2, T, 3) camera space rotations (angle axis)
        pred_hand_pose: (2, T, 45) camera space hand poses (angle axis)
        pred_betas: (2, T, 10) shape parameters
        pred_valid: (2, T) validity mask
    """
    pred_trans = torch.zeros(2, num_frames, 3)
    pred_rot = torch.zeros(2, num_frames, 3)
    pred_hand_pose = torch.zeros(2, num_frames, 45)
    pred_betas = torch.zeros(2, num_frames, 10)
    pred_valid = torch.zeros((2, num_frames))
    
    # Load camera space predictions for each hand
    for idx in [0, 1]:  # 0: left, 1: right
        frame_chunks = frame_chunks_all[idx]
        
        if len(frame_chunks) == 0:
            continue
        
        for frame_ck in frame_chunks:
            pred_path = os.path.join(seq_folder, 'cam_space', str(idx), f"{frame_ck[0]}_{frame_ck[-1]}.json")
            
            if not os.path.exists(pred_path):
                print(f"Warning: {pred_path} not found")
                continue
            
            with open(pred_path, "r") as f:
                pred_dict = json.load(f)
            
            data_out = {
                k: torch.tensor(v) for k, v in pred_dict.items()
            }
            
            # Convert rotation matrices to angle axis
            root_orient_aa = rotation_matrix_to_angle_axis(data_out["init_root_orient"])
            hand_pose_aa = rotation_matrix_to_angle_axis(data_out["init_hand_pose"])
            
            # Store predictions
            pred_trans[[idx], frame_ck] = data_out["init_trans"]
            pred_rot[[idx], frame_ck] = root_orient_aa
            pred_hand_pose[[idx], frame_ck] = hand_pose_aa.flatten(-2)  # (B, T, 15*3)
            pred_betas[[idx], frame_ck] = data_out["init_betas"]
            pred_valid[[idx], frame_ck] = 1
    
    return pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid


def process_video(video_path, checkpoint, output_dir=None, keep_intermediates=False, 
                  fps=None, img_focal=None, model=None, model_cfg=None):
    """
    Process a single video - can be called directly from batch script.
    
    Args:
        video_path: Path to input video
        checkpoint: Path to checkpoint (used if model is None)
        output_dir: Output directory (default: same as video)
        keep_intermediates: Keep intermediate files
        fps: Frames per second (default: auto-detect from video)
        img_focal: Focal length (default: auto-detect)
        model: Optional pre-loaded model (for batch processing optimization)
        model_cfg: Optional model config (for batch processing optimization)
    
    Returns:
        True if successful, False otherwise
    """
    # Validate video exists
    if not os.path.exists(video_path):
        print(f"Error: Video not found: {video_path}")
        return False
    
    # Set output directory
    video_dir = os.path.dirname(os.path.abspath(video_path))
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    if output_dir is None:
        output_dir = video_dir
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("HaWoR Hand Rendering - Camera Space Only")
    print("="*60)
    print(f"Input video: {video_path}")
    print(f"Output directory: {output_dir}")
    print(f"Keep intermediates: {keep_intermediates}")
    print("="*60)
    
    # Create args object for compatibility with existing functions
    class Args:
        def __init__(self):
            self.video_path = video_path
            self.input_type = 'file'
            self.checkpoint = checkpoint
            self.img_focal = img_focal
            self.fps = fps  # Add fps attribute
    
    args = Args()
    
    # Step 1: Detection and tracking
    print("\n[1/3] Hand detection and tracking...")
    start_idx, end_idx, seq_folder, imgfiles = detect_track_video(args)
    
    # Step 2: Motion estimation (saves to cam_space folder)
    print("\n[2/3] Hand motion estimation...")
    frame_chunks_all, img_focal = hawor_motion_estimation(
        args, start_idx, end_idx, seq_folder, 
        model=model, model_cfg=model_cfg  # Pass pre-loaded model if available
    )
    
    # Step 3: Load camera space predictions and render
    print("\n[3/3] Loading camera space predictions and rendering...")
    
    num_frames = len(imgfiles)
    pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = load_camera_space_predictions(
        seq_folder, frame_chunks_all, num_frames
    )
    
    hand2idx = {"right": 1, "left": 0}
    vis_start = 0
    vis_end = pred_trans.shape[1]
    
    # Get faces
    faces = get_mano_faces()
    faces_new = np.array([
        [92, 38, 234], [234, 38, 239], [38, 122, 239], [239, 122, 279],
        [122, 118, 279], [279, 118, 215], [118, 117, 215], [215, 117, 214],
        [117, 119, 214], [214, 119, 121], [119, 120, 121], [121, 120, 78],
        [120, 108, 78], [78, 108, 79]
    ])
    faces_right = np.concatenate([faces, faces_new], axis=0)
    faces_left = faces_right[:, [0, 2, 1]]
    
    # Get right hand vertices (camera space)
    hand_idx = hand2idx['right']
    pred_glob_r = run_mano(
        pred_trans[hand_idx:hand_idx+1, vis_start:vis_end],
        pred_rot[hand_idx:hand_idx+1, vis_start:vis_end],
        pred_hand_pose[hand_idx:hand_idx+1, vis_start:vis_end],
        betas=pred_betas[hand_idx:hand_idx+1, vis_start:vis_end]
    )
    right_verts = pred_glob_r['vertices'][0]
    
    # Get left hand vertices (camera space)
    hand_idx = hand2idx['left']
    pred_glob_l = run_mano_left(
        pred_trans[hand_idx:hand_idx+1, vis_start:vis_end],
        pred_rot[hand_idx:hand_idx+1, vis_start:vis_end],
        pred_hand_pose[hand_idx:hand_idx+1, vis_start:vis_end],
        betas=pred_betas[hand_idx:hand_idx+1, vis_start:vis_end]
    )
    left_verts = pred_glob_l['vertices'][0]
    
    # Note: vertices are already in camera space (OpenCV coordinate system)
    # No coordinate transformation needed for camera space - just convert to PyTorch3D coord system in renderer
    
    # Load background images
    num_render_frames = vis_end - vis_start
    print(f"Loading {num_render_frames} background images...")
    image_names = imgfiles[vis_start:vis_end]
    background_images = []
    for img_path in image_names:
        img = cv2.imread(img_path)
        if img is not None:
            background_images.append(img)
        else:
            img0 = cv2.imread(image_names[0]) if image_names else None
            background_images.append(np.zeros_like(img0) if img0 is not None else np.zeros((480, 640, 3), dtype=np.uint8))
    
    img0 = background_images[0]
    height, width = img0.shape[:2]
    
    # Get FPS from input video
    if fps is None:
        fps = 10
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        if input_fps > 0:
            fps = input_fps
        cap.release()
    print(f"Output FPS: {fps}")
    
    # Render hands
    print("Rendering hand meshes...")
    rendered_frames, mask_frames = render_hands_pytorch3d_cam_only(
        left_verts, right_verts, faces_left, faces_right, 
        img_focal, width, height,
        pred_valid=pred_valid[:, vis_start:vis_end]  # Pass validity mask
    )
    
    # Create overlay videos using mask (hand area only)
    print("Creating overlay videos...")
    overlay_frames = []
    for i, (bg_img, rendered_img, mask) in enumerate(zip(background_images, rendered_frames, mask_frames)):
        # Normalize mask to [0, 1]
        alpha = mask.astype(np.float32) / 255.0
        alpha = alpha[:, :, None]  # Add channel dimension
        
        # Blend mesh with background using mask (hand area only)
        overlay = (rendered_img * alpha + bg_img * (1 - alpha)).astype(np.uint8)
        overlay_frames.append(overlay)
    
    # Save videos
    video_name_base = os.path.splitext(os.path.basename(video_path))[0]
    output_overlay = os.path.join(output_dir, f"{video_name_base}_hand_overlay_cam_only.mp4")
    output_mesh = os.path.join(output_dir, f"{video_name_base}_hand_mesh_cam_only.mp4")
    output_mask = os.path.join(output_dir, f"{video_name_base}_hand_mask_cam_only.mp4")
    
    print(f"Saving overlay video: {output_overlay}")
    create_video_from_frames(overlay_frames, output_overlay, fps)
    
    print(f"Saving mesh video: {output_mesh}")
    create_video_from_frames(rendered_frames, output_mesh, fps)
    
    print(f"Saving mask video: {output_mask}")
    create_video_from_frames(mask_frames, output_mask, fps)
    
    # Clean up intermediate files if requested
    if not keep_intermediates:
        print("\nCleaning up intermediate files...")
        try:
            # Remove intermediate folders
            intermediate_dirs = [
                os.path.join(seq_folder, 'cam_space'),
                os.path.join(seq_folder, 'extracted_images'),
            ]
            
            # Remove tracks_* folders
            tracks_pattern = os.path.join(seq_folder, 'tracks_*')
            tracks_dirs = glob(tracks_pattern)
            intermediate_dirs.extend(tracks_dirs)
            
            # Remove est_focal.txt
            est_focal_file = os.path.join(seq_folder, 'est_focal.txt')
            if os.path.exists(est_focal_file):
                os.remove(est_focal_file)
            
            for dir_path in intermediate_dirs:
                if os.path.exists(dir_path):
                    shutil.rmtree(dir_path, ignore_errors=True)
                    print(f"  Removed: {dir_path}")
            
            # Remove seq_folder if it's empty (only if it's a subdirectory)
            if seq_folder != video_dir and os.path.exists(seq_folder):
                try:
                    if not os.listdir(seq_folder):  # Empty directory
                        os.rmdir(seq_folder)
                        print(f"  Removed empty directory: {seq_folder}")
                except OSError:
                    pass  # Directory not empty or other error
                    
        except Exception as e:
            print(f"  Warning: Error cleaning up intermediate files: {e}")
    
    print("\n" + "="*60)
    print("✅ Processing complete!")
    print(f"Output videos saved to: {output_dir}")
    print("="*60)
    
    return True


def main():
    parser = argparse.ArgumentParser(description='HaWoR Hand Rendering - Camera Space Only')
    parser.add_argument("--img_focal", type=float, default=None)
    parser.add_argument("--video_path", type=str, required=True, help='Input video path')
    parser.add_argument("--checkpoint", type=str, default='./weights/hawor/checkpoints/hawor.ckpt')
    parser.add_argument("--output_dir", type=str, default=None, help='Output directory (default: same as video)')
    parser.add_argument("--keep_intermediates", action='store_true', help='Keep intermediate files (tracks, images, etc.)')
    parser.add_argument("--fps", type=int, default=8, help='Frames per second for video extraction')
    args = parser.parse_args()
    
    success = process_video(
        args.video_path,
        args.checkpoint,
        args.output_dir,
        args.keep_intermediates,
        args.fps,
        args.img_focal
    )
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()

