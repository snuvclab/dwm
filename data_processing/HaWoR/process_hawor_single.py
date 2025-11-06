#!/usr/bin/env python3
"""
HaWoR Video Rendering Script
Renders hand meshes overlaid on video and saves two outputs:
1. Mesh overlaid on original video
2. Mesh only on black background

Cleans up all intermediate files after processing.
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
from glob import glob
from natsort import natsorted

from scripts.scripts_test_video.detect_track_video import detect_track_video
from scripts.scripts_test_video.hawor_video import hawor_motion_estimation, hawor_infiller
from scripts.scripts_test_video.hawor_slam import hawor_slam
from hawor.utils.process import get_mano_faces, run_mano, run_mano_left
from lib.eval_utils.custom_utils import load_slam_cam

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


def render_hands_pytorch3d_cam(left_verts, right_verts, left_faces, right_faces, 
                                focal_length, img_width, img_height, 
                                R_w2c, t_w2c):
    """
    Render hands using PyTorch3D in camera space.
    
    Args:
        left_verts: (T, V, 3) left hand vertices in world space
        right_verts: (T, V, 3) right hand vertices in world space
        left_faces: (F, 3) left hand faces
        right_faces: (F, 3) right hand faces
        focal_length: camera focal length
        img_width, img_height: image dimensions
        R_w2c: (T, 3, 3) world to camera rotation
        t_w2c: (T, 3) world to camera translation
    
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
    
    # Debug: print shapes
    if num_frames > 0:
        print(f"Debug - Left verts shape: {left_verts.shape}, Right verts shape: {right_verts.shape}")
        print(f"Debug - Left faces shape: {left_faces_t.shape}, Right faces shape: {right_faces_t.shape}")
    
    # Hand colors (from blender_ego_hand.py)
    left_color = (0.5, 0.8, 0.5)   # Vibrant green
    right_color = (0.8, 0.4, 0.4)  # Deep red
    
    # Coordinate conversion: OpenCV (X-right, Y-down, Z-out) -> PyTorch3D (X-left, Y-up, Z-out)
    coord_convert = np.diag([-1.0, -1.0, 1.0]).astype(np.float32)
    
    for frame_idx in range(num_frames):
        if frame_idx % 10 == 0 or frame_idx == num_frames - 1:
            print(f"  Rendering frame {frame_idx + 1}/{num_frames} ({(frame_idx + 1) / num_frames * 100:.1f}%)...")
        
        # Transform vertices to camera space
        R_cam = R_w2c[frame_idx].numpy()
        t_cam = t_w2c[frame_idx].numpy()
        
        # Convert to PyTorch3D coordinate system
        R_p3d = coord_convert @ R_cam
        t_p3d = coord_convert @ t_cam
        R_p3d_t = torch.from_numpy(R_p3d).to(device).float()
        t_p3d_t = torch.from_numpy(t_p3d).to(device).float()
        
        # Transform vertices
        left_v_world = left_verts[frame_idx].numpy()
        left_v_world_t = torch.from_numpy(left_v_world).to(device).float()
        left_v_cam = (left_v_world_t @ R_p3d_t.t()) + t_p3d_t
        left_v_tensor = left_v_cam.unsqueeze(0)
        
        right_v_world = right_verts[frame_idx].numpy()
        right_v_world_t = torch.from_numpy(right_v_world).to(device).float()
        right_v_cam = (right_v_world_t @ R_p3d_t.t()) + t_p3d_t
        right_v_tensor = right_v_cam.unsqueeze(0)
        
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
        
        # Render left hand
        left_colors = torch.tensor(left_color, device=device).view(1, 1, 3).expand(1, left_v_tensor.shape[1], 3)
        # Ensure faces have batch dimension: (1, F, 3)
        left_faces_batch = left_faces_t.unsqueeze(0) if left_faces_t.ndim == 2 else left_faces_t
        left_mesh = Meshes(verts=left_v_tensor, faces=left_faces_batch, textures=TexturesVertex(left_colors))
        
        with torch.no_grad():
            left_images, left_frags = renderer(left_mesh)
        
        left_rgb = left_images[0, :, :, :3]
        left_zbuf = left_frags.zbuf[0, :, :, 0].float()
        left_valid = left_frags.pix_to_face[0, :, :, 0] >= 0
        
        closer = left_valid & (left_zbuf < final_z)
        closer3 = closer.unsqueeze(-1)
        final_rgb = torch.where(closer3, left_rgb, final_rgb)
        final_z = torch.where(closer, left_zbuf, final_z)
        
        # Render right hand
        right_colors = torch.tensor(right_color, device=device).view(1, 1, 3).expand(1, right_v_tensor.shape[1], 3)
        # Ensure faces have batch dimension: (1, F, 3)
        right_faces_batch = right_faces_t.unsqueeze(0) if right_faces_t.ndim == 2 else right_faces_t
        right_mesh = Meshes(verts=right_v_tensor, faces=right_faces_batch, textures=TexturesVertex(right_colors))
        
        with torch.no_grad():
            right_images, right_frags = renderer(right_mesh)
        
        right_rgb = right_images[0, :, :, :3]
        right_zbuf = right_frags.zbuf[0, :, :, 0].float()
        right_valid = right_frags.pix_to_face[0, :, :, 0] >= 0
        
        closer = right_valid & (right_zbuf < final_z)
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
            'ffmpeg', '-y',
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


def main():
    parser = argparse.ArgumentParser(description='HaWoR Hand Rendering - Video Only')
    parser.add_argument("--img_focal", type=float, default=None)
    parser.add_argument("--video_path", type=str, required=True, help='Input video path')
    parser.add_argument("--checkpoint", type=str, default='./weights/hawor/checkpoints/hawor.ckpt')
    parser.add_argument("--infiller_weight", type=str, default='./weights/hawor/checkpoints/infiller.pt')
    parser.add_argument("--output_dir", type=str, default=None, help='Output directory (default: same as video)')
    parser.add_argument("--keep_intermediates", action='store_true', help='Keep intermediate files (tracks, images, etc.)')
    args = parser.parse_args()
    
    # Validate video exists
    if not os.path.exists(args.video_path):
        print(f"Error: Video not found: {args.video_path}")
        sys.exit(1)
    
    # Set output directory
    video_dir = os.path.dirname(os.path.abspath(args.video_path))
    video_name = os.path.splitext(os.path.basename(args.video_path))[0]
    
    if args.output_dir is None:
        output_dir = video_dir
    else:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("HaWoR Hand Rendering - Video Only Mode")
    print("="*60)
    print(f"Input video: {args.video_path}")
    print(f"Output directory: {output_dir}")
    print(f"Keep intermediates: {args.keep_intermediates}")
    print("="*60)
    
    # Step 1: Detection and tracking
    print("\n[1/5] Hand detection and tracking...")
    args.input_type = 'file'
    start_idx, end_idx, seq_folder, imgfiles = detect_track_video(args)
    
    # Step 2: Motion estimation
    print("\n[2/5] Hand motion estimation...")
    frame_chunks_all, img_focal = hawor_motion_estimation(args, start_idx, end_idx, seq_folder)
    
    # Step 3: SLAM
    print("\n[3/5] Running SLAM...")
    slam_path = os.path.join(seq_folder, f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz")
    if not os.path.exists(slam_path):
        hawor_slam(args, start_idx, end_idx)
    R_w2c_sla_all, t_w2c_sla_all, R_c2w_sla_all, t_c2w_sla_all = load_slam_cam(slam_path)
    
    # Step 4: Infilling
    print("\n[4/5] Running infiller...")
    pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = hawor_infiller(
        args, start_idx, end_idx, frame_chunks_all
    )
    
    # Step 5: Rendering
    print("\n[5/5] Rendering hand meshes...")
    
    hand2idx = {"right": 1, "left": 0}
    vis_start = 0
    vis_end = pred_trans.shape[1] - 1
    
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
    
    # Get right hand vertices
    hand_idx = hand2idx['right']
    pred_glob_r = run_mano(
        pred_trans[hand_idx:hand_idx+1, vis_start:vis_end],
        pred_rot[hand_idx:hand_idx+1, vis_start:vis_end],
        pred_hand_pose[hand_idx:hand_idx+1, vis_start:vis_end],
        betas=pred_betas[hand_idx:hand_idx+1, vis_start:vis_end]
    )
    right_verts = pred_glob_r['vertices'][0]
    
    # Get left hand vertices
    hand_idx = hand2idx['left']
    pred_glob_l = run_mano_left(
        pred_trans[hand_idx:hand_idx+1, vis_start:vis_end],
        pred_rot[hand_idx:hand_idx+1, vis_start:vis_end],
        pred_hand_pose[hand_idx:hand_idx+1, vis_start:vis_end],
        betas=pred_betas[hand_idx:hand_idx+1, vis_start:vis_end]
    )
    left_verts = pred_glob_l['vertices'][0]
    
    # Apply coordinate transformation
    R_x = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).float()
    R_c2w_sla_all = torch.einsum('ij,njk->nik', R_x, R_c2w_sla_all)
    t_c2w_sla_all = torch.einsum('ij,nj->ni', R_x, t_c2w_sla_all)
    R_w2c_sla_all = R_c2w_sla_all.transpose(-1, -2)
    t_w2c_sla_all = -torch.einsum("bij,bj->bi", R_w2c_sla_all, t_c2w_sla_all)
    # Note: left_verts and right_verts are (T, V, 3), not (B, T, V, 3)
    left_verts = torch.einsum('ij,tnj->tni', R_x, left_verts.cpu())
    right_verts = torch.einsum('ij,tnj->tni', R_x, right_verts.cpu())
    
    # Load background images
    print(f"Loading {len(imgfiles[vis_start:vis_end])} background images...")
    image_names = imgfiles[vis_start:vis_end]
    background_images = []
    for img_path in image_names:
        img = cv2.imread(img_path)
        if img is not None:
            background_images.append(img)
        else:
            img0 = cv2.imread(image_names[0])
            background_images.append(np.zeros_like(img0))
    
    img0 = background_images[0]
    height, width = img0.shape[:2]
    
    # Get FPS from input video
    fps = 10
    cap = cv2.VideoCapture(args.video_path)
    if cap.isOpened():
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        if input_fps > 0:
            fps = input_fps
        cap.release()
    print(f"Output FPS: {fps}")
    
    # Extract vertices (remove batch dimension if present)
    # left_verts and right_verts should be (T, V, 3)
    if left_verts.ndim == 4:  # (B, T, V, 3)
        left_verts_render = left_verts[0]
        right_verts_render = right_verts[0]
    else:  # Already (T, V, 3)
        left_verts_render = left_verts
        right_verts_render = right_verts
    
    print(f"Vertices shape - Left: {left_verts_render.shape}, Right: {right_verts_render.shape}")
    print(f"Faces shape - Left: {faces_left.shape}, Right: {faces_right.shape}")
    
    # Render once to get mesh and mask
    print("\nRendering hand meshes (one pass)...")
    mesh_frames, mask_frames = render_hands_pytorch3d_cam(
        left_verts_render, right_verts_render, faces_left, faces_right,
        img_focal, width, height,
        R_w2c_sla_all[vis_start:vis_end],
        t_w2c_sla_all[vis_start:vis_end]
    )
    
    # Generate overlay frames using mask
    print("\nGenerating overlay frames...")
    overlay_frames = []
    for i, (mesh_frame, mask, bg_img) in enumerate(zip(mesh_frames, mask_frames, background_images)):
        # Normalize mask to [0, 1]
        alpha = mask.astype(np.float32) / 255.0
        alpha = alpha[:, :, None]  # Add channel dimension
        
        # Blend mesh with background using mask
        overlay = (mesh_frame * alpha + bg_img * (1 - alpha)).astype(np.uint8)
        overlay_frames.append(overlay)
    
    # Save three videos
    overlay_video_path = os.path.join(output_dir, f'{video_name}_hand_overlay.mp4')
    mesh_video_path = os.path.join(output_dir, f'{video_name}_hand_mesh.mp4')
    mask_video_path = os.path.join(output_dir, f'{video_name}_hand_mask.mp4')
    
    print(f"\nSaving overlay video: {overlay_video_path}")
    if create_video_from_frames(overlay_frames, overlay_video_path, fps):
        print(f"✓ Overlay video saved: {overlay_video_path}")
    else:
        print(f"✗ Failed to save overlay video")
    
    print(f"\nSaving mesh-only video: {mesh_video_path}")
    if create_video_from_frames(mesh_frames, mesh_video_path, fps):
        print(f"✓ Mesh video saved: {mesh_video_path}")
    else:
        print(f"✗ Failed to save mesh video")
    
    print(f"\nSaving mask video: {mask_video_path}")
    # Convert single channel mask to 3-channel for video
    mask_frames_3ch = [cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) for mask in mask_frames]
    if create_video_from_frames(mask_frames_3ch, mask_video_path, fps):
        print(f"✓ Mask video saved: {mask_video_path}")
    else:
        print(f"✗ Failed to save mask video")
    
    # Clean up intermediate files
    if not args.keep_intermediates:
        print("\nCleaning up intermediate files...")
        cleanup_folders = [
            os.path.join(seq_folder, 'extracted_images'),
            os.path.join(seq_folder, f'tracks_{start_idx}_{end_idx}'),
            os.path.join(seq_folder, 'cam_space'),
            os.path.join(seq_folder, 'SLAM'),
            os.path.join(seq_folder, f'vis_{vis_start}_{vis_end}'),
        ]
        
        for folder in cleanup_folders:
            if os.path.exists(folder):
                try:
                    shutil.rmtree(folder)
                    print(f"  ✓ Removed: {folder}")
                except Exception as e:
                    print(f"  ✗ Failed to remove {folder}: {e}")
        
        # Remove seq_folder if empty
        try:
            if os.path.exists(seq_folder) and not os.listdir(seq_folder):
                os.rmdir(seq_folder)
                print(f"  ✓ Removed empty folder: {seq_folder}")
        except Exception as e:
            print(f"  ✗ Failed to remove {seq_folder}: {e}")
    
    print("\n" + "="*60)
    print("COMPLETE!")
    print(f"Output videos:")
    print(f"  1. Overlay: {overlay_video_path}")
    print(f"  2. Mesh only: {mesh_video_path}")
    print(f"  3. Mask: {mask_video_path}")
    print("="*60)


if __name__ == '__main__':
    main()

