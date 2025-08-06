"""
Visualize camera poses and SMPL/SMPLX human meshes in 3D space using viser.

This script supports both ARIA (egoallo) and Trumans (SMPLX) datasets.
For Trumans datasets, it applies coordinate transformation from Blender to viser coordinates.
"""
import random
import time
import numpy as np
import os, sys
import cv2
from tqdm import tqdm
import pickle
import math
import argparse
from pathlib import Path
import imageio.v3 as iio

import viser
import viser.transforms as tf

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from training.aether.utils.postprocess_utils import postprocess_pointmap
from training.aether.utils.postprocess_utils import camera_pose_to_raymap

import trimesh
import torch

# Coordinate transformation matrix for Blender to viser conversion (absolute case)
# 90° rotation around X-axis: Y->Z, Z->-Y
BLENDER_TO_VISER_TRANSFORM = np.array([
    [1,  0,  0,  0],
    [0,  0, -1,  0],
    [0,  1,  0,  0],
    [0,  0,  0,  1]
])

# Coordinate transformation matrix for SMPL to camera coordinate system (relative case)
# SMPL: +Y up, +Z front
# Camera: -Y up, +X right, +Z front
# Transform: Rotate 180° around Z-axis to flip Y direction
SMPL_TO_CAMERA_TRANSFORM = np.array([
    [-1,  0,  0,  0],
    [ 0, -1,  0,  0],
    [ 0,  0,  1,  0],
    [ 0,  0,  0,  1]
])

def quat_inv(q):
    q_inv = q.clone()
    q_inv[..., 1:] *= -1  # invert vector part
    return q_inv

def quat_mul_wxyz(q1, q2):
    # Quaternion multiplication: (w, x, y, z)
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return torch.stack([w, x, y, z], dim=-1)

def quat_rotate_wxyz(q, v):
    # Rotate 3D vector v by quaternion q (w, x, y, z)
    q_conj = q.clone()
    q_conj[..., 1:] *= -1
    v_q = torch.cat([torch.zeros_like(v[..., :1]), v], dim=-1)
    return quat_mul_wxyz(quat_mul_wxyz(q, v_q), q_conj)[..., 1:]

def se3_inverse_qwxyz(pose):
    q, t = pose[..., :4], pose[..., 4:]
    q_inv = quat_inv(q)
    t_inv = -quat_rotate_wxyz(q_inv, t)
    return torch.cat([q_inv, t_inv], dim=-1)

def compose_se3_qwxyz(pose1, pose2):
    """
    Compose two poses in [qw, qx, qy, qz, tx, ty, tz] format
    """
    q1, t1 = pose1[..., :4], pose1[..., 4:]
    q2, t2 = pose2[..., :4], pose2[..., 4:]

    q_new = quat_mul_wxyz(q1, q2)
    t2_rot = quat_rotate_wxyz(q1, t2)
    t_new = t1 + t2_rot

    return torch.cat([q_new, t_new], dim=-1)

def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Converts axis-angle vectors to rotation matrices using Rodrigues' rotation formula.
    
    Args:
        axis_angle: Tensor of shape (..., 3), where the last dimension is the axis-angle.

    Returns:
        Rotation matrices of shape (..., 3, 3)
    """
    angle = torch.norm(axis_angle, dim=-1, keepdim=True) + 1e-8
    axis = axis_angle / angle

    x, y, z = axis.unbind(-1)
    zeros = torch.zeros_like(x)
    K = torch.stack([
        zeros, -z, y,
        z, zeros, -x,
        -y, x, zeros
    ], dim=-1).reshape(*axis.shape[:-1], 3, 3)

    I = torch.eye(3, device=axis.device, dtype=axis.dtype).expand(*axis.shape[:-1], 3, 3)
    sin = torch.sin(angle)[..., None]
    cos = torch.cos(angle)[..., None]

    R = I + sin * K + (1 - cos) * K @ K
    return R

def create_zero_pose_smplx_mesh(smpl_model, device, dataset_type, server):
    """
    Create a zero pose SMPLX mesh for debugging purposes.
    
    Args:
        smpl_model: SMPLX model instance
        device: PyTorch device
        dataset_type: Dataset type ("aria" or "trumans")
        server: Viser server instance
    
    Returns:
        None (adds mesh to server scene)
    """
    print("🔄 Creating zero pose SMPLX mesh for debugging...")
    
    # Create zero pose parameters
    zero_betas = torch.zeros(1, 10, device=device, dtype=torch.float32)
    zero_global_orient = torch.zeros(1, 3, device=device, dtype=torch.float32)
    zero_body_pose = torch.zeros(1, 63, device=device, dtype=torch.float32)  # SMPLX: 21 joints * 3
    zero_left_hand_pose = torch.zeros(1, 45, device=device, dtype=torch.float32)  # 15 joints * 3
    zero_right_hand_pose = torch.zeros(1, 45, device=device, dtype=torch.float32)  # 15 joints * 3
    zero_jaw_pose = torch.zeros(1, 3, device=device, dtype=torch.float32)
    zero_leye_pose = torch.zeros(1, 3, device=device, dtype=torch.float32)
    zero_reye_pose = torch.zeros(1, 3, device=device, dtype=torch.float32)
    zero_transl = torch.zeros(1, 3, device=device, dtype=torch.float32)
    
    # Create SMPLX output with zero pose
    zero_smplx_output = smpl_model(
        betas=zero_betas,
        global_orient=zero_global_orient,
        body_pose=zero_body_pose,
        left_hand_pose=zero_left_hand_pose,
        right_hand_pose=zero_right_hand_pose,
        jaw_pose=zero_jaw_pose,
        leye_pose=zero_leye_pose,
        reye_pose=zero_reye_pose,
        transl=zero_transl,
        return_verts=True,
        return_full_pose=True,
        use_pca=False
    )
    
    # Get vertices and faces
    zero_vertices = zero_smplx_output.vertices.detach().cpu().numpy().squeeze()
    zero_vertex_colors = np.array([255, 0, 0])  # Red color for zero pose
    
    # Get faces from the model
    if hasattr(smpl_model, 'faces'):
        if hasattr(smpl_model.faces, 'cpu'):
            zero_faces = smpl_model.faces.cpu().numpy()
        else:
            zero_faces = smpl_model.faces
    else:
        if hasattr(smpl_model.faces_tensor, 'cpu'):
            zero_faces = smpl_model.faces_tensor.cpu().numpy()
        else:
            zero_faces = smpl_model.faces_tensor
    
    # Apply coordinate transformation for Trumans dataset
    if dataset_type == "trumans":
        print("🔄 Applying coordinate transformation to zero pose SMPLX mesh...")
        
        # Apply coordinate transformation directly to vertices (avoid global_orient translation issue)
        vertices_homogeneous = np.concatenate([zero_vertices, np.ones((zero_vertices.shape[0], 1))], axis=1)
        vertices_transformed_homogeneous = (BLENDER_TO_VISER_TRANSFORM @ vertices_homogeneous.T).T
        zero_vertices_transformed = vertices_transformed_homogeneous[:, :3]
        
        zero_mesh_out = trimesh.Trimesh(
            vertices=zero_vertices_transformed,
            faces=zero_faces,
            vertex_colors=np.repeat(zero_vertex_colors[None, :], zero_vertices_transformed.shape[0], axis=0)
        )
        print("✅ Applied coordinate transformation to zero pose mesh")
    else:
        # For ARIA: Use original zero pose
        zero_mesh_out = trimesh.Trimesh(
            vertices=zero_vertices,
            faces=zero_faces,
            vertex_colors=np.repeat(zero_vertex_colors[None, :], zero_vertices.shape[0], axis=0)
        )
    
    server.scene.add_mesh_trimesh(
        "/zero_pose_smplx",
        mesh=zero_mesh_out,
    )
    print("✅ Added zero pose SMPLX mesh (red) for debugging")

def diagonal_fov(fov_x_deg, fov_y_deg):
    fov_x_rad = math.radians(fov_x_deg)
    fov_y_rad = math.radians(fov_y_deg)

    tan_x = math.tan(fov_x_rad / 2)
    tan_y = math.tan(fov_y_rad / 2)

    diag_fov_rad = 2 * math.atan(math.sqrt(tan_x**2 + tan_y**2))
    return math.degrees(diag_fov_rad)

def load_trimesh_obj(path: str):
    mesh = trimesh.load(path, force='mesh', process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()
    return mesh
                        
def main():

    parser = argparse.ArgumentParser(description="Visualize camera poses and point clouds.")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory containing data.")
    parser.add_argument("--name", type=str, required=True, help="Name of the sequence.")
    parser.add_argument("--dataset_type", type=str, choices=["aria", "trumans"], default=None,
                       help="Dataset type: 'aria' for egoallo data, 'trumans' for SMPLX data. Auto-detected if not specified.")
    parser.add_argument("--camera_type", type=str, choices=["relative", "absolute"], default="relative",
                       help="Camera trajectory type: 'relative' for relative poses, 'absolute' for absolute poses")
    args = parser.parse_args()

    data_root = Path(args.data_root)

    # Auto-detect dataset type if not specified
    egoallo_human_pose_file = data_root / "egoallo.npz"
    if args.dataset_type is None:
        if egoallo_human_pose_file.exists():
            args.dataset_type = "aria"
            print(f"🔍 Auto-detected dataset type: ARIA (found egoallo.npz)")
        else:
            args.dataset_type = "trumans"
            print(f"🔍 Auto-detected dataset type: Trumans (SMPLX processing enabled)")
    
    print(f"📊 Dataset type: {args.dataset_type.upper()}")

    # Import egoallo only for ARIA datasets
    if args.dataset_type == "aria":
        try:
            import projectaria_tools.core.mps as mps
            from projectaria_tools.core.mps.utils import filter_points_from_confidence
            print("📁 Imported projectaria_tools for ARIA dataset")
            from egoallo import fncsmpl, fncsmpl_extensions
            print("📁 Imported egoallo for ARIA dataset")
        except ImportError:
            print("❌ Error: egoallo library not found. Please install it for ARIA datasets.")
            return

    server = viser.ViserServer()
    num_frames = 49

    # Add some coordinate frames to the scene. These will be visualized in the viewer.
    server.scene.add_frame(
        "/tree",
        wxyz=(1.0, 0.0, 0.0, 0.0),
        position=(0, 0, 0),
    )

    # # Add oven.obj mesh to the scene
    # oven_mesh = load_trimesh_obj("./oven.obj")
    # server.scene.add_mesh_trimesh(
    #     "/oven",
    #     mesh=oven_mesh,
    # )
    # smpl_mesh = load_trimesh_obj("./smplx.obj")
    # server.scene.add_mesh_trimesh(
    #     "/smpl",
    #     mesh=smpl_mesh,
    # )
    
    # Initial camera pose.
    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client.camera.position = (0, -2.0, -2)
        client.camera.look_at = (0.0, -1.5, 0.0)

    # Add playback UI.
    with server.gui.add_folder("Playback"):
        
        gui_timestep = server.gui.add_slider(
            "Timestep",
            min=0,
            max=num_frames - 1,
            step=1,
            initial_value=0,
            disabled=True,
        )
        gui_next_frame = server.gui.add_button("Next Frame", disabled=True)
        gui_prev_frame = server.gui.add_button("Prev Frame", disabled=True)
        gui_playing = server.gui.add_checkbox("Playing", False)
        gui_framerate = server.gui.add_slider(
            "FPS", min=1, max=60, step=0.1, initial_value=8
        )
        gui_framerate_options = server.gui.add_button_group(
            "FPS options", ("8", "20", "30", "60")
        )

    # Frame step buttons.
    @gui_next_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value + 1) % num_frames

    @gui_prev_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value - 1) % num_frames

    # Disable frame controls when we're playing.
    @gui_playing.on_update
    def _(_) -> None:
        gui_timestep.disabled = gui_playing.value
        gui_next_frame.disabled = gui_playing.value
        gui_prev_frame.disabled = gui_playing.value

    # Set the framerate when we click one of the options.
    @gui_framerate_options.on_click
    def _(_) -> None:
        gui_framerate.value = int(gui_framerate_options.value)

    prev_timestep = gui_timestep.value

    # Toggle frame visibility when the timestep slider changes.
    @gui_timestep.on_update
    def _(_) -> None:
        nonlocal prev_timestep
        current_timestep = gui_timestep.value
        with server.atomic():
            # Toggle visibility.
            frame_nodes[current_timestep].visible = True
            frame_nodes[prev_timestep].visible = False
        prev_timestep = current_timestep
        server.flush()  # Optional!

    server.scene.add_frame(
        "/frames",
        wxyz=(1.0, 0.0, 0.0, 0.0),
        position=(0, 0, 0),
        show_axes=False,
    )
       
    
    frame_nodes: list[viser.FrameHandle] = []
    point_nodes: list[viser.PointCloudHandle] = []
    smpl_nodes: list[viser.FrameHandle] = []
    
    # GT camera poses
    trajectory_suffix = "_abs" if args.camera_type == "absolute" else ""
    with open(data_root / "trajectory" / f"{args.name}{trajectory_suffix}.npy", "rb") as f:
        gt_trajectory = np.load(f)
    gt_intrinsics = np.load(data_root.parent / "cam_params" / "intrinsics.npy")
    gt_frames = iio.imread(data_root / "videos" / f"{args.name}.mp4")

    disparities = np.load(data_root / "disparity" / f"{args.name}.npz")["disparity"]

    raymaps = np.load(data_root / "raymaps" / f"{args.name}{trajectory_suffix}.npz")["raymap"]
    human_motions = np.load(data_root / "human_motions" / f"{args.name}.npz")

    # Calculate point maps using disparities and raymaps
    point_maps = postprocess_pointmap(
        disparity= disparities,
        raymap=raymaps.copy(),
        ray_o_scale_inv=0.1,
    )["pointmap"]

    # Create the SMPL model based on dataset type
    device = "cuda"
    if args.dataset_type == "aria":
        # Use SMPLH model for ARIA datasets (egoallo)
        model_path = '/media/taeksoo/SSD1/github1/egoallo/data/smplh/neutral/model.npz'
        smpl_model = fncsmpl.SmplhModel.load(model_path).to(device)
        print(f"📁 Loaded SMPLH model for ARIA dataset")
    elif args.dataset_type == "trumans":
        # Use SMPLX model for Trumans datasets
        try:
            import smplx
            smpl_model = smplx.create('smpl_models/smplx/SMPLX_MALE.npz', model_type='smplx', gender='male', use_pca=False).to(device)
            print(f"📁 Loaded SMPLX model for Trumans dataset")
        except ImportError:
            raise ImportError("⚠️  Warning: smplx library not found. Please install it with: pip install smplx")
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")


    # unify coordinates between camera and human motions
    # axis_tf = torch.tensor([0, 0, 0, 1, 0, 0, 0], dtype=torch.float32)
    # human_motions_axis = {}
    # global_orient_axis = []
    # transl_axis = []
    # for global_orient, trans in zip(human_motions['global_orient_quat'], human_motions['transl']):
    #     T_world_root = np.concatenate([global_orient, trans])
    #     T_world_root = compose_se3_qwxyz(axis_tf, torch.tensor(T_world_root))
    #     global_orient_axis.append(T_world_root[:4])
    #     transl_axis.append(T_world_root[4:])

    # # Create zero pose SMPLX mesh for debugging
    # create_zero_pose_smplx_mesh(smpl_model, device, args.dataset_type, server)


    for i in tqdm(range(num_frames)):

        frame = gt_frames[i]

        frame_nodes.append(server.scene.add_frame(f"/frames/t{i}", show_axes=False))

        # Add the point cloud.
        point_nodes.append(
            server.scene.add_point_cloud(
                name=f"/frames/t{i}/point_cloud",
                points=point_maps[i].reshape(-1, 3),
                colors=frame.reshape(-1, 3),
                point_size=0.005,
                point_shape="rounded",
            )
        )

        # Add camera poses
        fov = 2 * np.arctan2(frame.shape[0] / 2, gt_intrinsics[0, 0])
        aspect = frame.shape[1] / frame.shape[0]
        server.scene.add_camera_frustum(
            f"/frames/t{i}/frustum_gt",
            fov=fov,
            aspect=aspect,
            scale=0.15,
            image=frame[::4, ::4],
            wxyz=tf.SO3.from_matrix(gt_trajectory[i][:3, :3]).wxyz,
            position=gt_trajectory[i][:3, 3],
        )
        # Add some axes.
        server.scene.add_frame(
            f"/frames/t{i}/frustum_gt/axes",
            axes_length=0.3,
            axes_radius=0.01,
            origin_color=(0, 0, 0),  # black for ground truth
        )


        # Add SMPL mesh based on dataset type
        if args.dataset_type == "aria":
            # Process egoallo data for ARIA datasets
            smpl_shape = smpl_model.with_shape(betas=torch.tensor(human_motions['betas'][i, :], device=device))

            local_quats = torch.cat([
                torch.tensor(human_motions['body_pose_quat'][i, :], device=device),
                torch.tensor(human_motions['left_hand_pose_quat'][i, :], device=device),
                torch.tensor(human_motions['right_hand_pose_quat'][i, :], device=device),
            ], dim=0)

            # Local visualization for ARIA
            smpl_pose = smpl_shape.with_pose(T_world_root=torch.tensor([1, 0, 0, 0, 0, 0, 0], device=device, dtype=torch.float32),
                                            local_quats=local_quats)
            cpf_pos = torch.tensor((tf.SE3.from_matrix((gt_trajectory[i]))).wxyz_xyz, dtype=torch.float32, device=device)
            cpf_pos = compose_se3_qwxyz(cpf_pos, torch.tensor([0, 0, 0, 1, 0, 0, 0], device=device, dtype=torch.float32))
            T_world_root = fncsmpl_extensions.get_T_world_root_from_cpf_pose(smpl_pose, cpf_pos)
            smpl_pose = smpl_shape.with_pose(T_world_root=T_world_root,
                                            local_quats=local_quats)

            mesh = smpl_pose.lbs()
            
        elif args.dataset_type == "trumans":
            # Process SMPLX data for Trumans datasets
            import smplx
            
            # Extract SMPLX parameters
            if 'betas' in human_motions and human_motions['betas'].size > 0:
                betas = torch.tensor(human_motions['betas'][i:i+1, :], device=device, dtype=torch.float32)
            else:
                # Initialize betas as zeros if empty or missing
                betas = torch.zeros(1, 10, device=device, dtype=torch.float32)  # SMPLX typically has 10 beta parameters
            
            global_orient = torch.tensor(human_motions['global_orient'][i:i+1, :], device=device, dtype=torch.float32) if 'global_orient' in human_motions else None
            body_pose = torch.tensor(human_motions['body_pose'][i:i+1, :], device=device, dtype=torch.float32) if 'body_pose' in human_motions else None
            left_hand_pose = torch.tensor(human_motions['left_hand_pose'][i:i+1, :], device=device, dtype=torch.float32) if 'left_hand_pose' in human_motions else None
            right_hand_pose = torch.tensor(human_motions['right_hand_pose'][i:i+1, :], device=device, dtype=torch.float32) if 'right_hand_pose' in human_motions else None
            jaw_pose = torch.tensor(human_motions['jaw_pose'][i:i+1, :], device=device, dtype=torch.float32) if 'jaw_pose' in human_motions else None
            leye_pose = torch.tensor(human_motions['leye_pose'][i:i+1, :], device=device, dtype=torch.float32) if 'leye_pose' in human_motions else None
            reye_pose = torch.tensor(human_motions['reye_pose'][i:i+1, :], device=device, dtype=torch.float32) if 'reye_pose' in human_motions else None
            transl = torch.tensor(human_motions['transl'][i:i+1, :], device=device, dtype=torch.float32) if 'transl' in human_motions else None
            
            # Handle different camera types for Trumans
            if args.camera_type == "relative":
                # For relative camera poses, use fncsmpl_extensions approach like ARIA
                # Create SMPLX output with local pose (no global transformation)
                smplx_output = smpl_model(
                    betas=betas,
                    global_orient=torch.zeros_like(global_orient) if global_orient is not None else None,  # Zero global orientation for local pose
                    body_pose=body_pose,
                    left_hand_pose=left_hand_pose,
                    right_hand_pose=right_hand_pose,
                    jaw_pose=jaw_pose,
                    leye_pose=leye_pose,
                    reye_pose=reye_pose,
                    transl=torch.zeros_like(transl) if transl is not None else None,  # Zero translation for local pose
                    return_verts=True,
                    return_full_pose=True,
                    use_pca=False
                )
                
                # Get local vertices and apply coordinate transformation
                vertices = smplx_output.vertices.detach().cpu().numpy().squeeze()
                
                # Apply coordinate transformation to match camera coordinate system
                vertices_homogeneous = np.concatenate([vertices, np.ones((vertices.shape[0], 1))], axis=1)
                vertices_transformed_homogeneous = (SMPL_TO_CAMERA_TRANSFORM @ vertices_homogeneous.T).T
                vertices_transformed = vertices_transformed_homogeneous[:, :3]
                
                # Align SMPLX mesh with camera pose using eye center and head orientation
                camera_pose = gt_trajectory[i]
                
                # Load vertex segmentation for eye indices
                import json
                with open("smplx_vert_segmentation.json") as f:
                    vert_seg = json.load(f)
                eye_indices = vert_seg["leftEye"] + vert_seg["rightEye"]
                head_joint_index = 15

                # Compute eye center and head orientation
                eye_center = vertices_transformed[eye_indices].mean(axis=0)
                head_rotmat = axis_angle_to_matrix(smplx_output.full_pose[0].view(-1, 3)[head_joint_index]).detach().cpu().numpy()
                
                # Align head forward direction with camera forward direction
                head_forward = head_rotmat @ np.array([0, 0, 1])
                camera_forward = camera_pose[:3, 2]
                head_forward /= np.linalg.norm(head_forward)
                camera_forward /= np.linalg.norm(camera_forward)

                # Compute rotation to align head with camera
                v = np.cross(head_forward, camera_forward)
                c = np.dot(head_forward, camera_forward)
                s = np.linalg.norm(v)

                if s < 1e-8:
                    R_align = np.eye(3) if c > 0 else -np.eye(3)
                else:
                    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                    R_align = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))

                # Build transformation from SMPLX to camera frame
                T_smplx_to_camera = np.eye(4)
                T_smplx_to_camera[:3, :3] = R_align.T
                T_smplx_to_camera[:3, 3] = -R_align.T @ eye_center

                # Apply transformations
                vertices_homogeneous = np.concatenate([vertices_transformed, np.ones((vertices_transformed.shape[0], 1))], axis=1)
                vertices_world = (camera_pose @ T_smplx_to_camera @ vertices_homogeneous.T).T[:, :3]
                mesh = torch.tensor(vertices_world, device=device, dtype=torch.float32).unsqueeze(0)
            else:  # absolute camera poses
                # For absolute camera poses, use the given global pose (original logic)
                smplx_output = smpl_model(
                    betas=betas,
                    global_orient=global_orient,
                    body_pose=body_pose,
                    left_hand_pose=left_hand_pose,
                    right_hand_pose=right_hand_pose,
                    jaw_pose=jaw_pose,
                    leye_pose=leye_pose,
                    reye_pose=reye_pose,
                    transl=transl,
                    return_verts=True,
                    return_full_pose=True,
                    use_pca=False
                )
                
                # Apply coordinate transformation for Trumans dataset
                vertices = smplx_output.vertices.detach().cpu().numpy().squeeze()
                
                # Apply transformation directly to vertices (avoid global_orient translation issue)
                vertices_homogeneous = np.concatenate([vertices, np.ones((vertices.shape[0], 1))], axis=1)
                vertices_transformed_homogeneous = (BLENDER_TO_VISER_TRANSFORM @ vertices_homogeneous.T).T
                vertices_transformed = vertices_transformed_homogeneous[:, :3]
                
                mesh = torch.tensor(vertices_transformed, device=device, dtype=torch.float32).unsqueeze(0)
        
        # Create mesh visualization
        vertex_colors = np.array([180, 248, 200])
        if hasattr(mesh, 'cpu'):
            vertices = mesh.detach().cpu().numpy()
        else:
            vertices = mesh.detach().cpu().numpy()
            
        # Get faces from the model
        if hasattr(smpl_model, 'faces'):
            if hasattr(smpl_model.faces, 'cpu'):
                # If it's a tensor, convert to numpy
                faces = smpl_model.faces.cpu().numpy()
            else:
                # If it's already a numpy array
                faces = smpl_model.faces
        else:
            # Fallback for SMPLX
            if hasattr(smpl_model.faces_tensor, 'cpu'):
                faces = smpl_model.faces_tensor.cpu().numpy()
            else:
                faces = smpl_model.faces_tensor
            
        mesh_out = trimesh.Trimesh(vertices=vertices.squeeze() * 1,
                                faces=faces,
                                vertex_colors=np.repeat(vertex_colors[None, :], vertices.shape[0], axis=0))
        smpl_nodes.append(
            server.scene.add_mesh_trimesh(
                f"/frames/t{i}/smpl",
                mesh=mesh_out,
            )
        )

        pass

    

    # Hide all but the current frame.
    for i, frame_node in enumerate(frame_nodes):
        frame_node.visible = i == gui_timestep.value
    # Hide all point clouds except the current one.
    # for i, point_node in enumerate(point_nodes):
    #     point_node.visible = i == gui_timestep.value


    # Playback update loop.
    prev_timestep = gui_timestep.value
    while True:
        # Update the timestep if we're playing.
        if gui_playing.value:
            gui_timestep.value = (gui_timestep.value + 1) % num_frames

        # Update point size of both this timestep and the next one! There's
        # redundancy here, but this will be optimized out internally by viser.
        #
        # We update the point size for the next timestep so that it will be
        # immediately available when we toggle the visibility.
        # point_nodes[gui_timestep.value].point_size = gui_point_size.value
        # point_nodes[
        #     (gui_timestep.value + 1) % num_frames
        # ].point_size = gui_point_size.value

        time.sleep(1.0 / gui_framerate.value)

if __name__ == "__main__":
    main()
