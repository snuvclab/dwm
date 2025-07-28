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
import numpy as np


import viser
import viser.transforms as tf

import projectaria_tools.core.mps as mps
from projectaria_tools.core.mps.utils import filter_points_from_confidence

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from training.aether.utils.postprocess_utils import postprocess_pointmap

from egoallo import fncsmpl, fncsmpl_extensions
import trimesh
import torch

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


def diagonal_fov(fov_x_deg, fov_y_deg):
    fov_x_rad = math.radians(fov_x_deg)
    fov_y_rad = math.radians(fov_y_deg)

    tan_x = math.tan(fov_x_rad / 2)
    tan_y = math.tan(fov_y_rad / 2)

    diag_fov_rad = 2 * math.atan(math.sqrt(tan_x**2 + tan_y**2))
    return math.degrees(diag_fov_rad)

                        
def main():

    parser = argparse.ArgumentParser(description="Visualize camera poses and point clouds.")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory containing data.")
    parser.add_argument("--name", type=str, required=True, help="Name of the sequence.")
    args = parser.parse_args()

    data_root = Path(args.data_root)

    server = viser.ViserServer()
    num_frames = 49

    # Add some coordinate frames to the scene. These will be visualized in the viewer.
    server.scene.add_frame(
        "/tree",
        wxyz=(1.0, 0.0, 0.0, 0.0),
        position=(0, 0, 0),
    )


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
    with open(data_root / "trajectory" / f"{args.name}.npy", "rb") as f:
        gt_trajectory = np.load(f)
    gt_intrinsics = np.load(data_root.parent / "cam_params" / "intrinsics.npy")
    gt_frames = iio.imread(data_root / "videos" / f"{args.name}.mp4")

    disparities = np.load(data_root / "disparity" / f"{args.name}.npz")["disparity"]
    raymaps = np.load(data_root / "raymaps" / f"{args.name}.npz")["raymap"]
    human_motions = np.load(data_root / "human_motions" / f"{args.name}.npz")

    # Calculate point maps using disparities and raymaps
    point_maps = postprocess_pointmap(
        disparity= disparities,
        raymap=raymaps,
        ray_o_scale_inv=0.1,
    )["pointmap"]

    # Create the SMPLH model
    model_path = '/media/taeksoo/SSD1/github1/egoallo/data/smplh/neutral/model.npz'
    device = "cuda"
    smplh_model = fncsmpl.SmplhModel.load(model_path).to(device)


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

        # Add SMPLH mesh
        smplh_shape = smplh_model.with_shape(betas=torch.tensor(human_motions['betas'][i, :], device=device))

        local_quats = torch.cat([
            torch.tensor(human_motions['body_pose_quat'][i, :], device=device),
            torch.tensor(human_motions['left_hand_pose_quat'][i, :], device=device),
            torch.tensor(human_motions['right_hand_pose_quat'][i, :], device=device),
        ], dim=0)

        # first_frame_Ts_world_root = torch.cat([
        #     torch.tensor(global_orient_axis[0], device=device),
        #     torch.tensor(transl_axis[0], device=device),
        # ])
        # first_frame_inv = se3_inverse_qwxyz(first_frame_Ts_world_root)

        # Ts_world_root = torch.cat([
        #     torch.tensor(global_orient_axis[i], device=device),
        #     torch.tensor(transl_axis[i], device=device),
        # ])
        # Ts_world_root = compose_se3_qwxyz(first_frame_inv, Ts_world_root)
        # smplh_pose = smplh_shape.with_pose(T_world_root=Ts_world_root,
        #                                 local_quats=local_quats)


        # # Global visualization
        # Ts_world_root = torch.cat([
        #     torch.tensor(human_motions['global_orient_quat'][i, :], device=device),
        #     torch.tensor(human_motions['transl'][i, :], device=device),
        # ])
        # smplh_pose = smplh_shape.with_pose(T_world_root=Ts_world_root,
        #                                 local_quats=local_quats)
        

        # Local visualization
        smplh_pose = smplh_shape.with_pose(T_world_root=torch.tensor([1, 0, 0, 0, 0, 0, 0], device=device, dtype=torch.float32),
                                        local_quats=local_quats)
        cpf_pos = torch.tensor((tf.SE3.from_matrix((gt_trajectory[i]))).wxyz_xyz, dtype=torch.float32, device=device)
        cpf_pos = compose_se3_qwxyz(cpf_pos, torch.tensor([0, 0, 0, 1, 0, 0, 0], device=device, dtype=torch.float32))
        T_world_root = fncsmpl_extensions.get_T_world_root_from_cpf_pose(smplh_pose, cpf_pos)
        smplh_pose = smplh_shape.with_pose(T_world_root=T_world_root,
                                        local_quats=local_quats)
        

        mesh = smplh_pose.lbs()
        vertex_colors = np.array([180, 248, 200])
        mesh_out = trimesh.Trimesh(vertices=mesh.verts.cpu().numpy() * 1,
                                faces=mesh.faces.cpu().numpy(),
                                vertex_colors=np.repeat(vertex_colors[None, :], mesh.verts.shape[0], axis=0))
        smpl_nodes.append(
            server.scene.add_mesh_trimesh(
                f"/frames/t{i}/smplh",
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
