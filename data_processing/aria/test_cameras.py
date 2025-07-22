import random
import time
import numpy as np
import os, sys
import cv2
from tqdm import tqdm

import viser
import viser.transforms as tf

import projectaria_tools.core.mps as mps
from projectaria_tools.core.mps.utils import filter_points_from_confidence

from egoallo import fncsmpl, fncsmpl_extensions
import trimesh
import torch
                        
def main():
    global_points_path = "/media/taeksoo/HDD3/aria/lab_01/mps_lab_01_vrs/slam/semidense_points.csv.gz"
    data_path = "/media/taeksoo/HDD3/aria/lab_01/lab_01_data"
    num_frames = 300
    points = mps.read_global_point_cloud(global_points_path)

    # filter the point cloud using thresholds on the inverse depth and distance standard deviation
    inverse_distance_std_threshold = 0.001
    distance_std_threshold = 0.15

    filtered_points = filter_points_from_confidence(points, inverse_distance_std_threshold, distance_std_threshold)

    # example: get position of this point in the world coordinate frame
    pcd = []
    for point in filtered_points:
        pcd.append(point.position_world)
    pcd = np.array(pcd)

    

    server = viser.ViserServer()


    # Add some coordinate frames to the scene. These will be visualized in the viewer.
    server.scene.add_frame(
        "/tree",
        wxyz=(1.0, 0.0, 0.0, 0.0),
        position=(0, 0, 0),
    )

    # Create colors based on height (z-coordinate).
    z_min, z_max = pcd[:, 2].min(), pcd[:, 2].max()
    normalized_z = (pcd[:, 2] - z_min) / (z_max - z_min)

    # Color gradient from blue (bottom) to red (top).
    colors = np.zeros((len(pcd), 3), dtype=np.uint8)
    colors[:, 0] = (normalized_z * 255).astype(np.uint8)  # Red channel.
    colors[:, 2] = ((1 - normalized_z) * 255).astype(np.uint8)  # Blue channel.

    # Add the point cloud to the scene.
    server.scene.add_point_cloud(
        name="/tree/pcd",
        points=pcd,
        colors=colors,
        point_size=0.002,
    )

    # Initial camera pose.
    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client.camera.position = (-1.554, -1.013, 1.142)
        client.camera.look_at = (-0.005, 2.283, -0.156)

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
            "FPS", min=1, max=60, step=0.1, initial_value=10
        )
        gui_framerate_options = server.gui.add_button_group(
            "FPS options", ("10", "20", "30", "60")
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
    smpl_nodes: list[viser.FrameHandle] = []

    model_path = '/media/taeksoo/SSD1/github1/egoallo/data/smplh/neutral/model.npz'
    device = "cuda"

    # Create the SMPLH model
    smplh_model = fncsmpl.SmplhModel.load(model_path).to(device)

    # load parameter
    params = np.load("/media/taeksoo/HDD3/aria/lab_01/egoallo.npz")

    
    for i in tqdm(range(0, 0+num_frames)):
        j = 2 * i
        frame = cv2.imread(f"{data_path}/images/{j:05d}.png")
        cv2.imwrite(f"/home/taeksoo/Desktop/temp/{i:05d}.png", frame)

        frame_nodes.append(server.scene.add_frame(f"/frames/t{j}", show_axes=False))

        intrinsics = np.load(f"{data_path}/cam_params/intrinsics.npy")
        extrinsics = np.load(f"{data_path}/cam_params/{j:05d}.npy")

        # Place the frustum.
        fov = 2 * np.arctan2(frame.shape[0] / 2, intrinsics[0, 0])
        aspect = frame.shape[1] / frame.shape[0]
        server.scene.add_camera_frustum(
            f"/frames/t{j}/frustum",
            fov=fov,
            aspect=aspect,
            scale=0.15,
            image=frame[::4, ::4],
            wxyz=tf.SO3.from_matrix(extrinsics[:3, :3]).wxyz,
            position=extrinsics[:3, 3],
        )

        # Add some axes.
        server.scene.add_frame(
            f"/frames/t{j}/frustum/axes",
            axes_length=0.3,
            axes_radius=0.01,
        )


        # Add SMPLH mesh
        # with shape
        k = 1 * j
        smplh_shape = smplh_model.with_shape(betas=torch.tensor(params['betas'][:, k, :], device=device))

        local_quats = torch.cat([
            torch.tensor(params['body_quats'][:, k, :], device=device),
            torch.tensor(params['left_hand_quats'][:, k, :], device=device),
            torch.tensor(params['right_hand_quats'][:, k, :], device=device),
        ], dim=1)
        smplh_pose = smplh_shape.with_pose(T_world_root=torch.tensor(params['Ts_world_root'][:, k, :], device=device),
                                        local_quats=local_quats)
        mesh = smplh_pose.lbs()

        vertex_colors = np.array([180, 248, 200])
        mesh_out = trimesh.Trimesh(vertices=mesh.verts.cpu().numpy()[0],
                                faces=mesh.faces.cpu().numpy(),
                                vertex_colors=np.repeat(vertex_colors[None, :], mesh.verts.shape[1], axis=0))
        smpl_nodes.append(
            server.scene.add_mesh_trimesh(
                f"/frames/t{k}/smplh",
                mesh=mesh_out,
            )
        )

    

    # Hide all but the current frame.
    for i, frame_node in enumerate(frame_nodes):
        frame_node.visible = i == gui_timestep.value

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
