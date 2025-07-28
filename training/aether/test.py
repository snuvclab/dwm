import random
import time
import numpy as np
import os
import cv2
from tqdm import tqdm
import pickle
import math

import viser
import viser.transforms as tf

import projectaria_tools.core.mps as mps
from projectaria_tools.core.mps.utils import filter_points_from_confidence

from 


def diagonal_fov(fov_x_deg, fov_y_deg):
    fov_x_rad = math.radians(fov_x_deg)
    fov_y_rad = math.radians(fov_y_deg)

    tan_x = math.tan(fov_x_rad / 2)
    tan_y = math.tan(fov_y_rad / 2)

    diag_fov_rad = 2 * math.atan(math.sqrt(tan_x**2 + tan_y**2))
    return math.degrees(diag_fov_rad)

                        
def main():   

    server = viser.ViserServer()
    num_frames = 41

    # Add some coordinate frames to the scene. These will be visualized in the viewer.
    # server.scene.add_frame(
    #     "/tree",
    #     wxyz=(1.0, 0.0, 0.0, 0.0),
    #     position=(0, 0, 0),
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
    
    # GT camera poses
    with open(f"//media/taeksoo/HDD3/aria/WM_lab_00_data/trajectories/00038.npy", "rb") as f:
        gt_pose = np.load(f)

    for i in tqdm(range(num_frames)):
        with open(f"/media/taeksoo/SSD1/github1/world_model/outputs/00038/prediction_00038_predictions_{i:02d}.pkl", "rb") as f:
            data = pickle.load(f)
        frame = cv2.imread(f"/media/taeksoo/SSD1/github1/world_model/outputs/00038/frames/{i+1:02d}.png")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_nodes.append(server.scene.add_frame(f"/frames/t{i}", show_axes=False))

        # Add the point cloud.
        point_nodes.append(
            server.scene.add_point_cloud(
                name=f"/frames/t{i}/point_cloud",
                points=data['world_points'].reshape(-1, 3),
                colors=(data['images'].reshape(-1, 3) * 255).astype(np.uint8),
                point_size=0.005,
                point_shape="rounded",
            )
        )

        intrinsics = data['intrinsics']
        extrinsics = data['camera_poses'][0]

        # Estimated camera poses
        fov = 2 * np.arctan2(frame.shape[0] / 2, intrinsics[0, 0])
        aspect = frame.shape[1] / frame.shape[0]
        server.scene.add_camera_frustum(
            f"/frames/t{i}/frustum",
            fov=fov,
            aspect=aspect,
            scale=0.15,
            image=frame[::4, ::4],
            wxyz=tf.SO3.from_matrix(extrinsics[:3, :3]).wxyz,
            position=extrinsics[:3, 3],
        )
        # Add some axes.
        server.scene.add_frame(
            f"/frames/t{i}/frustum/axes",
            axes_length=0.3,
            axes_radius=0.01,
        )


        # Ground truth camera poses
        # fov = gt_pose[i]
        aspect = frame.shape[1] / frame.shape[0]
        server.scene.add_camera_frustum(
            f"/frames/t{i}/frustum_gt",
            fov=fov,
            aspect=aspect,
            scale=0.15,
            image=frame[::4, ::4],
            wxyz=tf.SO3.from_matrix(gt_pose[i][:3, :3]).wxyz,
            position=gt_pose[i][:3, 3],
        )
        # Add some axes.
        server.scene.add_frame(
            f"/frames/t{i}/frustum_gt/axes",
            axes_length=0.3,
            axes_radius=0.01,
            origin_color=(0, 0, 0),  # black for ground truth
        )

    

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
