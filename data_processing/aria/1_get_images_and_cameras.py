from projectaria_tools.core import data_provider, image, mps, calibration
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.data_provider import VrsDataProvider, create_vrs_data_provider
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core.sophus import SE3
from typing import Any, Dict, List, cast
from dataclasses import dataclass
import cv2
from PIL import Image
import os
from tqdm import tqdm
import numpy as np
from scipy.spatial.transform import Rotation as R
import pickle

@dataclass
class AriaCameraCalibration:
    fx: float
    fy: float
    cx: float
    cy: float
    distortion_params: np.ndarray
    width: int
    height: int
    t_device_camera: SE3

@dataclass
class AriaImageFrame:
    camera: AriaCameraCalibration
    file_path: str
    t_world_camera: SE3
    timestamp_ns: float

@dataclass
class TimedPoses:
    timestamps_ns: np.ndarray
    t_world_devices: List[SE3]

def get_camera_calibs(provider: VrsDataProvider) -> Dict[str, AriaCameraCalibration]:
    """Retrieve the per-camera factory calibration from within the VRS."""

    factory_calib = {}
    name = "camera-rgb"
    device_calib = provider.get_device_calibration()
    assert device_calib is not None, "Could not find device calibration"
    sensor_calib = device_calib.get_camera_calib(name)
    assert sensor_calib is not None, f"Could not find sensor calibration for {name}"

    width = sensor_calib.get_image_size()[0].item()
    height = sensor_calib.get_image_size()[1].item()

    width = sensor_calib.get_image_size()[0].item()
    height = sensor_calib.get_image_size()[1].item()

    # if width > max_output_size or height > max_output_size:
    #     sensor_calib = sensor_calib.rescale(
    #         np.array([max_output_size, max_output_size]).astype(np.int64),
    #         max_output_size / width,
    #     )
    #     width = sensor_calib.get_image_size()[0].item()
    #     height = sensor_calib.get_image_size()[1].item()

    intrinsics = sensor_calib.get_projection_params()

    factory_calib[name] = AriaCameraCalibration(
        fx=intrinsics[0],
        fy=intrinsics[0],
        cx=intrinsics[1],
        cy=intrinsics[2],
        distortion_params=intrinsics[3:15],
        width=width,
        height=height,
        t_device_camera=sensor_calib.get_transform_device_camera(),
    )

    return factory_calib

def read_trajectory_csv_to_dict(file_iterable_csv: str) -> TimedPoses:
    closed_loop_traj = mps.read_closed_loop_trajectory(file_iterable_csv)  # type: ignore

    timestamps_secs, poses = zip(
        *[(it.tracking_timestamp.total_seconds(), it.transform_world_device) for it in closed_loop_traj]
    )

    SEC_TO_NANOSEC = 1e9
    return TimedPoses(
        timestamps_ns=(np.array(timestamps_secs) * SEC_TO_NANOSEC).astype(int),
        t_world_devices=poses,
    )



vrsfile = "/media/taeksoo/HDD3/aria/lab_01/lab_01.vrs"
provider = data_provider.create_vrs_data_provider(vrsfile)
stream_id = provider.get_stream_id_from_label("camera-rgb")

name = vrsfile.split("/")[-1].split(".")[0]
image_output_dir = f"/media/taeksoo/HDD3/aria/lab_01/{name}_data/images"
camera_output_dir = f"/media/taeksoo/HDD3/aria/lab_01/{name}_data/cam_params"
os.makedirs(image_output_dir, exist_ok=True)
os.makedirs(camera_output_dir, exist_ok=True)

num_frames = provider.get_num_data(stream_id)
closed_loop_traj_path = f"/media/taeksoo/HDD3/aria/lab_01/mps_{name}_vrs/slam/closed_loop_trajectory.csv"
t_world_devices = read_trajectory_csv_to_dict(closed_loop_traj_path)
name_to_camera = get_camera_calibs(provider)

# https://facebookresearch.github.io/projectaria_tools/docs/data_utilities/advanced_code_snippets/image_utilities
calib = provider.get_device_calibration().get_camera_calib("camera-rgb")
pinhole = calibration.get_linear_camera_calibration(1024, 1024, 300, "camera-rgb", calib.get_transform_device_camera()) # width, height, focal_length
pinhole_cw90 = calibration.rotate_camera_calib_cw90deg(pinhole)


for i in tqdm(range(num_frames)):
    image_data =  provider.get_image_data_by_index(stream_id, i)
    frame_rgb = image_data[0].to_numpy_array()
    undistored_frame_rgb = calibration.distort_by_calibration(frame_rgb, pinhole, calib)

    # Step 1: Rotate image 90 degrees clockwise
    rotated = np.rot90(undistored_frame_rgb, k=3)

    # Step 2: Center crop to 1080x720 (W x H)
    crop_width, crop_height = 720, 480
    h, w = rotated.shape[:2]
    start_x = (w - crop_width) // 2
    start_y = (h - crop_height) // 2
    cropped = rotated[start_y:start_y + crop_height, start_x:start_x + crop_width]


    # Save the processed image
    output_path = os.path.join(image_output_dir, f"{i:05d}.png")
    cv2.imwrite(output_path, cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))

    # Pose information
    capture_time_ns = image_data[1].capture_timestamp_ns

    # Find the nearest neighbor pose with the closest timestamp to the capture time.
    nearest_pose_idx = np.searchsorted(t_world_devices.timestamps_ns, capture_time_ns)
    nearest_pose_idx = np.minimum(nearest_pose_idx, len(t_world_devices.timestamps_ns) - 1)
    assert nearest_pose_idx != -1, f"Could not find pose for {capture_time_ns}"

    t_world_device = t_world_devices.t_world_devices[nearest_pose_idx] # world to aria glasses device pose
    t_world_camera = t_world_device @ pinhole_cw90.get_transform_device_camera() # world to rgb camera pose

    extrinsics = t_world_camera.to_matrix()
    np.save(os.path.join(camera_output_dir, f"{i:05d}.npy"), extrinsics)

fx, fy, cx, cy = pinhole_cw90.get_projection_params()
intrinsics = np.array([
    [fx, 0, cx - start_x],
    [0, fy, cy - start_y],
    [0, 0, 1]
])
np.save(os.path.join(camera_output_dir, f"intrinsics.npy"), intrinsics)
    
