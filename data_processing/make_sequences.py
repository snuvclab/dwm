import cv2
import os
from glob import glob
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="Generate video and trajectory sequences from images and camera parameters.")
parser.add_argument("--data_root", type=str, required=True, help="Root directory containing folders of image and camera parameter data.")
parser.add_argument("--start_idx", type=int, default=0, help="Starting index for image sequence.")
parser.add_argument("--end_idx", type=int, default=-1, help="Ending index for image sequence.")
args = parser.parse_args()


# input folders
image_folder = Path(args.data_root) / "images"
disparity_folder = Path(args.data_root) / "disparity"
cam_param_folder = Path(args.data_root) / "cam_params"
human_pose_folder = Path(args.data_root) / "human_poses"

# output folders
output_folder = Path(args.data_root) / "sequences"
video_output_folder = Path(output_folder) / "videos"
disparity_output_folder = Path(output_folder) / "disparity"
trajectory_output_folder = Path(output_folder) / "trajectory"
human_pose_output_folder = Path(output_folder) / "human_motions"

# parameters
clip_length = 49
stride = 10
fps = 8
start_image_idx = args.start_idx  # Start from
end_image_idx = args.end_idx  # End at
output_size = None  # Set this if you want to resize (e.g., (720, 480))

# Create output directory
if image_folder.exists(): os.makedirs(video_output_folder, exist_ok=True)
if disparity_folder.exists(): os.makedirs(disparity_output_folder, exist_ok=True)
if cam_param_folder.exists(): os.makedirs(trajectory_output_folder, exist_ok=True)
if human_pose_folder.exists(): os.makedirs(human_pose_output_folder, exist_ok=True)

# Get sorted list of image paths
image_paths = sorted(glob(os.path.join(image_folder, "*.png")))
image_paths = image_paths[start_image_idx:end_image_idx]
cam_param_paths = [image_path.replace("images", "cam_params").replace(".png", ".npy") for image_path in image_paths]

num_images = len(image_paths)

# Generate clips
clip_idx = 0
for start_idx in tqdm(range(0, num_images - clip_length + 1, stride)):
    clip_frames = image_paths[start_idx : start_idx + clip_length]
    clip_cam_params = cam_param_paths[start_idx : start_idx + clip_length]

    # Read first frame to get video dimensions
    frame = cv2.imread(clip_frames[0])
    if output_size:
        frame = cv2.resize(frame, output_size)
    height, width = frame.shape[:2]

    # Define video writer
    out_path = os.path.join(video_output_folder, f"{clip_idx:05}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    for frame_path in clip_frames:
        frame = cv2.imread(frame_path)
        if output_size:
            frame = cv2.resize(frame, output_size)
        out.write(frame)
    out.release()

    # Save trajectory
    cam_params = [np.load(cam_param_path) for cam_param_path in clip_cam_params]
    T0_inv = np.linalg.inv(cam_params[0])
    relative_trajectory = np.array([T0_inv @ T for T in cam_params]) # relative to the first frame
    absolute_trajectory = np.array(cam_params) # absolute trajectory in world coordinates

    np.save(os.path.join(trajectory_output_folder, f"{clip_idx:05}.npy"), relative_trajectory)
    np.save(os.path.join(trajectory_output_folder, f"{clip_idx:05}_abs.npy"), absolute_trajectory)

    clip_idx += 1
