import cv2
import os
from glob import glob
import numpy as np
from tqdm import tqdm

vrsfile = "/media/taeksoo/HDD3/aria/WM_lab_00.vrs"

# Parameters
name = vrsfile.split("/")[-1].split(".")[0]
image_folder = f"/media/taeksoo/HDD3/aria/{name}_data/images"
video_output_folder = f"/media/taeksoo/HDD3/aria/{name}_data/videos"
trajectory_output_folder = f"/media/taeksoo/HDD3/aria/{name}_data/trajectories"

clip_length = 49
stride = 10
fps = 8
start_image_idx = 42  # Start from the first image
end_image_idx = 2704  # End at the 1000th image

output_size = None  # Set this if you want to resize (e.g., (720, 480))

# Create output directory
os.makedirs(video_output_folder, exist_ok=True)
os.makedirs(trajectory_output_folder, exist_ok=True)

# Get sorted list of image paths
image_paths = sorted(glob(os.path.join(image_folder, "*.png")))  # or .jpg
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
