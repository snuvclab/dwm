import cv2
import imageio.v3 as iio
import os
from glob import glob
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
import json

parser = argparse.ArgumentParser(description="Generate video and trajectory sequences from images and camera parameters.")
parser.add_argument("--data_root", type=str, required=True, help="Root directory containing folders of image and camera parameter data.")
parser.add_argument("--start_idx", type=int, default=1, help="Starting index for image sequence.")
parser.add_argument("--end_idx", type=int, default=-1, help="Ending index for image sequence.")
parser.add_argument("--stride", type=int, default=10, help="Stride for selecting frames in the sequence.")
args = parser.parse_args()


# input folders
image_folder = Path(args.data_root) / "images"
disparity_folder = Path(args.data_root) / "disparity"
cam_param_folder = Path(args.data_root) / "cam_params"
egoallo_human_pose_file = Path(args.data_root) / "egoallo.npz"

# output folders
output_folder = Path(args.data_root) / "sequences"
video_output_folder = Path(output_folder) / "videos"
disparity_output_folder = Path(output_folder) / "disparity"
trajectory_output_folder = Path(output_folder) / "trajectory"
human_pose_output_folder = Path(output_folder) / "human_motions"

# parameters
clip_length = 49
stride = args.stride
fps = 8
start_image_idx = args.start_idx  # Start from
end_image_idx = args.end_idx  # End at
output_size = None  # Set this if you want to resize (e.g., (720, 480))

# Create output directory
if not image_folder.exists():
    raise FileNotFoundError(f"Image folder {image_folder} does not exist.")
if image_folder.exists(): os.makedirs(video_output_folder, exist_ok=True)
if disparity_folder.exists(): os.makedirs(disparity_output_folder, exist_ok=True)
if cam_param_folder.exists(): os.makedirs(trajectory_output_folder, exist_ok=True)
if egoallo_human_pose_file.exists(): os.makedirs(human_pose_output_folder, exist_ok=True)

# save arguments to a JSON file
with open(f"{str(output_folder)}/args.json", "w") as f:
    args = {
        "clip_length": clip_length,
        "stride": stride,
        "fps": fps,
        "start_image_idx": start_image_idx,
        "end_image_idx": end_image_idx,
    }
    json.dump(args, f, indent=4)

# Get sorted list of paths
image_paths = sorted(image_folder.glob("*.png"))
image_paths = image_paths[start_image_idx:end_image_idx]

if disparity_folder.exists():
    disparity_paths = sorted(disparity_folder.glob("*.png"))
    disparity_paths = disparity_paths[start_image_idx:end_image_idx]

if cam_param_folder.exists():
    cam_param_paths = sorted(cam_param_folder.glob("*.npy"))
    cam_param_paths = cam_param_paths[start_image_idx:end_image_idx]

if egoallo_human_pose_file.exists():
    smplh_data = np.load(egoallo_human_pose_file)
    smplh_data = {key: smplh_data[key].astype(np.float32) for key in smplh_data.files} 
    for key in smplh_data:
        if key in ["Ts_world_root", "body_quats", "left_hand_quats", "right_hand_quats", "betas"]:
            smplh_data[key] = smplh_data[key][:, start_image_idx-1:end_image_idx-1] # egoallo predicts from the second frame

print(f"Total images: {len(image_paths)}")
print(f"Total poses: {smplh_data['Ts_world_root'].shape[1] if egoallo_human_pose_file.exists() else 0}")


num_images = len(image_paths)
# Generate clips
clip_idx = 0
for start_idx in tqdm(range(0, num_images - clip_length + 1, stride)):
    clip_frames = image_paths[start_idx : start_idx + clip_length]
    if disparity_folder.exists():
        clip_disparity = disparity_paths[start_idx : start_idx + clip_length]
    if cam_param_folder.exists():
        clip_cam_params = cam_param_paths[start_idx : start_idx + clip_length]
    if egoallo_human_pose_file.exists():
        clip_smplh = {}
        for key in smplh_data:
            if key == 'Ts_world_root':
                clip_smplh["global_orient_quat"] = smplh_data[key][0, start_idx : start_idx + clip_length, :4] # global orientation in quaternion format Fx4
                clip_smplh["transl"] = smplh_data[key][0, start_idx : start_idx + clip_length, 4:] # translation Fx3
            elif key == 'body_quats':
                clip_smplh["body_pose_quat"] = smplh_data[key][0, start_idx : start_idx + clip_length, :] # body pose in quaternion format FxN_jx4
            elif key == 'left_hand_quats':
                clip_smplh["left_hand_pose_quat"] = smplh_data[key][0, start_idx : start_idx + clip_length, :] # left hand pose in quaternion format FxN_jx4
            elif key == 'right_hand_quats':
                clip_smplh["right_hand_pose_quat"] = smplh_data[key][0, start_idx : start_idx + clip_length, :] # right hand pose in quaternion format FxN_jx4
            elif key == 'betas':
                clip_smplh["betas"] = smplh_data[key][0, start_idx : start_idx + clip_length, :] # shape parameters FxN_betas_smplh
            else:
                continue # Skip any other keys
    
    # RGB video output
    out_path = os.path.join(video_output_folder, f"{clip_idx:05}.mp4")
    frames = []
    for frame_path in clip_frames:
        frame = iio.imread(frame_path)
        if output_size:
            frame = cv2.resize(frame, output_size)
        frames.append(frame)
    iio.imwrite(
        out_path,
        np.stack(frames),
        fps=fps,
        codec='libx264'
    )

    # Disparity video output
    if disparity_folder.exists():
        out_path = os.path.join(disparity_output_folder, f"{clip_idx:05}_disparity.mp4")
        disparity_frames = []
        for disparity_path in clip_disparity:
            disparity_frame = iio.imread(disparity_path)
            if output_size:
                disparity_frame = cv2.resize(disparity_frame, output_size)
            disparity_frames.append(disparity_frame)
        iio.imwrite(
            out_path,
            np.stack(disparity_frames),
            fps=fps,
            codec='libx264'
        )

    # Camera trajectory output
    if cam_param_folder.exists():
        cam_params = [np.load(cam_param_path) for cam_param_path in clip_cam_params]
        T0_inv = np.linalg.inv(cam_params[0])
        relative_trajectory = np.array([T0_inv @ T for T in cam_params]) # relative to the first frame
        absolute_trajectory = np.array(cam_params) # absolute trajectory in world coordinates

        np.save(os.path.join(trajectory_output_folder, f"{clip_idx:05}.npy"), relative_trajectory)
        np.save(os.path.join(trajectory_output_folder, f"{clip_idx:05}_abs.npy"), absolute_trajectory)

    # Human pose output
    if egoallo_human_pose_file.exists():
        np.savez_compressed(os.path.join(human_pose_output_folder, f"{clip_idx:05}.npz"), **clip_smplh)

    clip_idx += 1

with open(f"{output_folder}/videos.txt", "w") as f:
    for i in range(clip_idx):
        f.write(f"videos/{i:05}.mp4\n")
with open(f"{output_folder}/prompts.txt", "w") as f:
    for i in range(clip_idx):
        f.write(f"\n")