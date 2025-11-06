import sys
import os
sys.path.insert(0, os.path.dirname(__file__) + '/../..')

import argparse
import numpy as np
from glob import glob
from lib.pipeline.tools import detect_track
from natsort import natsorted
import subprocess


def extract_frames(video_path, output_folder, fps=30):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    command = [
        'ffmpeg',               
        '-i', video_path,       
        '-vf', f'fps={fps}',         
        '-start_number', '0',
        os.path.join(output_folder, '%04d.jpg')  
    ]

    # Suppress ffmpeg output (stdout and stderr)
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def detect_track_video(args):
    file = args.video_path
    root = os.path.dirname(file)
    seq = os.path.basename(file).split('.')[0]

    seq_folder = f'{root}/{seq}'
    img_folder = f'{seq_folder}/extracted_images'
    os.makedirs(seq_folder, exist_ok=True)
    os.makedirs(img_folder, exist_ok=True)
    print(f'Running detect_track on {file} ...')

    ##### Extract Frames #####
    imgfiles = natsorted(glob(f'{img_folder}/*.jpg'))
    # print(imgfiles[:10])
    if len(imgfiles) > 0:
        print("Skip extracting frames")
    else:
        _ = extract_frames(file, img_folder, fps=args.fps)
    imgfiles = natsorted(glob(f'{img_folder}/*.jpg'))

    ##### Detection + Track #####
    print('Detect and Track ...')

    start_idx = 0
    end_idx = len(imgfiles)

    if os.path.exists(f'{seq_folder}/tracks_{start_idx}_{end_idx}/model_boxes.npy'):
        print(f"skip track for {start_idx}_{end_idx}")
        return start_idx, end_idx, seq_folder, imgfiles
    os.makedirs(f"{seq_folder}/tracks_{start_idx}_{end_idx}", exist_ok=True)
    boxes_, tracks_ = detect_track(imgfiles, thresh=0.2)
    np.save(f'{seq_folder}/tracks_{start_idx}_{end_idx}/model_boxes.npy', boxes_)
    np.save(f'{seq_folder}/tracks_{start_idx}_{end_idx}/model_tracks.npy', tracks_)

    return start_idx, end_idx, seq_folder, imgfiles

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--img_focal", type=float)
    parser.add_argument("--video_path", type=str, default='')
    parser.add_argument("--input_type", type=str, default='file')
    args = parser.parse_args()

    detect_track_video(args)