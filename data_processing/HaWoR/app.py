import gradio as gr
# import spaces
import sys
import os
import torch
import numpy as np
import joblib
from easydict import EasyDict
from scripts.scripts_test_video.detect_track_video import detect_track_video
from scripts.scripts_test_video.hawor_video import hawor_motion_estimation, hawor_infiller
from scripts.scripts_test_video.hawor_slam import hawor_slam
from hawor.utils.process import get_mano_faces, run_mano, run_mano_left
from lib.eval_utils.custom_utils import load_slam_cam
from lib.vis.run_vis2 import run_vis2_on_video, run_vis2_on_video_cam


def render_reconstruction(input_video, img_focal): 
    args = EasyDict()
    args.video_path = input_video
    args.input_type = 'file'
    args.checkpoint = './weights/hawor/checkpoints/hawor.ckpt'
    args.infiller_weight = './weights/hawor/checkpoints/infiller.pt'
    args.vis_mode = 'world'
    args.img_focal = img_focal
    
    start_idx, end_idx, seq_folder, imgfiles = detect_track_video(args)

    frame_chunks_all, img_focal = hawor_motion_estimation(args, start_idx, end_idx, seq_folder)

    slam_path = os.path.join(seq_folder, f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz")
    if not os.path.exists(slam_path):
        hawor_slam(args, start_idx, end_idx)
    slam_path = os.path.join(seq_folder, f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz")
    R_w2c_sla_all, t_w2c_sla_all, R_c2w_sla_all, t_c2w_sla_all = load_slam_cam(slam_path)

    pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = hawor_infiller(args, start_idx, end_idx, frame_chunks_all)

    # vis sequence for this video
    hand2idx = {
        "right": 1,
        "left": 0
    }
    vis_start = 0
    vis_end = pred_trans.shape[1] - 1
            
    # get faces
    faces = get_mano_faces()
    faces_new = np.array([[92, 38, 234],
            [234, 38, 239],
            [38, 122, 239],
            [239, 122, 279],
            [122, 118, 279],
            [279, 118, 215],
            [118, 117, 215],
            [215, 117, 214],
            [117, 119, 214],
            [214, 119, 121],
            [119, 120, 121],
            [121, 120, 78],
            [120, 108, 78],
            [78, 108, 79]])
    faces_right = np.concatenate([faces, faces_new], axis=0)

    # get right hand vertices
    hand = 'right'
    hand_idx = hand2idx[hand]
    pred_glob_r = run_mano(pred_trans[hand_idx:hand_idx+1, vis_start:vis_end], pred_rot[hand_idx:hand_idx+1, vis_start:vis_end], pred_hand_pose[hand_idx:hand_idx+1, vis_start:vis_end], betas=pred_betas[hand_idx:hand_idx+1, vis_start:vis_end])
    right_verts = pred_glob_r['vertices'][0]
    right_dict = {
            'vertices': right_verts.unsqueeze(0),
            'faces': faces_right,
        }

    # get left hand vertices
    faces_left = faces_right[:,[0,2,1]]
    hand = 'left'
    hand_idx = hand2idx[hand]
    pred_glob_l = run_mano_left(pred_trans[hand_idx:hand_idx+1, vis_start:vis_end], pred_rot[hand_idx:hand_idx+1, vis_start:vis_end], pred_hand_pose[hand_idx:hand_idx+1, vis_start:vis_end], betas=pred_betas[hand_idx:hand_idx+1, vis_start:vis_end])
    left_verts = pred_glob_l['vertices'][0]
    left_dict = {
            'vertices': left_verts.unsqueeze(0),
            'faces': faces_left,
        }

    R_x = torch.tensor([[1,  0,  0],
                        [0, -1,  0],
                        [0,  0, -1]]).float()
    R_c2w_sla_all = torch.einsum('ij,njk->nik', R_x, R_c2w_sla_all)
    t_c2w_sla_all = torch.einsum('ij,nj->ni', R_x, t_c2w_sla_all)
    R_w2c_sla_all = R_c2w_sla_all.transpose(-1, -2)
    t_w2c_sla_all = -torch.einsum("bij,bj->bi", R_w2c_sla_all, t_c2w_sla_all)
    left_dict['vertices'] = torch.einsum('ij,btnj->btni', R_x, left_dict['vertices'].cpu())
    right_dict['vertices'] = torch.einsum('ij,btnj->btni', R_x, right_dict['vertices'].cpu())
    
    # Here we use aitviewer(https://github.com/eth-ait/aitviewer) for simple visualization.
    if args.vis_mode == 'world': 
        output_pth = os.path.join(seq_folder, f"vis_{vis_start}_{vis_end}")
        if not os.path.exists(output_pth):
            os.makedirs(output_pth)
        image_names = imgfiles[vis_start:vis_end]
        print(f"vis {vis_start} to {vis_end}")
        vis_video_path = run_vis2_on_video(left_dict, right_dict, output_pth, img_focal, image_names, R_c2w=R_c2w_sla_all[vis_start:vis_end], t_c2w=t_c2w_sla_all[vis_start:vis_end], interactive=False)
    elif args.vis_mode == 'cam':
        # output_pth = os.path.join(seq_folder, f"vis_{vis_start}_{vis_end}")
        # if not os.path.exists(output_pth):
        #     os.makedirs(output_pth)
        # image_names = imgfiles[vis_start:vis_end]
        # print(f"vis {vis_start} to {vis_end}")
        # run_vis2_on_video_cam(left_dict, right_dict, output_pth, img_focal, image_names, R_w2c=R_w2c_sla_all[vis_start:vis_end], t_w2c=t_w2c_sla_all[vis_start:vis_end])
        raise NotImplementedError

    return vis_video_path  

# @spaces.GPU()
def run_wilow_model(image, conf, IoU_threshold=0.5):
    img_cv2 = image[...,::-1]
    return img_vis.astype(np.float32)/255.0, len(detections), None       



header = ('''
<div class="embed_hidden" style="text-align: center;">
    <h1> <b>HaWoR</b>: World-Space Hand Motion Reconstruction from Egocentric Videos</h1>
    <h3>
        <a href="" target="_blank" rel="noopener noreferrer">Jinglei Zhang</a><sup>1</sup>,        
        <a href="https://jiankangdeng.github.io/" target="_blank" rel="noopener noreferrer">Jiankang Deng</a><sup>2</sup>,
        <br>
        <a href="https://scholar.google.com/citations?user=syoPhv8AAAAJ&hl=en" target="_blank" rel="noopener noreferrer">Chao Ma</a><sup>1</sup>
        <a href="https://rolpotamias.github.io" target="_blank" rel="noopener noreferrer">Rolandos Alexandros Potamias</a><sup>2</sup>
    </h3>
    <h3>
        <sup>1</sup>Shanghai Jiao Tong University;
        <sup>2</sup>Imperial College London
    </h3>
</div>
<div style="display:flex; gap: 0.3rem; justify-content: center; align-items: center;" align="center">
<a href='https://arxiv.org/abs/xxxx.xxxxx'><img src='https://img.shields.io/badge/Arxiv-xxxx.xxxxx-A42C25?style=flat&logo=arXiv&logoColor=A42C25'></a> 
<a href=''><img src='https://img.shields.io/badge/Paper-PDF-yellow?style=flat&logo=arXiv&logoColor=yellow'></a> 
<a href='https://hawor-project.github.io/'><img src='https://img.shields.io/badge/Project-Page-%23df5b46?style=flat&logo=Google%20chrome&logoColor=%23df5b46'></a> 
<a href='https://github.com/ThunderVVV/HaWoR'><img src='https://img.shields.io/badge/GitHub-Code-black?style=flat&logo=github&logoColor=white'></a> 
''')


with gr.Blocks(title="HaWoR: World-Space Hand Motion Reconstruction from Egocentric Videos", css=".gradio-container") as demo:

    gr.Markdown(header)

    with gr.Row():
        with gr.Column():
            input_video = gr.Video(label="Input video", sources=["upload"])
            img_focal = gr.Number(label="Focal Length", value=600)
            # threshold = gr.Slider(value=0.3, minimum=0.05, maximum=0.95, step=0.05, label='Detection Confidence Threshold')
            #nms = gr.Slider(value=0.5, minimum=0.05, maximum=0.95, step=0.05, label='IoU NMS Threshold')
            submit = gr.Button("Submit", variant="primary")
        
        
        with gr.Column():
            reconstruction = gr.Video(label="Reconstruction",show_download_button=True)
            # hands_detected = gr.Textbox(label="Hands Detected")
    
        submit.click(fn=render_reconstruction, inputs=[input_video, img_focal], outputs=[reconstruction])

    with gr.Row():
        
        example_images = gr.Examples([
            ['./example/video_0.mp4'] 
            ], 
            inputs=input_video)
    
demo.launch(debug=True)