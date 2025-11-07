import os
import cv2
import numpy as np
import torch
import trimesh

import lib.vis.viewer as viewer_utils
from lib.vis.wham_tools.tools import checkerboard_geometry

def camera_marker_geometry(radius, height):
    vertices = np.array(
        [
            [-radius, -radius, 0],
            [radius, -radius, 0],
            [radius, radius, 0],
            [-radius, radius, 0],
            [0, 0, - height],
        ]
    )


    faces = np.array(
        [[0, 1, 2], [0, 2, 3], [1, 0, 4], [2, 1, 4], [3, 2, 4], [0, 3, 4],]
    )

    face_colors = np.array(
        [
            [0.5, 0.5, 0.5, 1.0],
            [0.5, 0.5, 0.5, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
        ]
    )
    return vertices, faces, face_colors


def run_vis2_on_video(res_dict, res_dict2, output_pth, focal_length, image_names, R_c2w=None, t_c2w=None, interactive=True):
    
    img0 = cv2.imread(image_names[0])
    height, width, _ = img0.shape

    world_mano = {}
    world_mano['vertices'] = res_dict['vertices']
    world_mano['faces'] = res_dict['faces']

    world_mano2 = {}
    world_mano2['vertices'] = res_dict2['vertices']
    world_mano2['faces'] = res_dict2['faces']

    vis_dict = {}
    color_idx = 0
    world_mano['vertices'] = world_mano['vertices']
    for _id, _verts in enumerate(world_mano['vertices']):
        verts = _verts.cpu().numpy() # T, N, 3
        body_faces = world_mano['faces']
        body_meshes = {
            "v3d": verts,
            "f3d": body_faces,
            "vc": None,
            "name": f"hand_{_id}",
            # "color": "pace-green",
            "color": "director-purple",
        }
        vis_dict[f"hand_{_id}"] = body_meshes
        color_idx += 1
    
    world_mano2['vertices'] = world_mano2['vertices']
    for _id, _verts in enumerate(world_mano2['vertices']):
        verts = _verts.cpu().numpy() # T, N, 3
        body_faces = world_mano2['faces']
        body_meshes = {
            "v3d": verts,
            "f3d": body_faces,
            "vc": None,
            "name": f"hand2_{_id}",
            # "color": "pace-blue",
            "color": "director-blue",
        }
        vis_dict[f"hand2_{_id}"] = body_meshes
        color_idx += 1
    
    v, f, vc, fc = checkerboard_geometry(length=100, c1=0, c2=0, up="z")
    v[:, 2] -= 2 # z plane
    gound_meshes = {
        "v3d": v,
        "f3d": f,
        "vc": vc,
        "name": "ground",
        "fc": fc,
        "color": -1,
    }
    vis_dict["ground"] = gound_meshes

    num_frames = len(world_mano['vertices'][_id])
    Rt = np.zeros((num_frames, 3, 4))
    Rt[:, :3, :3] = R_c2w[:num_frames]
    Rt[:, :3, 3] = t_c2w[:num_frames]

    verts, faces, face_colors = camera_marker_geometry(0.05, 0.1)
    verts = np.einsum("tij,nj->tni", Rt[:, :3, :3], verts) + Rt[:, None, :3, 3]
    camera_meshes = {
        "v3d": verts,
        "f3d": faces,
        "vc": None,
        "name": "camera",
        "fc": face_colors,
        "color": -1,
    }
    vis_dict["camera"] = camera_meshes

    side_source = torch.tensor([0.463, -0.478, 2.456])
    side_target = torch.tensor([0.026, -0.481, -3.184])
    up = torch.tensor([1.0, 0.0, 0.0])
    view_camera = lookat_matrix(side_source, side_target, up)
    viewer_Rt = np.tile(view_camera[:3, :4], (num_frames, 1, 1))

    meshes = viewer_utils.construct_viewer_meshes(
        vis_dict, draw_edges=False, flat_shading=False
    )

    vis_h, vis_w = (height, width)
    K = np.array(
        [
            [1000, 0, vis_w / 2],
            [0, 1000, vis_h / 2],
            [0, 0, 1]
        ]
    )
    
    data = viewer_utils.ViewerData(viewer_Rt, K, vis_w, vis_h)
    batch = (meshes, data)

    if interactive:
        viewer = viewer_utils.ARCTICViewer(interactive=True, size=(vis_w, vis_h))
        viewer.render_seq(batch, out_folder=os.path.join(output_pth, 'aitviewer'))
    else:
        viewer = viewer_utils.ARCTICViewer(interactive=False, size=(vis_w, vis_h), render_types=['video'])
        if os.path.exists(os.path.join(output_pth, 'aitviewer', "video_0.mp4")):
            os.remove(os.path.join(output_pth, 'aitviewer', "video_0.mp4"))
        viewer.render_seq(batch, out_folder=os.path.join(output_pth, 'aitviewer'))
        return os.path.join(output_pth, 'aitviewer', "video_0.mp4")

def run_vis2_on_video_cam(res_dict, res_dict2, output_pth, focal_length, image_names, R_w2c=None, t_w2c=None):
    
    img0 = cv2.imread(image_names[0])
    height, width, _ = img0.shape

    world_mano = {}
    world_mano['vertices'] = res_dict['vertices']
    world_mano['faces'] = res_dict['faces']

    world_mano2 = {}
    world_mano2['vertices'] = res_dict2['vertices']
    world_mano2['faces'] = res_dict2['faces']

    vis_dict = {}
    color_idx = 0
    world_mano['vertices'] = world_mano['vertices']
    for _id, _verts in enumerate(world_mano['vertices']):
        verts = _verts.cpu().numpy() # T, N, 3
        body_faces = world_mano['faces']
        body_meshes = {
            "v3d": verts,
            "f3d": body_faces,
            "vc": None,
            "name": f"hand_{_id}",
            # "color": "pace-green",
            "color": "director-purple",
        }
        vis_dict[f"hand_{_id}"] = body_meshes
        color_idx += 1
    
    world_mano2['vertices'] = world_mano2['vertices']
    for _id, _verts in enumerate(world_mano2['vertices']):
        verts = _verts.cpu().numpy() # T, N, 3
        body_faces = world_mano2['faces']
        body_meshes = {
            "v3d": verts,
            "f3d": body_faces,
            "vc": None,
            "name": f"hand2_{_id}",
            # "color": "pace-blue",
            "color": "director-blue",
        }
        vis_dict[f"hand2_{_id}"] = body_meshes
        color_idx += 1

    meshes = viewer_utils.construct_viewer_meshes(
        vis_dict, draw_edges=False, flat_shading=False
    )

    num_frames = len(world_mano['vertices'][_id])
    Rt = np.zeros((num_frames, 3, 4))
    Rt[:, :3, :3] = R_w2c[:num_frames]
    Rt[:, :3, 3] = t_w2c[:num_frames]

    cols, rows = (width, height)
    K = np.array(
        [
            [focal_length, 0, width / 2],
            [0, focal_length, height / 2],
            [0, 0, 1]
        ]
    )
    vis_h = height
    vis_w = width

    data = viewer_utils.ViewerData(Rt, K, cols, rows, imgnames=image_names)
    batch = (meshes, data)

    viewer = viewer_utils.ARCTICViewer(interactive=True, size=(vis_w, vis_h))
    viewer.render_seq(batch, out_folder=os.path.join(output_pth, 'aitviewer'))

def lookat_matrix(source_pos, target_pos, up):
    """
    IMPORTANT: USES RIGHT UP BACK XYZ CONVENTION
    :param source_pos (*, 3)
    :param target_pos (*, 3)
    :param up (3,)
    """
    *dims, _ = source_pos.shape
    up = up.reshape(*(1,) * len(dims), 3)
    up = up / torch.linalg.norm(up, dim=-1, keepdim=True)
    back = normalize(target_pos - source_pos)
    right = normalize(torch.linalg.cross(up, back))
    up = normalize(torch.linalg.cross(back, right))
    R = torch.stack([right, up, back], dim=-1)
    return make_4x4_pose(R, source_pos)

def make_4x4_pose(R, t):
    """
    :param R (*, 3, 3)
    :param t (*, 3)
    return (*, 4, 4)
    """
    dims = R.shape[:-2]
    pose_3x4 = torch.cat([R, t.view(*dims, 3, 1)], dim=-1)
    bottom = (
        torch.tensor([0, 0, 0, 1], device=R.device)
        .reshape(*(1,) * len(dims), 1, 4)
        .expand(*dims, 1, 4)
    )
    return torch.cat([pose_3x4, bottom], dim=-2)

def normalize(x):
    return x / torch.linalg.norm(x, dim=-1, keepdim=True)

def save_mesh_to_obj(vertices, faces, file_path):
    # 创建一个 Trimesh 对象
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # 导出为 .obj 文件
    mesh.export(file_path)
    print(f"Mesh saved to {file_path}")

