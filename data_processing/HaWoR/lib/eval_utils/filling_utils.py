import copy
import os
import joblib
import numpy as np
from scipy.spatial.transform import Slerp, Rotation
import torch

from hawor.utils.process import run_mano, run_mano_left
from hawor.utils.rotation import angle_axis_to_quaternion, angle_axis_to_rotation_matrix, quaternion_to_rotation_matrix, rotation_matrix_to_angle_axis
from lib.utils.geometry import rotmat_to_rot6d
from lib.utils.geometry import rot6d_to_rotmat

def slerp_interpolation_aa(pos, valid):

    B, T, N, _ = pos.shape  # B: 批次大小, T: 时间步长, N: 关节数, 4: 四元数维度
    pos_interp = pos.copy()  # 创建副本以存储插值结果
    
    for b in range(B):
        for n in range(N):
            quat_b_n = pos[b, :, n, :]
            valid_b_n = valid[b, :]
            
            invalid_idxs = np.where(~valid_b_n)[0]
            valid_idxs = np.where(valid_b_n)[0]

            if len(invalid_idxs) == 0:
                continue
            
            if len(valid_idxs) > 1:
                valid_times = valid_idxs  # 有效时间步
                valid_rots = Rotation.from_rotvec(quat_b_n[valid_idxs])  # 有效四元数
                
                slerp = Slerp(valid_times, valid_rots)
                
                for idx in invalid_idxs:
                    if idx < valid_idxs[0]:  # 时间步小于第一个有效时间步，进行外推
                        pos_interp[b, idx, n, :] = quat_b_n[valid_idxs[0]]  # 复制第一个有效四元数
                    elif idx > valid_idxs[-1]:  # 时间步大于最后一个有效时间步，进行外推
                        pos_interp[b, idx, n, :] = quat_b_n[valid_idxs[-1]]  # 复制最后一个有效四元数
                    else:
                        interp_rot = slerp([idx])
                        pos_interp[b, idx, n, :] = interp_rot.as_rotvec()[0]
    # print("#######")
    # if N > 1:
    #     print(pos[1,0,11])
    #     print(pos_interp[1,0,11])
    
    return pos_interp

def slerp_interpolation_quat(pos, valid):

    # wxyz to xyzw
    pos = pos[:, :, :, [1, 2, 3, 0]]

    B, T, N, _ = pos.shape  # B: 批次大小, T: 时间步长, N: 关节数, 4: 四元数维度
    pos_interp = pos.copy()  # 创建副本以存储插值结果
    
    for b in range(B):
        for n in range(N):
            quat_b_n = pos[b, :, n, :]
            valid_b_n = valid[b, :]
            
            invalid_idxs = np.where(~valid_b_n)[0]
            valid_idxs = np.where(valid_b_n)[0]

            if len(invalid_idxs) == 0:
                continue
            
            if len(valid_idxs) > 1:
                valid_times = valid_idxs  # 有效时间步
                valid_rots = Rotation.from_quat(quat_b_n[valid_idxs])  # 有效四元数
                
                slerp = Slerp(valid_times, valid_rots)
                
                for idx in invalid_idxs:
                    if idx < valid_idxs[0]:  # 时间步小于第一个有效时间步，进行外推
                        pos_interp[b, idx, n, :] = quat_b_n[valid_idxs[0]]  # 复制第一个有效四元数
                    elif idx > valid_idxs[-1]:  # 时间步大于最后一个有效时间步，进行外推
                        pos_interp[b, idx, n, :] = quat_b_n[valid_idxs[-1]]  # 复制最后一个有效四元数
                    else:
                        interp_rot = slerp([idx])
                        pos_interp[b, idx, n, :] = interp_rot.as_quat()[0]
    
    # xyzw to wxyz
    pos_interp = pos_interp[:, :, :, [3, 0, 1, 2]]
    return pos_interp


def linear_interpolation_nd(pos, valid):
    B, T = pos.shape[:2]  # 取出批次大小B和时间步长T
    feature_dim = pos.shape[2]  # ** 代表的任意维度
    pos_interp = pos.copy()  # 创建一个副本，用来保存插值结果
    
    for b in range(B):
        for idx in range(feature_dim):  # 针对任意维度
            pos_b_idx = pos[b, :, idx]  # 取出第b批次对应的**维度下的一个时间序列
            valid_b = valid[b, :]  # 当前批次的有效标志
            
            # 找到无效的索引（False）
            invalid_idxs = np.where(~valid_b)[0]
            valid_idxs = np.where(valid_b)[0]

            if len(invalid_idxs) == 0:
                continue
            
            # 对无效部分进行线性插值
            if len(valid_idxs) > 1:  # 确保有足够的有效点用于插值
                pos_b_idx[invalid_idxs] = np.interp(invalid_idxs, valid_idxs, pos_b_idx[valid_idxs])
                pos_interp[b, :, idx] = pos_b_idx  # 保存插值结果
    
    return pos_interp

def world2canonical_convert(R_c2w_sla, t_c2w_sla, data_out, handedness):
    init_rot_mat = copy.deepcopy(data_out["init_root_orient"])
    init_rot_mat = torch.einsum("tij,btjk->btik", R_c2w_sla, init_rot_mat)
    init_rot = rotation_matrix_to_angle_axis(init_rot_mat)
    init_rot_quat = angle_axis_to_quaternion(init_rot)
    # data_out["init_root_orient"] = rotation_matrix_to_angle_axis(data_out["init_root_orient"])
    # data_out["init_hand_pose"] = rotation_matrix_to_angle_axis(data_out["init_hand_pose"])
    data_out_init_root_orient = rotation_matrix_to_angle_axis(data_out["init_root_orient"])
    data_out_init_hand_pose = rotation_matrix_to_angle_axis(data_out["init_hand_pose"])

    init_trans = data_out["init_trans"] # (B, T, 3)
    if handedness == "left":
        outputs = run_mano_left(data_out["init_trans"], data_out_init_root_orient, data_out_init_hand_pose, betas=data_out["init_betas"])

    elif handedness == "right":
        outputs = run_mano(data_out["init_trans"], data_out_init_root_orient, data_out_init_hand_pose, betas=data_out["init_betas"])
    root_loc = outputs["joints"][..., 0, :].cpu()  # (B, T, 3)
    offset = init_trans - root_loc  # It is a constant, no matter what the rotation is.
    init_trans = (
        torch.einsum("tij,btj->bti", R_c2w_sla, root_loc)
        + t_c2w_sla[None, :]
        + offset
    )

    data_world = {
        "init_root_orient": init_rot, # (B, T, 3)
        "init_hand_pose": data_out_init_hand_pose, # (B, T, 15, 3)
        "init_trans": init_trans,  # (B, T, 3)
        "init_betas": data_out["init_betas"]  # (B, T, 10)
    }

    return data_world

def filling_preprocess(item):

    num_joints = 15

    global_trans = item['trans'] # (2, seq_len, 3)
    global_rot = item['rot'] #(2, seq_len, 3)
    hand_pose = item['hand_pose'] # (2, seq_len, 45)
    betas = item['betas'] # (2, seq_len, 10)
    valid = item['valid'] # (2, seq_len)

    N, T, _ = global_trans.shape
    R_canonical2world_left_aa = torch.from_numpy(global_rot[0, 0])
    R_canonical2world_right_aa = torch.from_numpy(global_rot[1, 0])
    R_world2canonical_left = angle_axis_to_rotation_matrix(R_canonical2world_left_aa).t()
    R_world2canonical_right = angle_axis_to_rotation_matrix(R_canonical2world_right_aa).t()
    

    # transform left hand to canonical
    hand_pose = hand_pose.reshape(N, T, num_joints, 3)
    data_world_left = {
        "init_trans": torch.from_numpy(global_trans[0:1]),
        "init_root_orient": angle_axis_to_rotation_matrix(torch.from_numpy(global_rot[0:1])),
        "init_hand_pose": angle_axis_to_rotation_matrix(torch.from_numpy(hand_pose[0:1])),
        "init_betas": torch.from_numpy(betas[0:1]),
    }

    data_left_init_root_orient = rotation_matrix_to_angle_axis(data_world_left["init_root_orient"])
    data_left_init_hand_pose = rotation_matrix_to_angle_axis(data_world_left["init_hand_pose"])
    outputs = run_mano_left(data_world_left["init_trans"], data_left_init_root_orient, data_left_init_hand_pose, betas=data_world_left["init_betas"])
    init_trans = data_world_left["init_trans"][0, 0] # (3,)
    root_loc = outputs["joints"][0, 0, 0, :].cpu()  # (3,)
    offset = init_trans - root_loc  # It is a constant, no matter what the rotation is.
    t_world2canonical_left = -torch.einsum("ij,j->i", R_world2canonical_left, root_loc) - offset

    R_world2canonical_left = R_world2canonical_left.repeat(T, 1, 1)
    t_world2canonical_left = t_world2canonical_left.repeat(T, 1)
    data_canonical_left = world2canonical_convert(R_world2canonical_left, t_world2canonical_left, data_world_left, "left")

    # transform right hand to canonical
    data_world_right = {
        "init_trans": torch.from_numpy(global_trans[1:2]),
        "init_root_orient": angle_axis_to_rotation_matrix(torch.from_numpy(global_rot[1:2])),
        "init_hand_pose": angle_axis_to_rotation_matrix(torch.from_numpy(hand_pose[1:2])),
        "init_betas": torch.from_numpy(betas[1:2]),
    }

    data_right_init_root_orient = rotation_matrix_to_angle_axis(data_world_right["init_root_orient"])
    data_right_init_hand_pose = rotation_matrix_to_angle_axis(data_world_right["init_hand_pose"])
    outputs = run_mano(data_world_right["init_trans"], data_right_init_root_orient, data_right_init_hand_pose, betas=data_world_right["init_betas"])
    init_trans = data_world_right["init_trans"][0, 0] # (3,)
    root_loc = outputs["joints"][0, 0, 0, :].cpu()  # (3,)
    offset = init_trans - root_loc  # It is a constant, no matter what the rotation is.
    t_world2canonical_right = -torch.einsum("ij,j->i", R_world2canonical_right, root_loc) - offset

    R_world2canonical_right = R_world2canonical_right.repeat(T, 1, 1)
    t_world2canonical_right = t_world2canonical_right.repeat(T, 1)
    data_canonical_right = world2canonical_convert(R_world2canonical_right, t_world2canonical_right, data_world_right, "right")

    # merge left and right canonical data
    global_rot = torch.cat((data_canonical_left['init_root_orient'], data_canonical_right['init_root_orient']))
    global_trans = torch.cat((data_canonical_left['init_trans'], data_canonical_right['init_trans'])).numpy()
    
    # global_rot = angle_axis_to_quaternion(global_rot).numpy().reshape(N, T, 1, 4)
    global_rot = global_rot.reshape(N, T, 1, 3).numpy()
    
    hand_pose = hand_pose.reshape(N, T, 15, 3)
    # hand_pose = angle_axis_to_quaternion(torch.from_numpy(hand_pose)).numpy()

    # lerp and slerp
    global_trans_lerped = linear_interpolation_nd(global_trans, valid)
    betas_lerped = linear_interpolation_nd(betas, valid)
    global_rot_slerped = slerp_interpolation_aa(global_rot, valid)
    hand_pose_slerped = slerp_interpolation_aa(hand_pose, valid)
    

    # convert to rot6d

    global_rot_slerped_mat = angle_axis_to_rotation_matrix(torch.from_numpy(global_rot_slerped.reshape(N*T, -1)))
    # global_rot_slerped_mat = quaternion_to_rotation_matrix(torch.from_numpy(global_rot_slerped.reshape(N*T, -1)))
    global_rot_slerped_rot6d = rotmat_to_rot6d(global_rot_slerped_mat).reshape(N, T, -1).numpy()
    hand_pose_slerped_mat = angle_axis_to_rotation_matrix(torch.from_numpy(hand_pose_slerped.reshape(N*T*num_joints, -1)))
    # hand_pose_slerped_mat = quaternion_to_rotation_matrix(torch.from_numpy(hand_pose_slerped.reshape(N*T*num_joints, -1)))
    hand_pose_slerped_rot6d = rotmat_to_rot6d(hand_pose_slerped_mat).reshape(N, T, -1).numpy()


    # concat to (T, concat_dim)
    global_pose_vec_input = np.concatenate((global_trans_lerped, betas_lerped, global_rot_slerped_rot6d, hand_pose_slerped_rot6d), axis=-1).transpose(1, 0, 2).reshape(T, -1)

    R_canon2w_left = R_world2canonical_left.transpose(-1, -2)
    t_canon2w_left = -torch.einsum("tij,tj->ti", R_canon2w_left, t_world2canonical_left)
    R_canon2w_right = R_world2canonical_right.transpose(-1, -2)
    t_canon2w_right = -torch.einsum("tij,tj->ti", R_canon2w_right, t_world2canonical_right)

    transform_w_canon = {
        "R_w2canon_left": R_world2canonical_left,
        "t_w2canon_left": t_world2canonical_left,
        "R_canon2w_left": R_canon2w_left,
        "t_canon2w_left": t_canon2w_left,

        "R_w2canon_right": R_world2canonical_right,
        "t_w2canon_right": t_world2canonical_right,
        "R_canon2w_right": R_canon2w_right,
        "t_canon2w_right": t_canon2w_right,
    }
    
    return global_pose_vec_input, transform_w_canon

def custom_rot6d_to_rotmat(rot6d):
    original_shape = rot6d.shape[:-1]
    rot6d = rot6d.reshape(-1, 6)
    mat = rot6d_to_rotmat(rot6d)
    mat = mat.reshape(*original_shape, 3, 3)
    return mat

def filling_postprocess(output, transform_w_canon):
    # output = output.numpy()
    output = output.permute(1, 0, 2) # (2, T, -1)
    N, T, _ = output.shape
    canon_trans = output[:, :, :3]
    betas = output[:, :, 3:13]
    canon_rot_rot6d = output[:, :, 13:19]
    hand_pose_rot6d = output[:, :, 19:109].reshape(N, T, 15, 6)

    canon_rot_mat = custom_rot6d_to_rotmat(canon_rot_rot6d)
    hand_pose_mat = custom_rot6d_to_rotmat(hand_pose_rot6d)

    data_canonical_left = {
        "init_trans": canon_trans[[0], :, :],
        "init_root_orient": canon_rot_mat[[0], :, :, :],
        "init_hand_pose": hand_pose_mat[[0], :, :, :, :],
        "init_betas": betas[[0], :, :]
    }

    data_canonical_right = {
        "init_trans": canon_trans[[1], :, :],
        "init_root_orient": canon_rot_mat[[1], :, :, :],
        "init_hand_pose": hand_pose_mat[[1], :, :, :, :],
        "init_betas": betas[[1], :, :]
    }

    R_canon2w_left = transform_w_canon['R_canon2w_left']
    t_canon2w_left = transform_w_canon['t_canon2w_left']
    R_canon2w_right = transform_w_canon['R_canon2w_right']
    t_canon2w_right = transform_w_canon['t_canon2w_right']


    world_left = world2canonical_convert(R_canon2w_left, t_canon2w_left, data_canonical_left, "left")
    world_right = world2canonical_convert(R_canon2w_right, t_canon2w_right, data_canonical_right, "right")

    global_rot = torch.cat((world_left['init_root_orient'], world_right['init_root_orient'])).numpy()
    global_trans = torch.cat((world_left['init_trans'], world_right['init_trans'])).numpy()

    pred_data = {
        "trans": global_trans, # (2, T, 3)
        "rot": global_rot, # (2, T, 3)
        "hand_pose": rotation_matrix_to_angle_axis(hand_pose_mat).flatten(-2).numpy(), # (2, T, 45)
        "betas": betas.numpy(), # (2, T, 10)
    }

    return pred_data

