"""
Try out for joint rotation format change
"""
import pickle
import smplx
from pathlib import Path

import torch
# import open3d as o3d
#from smplx_optims import *
from scipy.spatial.transform import Rotation as rot
from pytorch3d.transforms import matrix_to_axis_angle, rotation_6d_to_matrix, axis_angle_to_matrix, matrix_to_euler_angles, \
                                euler_angles_to_matrix, quaternion_to_matrix
import numpy as np
import trimesh

"""
 For simple visualization, change it to batched setting
 1. Rotation axis change
 2. global => local rotation change (축의 이동 고려)
 3. transition to axis-angle rotation

    # xsens_rel_rotmat = torch.cat([
    #             smplx_glb_rotmat[:, [0]],
    #             recursive_mult(smplx_glb_rotmat, smplx_glb_rotmat_inv, [1,0]),
    #             recursive_mult(smplx_glb_rotmat, smplx_glb_rotmat_inv, [2,1,0]),
    #             recursive_mult(smplx_glb_rotmat, smplx_glb_rotmat_inv, [3,2,1,0]),
    #             recursive_mult(smplx_glb_rotmat, smplx_glb_rotmat_inv, [4,3,2,1,0]),
    #             recursive_mult(smplx_glb_rotmat, smplx_glb_rotmat_inv, [5,4,3,2,1,0]),
    #             recursive_mult(smplx_glb_rotmat, smplx_glb_rotmat_inv, [6,5,4,3,2,1,0]),
    #             recursive_mult(smplx_glb_rotmat, smplx_glb_rotmat_inv, [7,5,4,3,2,1,0]),
    #             recursive_mult(smplx_glb_rotmat, smplx_glb_rotmat_inv, [8,7,5,4,3,2,1,0]),
    #             recursive_mult(smplx_glb_rotmat, smplx_glb_rotmat_inv, [9,8,7,5,4,3,2,1,0]),
    #             recursive_mult(smplx_glb_rotmat, smplx_glb_rotmat_inv, [10,9,8,7,5,4,3,2,1,0]),
    #             recursive_mult(smplx_glb_rotmat, smplx_glb_rotmat_inv, [11,5,4,3,2,1,0]),
    #             recursive_mult(smplx_glb_rotmat, smplx_glb_rotmat_inv, [12,11,5,4,3,2,1,0]),
    #             recursive_mult(smplx_glb_rotmat, smplx_glb_rotmat_inv, [13,12,11,5,4,3,2,1,0]),
    #             recursive_mult(smplx_glb_rotmat, smplx_glb_rotmat_inv, [14,13,12,11,5,4,3,2,1,0]),
    #             recursive_mult(smplx_glb_rotmat, smplx_glb_rotmat_inv, [15,0]),
    #             recursive_mult(smplx_glb_rotmat, smplx_glb_rotmat_inv, [16,15,0]),
    #             recursive_mult(smplx_glb_rotmat, smplx_glb_rotmat_inv, [17,16,15,0]),
    #             recursive_mult(smplx_glb_rotmat, smplx_glb_rotmat_inv, [18,17,16,15,0]),
    #             recursive_mult(smplx_glb_rotmat, smplx_glb_rotmat_inv, [19,0]),
    #             recursive_mult(smplx_glb_rotmat, smplx_glb_rotmat_inv, [20,19,0]),
    #             recursive_mult(smplx_glb_rotmat, smplx_glb_rotmat_inv, [21,20,19,0]),
    #             recursive_mult(smplx_glb_rotmat, smplx_glb_rotmat_inv, [22,21,20,19,0]),
    #             ], axis=1)

"""



def xsens2smplx_body(xsens_tensor):
    """ Body """
    #xsens_tensor = torch.FloatTensor(xsens_rotation_arr)
    # Neck, Head adjustment 
    # neck_euler = matrix_to_euler_angles(rotation_6d_to_matrix(xsens_tensor), "XYZ")
    xsens_tensor = torch.Tensor(xsens_tensor)
    xsens_tensor = quaternion_to_matrix(xsens_tensor) # convert to rotation matrix
    xsens_tensor[:,5,1] -= 0.05 # adjust neck
    xsens_tensor[:,6,1] += 0.05 # adjust head
    # xsens_tensor = euler_angles_to_matrix(xsens_tensor, "ZXY")

    xsens_rotvec = matrix_to_axis_angle(xsens_tensor) #rotation_6d_to_matrix(xsens_tensor))
    # global rotation, local frame unchanged
    smplx_glb_rotvec = torch.cat([xsens_rotvec[:,:,[1]], xsens_rotvec[:,:,[2]], xsens_rotvec[:,:,[0]]], dim=2)
    smplx_glb_rotmat = axis_angle_to_matrix(smplx_glb_rotvec)
    smplx_glb_rotmat_inv = torch.linalg.inv(smplx_glb_rotmat) # inversed rotmat
    xsens_rel_rotmat = torch.cat([
                torch.einsum('FJMN, FJNK -> FJMK', smplx_glb_rotmat_inv[:,[0]], smplx_glb_rotmat[:,[19]]),
                torch.einsum('FJMN, FJNK -> FJMK', smplx_glb_rotmat_inv[:,[0]], smplx_glb_rotmat[:,[15]]),                
                torch.einsum('FJMN, FJNK -> FJMK', smplx_glb_rotmat_inv[:,[0]], smplx_glb_rotmat[:,[1]]),                
                torch.einsum('FJMN, FJNK -> FJMK', smplx_glb_rotmat_inv[:,[19]], smplx_glb_rotmat[:,[20]]),
                torch.einsum('FJMN, FJNK -> FJMK', smplx_glb_rotmat_inv[:,[15]], smplx_glb_rotmat[:,[16]]),
                torch.einsum('FJMN, FJNK -> FJMK', smplx_glb_rotmat_inv[:,[1]], smplx_glb_rotmat[:,[3]]), # spine2 @ spine1 inv 
                torch.einsum('FJMN, FJNK -> FJMK', smplx_glb_rotmat_inv[:,[20]], smplx_glb_rotmat[:,[21]]), # left_ankle
                torch.einsum('FJMN, FJNK -> FJMK', smplx_glb_rotmat_inv[:,[16]], smplx_glb_rotmat[:,[17]]),     
                torch.einsum('FJMN, FJNK -> FJMK', smplx_glb_rotmat_inv[:,[3]], smplx_glb_rotmat[:,[4]]), # spine3
                torch.einsum('FJMN, FJNK -> FJMK', smplx_glb_rotmat_inv[:,[21]], smplx_glb_rotmat[:,[22]] ,), # left_foot
                torch.einsum('FJMN, FJNK -> FJMK', smplx_glb_rotmat_inv[:,[17]], smplx_glb_rotmat[:,[18]]),
                torch.einsum('FJMN, FJNK -> FJMK', smplx_glb_rotmat_inv[:,[4]], smplx_glb_rotmat[:,[5]]), # neck
                torch.einsum('FJMN, FJNK -> FJMK', smplx_glb_rotmat_inv[:,[4]], smplx_glb_rotmat[:,[11]]), # left collar
                torch.einsum('FJMN, FJNK -> FJMK', smplx_glb_rotmat_inv[:,[4]], smplx_glb_rotmat[:,[7]]),
                torch.einsum('FJMN, FJNK -> FJMK', smplx_glb_rotmat_inv[:,[5]], smplx_glb_rotmat[:,[6]]), # head
                torch.einsum('FJMN, FJNK -> FJMK', smplx_glb_rotmat_inv[:,[11]], smplx_glb_rotmat[:,[12]]), # left_shoulder
                torch.einsum('FJMN, FJNK -> FJMK', smplx_glb_rotmat_inv[:,[7]], smplx_glb_rotmat[:,[8]]),
                torch.einsum('FJMN, FJNK -> FJMK', smplx_glb_rotmat_inv[:,[12]], smplx_glb_rotmat[:,[13]]), # left elbow
                torch.einsum('FJMN, FJNK -> FJMK', smplx_glb_rotmat_inv[:,[8]], smplx_glb_rotmat[:,[9]]),
                torch.einsum('FJMN, FJNK -> FJMK', smplx_glb_rotmat_inv[:,[13]], smplx_glb_rotmat[:,[14]]), # left wrist
                torch.einsum('FJMN, FJNK -> FJMK', smplx_glb_rotmat_inv[:,[9]], smplx_glb_rotmat[:,[10]])
                ], axis=1)
    xsens_rel_rotvec = matrix_to_axis_angle(xsens_rel_rotmat)    
    return xsens_rel_rotvec    


def xsens2smplx(xsens_rotation_arr, manus_hand_arr):
    """
    xsens_rotation_arr : (F, J, 6) array of rotation 6D parameters
    """
    # def recursive_mult(glb_rotmat, glb_rotmat_inv, target_indices):
    #     """
    #     Last of target_indices is target joint rotation
    #     """
    #     if len(target_indices) == 1:
    #         return glb_rotmat[:, [target_indices[0]]]
    #     else:
    #         idx = target_indices.pop(-1)
    #         relative_rot = torch.einsum('FJMN, FJNK -> FJMK',
    #                                     recursive_mult(glb_rotmat, glb_rotmat_inv, target_indices),
    #                                     glb_rotmat_inv[:, [idx]])
    #         return relative_rot
    """ Body """
    Framenum = xsens_rotation_arr.shape[0]
    xsens_tensor = torch.FloatTensor(xsens_rotation_arr)
    # Neck, Head adjustment 
    neck_euler = matrix_to_euler_angles(rotation_6d_to_matrix(xsens_tensor), "XYZ")
    neck_euler[:,5,1] -= 0.05 # adjust neck
    neck_euler[:,6,1] += 0.05 # adjust head
    xsens_tensor = euler_angles_to_matrix(neck_euler, "XYZ")

    xsens_rotvec = matrix_to_axis_angle(xsens_tensor) #rotation_6d_to_matrix(xsens_tensor))
    # global rotation, local frame unchanged
    smplx_glb_rotvec = torch.cat([xsens_rotvec[:,:,[1]], xsens_rotvec[:,:,[2]], xsens_rotvec[:,:,[0]]], dim=2)
    smplx_glb_rotmat = axis_angle_to_matrix(smplx_glb_rotvec)
    smplx_glb_rotmat_inv = torch.linalg.inv(smplx_glb_rotmat) # inversed rotmat

    """ Hand """
    manus_tensor = torch.FloatTensor(manus_hand_arr)
    manus_rotvec = matrix_to_axis_angle(rotation_6d_to_matrix(manus_tensor))
    # global rotation, local frame unchanged
    smplx_hand_glb_rotvec = torch.cat([manus_rotvec[:,:,[1]], manus_rotvec[:,:,[2]], manus_rotvec[:,:,[0]]], dim=2)
    smplx_hand_glb_rotmat = axis_angle_to_matrix(smplx_hand_glb_rotvec)
    smplx_hand_glb_rotmat_inv = torch.linalg.inv(smplx_hand_glb_rotmat) # inversed rotmat

    """
    reform to relative rotations (predefined with indices)
    xsens 연결 기준으로 생각하지 말고, smplx 연결 가지고 relative rotation 구하기

    # hand 고치기
    """
    xsens_rel_rotmat = torch.cat([
                torch.einsum('FJMN, FJNK -> FJMK', smplx_glb_rotmat_inv[:,[0]], smplx_glb_rotmat[:,[19]]),
                torch.einsum('FJMN, FJNK -> FJMK', smplx_glb_rotmat_inv[:,[0]], smplx_glb_rotmat[:,[15]]),                
                torch.einsum('FJMN, FJNK -> FJMK', smplx_glb_rotmat_inv[:,[0]], smplx_glb_rotmat[:,[1]]),                
                torch.einsum('FJMN, FJNK -> FJMK', smplx_glb_rotmat_inv[:,[19]], smplx_glb_rotmat[:,[20]]),
                torch.einsum('FJMN, FJNK -> FJMK', smplx_glb_rotmat_inv[:,[15]], smplx_glb_rotmat[:,[16]]),
                torch.einsum('FJMN, FJNK -> FJMK', smplx_glb_rotmat_inv[:,[1]], smplx_glb_rotmat[:,[3]]), # spine2 @ spine1 inv 
                torch.einsum('FJMN, FJNK -> FJMK', smplx_glb_rotmat_inv[:,[20]], smplx_glb_rotmat[:,[21]]), # left_ankle
                torch.einsum('FJMN, FJNK -> FJMK', smplx_glb_rotmat_inv[:,[16]], smplx_glb_rotmat[:,[17]]),     
                torch.einsum('FJMN, FJNK -> FJMK', smplx_glb_rotmat_inv[:,[3]], smplx_glb_rotmat[:,[4]]), # spine3
                torch.einsum('FJMN, FJNK -> FJMK', smplx_glb_rotmat_inv[:,[21]], smplx_glb_rotmat[:,[22]] ,), # left_foot
                torch.einsum('FJMN, FJNK -> FJMK', smplx_glb_rotmat_inv[:,[17]], smplx_glb_rotmat[:,[18]]),
                torch.einsum('FJMN, FJNK -> FJMK', smplx_glb_rotmat_inv[:,[4]], smplx_glb_rotmat[:,[5]]), # neck
                torch.einsum('FJMN, FJNK -> FJMK', smplx_glb_rotmat_inv[:,[4]], smplx_glb_rotmat[:,[11]]), # left collar
                torch.einsum('FJMN, FJNK -> FJMK', smplx_glb_rotmat_inv[:,[4]], smplx_glb_rotmat[:,[7]]),
                torch.einsum('FJMN, FJNK -> FJMK', smplx_glb_rotmat_inv[:,[5]], smplx_glb_rotmat[:,[6]]), # head
                torch.einsum('FJMN, FJNK -> FJMK', smplx_glb_rotmat_inv[:,[11]], smplx_glb_rotmat[:,[12]]), # left_shoulder
                torch.einsum('FJMN, FJNK -> FJMK', smplx_glb_rotmat_inv[:,[7]], smplx_glb_rotmat[:,[8]]),
                torch.einsum('FJMN, FJNK -> FJMK', smplx_glb_rotmat_inv[:,[12]], smplx_glb_rotmat[:,[13]]), # left elbow
                torch.einsum('FJMN, FJNK -> FJMK', smplx_glb_rotmat_inv[:,[8]], smplx_glb_rotmat[:,[9]]),
                torch.einsum('FJMN, FJNK -> FJMK', smplx_glb_rotmat_inv[:,[13]], smplx_glb_rotmat[:,[14]]), # left wrist
                torch.einsum('FJMN, FJNK -> FJMK', smplx_glb_rotmat_inv[:,[9]], smplx_glb_rotmat[:,[10]])
                ], axis=1)
    # hand relative from 
    # identity_matrix = torch.eye(3)
    # eyemat = identity_matrix.unsqueeze(0).unsqueeze(0).expand(Framenum, 1, 3, 3)    
    # manus_lhand_rel = torch.cat([
    #             #smplx_hand_glb_rotmat[:,[15]],
    #             eyemat,
    #             smplx_hand_glb_rotmat[:,[16]], # index 2 
    #             #torch.einsum('FJMN, FJNK -> FJMK', smplx_hand_glb_rotmat_inv[:,[15]], smplx_hand_glb_rotmat[:,[16]]), # index 2 
    #             torch.einsum('FJMN, FJNK -> FJMK', smplx_hand_glb_rotmat_inv[:,[16]], smplx_hand_glb_rotmat[:,[17]]),
    #             #smplx_hand_glb_rotmat[:,[12]],                
    #             eyemat,
    #             smplx_hand_glb_rotmat[:,[13]],
    #             #torch.einsum('FJMN, FJNK -> FJMK', smplx_hand_glb_rotmat_inv[:,[12]], smplx_hand_glb_rotmat[:,[13]]),
    #             torch.einsum('FJMN, FJNK -> FJMK', smplx_hand_glb_rotmat_inv[:,[13]], smplx_hand_glb_rotmat[:,[14]]), 
    #             eyemat,
    #             #smplx_hand_glb_rotmat[:,[6]],
    #             smplx_hand_glb_rotmat[:,[7]],
    #             #torch.einsum('FJMN, FJNK -> FJMK', smplx_hand_glb_rotmat_inv[:,[6]], smplx_hand_glb_rotmat[:,[7]]),     
    #             torch.einsum('FJMN, FJNK -> FJMK', smplx_hand_glb_rotmat_inv[:,[7]], smplx_hand_glb_rotmat[:,[8]]),
    #             eyemat,
    #             #smplx_hand_glb_rotmat[:,[9]], 
    #             smplx_hand_glb_rotmat[:,[10]],
    #             #torch.einsum('FJMN, FJNK -> FJMK', smplx_hand_glb_rotmat_inv[:,[9]], smplx_hand_glb_rotmat[:,[10]]),
    #             torch.einsum('FJMN, FJNK -> FJMK', smplx_hand_glb_rotmat_inv[:,[10]], smplx_hand_glb_rotmat[:,[11]]),
    #             eyemat,
    #             #smplx_hand_glb_rotmat[:,[1]],
    #             smplx_hand_glb_rotmat[:,[18]],
    #             #torch.einsum('FJMN, FJNK -> FJMK', smplx_hand_glb_rotmat_inv[:,[1]], smplx_hand_glb_rotmat[:,[18]]),
    #             torch.einsum('FJMN, FJNK -> FJMK', smplx_hand_glb_rotmat_inv[:,[18]], smplx_hand_glb_rotmat[:,[19]]),
    #             ], axis=1)
    manus_lhand_rel = torch.cat([
                smplx_hand_glb_rotmat[:,[15]], # SecondMCP
                torch.einsum('FJMN, FJNK -> FJMK', smplx_hand_glb_rotmat_inv[:,[15]], smplx_hand_glb_rotmat[:,[16]]), # index 2 
                torch.einsum('FJMN, FJNK -> FJMK', smplx_hand_glb_rotmat_inv[:,[16]], smplx_hand_glb_rotmat[:,[17]]),
                smplx_hand_glb_rotmat[:,[12]], # FifthMCP                
                torch.einsum('FJMN, FJNK -> FJMK', smplx_hand_glb_rotmat_inv[:,[12]], smplx_hand_glb_rotmat[:,[13]]),
                torch.einsum('FJMN, FJNK -> FJMK', smplx_hand_glb_rotmat_inv[:,[13]], smplx_hand_glb_rotmat[:,[14]]), 
                smplx_hand_glb_rotmat[:,[6]], # FifthMCP
                torch.einsum('FJMN, FJNK -> FJMK', smplx_hand_glb_rotmat_inv[:,[6]], smplx_hand_glb_rotmat[:,[7]]),     
                torch.einsum('FJMN, FJNK -> FJMK', smplx_hand_glb_rotmat_inv[:,[7]], smplx_hand_glb_rotmat[:,[8]]),
                smplx_hand_glb_rotmat[:,[9]], # FourthMCP
                torch.einsum('FJMN, FJNK -> FJMK', smplx_hand_glb_rotmat_inv[:,[9]], smplx_hand_glb_rotmat[:,[10]]),
                torch.einsum('FJMN, FJNK -> FJMK', smplx_hand_glb_rotmat_inv[:,[10]], smplx_hand_glb_rotmat[:,[11]]),
                smplx_hand_glb_rotmat[:,[1]], # firstCMC
                torch.einsum('FJMN, FJNK -> FJMK', smplx_hand_glb_rotmat_inv[:,[1]], smplx_hand_glb_rotmat[:,[18]]),
                torch.einsum('FJMN, FJNK -> FJMK', smplx_hand_glb_rotmat_inv[:,[18]], smplx_hand_glb_rotmat[:,[19]]),
                ], axis=1)    
    rhand_glb_rotmat = smplx_hand_glb_rotmat[:,20:]
    rhand_glb_rotmat_inv = smplx_hand_glb_rotmat_inv[:,20:]

    manus_rhand_rel = torch.cat([
                rhand_glb_rotmat[:,[15]],
                #torch.einsum('FJMN, FJNK -> FJMK', smplx_glb_rotmat_inv[:,[10]], rhand_glb_rotmat[:,[15]]), # index 1
                torch.einsum('FJMN, FJNK -> FJMK', rhand_glb_rotmat_inv[:,[15]], rhand_glb_rotmat[:,[16]]),
                torch.einsum('FJMN, FJNK -> FJMK', rhand_glb_rotmat_inv[:,[16]], rhand_glb_rotmat[:,[17]]),
                rhand_glb_rotmat[:,[12]],                
                #torch.einsum('FJMN, FJNK -> FJMK', smplx_glb_rotmat_inv[:,[10]], rhand_glb_rotmat[:,[12]]), # middle 1 
                torch.einsum('FJMN, FJNK -> FJMK', rhand_glb_rotmat_inv[:,[12]], rhand_glb_rotmat[:,[13]]),
                torch.einsum('FJMN, FJNK -> FJMK', rhand_glb_rotmat_inv[:,[13]], rhand_glb_rotmat[:,[14]]),
                rhand_glb_rotmat[:,[6]], 
                #torch.einsum('FJMN, FJNK -> FJMK', smplx_glb_rotmat_inv[:,[10]], rhand_glb_rotmat[:,[6]]), # pinky 1
                torch.einsum('FJMN, FJNK -> FJMK', rhand_glb_rotmat_inv[:,[6]], rhand_glb_rotmat[:,[7]]),     
                torch.einsum('FJMN, FJNK -> FJMK', rhand_glb_rotmat_inv[:,[7]], rhand_glb_rotmat[:,[8]]), 
                rhand_glb_rotmat[:,[9]],
                #torch.einsum('FJMN, FJNK -> FJMK', smplx_glb_rotmat_inv[:,[10]], rhand_glb_rotmat[:,[9]] ,), # ring 1 
                torch.einsum('FJMN, FJNK -> FJMK', rhand_glb_rotmat_inv[:,[9]], rhand_glb_rotmat[:,[10]]),
                torch.einsum('FJMN, FJNK -> FJMK', rhand_glb_rotmat_inv[:,[10]], rhand_glb_rotmat[:,[11]]),
                rhand_glb_rotmat[:,[1]],
                #torch.einsum('FJMN, FJNK -> FJMK', smplx_glb_rotmat_inv[:,[10]], rhand_glb_rotmat[:,[1]]), # thumb 1
                torch.einsum('FJMN, FJNK -> FJMK', rhand_glb_rotmat_inv[:,[1]], rhand_glb_rotmat[:,[18]]),
                torch.einsum('FJMN, FJNK -> FJMK', rhand_glb_rotmat_inv[:,[18]], rhand_glb_rotmat[:,[19]]),
                ], axis=1)


    xsens_rel_rotvec = matrix_to_axis_angle(xsens_rel_rotmat)
    manus_lhand_rel_rotvec = matrix_to_axis_angle(manus_lhand_rel)
    manus_rhand_rel_rotvec = matrix_to_axis_angle(manus_rhand_rel)

    return xsens_rel_rotvec, manus_lhand_rel_rotvec, manus_rhand_rel_rotvec # smplx_glb_rotvec


def manus2smplxhand(manus_tensor):
    manus_rotvec = matrix_to_axis_angle(rotation_6d_to_matrix(manus_tensor))
    # global rotation, local frame unchanged
    smplx_hand_glb_rotvec = torch.cat([manus_rotvec[:,:,[1]], manus_rotvec[:,:,[2]], manus_rotvec[:,:,[0]]], dim=2)

    # smplx_hand_glb_rotvec[:,12,1] -= 0.13
    # smplx_hand_glb_rotvec[:,6,1] -= 0.6
    # smplx_hand_glb_rotvec[:,9,1] -= -0.3

    # smplx_hand_glb_rotvec[:,12,0] = 0.
    # smplx_hand_glb_rotvec[:,6,0] = 0.
    # smplx_hand_glb_rotvec[:,9,0] = 0.
    #smplx_hand_glb_rotvec[:,1,1] -= 40*np.pi/180

    smplx_hand_glb_rotmat = axis_angle_to_matrix(smplx_hand_glb_rotvec)
    smplx_hand_glb_rotmat_inv = torch.linalg.inv(smplx_hand_glb_rotmat) # inversed rotmat    
    manus_hand_rel = torch.cat([
                smplx_hand_glb_rotmat[:,[15]], # SecondMCP
                torch.einsum('FJMN, FJNK -> FJMK', smplx_hand_glb_rotmat_inv[:,[15]], smplx_hand_glb_rotmat[:,[16]]), # index 2 
                torch.einsum('FJMN, FJNK -> FJMK', smplx_hand_glb_rotmat_inv[:,[16]], smplx_hand_glb_rotmat[:,[17]]),
                smplx_hand_glb_rotmat[:,[12]], # ThirdMCP                
                torch.einsum('FJMN, FJNK -> FJMK', smplx_hand_glb_rotmat_inv[:,[12]], smplx_hand_glb_rotmat[:,[13]]),
                torch.einsum('FJMN, FJNK -> FJMK', smplx_hand_glb_rotmat_inv[:,[13]], smplx_hand_glb_rotmat[:,[14]]), 
                smplx_hand_glb_rotmat[:,[6]], # FifthMCP
                torch.einsum('FJMN, FJNK -> FJMK', smplx_hand_glb_rotmat_inv[:,[6]], smplx_hand_glb_rotmat[:,[7]]),     
                torch.einsum('FJMN, FJNK -> FJMK', smplx_hand_glb_rotmat_inv[:,[7]], smplx_hand_glb_rotmat[:,[8]]),
                smplx_hand_glb_rotmat[:,[9]], # FourthMCP
                torch.einsum('FJMN, FJNK -> FJMK', smplx_hand_glb_rotmat_inv[:,[9]], smplx_hand_glb_rotmat[:,[10]]),
                torch.einsum('FJMN, FJNK -> FJMK', smplx_hand_glb_rotmat_inv[:,[10]], smplx_hand_glb_rotmat[:,[11]]),
                smplx_hand_glb_rotmat[:,[1]], # firstCMC
                torch.einsum('FJMN, FJNK -> FJMK', smplx_hand_glb_rotmat_inv[:,[1]], smplx_hand_glb_rotmat[:,[18]]),
                torch.einsum('FJMN, FJNK -> FJMK', smplx_hand_glb_rotmat_inv[:,[18]], smplx_hand_glb_rotmat[:,[19]]),
                ], axis=1) 
    smplx_hand_rel_rotvec = matrix_to_axis_angle(manus_hand_rel)
    return smplx_hand_rel_rotvec


if __name__ == "__main__":
    xsens_data = np.load('part_qwxyz.npy')
    xsens_data = xsens_data.reshape(-1, 4)
    # xsens_data = np.concatenate([np.array([[0, 0, 0]]), xsens_data], axis=0)  # add zero global Rt

    smplx_params = xsens2smplx_body(xsens_data[None])
    smplx_model = smplx.SMPLX("data_processing/aria/smpl_files/smplx/SMPLX_NEUTRAL.npz")
    smplx_output = smplx_model(betas=torch.zeros(1, 10), global_orient=torch.zeros(1, 3), body_pose=smplx_params)
    trimesh.Trimesh(vertices=smplx_output.vertices[0].detach().numpy(), faces=smplx_model.faces).export("test.obj")
    pass