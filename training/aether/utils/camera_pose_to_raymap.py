import argparse
import numpy as np
import os
from tqdm import tqdm
from postprocess_utils import camera_pose_to_raymap

def main(args):
    ext_dir = os.path.join(args.data_root, "trajectories")
    int_path = os.path.join(args.data_root, "cam_params", "intrinsics.npy")
    disp_dir = os.path.join(args.data_root, "disparity")

    out_dir = os.path.join(args.data_root, "raymaps")
    os.makedirs(out_dir, exist_ok=True)

    K = np.load(int_path)

    for traj in tqdm(sorted(os.listdir(ext_dir))):
        Rt = np.load(os.path.join(ext_dir, traj))

        idx = traj[:5] + ".npy"
        disparity = np.load(os.path.join(disp_dir, idx))
        dmax = disparity.max()
        print(dmax)

        raymap = camera_pose_to_raymap(Rt, np.tile(K, (len(Rt), 1, 1)), ray_o_scale_factor=10.0, dmax=dmax)
        # print(os.path.join(out_dir, idx))
        np.save(os.path.join(out_dir, traj), raymap)
        
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert camera pose to raymap.")
    parser.add_argument("--data_root", type=str, required=True, help="Data root dir.")
    args = parser.parse_args()

    main(args)