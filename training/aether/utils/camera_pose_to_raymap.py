import argparse
import numpy as np
import os
from tqdm import tqdm
from postprocess_utils import camera_pose_to_raymap
from pathlib import Path

def main(args):
    ext_dir = Path(args.data_root) / "sequences" / "trajectory"  # Load Fx4x4 extrinsics
    int_path = Path(args.data_root) / "cam_params" / "intrinsics.npy"  # Load 3x3 intrinsic parameters
    disp_dir = Path(args.data_root) / "sequences" / "disparity"

    out_dir = Path(args.data_root) / "sequences" / "raymaps"
    out_dir.mkdir(parents=True, exist_ok=True)

    K = np.load(int_path)



    for traj in tqdm(sorted(ext_dir.glob("*.npy"))):
        Rt = np.load(traj)

        name = traj.stem.replace("_abs", "")
        disparity = np.load(disp_dir / f"{name}.npy")
        dmax = disparity.max()

        raymap = camera_pose_to_raymap(Rt, np.tile(K, (len(Rt), 1, 1)), ray_o_scale_factor=10.0, dmax=dmax)
        np.save(out_dir / f"{traj.stem}.npy", raymap)

        break
        
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert camera pose to raymap.")
    parser.add_argument("--data_root", type=str, required=True, help="Data root dir.")
    args = parser.parse_args()

    main(args)