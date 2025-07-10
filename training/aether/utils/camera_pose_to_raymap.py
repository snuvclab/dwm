import argparse
import numpy as np
from postprocess_utils import camera_pose_to_raymap

def main(args):
    Rt = np.load(args.Rt)
    K = np.load(args.K)

    raymap = camera_pose_to_raymap(Rt, np.tile(K, (len(Rt), 1, 1)), ray_o_scale_factor=10.0)
    np.save("raymap.npy", raymap)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert camera pose to raymap.")
    parser.add_argument("--Rt", type=str, required=True, help="Path to the trajectory file.")
    parser.add_argument("--K", type=str, required=True, help="Path to the intrinsics file.")
    args = parser.parse_args()

    main(args)