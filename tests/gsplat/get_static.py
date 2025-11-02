import os
import argparse
from pathlib import Path
import json


parser = argparse.ArgumentParser()
parser.add_argument('--vrs_file', type=str, required=True)
parser.add_argument('--num_frames', type=int, default=100)
args = parser.parse_args()

# # data processing
# vrs_path = Path(args.vrs_file)
# mps_data_dir = vrs_path.parent / "mps" / "slam"
# data_dir = vrs_path.parent / "gsplat" / "data"
# output_dir = vrs_path.parent / "gsplat" / "output"
# data_dir.mkdir(parents=True, exist_ok=True)
# output_dir.mkdir(parents=True, exist_ok=True)
# os.system(f"conda activate nerfstudio; ns-process-data aria --vrs-file {vrs_path} --mps-data-dir {mps_data_dir} --output-dir {data_dir} --max-frames 10000")

# edit transforms.json to use only the former frames
transforms = json.load(open("/media/taeksoo/HDD3/aria/test_bj_coke/gsplat/data/transforms.json", 'r'))
transforms['frames'] = transforms['frames'][:args.num_frames]
with open(data_dir / "transforms.json", 'w') as f:
    json.dump(transforms, f, indent=4)

# perform gsplat
os.system(f"ns-train splatfacto --data {data_dir} --output-dir {output_dir} --train-split-fraction 1")

pass