# Data Processing Pipeline

This repository contains utilities and scripts for processing visual and motion data to prepare it for model training.

## Steps

1. **Generate sequences from frames**
   - Script: `python data_processing/make_sequences.py --data_root DATA_ROOT`
   - Function: Constructs video/trajectory/motion sequences from:
     - Image frames (required)
     - Disparity maps (not required)
     - Camera poses (not required)
     - Human poses (not required)

    ```
    DATA_ROOT/
    ├── image/
    │ └── 00000.png
    ├── disparity/
    │ └── 00000.png
    ├── cam_param/
    │ └── 00000.npy
    ├── human_pose/
    │ └── 00000.npy
    ```
2. **Estimate disparity if needed**
   - Script: `python data_processing/DepthAnyVideo/run_infer.py --data_path PATH_TO_VIDEOS_FOLDER`
   - Function: Estimate disparity from rgb videos.
   - Requirements: Create a new env using "data_processing/DepthAnyVideo/requirements.txt"

3. **Convert camera pose to raymaps**
   - Script: `python training/aether/utils/camera_pose_to_raymap.py --data_root DATA_ROOT`
   - Function: Transforms Fx4x4 camera trajectories (world2cam) into Fx6xHxW raymaps.
   - Requirements: "Path(args.data_root) / "cam_params" / "intrinsics.npy" (3x3) should exist.

4. **Encode data into latents for training**
   - Script: `training/prepare_dataset.sh`
   - Function: Processes and encodes the data into latent representations for more efficient training.

   ```
   DATA_ROOT/
    ├── videos.txt (lists videos for training videos/00000.mp4)
    ├── prompts.txt (lists prompts for training, should have equal number of lines with videos.txt)
    ├── videos/
    │ └── 00000.mp4
    ├── disparity/
    │ └── 00000.mp4
    ├── raymaps/
    │ └── 00000.npy
    ├── human_motions/
    │ └── 00000.npy
    ```

## Notes

- All scripts are designed for internal use and may assume specific data formats or folder structures.
- Please align on dependencies and environments before running any scripts.
