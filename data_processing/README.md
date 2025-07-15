# Data Processing Pipeline

This repository contains utilities and scripts for processing visual and motion data to prepare it for model training.

## Overview

1. **Generate motion sequences**
   - Script: `python data_processing/make_sequences.py --data_root DATA_ROOT`
   - Function: Constructs video/trajectory/motion sequences from:
     - Image frames
     - Disparity maps
     - Camera poses
     - Human poses

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

2. **Convert camera pose to raymaps**
   - Script: `training/aether/utils/camera_pose_to_raymap.py`
   - Function: Transforms 6-DoF camera poses into per-pixel ray direction maps.

3. **Encode data into latents for training**
   - Script: `training/prepare_dataset.sh`
   - Function: Processes and encodes the data into latent representations for more efficient training.

## Notes

- All scripts are designed for internal use and may assume specific data formats or folder structures.
- Please align on dependencies and environments before running any scripts.

## Contact

For any clarifications or setup help, please reach out via your usual comms channel.

---

*This README is for internal collaboration purposes only.*
