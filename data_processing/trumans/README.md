# Trumans Dataset Processing

This directory contains scripts for processing the Trumans dataset to generate egocentric videos with RGB, depth, and camera pose data from Blender `.blend` files.

## Overview

The Trumans dataset contains 3D human action sequences stored as Blender files. This processing pipeline extracts:
- **RGB frames**: Standard video frames from the character's perspective
- **Depth maps**: Per-pixel depth information (32-bit EXR format)
- **Camera poses**: 4x4 transformation matrices for each frame
- **Camera intrinsics**: Camera calibration parameters

## Prerequisites

- Blender (tested with version 3.0+)
- Python 3.7+
- HSI Add-on for Blender (see installation below)

## Installation

### 1. Install HSI Add-on

The HSI Add-on is required to loop through multiple action sequences in the Trumans `.blend` files.

1. Download [HSI_addon-zzy.zip](https://github.com/jnnan/trumans_utils/blob/main/HSI_addon-zzy.zip)
2. Extract it to your Blender addons directory:
   ```bash
   unzip HSI_addon-zzy.zip -d /path/to/blender/{version}/scripts/addons/
   ```
   Replace `{version}` with your Blender version (e.g., `3.6`, `4.0`)

### 2. Activate HSI Add-on

#### Headless Server (Recommended)
```bash
/path/to/blender/blender --background --python ./activate_hsi_addon.py
```

#### GUI Installation
If you have access to Blender GUI:
1. Open Blender
2. Go to Edit → Preferences → Add-ons
3. Click "Install" and select the extracted HSI add-on
4. Enable the add-on by checking its checkbox

## Usage

### Output Structure

The processing creates the following directory structure:
```
output_folder/
├── blend_filename/
│   ├── animation_name_1/
│   │   ├── rgb/                    # RGB frames (PNG)
│   │   ├── depth/                  # Depth maps (EXR)
│   │   ├── cam_params/             # Camera poses (NPY)
│   │   └── cam_intrinsics.npy      # Camera calibration
│   ├── animation_name_2/
│   │   └── ...
│   └── ...
```

### Rendering Options

#### 1. GUI Testing (Recommended for First Use)

1. Open Blender and load a `.blend` file from the `Recording_blend/` directory
2. Switch to Scripting workspace
3. Open `blender_ego_rgb_depth.py` and copy its contents
4. Adjust the `output_folder` variable in the script
5. Run the script in Blender's text editor

**Note**: To change action sequences, use the HSI add-on. See [this tutorial video](https://github.com/jnnan/trumans_utils/blob/main/tutorial.mp4) for guidance.

#### 2. Headless Rendering

##### Single Animation Sequence
```bash
# Render all frames of animation index 9
/path/to/blender --background {path/to/file.blend} --python blender_ego_rgb_depth.py -- --animation_index 9

# Render specific frame range (frames 1-100)
/path/to/blender --background {path/to/file.blend} --python blender_ego_rgb_depth.py -- --start_frame 1 --end_frame 100 --animation_index 9
```

##### Batch Processing
```bash
# Process all .blend files in parallel
python launch_blender_jobs.py
```

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--animation_index` | int | None | Specific animation to render (0-based index) |
| `--start_frame` | int | Scene start | First frame to render |
| `--end_frame` | int | Scene end | Last frame to render |

**Note**: If `--animation_index` is not specified, all animations in the file will be rendered.

## Configuration

### Camera Settings

The script automatically configures:
- **FOV**: 60 degrees (vertical)
- **Resolution**: 720×480 pixels
- **Position**: Geometric center of the eye mesh
- **Orientation**: Looking forward (-Y direction, up +Z)

### Render Settings

- **Engine**: Cycles (GPU-accelerated when available)
- **Samples**: 128 (configurable)
- **Denoising**: Enabled
- **Output Formats**:
  - RGB: PNG (RGBA)
  - Depth: EXR (32-bit float)

## Troubleshooting

### Common Issues

1. **HSI Add-on not found**: Ensure the add-on is properly installed and activated
2. **GPU rendering fails**: The script automatically falls back to CPU rendering
3. **Missing objects**: Verify the `.blend` file contains the expected armature and mesh objects
4. **Permission errors**: Ensure write permissions for the output directory

### Debug Information

The script provides detailed logging including:
- Animation set discovery
- Camera configuration
- Render progress
- File output locations

## File Descriptions

- `blender_ego_rgb_depth.py`: Main rendering script
- `activate_hsi_addon.py`: HSI add-on activation script
- `launch_blender_jobs.py`: Batch processing script
- `exr_to_disparity.py`: Convert .exr depth files to disparity images
- `example_exr_to_disparity.sh`: Example usage of disparity conversion

## Output Format Details

### Camera Poses (`cam_params/`)
- **Format**: NumPy arrays (.npy)
- **Shape**: (4, 4) transformation matrices
- **Coordinate System**: World to camera transformation

### Depth Maps (`depth/`)
- **Format**: OpenEXR (.exr)
- **Bit Depth**: 32-bit float
- **Values**: Linear depth in Blender units

### RGB Frames (`rgb/`)
- **Format**: PNG
- **Channels**: RGBA
- **Resolution**: 720×480 (configurable)

### Camera Intrinsics (`cam_intrinsics.npy`)
- **Format**: NumPy array (.npy)
- **Shape**: (3, 3) intrinsic matrix
- **Contains**: Focal length, principal point, skew

## Post-Processing: Depth to Disparity Conversion

After rendering, you can convert the depth maps to disparity format for training or visualization:

### Convert .exr Depth Files to Disparity

The `exr_to_disparity.py` script converts the .exr depth files to disparity images:

```bash
# Process single file
python exr_to_disparity.py --single_file path/to/depth_0001.exr --output_dir output/

# Process directory
python exr_to_disparity.py --input_dir path/to/depth/ --output_dir output/

# Batch process entire Blender output structure
python exr_to_disparity.py --input_dir /path/to/blender/output/ --output_dir output/ --batch_process

# With custom settings
python exr_to_disparity.py --input_dir /path/to/blender/output/ --output_dir output/ --batch_process --fps 12
```

### Disparity Output Format

The script generates:
- **Disparity maps** (`.npy`): Normalized disparity values [0, 1] with square root applied
- **Colored videos** (`.mp4`): Visualized disparity maps using spectral colormap
- **Metadata**: Maximum disparity value (`dmax`) for each sequence

### Disparity Conversion Details

- **Input**: 32-bit EXR depth files from Blender
- **Output**: Normalized disparity maps (1/depth, scaled to [0,1])
- **Processing**: Square root transformation applied for better distribution
- **Visualization**: Spectral colormap for depth visualization