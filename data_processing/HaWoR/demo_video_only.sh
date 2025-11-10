#!/bin/bash

# HaWoR Video-Only Rendering Script
# Usage: ./demo_video_only.sh <video_path>
# Output: Two videos saved in the same directory as input video
#   - <video_name>_hand_overlay.mp4 (mesh overlaid on video)
#   - <video_name>_hand_mesh.mp4 (mesh only on black background)

if [ -z "$1" ]; then
    echo "Error: Please provide a video path"
    echo ""
    echo "Usage: $0 <video_path>"
    echo ""
    echo "Example:"
    echo "  $0 ./example/video_0.mp4"
    echo ""
    echo "Output videos will be saved in the same directory:"
    echo "  - <video_name>_hand_overlay.mp4 (mesh on video)"
    echo "  - <video_name>_hand_mesh.mp4 (mesh only, colored)"
    echo "  - <video_name>_hand_mask.mp4 (binary mask)"
    exit 1
fi

VIDEO_PATH=$1
export CUDA_VISIBLE_DEVICES=1

# Check if video exists
if [ ! -f "$VIDEO_PATH" ]; then
    echo "Error: Video file not found: $VIDEO_PATH"
    exit 1
fi

echo "Processing video: $VIDEO_PATH"
echo ""

# Run process_hawor_single.py
python process_hawor_single_cam_only.py \
    --video_path "$VIDEO_PATH"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "Success! Videos saved to: $(dirname "$VIDEO_PATH")"
else
    echo ""
    echo "Error: Processing failed"
    exit 1
fi

