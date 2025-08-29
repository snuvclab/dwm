#!/bin/bash

# Change to the EgoGPT directory
source ~/.bashrc
source $(conda info --base)/etc/profile.d/conda.sh
conda activate egogpt
cd /virtual_lab/jhb_vclab/byungjun_vclab/world_model/data_processing/EgoLife/EgoGPT/

# Configuration
DATA_ROOT="/virtual_lab/jhb_vclab/byungjun_vclab/world_model/data"
DATASET_NAME="trumans"
RENDER_TYPE="ego_render_fov90"
OUTPUT_FILE="video_checklist.txt"

echo "Current configuration:"
echo "  Data root: $DATA_ROOT"
echo "  Dataset: $DATASET_NAME"
echo "  Render type: $RENDER_TYPE"
echo ""
echo "Choose an option:"
echo "1. Quick check - show summary only"
echo "2. Detailed check - show verbose output"
echo "3. Save checklist to file (paths only)"
echo "4. Save detailed checklist to file"
echo "5. Custom data root"
echo "6. Check with no_skip_existing (show all videos)"
echo "7. Custom dataset and render type"
read -p "Enter your choice (1-7): " choice

case $choice in
    1)
        echo "Running quick check..."
        python check_videos_to_process.py \
            --data_root $DATA_ROOT \
            --dataset_name $DATASET_NAME \
            --render_type $RENDER_TYPE \
            --skip_existing
        ;;
    2)
        echo "Running detailed check..."
        python check_videos_to_process.py \
            --data_root $DATA_ROOT \
            --dataset_name $DATASET_NAME \
            --render_type $RENDER_TYPE \
            --skip_existing \
            --verbose
        ;;
    3)
        echo "Running check and saving to file (paths only)..."
        python check_videos_to_process.py \
            --data_root $DATA_ROOT \
            --dataset_name $DATASET_NAME \
            --render_type $RENDER_TYPE \
            --skip_existing \
            --output_file $OUTPUT_FILE \
            --paths_only
        ;;
    4)
        echo "Running check and saving detailed checklist to file..."
        python check_videos_to_process.py \
            --data_root $DATA_ROOT \
            --dataset_name $DATASET_NAME \
            --render_type $RENDER_TYPE \
            --skip_existing \
            --output_file "detailed_${OUTPUT_FILE}"
        ;;
    5)
        read -p "Enter custom data root path: " custom_data_root
        echo "Running check with custom data root: $custom_data_root"
        python check_videos_to_process.py \
            --data_root "$custom_data_root" \
            --dataset_name $DATASET_NAME \
            --render_type $RENDER_TYPE \
            --skip_existing \
            --verbose
        ;;
    6)
        echo "Running check with no_skip_existing (will show all videos)..."
        python check_videos_to_process.py \
            --data_root $DATA_ROOT \
            --dataset_name $DATASET_NAME \
            --render_type $RENDER_TYPE \
            --no_skip_existing \
            --verbose
        ;;
    7)
        read -p "Enter custom dataset name: " custom_dataset
        read -p "Enter custom render type: " custom_render
        echo "Running check with custom dataset: $custom_dataset, render type: $custom_render"
        python check_videos_to_process.py \
            --data_root $DATA_ROOT \
            --dataset_name "$custom_dataset" \
            --render_type "$custom_render" \
            --skip_existing \
            --verbose
        ;;
    *)
        echo "Invalid choice. Running default (quick check)..."
        python check_videos_to_process.py \
            --data_root $DATA_ROOT \
            --dataset_name $DATASET_NAME \
            --render_type $RENDER_TYPE \
            --skip_existing
        ;;
esac
