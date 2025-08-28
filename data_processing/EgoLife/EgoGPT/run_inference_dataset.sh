#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1

# Change to the EgoGPT directory
source ~/.bashrc
source $(conda info --base)/etc/profile.d/conda.sh
conda activate egogpt
cd /virtual_lab/jhb_vclab/byungjun_vclab/world_model/data_processing/EgoLife/EgoGPT/
DATA_ROOT=/virtual_lab/jhb_vclab/byungjun_vclab/world_model/data
QUERY="Please describe the video in detail."
PRETRAINED_PATH="lmms-lab/EgoGPT-7b-EgoIT-EgoLife"

echo "Choose an option:"
echo "1. Dry run - check which files need processing"
echo "2. Normal processing - skip existing files"
echo "3. Reprocess all - overwrite existing files"
echo "4. Custom query"
echo "5. Use video checklist file"
read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        echo "Running dry run to check which files need processing..."
        python inference_dataset.py \
            --data_root $DATA_ROOT \
            --query "$QUERY" \
            --pretrained_path $PRETRAINED_PATH \
            --dry_run
        ;;
    2)
        echo "Running normal processing (skipping existing files)..."
        python inference_dataset.py \
            --data_root $DATA_ROOT \
            --query "$QUERY" \
            --pretrained_path $PRETRAINED_PATH \
            --skip_existing
            ;;
    3)
        echo "Running reprocessing (overwriting all existing files)..."
        python inference_dataset.py \
            --data_root $DATA_ROOT \
            --query "$QUERY" \
            --pretrained_path $PRETRAINED_PATH \
            --no_skip_existing
            ;;
    4)
                read -p "Enter your custom query: " custom_query
        echo "Running with custom query: $custom_query"
        python inference_dataset.py \
            --data_root $DATA_ROOT \
            --query "$custom_query" \
            --pretrained_path $PRETRAINED_PATH \
            --skip_existing
        ;;
    5)
        read -p "Enter path to video checklist file: " checklist_file
        read -p "Remove processed videos from checklist? (y/n): " remove_choice
        if [ "$remove_choice" = "y" ] || [ "$remove_choice" = "Y" ]; then
            echo "Running with checklist and removing processed videos..."
            python inference_dataset.py \
                --data_root $DATA_ROOT \
                --query "$QUERY" \
                --pretrained_path $PRETRAINED_PATH \
                --skip_existing \
                --checklist_file "$checklist_file" \
                --remove_from_checklist
        else
            echo "Running with checklist (keeping checklist unchanged)..."
            python inference_dataset.py \
                --data_root $DATA_ROOT \
                --query "$QUERY" \
                --pretrained_path $PRETRAINED_PATH \
                --skip_existing \
                --checklist_file "$checklist_file"
        fi
        ;;
    *)
        echo "Invalid choice. Running default (normal processing)..."
        python inference_dataset.py \
            --data_root $DATA_ROOT \
            --query "$QUERY" \
            --pretrained_path $PRETRAINED_PATH \
            --skip_existing
        ;;
esac
