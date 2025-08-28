#!/bin/bash

# Change to the EgoGPT directory
cd "$(dirname "$0")"

echo "=== Video Checklist Splitter ==="
echo ""

# Check if input file is provided as argument
if [ $# -eq 1 ]; then
    INPUT_FILE="$1"
else
    read -p "Enter path to video checklist file: " INPUT_FILE
fi

# Check if file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: File not found: $INPUT_FILE"
    exit 1
fi

echo "Input file: $INPUT_FILE"
echo ""

# Count lines in the file
TOTAL_LINES=$(wc -l < "$INPUT_FILE")
VIDEO_LINES=$(grep -v '^#' "$INPUT_FILE" | grep -v '^$' | wc -l)

echo "File statistics:"
echo "  Total lines: $TOTAL_LINES"
echo "  Video entries: $VIDEO_LINES"
echo ""

echo "Choose splitting method:"
echo "1. Split into 2 equal parts"
echo "2. Split into 3 equal parts"
echo "3. Split into 4 equal parts"
echo "4. Split for 2 GPUs"
echo "5. Split for 4 GPUs"
echo "6. Custom number of parts"
echo "7. Custom number of GPUs"
read -p "Enter your choice (1-7): " choice

case $choice in
    1)
        echo "Splitting into 2 parts..."
        python split_checklist.py "$INPUT_FILE" --num_parts 2
        ;;
    2)
        echo "Splitting into 3 parts..."
        python split_checklist.py "$INPUT_FILE" --num_parts 3
        ;;
    3)
        echo "Splitting into 4 parts..."
        python split_checklist.py "$INPUT_FILE" --num_parts 4
        ;;
    4)
        echo "Splitting for 2 GPUs..."
        python split_checklist.py "$INPUT_FILE" --num_gpus 2
        ;;
    5)
        echo "Splitting for 4 GPUs..."
        python split_checklist.py "$INPUT_FILE" --num_gpus 4
        ;;
    6)
        read -p "Enter number of parts: " num_parts
        echo "Splitting into $num_parts parts..."
        python split_checklist.py "$INPUT_FILE" --num_parts "$num_parts"
        ;;
    7)
        read -p "Enter number of GPUs: " num_gpus
        echo "Splitting for $num_gpus GPUs..."
        python split_checklist.py "$INPUT_FILE" --num_gpus "$num_gpus"
        ;;
    *)
        echo "Invalid choice. Exiting..."
        exit 1
        ;;
esac

echo ""
echo "Split complete! Check the generated files in the current directory."
