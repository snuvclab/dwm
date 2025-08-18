#!/bin/bash

# Parallel sequence generation for 58 scenes split into 6 groups
# Each group processes ~10 scenes (9-10 scenes per group)

# Define shared command parts
BASE_CMD="python data_processing/run_sequence_generation.py"
SHARED_ARGS="--force_depth_reprocessing --data_root ../ego_render_new/ --save_root ../../nas1/public_dataset/trumans/ego_render_new/ --smplx_base_path ../../nas1/public_dataset/trumans/smplx_result/ --skip_existing_clips"

echo "🚀 Starting parallel sequence generation with 6 processes..."
echo "📊 Total scenes: 58, Scenes per process: ~10"
echo "⏰ Started at: $(date)"
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs

# Group 1: Scenes 0-9 (10 scenes)
echo "🔄 Starting Group 1: Scenes 0-9"
$BASE_CMD $SHARED_ARGS --scene_start 0 --scene_end 10 > logs/group1.log 2>&1 &
PID1=$!

# Group 2: Scenes 10-19 (10 scenes)
echo "🔄 Starting Group 2: Scenes 10-19"
$BASE_CMD $SHARED_ARGS --scene_start 10 --scene_end 20 > logs/group2.log 2>&1 &
PID2=$!

# Group 3: Scenes 20-29 (10 scenes)
echo "🔄 Starting Group 3: Scenes 20-29"
$BASE_CMD $SHARED_ARGS --scene_start 20 --scene_end 30 > logs/group3.log 2>&1 &
PID3=$!

# Group 4: Scenes 30-39 (10 scenes)
echo "🔄 Starting Group 4: Scenes 30-39"
$BASE_CMD $SHARED_ARGS --scene_start 30 --scene_end 40 > logs/group4.log 2>&1 &
PID4=$!

# Group 5: Scenes 40-49 (10 scenes)
echo "🔄 Starting Group 5: Scenes 40-49"
$BASE_CMD $SHARED_ARGS --scene_start 40 --scene_end 50 > logs/group5.log 2>&1 &
PID5=$!

# Group 6: Scenes 50-57 (8 scenes)
echo "🔄 Starting Group 6: Scenes 50-57"
$BASE_CMD $SHARED_ARGS --scene_start 50 > logs/group6.log 2>&1 &
PID6=$!

echo ""
echo "✅ All 6 processes started!"
echo "📝 Logs are being saved to logs/group*.log"
echo "🔍 Monitor progress with: tail -f logs/group*.log"
echo ""

# Wait for all processes to complete
echo "⏳ Waiting for all processes to complete..."
wait $PID1 $PID2 $PID3 $PID4 $PID5 $PID6

echo ""
echo "🎉 All processes completed!"
echo "⏰ Finished at: $(date)"
echo ""
echo "📊 Summary of results:"
echo "Group 1 (scenes 0-9):   $(grep -c 'Success\|Failed' logs/group1.log 2>/dev/null || echo 'Check logs/group1.log')"
echo "Group 2 (scenes 10-19): $(grep -c 'Success\|Failed' logs/group2.log 2>/dev/null || echo 'Check logs/group2.log')"
echo "Group 3 (scenes 20-29): $(grep -c 'Success\|Failed' logs/group3.log 2>/dev/null || echo 'Check logs/group3.log')"
echo "Group 4 (scenes 30-39): $(grep -c 'Success\|Failed' logs/group4.log 2>/dev/null || echo 'Check logs/group4.log')"
echo "Group 5 (scenes 40-49): $(grep -c 'Success\|Failed' logs/group5.log 2>/dev/null || echo 'Check logs/group5.log')"
echo "Group 6 (scenes 50-57): $(grep -c 'Success\|Failed' logs/group6.log 2>/dev/null || echo 'Check logs/group6.log')"