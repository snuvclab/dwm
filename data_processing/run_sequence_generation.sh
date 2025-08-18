#!/bin/bash

# Parallel sequence generation for 58 scenes split into 6 groups
# Each group processes ~10 scenes (9-10 scenes per group)

# Function to kill all background processes
cleanup() {
    echo ""
    echo "🛑 Received interrupt signal. Killing all background processes..."
    
    # Kill all background processes
    if [ ! -z "$PID1" ]; then kill $PID1 2>/dev/null; fi
    if [ ! -z "$PID2" ]; then kill $PID2 2>/dev/null; fi
    if [ ! -z "$PID3" ]; then kill $PID3 2>/dev/null; fi
    if [ ! -z "$PID4" ]; then kill $PID4 2>/dev/null; fi
    if [ ! -z "$PID5" ]; then kill $PID5 2>/dev/null; fi
    if [ ! -z "$PID6" ]; then kill $PID6 2>/dev/null; fi
    
    # Wait a moment for processes to terminate
    sleep 2
    
    # Force kill if still running
    if [ ! -z "$PID1" ]; then kill -9 $PID1 2>/dev/null; fi
    if [ ! -z "$PID2" ]; then kill -9 $PID2 2>/dev/null; fi
    if [ ! -z "$PID3" ]; then kill -9 $PID3 2>/dev/null; fi
    if [ ! -z "$PID4" ]; then kill -9 $PID4 2>/dev/null; fi
    if [ ! -z "$PID5" ]; then kill -9 $PID5 2>/dev/null; fi
    if [ ! -z "$PID6" ]; then kill -9 $PID6 2>/dev/null; fi
    
    echo "✅ All processes terminated."
    echo "⏰ Interrupted at: $(date)"
    exit 1
}

# Set up signal handlers for graceful shutdown
trap cleanup SIGINT SIGTERM

# Define shared command parts
BASE_CMD="python data_processing/run_sequence_generation.py"
SHARED_ARGS="--data_root ./data/trumans/ego_render_fov90/ --save_root ./data/trumans/ego_render_fov90/ --smplx_base_path ./data/trumans/smplx_result/ --skip_existing_clips"

echo "🚀 Starting parallel sequence generation with 6 processes..."
echo "📊 Total scenes: 58, Scenes per process: ~10"
echo "⏰ Started at: $(date)"
echo "💡 Press Ctrl+C to stop all processes gracefully"
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