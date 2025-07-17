#!/bin/bash

# ==== Config ====
JOB_SCRIPT="scripts/train.sbatch"              # Your Slurm script
SNAPSHOT_ROOT="/scratch/$USER/jobs"            # Where snapshots are stored
JOB_NAME=$(basename "$JOB_SCRIPT" .sbatch)     # Default job name from script name

# Optional: Take optional job label as first argument
LABEL=${1:-$(date +%Y%m%d_%H%M%S)}
SNAPSHOT_DIR="${SNAPSHOT_ROOT}/${JOB_NAME}_${LABEL}"

# ==== Create snapshot ====
echo "[INFO] Creating snapshot at $SNAPSHOT_DIR ..."
mkdir -p "$SNAPSHOT_DIR" || { echo "[ERROR] Failed to create snapshot directory"; exit 1; }

git archive --format=tar HEAD | tar -x -C "$SNAPSHOT_DIR"

# ==== Submit job from snapshot ====
cd "$SNAPSHOT_DIR" || exit 1
echo "[INFO] Submitting job from snapshot directory..."
sbatch "$JOB_SCRIPT"
