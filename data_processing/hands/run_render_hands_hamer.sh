#!/usr/bin/env bash
set -euo pipefail

python data_processing/hands/render_videos_hands_hamer.py \
  --data_root ./data/taste_rob_resized \
  --gpus 0 \
  --skip_existing \
  "$@"
