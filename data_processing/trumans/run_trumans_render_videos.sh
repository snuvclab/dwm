#!/usr/bin/env bash
set -euo pipefail

python data_processing/trumans/run_trumans_render_batch.py \
  --script-path data_processing/trumans/blender_ego_video_render.py \
  --save-path ./data/trumans/ego_render_fov90/ \
  --fps 8 --width 720 --height 480 \
  --samples 64 \
  --clip-length 49 \
  --clip-stride 25 \
  --frame-skip 3 \
  --video-output --auto-split-clips --scenes trumans_all.txt \
  "$@"
