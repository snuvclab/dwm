#!/usr/bin/env bash
set -euo pipefail

BACKEND="original"
FORWARD_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend)
      BACKEND="$2"
      shift 2
      ;;
    *)
      FORWARD_ARGS+=("$1")
      shift
      ;;
  esac
done

case "$BACKEND" in
  original)
    SCRIPT="data_processing/hands/render_videos_hands_hamer_original.py"
    ;;
  mediapipe)
    SCRIPT="data_processing/hands/render_videos_hands_hamer.py"
    ;;
  *)
    echo "Unsupported backend: $BACKEND" >&2
    exit 1
    ;;
esac

python "$SCRIPT" \
  --data_root ./data/taste_rob \
  --skip_existing \
  "${FORWARD_ARGS[@]}"
