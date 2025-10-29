#!/usr/bin/env bash
set -euo pipefail

# --- Defaults ---
NUM_FRAMES=100

# --- Help ---
usage() {
  cat <<EOF
Usage: $(basename "$0") --vrs_file PATH [--num_frames N]

Options:
  --vrs_file     Path to the .vrs file (required)
  --num_frames   Number of frames to keep in transforms.json (default: 100)
EOF
}

# --- Arg parsing (GNU getopt style) ---
if ! OPTIONS=$(getopt -o '' --long vrs_file:,num_frames:,help -- "$@"); then
  usage; exit 1
fi
eval set -- "$OPTIONS"
while true; do
  case "$1" in
    --vrs_file)   VRS_FILE=$2; shift 2 ;;
    --num_frames) NUM_FRAMES=$2; shift 2 ;;
    --help)       usage; exit 0 ;;
    --)           shift; break ;;
    *)            usage; exit 1 ;;
  esac
done

# --- Validate required args ---
: "${VRS_FILE:?ERROR: --vrs_file is required}"
if [[ ! -f "$VRS_FILE" ]]; then
  echo "ERROR: VRS file not found: $VRS_FILE" >&2
  exit 1
fi

# --- Dependencies check ---
if ! command -v jq >/dev/null 2>&1; then
  echo "ERROR: 'jq' is required but not found. Install jq and retry." >&2
  exit 1
fi

# --- Resolve paths ---
VRS_PATH="$(realpath "$VRS_FILE")"
VRS_DIR="$(dirname "$VRS_PATH")"

MPS_DATA_DIR="$VRS_DIR/mps/slam"
DATA_DIR="$VRS_DIR/gsplat/data"
OUTPUT_DIR="$VRS_DIR/gsplat/output"

mkdir -p "$DATA_DIR" "$OUTPUT_DIR"

# --- Conda activation (works with most setups) ---
# Try the modern hook; fall back to common install path if needed.
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)" || true
fi
if ! conda activate nerfstudio >/dev/null 2>&1; then
  # Fallback: source the conda script directly if the hook didn't work
  if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate nerfstudio
  elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
    conda activate nerfstudio
  else
    echo "WARNING: Could not auto-activate 'nerfstudio' env. Ensure the commands below run in the right env."
  fi
fi

# # --- 1) Data processing ---
# echo "Running ns-process-data..."
# ns-process-data aria \
#   --vrs-file "$VRS_PATH" \
#   --mps-data-dir "$MPS_DATA_DIR" \
#   --output-dir "$DATA_DIR" \
#   --max-frames 10000

# # --- 2) Edit transforms.json to keep only the first NUM_FRAMES ---
# TRANSFORMS_SRC="$DATA_DIR/transforms.json"
# TRANSFORMS_SRC_="$DATA_DIR/transforms_full.json"
# TRANSFORMS_DST="$DATA_DIR/transforms_.json"

# cp "$TRANSFORMS_SRC" "$TRANSFORMS_SRC_"

# if [[ ! -f "$TRANSFORMS_SRC" ]]; then
#   echo "ERROR: transforms.json not found at $TRANSFORMS_SRC" >&2
#   exit 1
# fi

# echo "Trimming frames to first $NUM_FRAMES entries..."
# jq --argjson n "$NUM_FRAMES" '.frames = (.frames[:$n])' "$TRANSFORMS_SRC" > "$TRANSFORMS_DST"
# rm "$TRANSFORMS_SRC"
# mv "$TRANSFORMS_DST" "$TRANSFORMS_SRC"

# # --- 3) Train gsplat ---
# echo "Starting training with ns-train splatfacto..."
# ns-train splatfacto \
#   --data "$DATA_DIR" \
#   --output-dir "$OUTPUT_DIR" \
#   --max-num-iterations 30000 \
#   --viewer.quit-on-train-completion True \
#   nerfstudio-data --train-split-fraction 1 --orientation-method none --center-method none --auto-scale-poses False 
# echo "Done."

# # --- 4) Rendering ---
# rm "$TRANSFORMS_SRC"
# mv "$TRANSFORMS_SRC_" "$TRANSFORMS_SRC"
# echo "Starting rendering with ns-render gsplat..."
# LATEST_CONFIG=$(ls -dt "$OUTPUT_DIR"/data/splatfacto/*/ | head -n 1)config.yml
# echo "$LATEST_CONFIG"

# # modify camera parameters of transforms.json
# python tests/gsplat/modify_transforms.py "$TRANSFORMS_SRC"

# ns-render dataset \
#   --load-config "$LATEST_CONFIG" \
#   --output-path "$OUTPUT_DIR/static" \
#   --split=train --rendered-output-names=rgb

# crop to 720x480
echo "Cropping renderings to 720x480..."
python tests/gsplat/crop_renderings.py "$OUTPUT_DIR/static/train/rgb"

# get original gt frames from vrs
echo "Extracting original images and camera poses from VRS..."
python data_processing/aria/1_get_images_and_cameras.py \
  --data_root "$VRS_DIR"