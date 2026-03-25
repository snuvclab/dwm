# CogVideoX DWM Data Processing

This document describes the DWM-side smoke/full preprocessing path for the final CogVideoX static-hand-concat model.

All commands below assume you run them from the repository root.

## Prerequisites

- `ffmpeg` must be available on `PATH`
- `blender` must be available on `PATH` if you render TRUMANS
- the examples below assume the public smoke layout lives under `data_refactor/`

## Target Layout

All smoke outputs in this refactor go under `data_refactor/`.

- TRUMANS smoke root: `data_refactor/trumans/{scene}/{animation}/...`
- TASTE-Rob smoke root: `data_refactor/taste_rob_resized/{SingleHand,DoubleHand}/{scene}/...`

Each sample root is expected to contain:

- `videos/*.mp4`
- `videos_static/*.mp4`
- `videos_hands/*.mp4`
- `prompts/*.txt`
- `prompts_rewrite/*.txt`
- optional preencoded siblings:
  - `video_latents/*.pt`
  - `static_video_latents/*.pt`
  - `hand_video_latents/*.pt`
  - `prompt_embeds_prompts_rewrite/*.pt`

Common training target format:

- `49` frames
- `480x720`
- `8 fps`

## 1. TRUMANS Smoke Rendering

Smoke rendering now has a small wrapper that renders one scene + one animation and copies a few clips into `data_refactor`.

```bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate dwm

python data_processing/trumans/render_smoke_trumans.py \
  --recordings_root data/trumans/Recordings_blend \
  --output_root data_refactor/trumans \
  --clip_count 6 \
  --gpu 0
```

If TRUMANS is not linked under `data/`, replace `--recordings_root` with your absolute dataset path.

Useful overrides:

```bash
--scene <scene_name>
--animation <animation_name>
--clip_length 49
--clip_stride 25
--frame_skip 3
--dry_run
```

The wrapper renders RGB / static / hand clips with the existing Blender scripts and then copies a common subset into the smoke dataset layout.

## 2. TASTE-Rob Smoke Preparation

Resize only a few clips from your local TASTE-Rob copy into `data_refactor/taste_rob_resized`:

```bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate dwm

python data_processing/taste_rob/prepare_smoke_taste_rob.py \
  --input_dir /path/to/taste_rob \
  --output_dir data_refactor/taste_rob_resized \
  --clip_count 6
```

To resize and immediately render hand videos with original HaMeR:

```bash
python data_processing/taste_rob/prepare_smoke_taste_rob.py \
  --input_dir /path/to/taste_rob \
  --output_dir data_refactor/taste_rob_resized \
  --clip_count 6 \
  --render_hands_backend original
```

## 3. TASTE-Rob Hand Videos

### 3.1 Available backends

- `mediapipe`: existing `third_party/hamer-mediapipe` path
- `original`: original HaMeR + ViTPose path via `third_party/hamer`

### 3.2 Open-source install notes for `original`

Assuming `third_party/hamer` is cloned or symlinked:

```bash
cd third_party/hamer
pip install --no-build-isolation -e .[all]
pip install -e third-party/ViTPose
```

Notes:

- Do not reinstall torch here if it is already present in the environment.
- The renderer expects HaMeR assets under one of:
  - `$HAMER_DATA_DIR`
  - `third_party/hamer/_DATA`
  - `third_party/hamer-mediapipe/_DATA`

### 3.3 Run hand rendering

Default wrapper:

```bash
bash data_processing/hands/run_render_hands_hamer.sh --backend original
```

Direct original backend call:

```bash
python data_processing/hands/render_videos_hands_hamer_original.py \
  --data_root data_refactor/taste_rob_resized \
  --skip_existing
```

Direct mediapipe backend call:

```bash
python data_processing/hands/render_videos_hands_hamer.py \
  --data_root data_refactor/taste_rob_resized \
  --skip_existing
```

Multi-GPU launch remains available through the existing launcher. Use `--script` to switch the worker implementation:

```bash
python data_processing/hands/launch_render_hands_hamer_multi_gpu.py \
  --gpus 0,1 \
  --script data_processing/hands/render_videos_hands_hamer_original.py -- \
  --data_root data_refactor/taste_rob_resized \
  --skip_existing
```

## 4. Caption And Rewrite

Caption generation is documented in `data_processing/video_caption/README.md`.

Recommended open-source smoke flow uses the BLIP fallback because it works with the base `requirements.txt` environment and does not require `vllm`.

```bash
ROOT_DIR="$(pwd)/data_refactor/trumans" \
CAPTION_BACKEND=blip \
bash data_processing/video_caption/run_trumans_caption_ego.sh

ROOT_DIR="$(pwd)/data_refactor/taste_rob_resized" \
CAPTION_BACKEND=blip \
bash data_processing/video_caption/run_taste_rob_caption_ego.sh

ROOT_DIR="$(pwd)/data_refactor/trumans" bash data_processing/video_caption/run_rewrite_trumans.sh
ROOT_DIR="$(pwd)/data_refactor/taste_rob_resized" bash data_processing/video_caption/run_rewrite_taste_rob.sh
```

If you want the heavier InternVL2 caption path instead, see `data_processing/video_caption/README.md`.

## 5. CogVideoX Preencoding

Encode selected modalities with the standalone CogVideoX encoder:

```bash
bash data_processing/encode_with_cogvideox.sh \
  --dataset_type trumans \
  --data_root data_refactor/trumans \
  --modalities videos static_videos hand_videos prompts \
  --prompt_subdir prompts_rewrite
```

And for TASTE-Rob:

```bash
bash data_processing/encode_with_cogvideox.sh \
  --dataset_type taste_rob \
  --data_root data_refactor/taste_rob_resized \
  --modalities videos static_videos hand_videos prompts \
  --prompt_subdir prompts_rewrite
```

The encoder now samples each source clip to the training target length before VAE encoding, so video latents stay aligned with the `49`-frame training contract.

Optional quick sanity check:

```bash
bash data_processing/encode_with_cogvideox.sh \
  --dataset_type trumans \
  --data_root data_refactor/trumans \
  --modalities videos static_videos hand_videos prompts \
  --prompt_subdir prompts_rewrite \
  --debug
```

`--debug` now encodes one sample per modality instead of stopping after the first modality.

## 6. Dataset Files

Create train/val/test split files after preencoding:

```bash
python data_processing/create_dataset_file.py \
  --dataset_type trumans \
  --data_root data_refactor/trumans \
  --output_dir data_refactor/dataset_files/trumans_smoke \
  --output_base_dir "$(pwd)/data_refactor"

python data_processing/create_dataset_file.py \
  --dataset_type taste_rob \
  --data_root data_refactor/taste_rob_resized \
  --output_dir data_refactor/dataset_files/taste_rob_smoke \
  --output_base_dir "$(pwd)/data_refactor"
```

`output_base_dir` should point at the shared `data_refactor` root because training and inference consume split-file entries relative to that common data root.

Use `--allow_missing_processed` if you want to create split files before preencoding.

## 7. Continue To Training

After `train.txt`, `val.txt`, and `test.txt` are created under `data_refactor/dataset_files/`, continue with [`training/cogvideox/README.md`](../training/cogvideox/README.md).
