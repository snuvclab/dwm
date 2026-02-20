# Data Processing (Phase 0)

This document integrates preprocessing instructions for both TRUMANS and TASTE-Rob.

## Scope
- TRUMANS rendering: `videos`, `videos_static`, `videos_hands`
- TASTE-Rob preprocessing: resized `videos` + generated `videos_static` + generated `videos_hands`
- Phase 0 does **not** include latent preparation

## Common Target Format
- Backbone target: 49 frames, 480p, 8fps
- Frame sampling convention used here:
  - select frames every 3 source frames
  - encode output videos at 8fps

## 1) TRUMANS

### 1.1 Download / Place Data
Follow TRUMANS utilities:
- https://github.com/jnnan/trumans_utils?tab=readme-ov-file

Expected local structure:
```text
data/trumans/
  Recordings_blend/
  smplx_result/
  Actions/
  video_render/
  ...
```

### 1.2 Blender Headless Setup (Ubuntu)
You can download Blender from:
- https://www.blender.org/download/

```bash
# Blender install
sudo tar -xvJf blender-4.1.1-linux-x64.tar.xz -C /opt/
sudo ln -s /opt/blender-4.1.1-linux-x64/blender /usr/local/bin/blender

# HSI add-on
unzip HSI_addon-zzy.zip
sudo cp -r HSI_addon-zzy /opt/blender-4.1.1-linux-x64/4.1/scripts/addons/
blender -b --python data_processing/trumans/activate_hsi_addon.py
```

### 1.3 Single Scene Test
```bash
blender --background \
  ./data/trumans/Recordings_blend/4abcb667-c57f-4d8f-940a-d964152329d5/4abcb667-c57f-4d8f-940a-d964152329d5.blend \
  --python data_processing/trumans/blender_ego_video_render.py -- \
  --animation_index 4 \
  --start_frame 2112 \
  --end_frame 2329 \
  --save-path ./data/trumans/ego_render_fov90/
```

### 1.4 Batch Rendering Entrypoints
- Interaction videos: `data_processing/trumans/run_trumans_render_videos.sh`
- Static videos: `data_processing/trumans/run_trumans_render_static.sh`
- Hand videos: `data_processing/trumans/run_trumans_render_hands.sh`
- Shared runner: `data_processing/trumans/run_trumans_render_batch.py`

### 1.5 TRUMANS Output Layout
```text
data/trumans/ego_render_fov90/{scene}/{action}/
  videos/*.mp4
  videos_static/*.mp4
  videos_hands/*.mp4
```

## 2) TASTE-Rob

### 2.1 Download / Place Data
Dataset source:
- https://github.com/GAP-LAB-CUHK-SZ/TASTE-Rob

Paper setting note:
- Due to time constraints, the paper used only `DoubleHand/{Dinning,Kitchen,Office}`.

Repository assumption:
```text
data/taste_rob/{SingleHand,DoubleHand}/{scene_type}/*.mp4
```

### 2.2 Preprocess Command
```bash
python data_processing/taste_rob/resize_videos_taste_rob.py \
  --input_dir data/taste_rob/ \
  --output_dir data/taste_rob_resized/ \
  --target_frames 49 \
  --output_fps 8 \
  --target_width 720 \
  --target_height 480 \
  --workers 16 \
  --merge_scene_prefix
```

### 2.3 Hand Mesh Video Generation (`videos_hands`)
Unlike TRUMANS, TASTE-Rob does not provide hand mesh data directly.
So `videos_hands` is generated using 3D hand tracking from RGB clips.

Initialize submodule once:

```bash
git submodule update --init --recursive third_party/hamer-mediapipe
```

Single GPU:

```bash
bash data_processing/hands/run_render_hands_hamer.sh
```

Multi GPU:

```bash
python data_processing/hands/render_videos_hands_hamer.py \
  --data_root data/taste_rob_resized \
  --gpus 0,1 \
  --workers_per_gpu 1 \
  --skip_existing
```

Subset by type + scene:

```bash
python data_processing/hands/render_videos_hands_hamer.py \
  --data_root data/taste_rob_resized \
  --hand_type SingleHand \
  --scene Dinning Kitchen Office \
  --gpus 0 \
  --skip_existing
```

Use a relative video list file (intersection with `--hand_type/--scene` when both are set):

```bash
python data_processing/hands/render_videos_hands_hamer.py \
  --data_root data/taste_rob_resized \
  --video_list_file data/taste_rob_video_subset.txt \
  --hand_type all \
  --gpus 0 \
  --skip_existing \
  --save_video_list data/taste_rob_selected_videos.txt
```

Use another environment/interpreter for `hamer-mediapipe` if needed:

```bash
python data_processing/hands/render_videos_hands_hamer.py \
  --data_root data/taste_rob_resized \
  --gpus 0 \
  --python_bin /path/to/venv/bin/python \
  --skip_existing
```

### 2.4 TASTE-Rob Output Layout
```text
data/taste_rob_resized/
  {SingleHand,DoubleHand}/
    {scene}/
      videos/*.mp4
      videos_static/*.mp4
      videos_hands/*.mp4
    ...
  ...
```

`videos_static` is created by repeating the first sampled frame to match clip length.

## 3) Prompt Preparation

`prompts_aux` is an auxiliary prompt stage used before generating final prompts.

All prompt-related pipelines are documented in:
- [`data_processing/video_caption/README.md`](./video_caption/README.md)
