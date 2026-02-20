# Video Captioning

Parts of this pipeline were adapted from VideoX-Fun video captioning:
- https://github.com/aigc-apps/VideoX-Fun/tree/main/videox_fun/video_caption

## Goal

Primary goal is to generate ego-video prompts from:
- `videos/*.mp4` -> `prompts/*.txt`

`prompts_aux` is **optional, but strongly recommended** as additional hint/context.

## Build `prompts_aux` (Optional, but Recommended)

Why:
- Ego-only captioning can be unstable for occluded/ambiguous actions.
- `prompts_aux` provides extra action signal.

### TRUMANS `prompts_aux`

1. Create third-person clips aligned to ego clips:

```bash
python data_processing/trumans/create_videos_third_trumans.py \
  --root_dir data/trumans/ego_render_fov90 \
  --third_video_root data/trumans/video_render \
  --ego_videos_dirname videos \
  --output_dirname videos_third \
  --clip_length 49 \
  --clip_stride 25 \
  --frame_skip 3 \
  --fps 8 \
  --skip_existing
```

2. Caption third-person clips:

```bash
bash data_processing/video_caption/run_trumans_caption_third.sh
```

3. Attach action hints from `Actions/*.txt`:

```bash
python data_processing/trumans/build_prompts_aux_trumans.py \
  --root_dir data/trumans/ego_render_fov90 \
  --actions_root data/trumans/Actions \
  --third_prompt_dirname prompts_aux \
  --third_video_dirname videos_third \
  --clip_length 49 \
  --clip_stride 25 \
  --frame_skip 3 \
  --skip_existing
```

Final schema:

```json
{
  "prompt": "action-focused description text",
  "action_hints": ["..."]
}
```

### TASTE-Rob `prompts_aux`

For TASTE-Rob, `prompts_aux` is exported from `captions.xlsx`:

```bash
python data_processing/taste_rob/export_prompts_from_captions_xlsx.py \
  --data_root data/taste_rob_resized \
  --captions_xlsx /virtual_lab/dataset/taste_rob/captions.xlsx \
  --prompt_dir_name prompts_aux
```

## Main Pipeline (Ego Prompt Generation)

### TRUMANS

Run:

```bash
bash data_processing/video_caption/run_trumans_caption_ego.sh
```

This uses:
- input prompt: `data_processing/video_caption/prompt/caption_ego.txt`
- video folder: `videos`
- output folder: `prompts`
- context (if exists): `prompts_aux`

### TASTE-Rob

Run:

```bash
bash data_processing/video_caption/run_taste_rob_caption_ego.sh
```

This uses:
- input prompt: `data_processing/video_caption/prompt/caption_ego.txt`
- video folder: `videos`
- output folder: `prompts`
- context (if exists): `prompts_aux`

## Runtime Options

The bash runners support environment-variable overrides, for example:

```bash
ROOT_DIR=/path/to/data/trumans/ego_render_fov90 \
MODEL_PATH=OpenGVLab/InternVL2-40B-AWQ \
NUM_SPLITS=8 ARRAY_INDEX=0 \
bash data_processing/video_caption/run_trumans_caption_ego.sh
```

### Multi-GPU

`internvl2_video_recaptioning.py` automatically uses all visible GPUs
by setting tensor parallel size from visible CUDA devices.

Examples:

```bash
# Use all visible GPUs
bash data_processing/video_caption/run_trumans_caption_ego.sh

# Restrict to selected GPUs
CUDA_VISIBLE_DEVICES=0,1 \
bash data_processing/video_caption/run_trumans_caption_ego.sh
```

## VLM Instruction Prompts

- `data_processing/video_caption/prompt/caption_third.txt`
  - VLM instruction used when captioning TRUMANS `videos_third` clips to build `prompts_aux`.
  - Focuses on action-centric description quality for third-person aligned clips.

- `data_processing/video_caption/prompt/caption_ego.txt`
  - VLM instruction used when captioning ego `videos` clips to build final `prompts`.
  - Used for both TRUMANS and TASTE-Rob ego prompt generation.
