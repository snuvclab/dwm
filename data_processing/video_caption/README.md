# Video Captioning And Prompt Rewrite

Parts of this path were adapted from VideoX-Fun video captioning.

All commands below assume you run them from the repository root.

## Goal

Generate prompt text from ego videos and then optionally rewrite it into a cleaner generation-oriented prompt.

Main stages:

- `videos/*.mp4 -> prompts/*.txt`
- `prompts/*.txt -> prompts_rewrite/*.txt`

`prompts_aux` remains optional additional context.

## 1. Build `prompts_aux` (optional)

### TRUMANS

```bash
python data_processing/trumans/create_videos_third_trumans.py \
  --root_dir data_refactor/trumans \
  --third_video_root data/trumans/video_render \
  --ego_videos_dirname videos \
  --output_dirname videos_third \
  --clip_length 49 \
  --clip_stride 25 \
  --frame_skip 3 \
  --fps 8 \
  --skip_existing

bash data_processing/video_caption/run_trumans_caption_third.sh
```

### TASTE-Rob

```bash
python data_processing/taste_rob/export_prompts_from_captions_xlsx.py \
  --data_root data_refactor/taste_rob_resized \
  --captions_xlsx /path/to/taste_rob/captions.xlsx \
  --prompt_dir_name prompts_aux
```

## 2. Ego Captioning

### Recommended open-source smoke path: BLIP fallback

This path works with the base `requirements.txt` environment and does not require `vllm`.

### TRUMANS

```bash
ROOT_DIR="$(pwd)/data_refactor/trumans" \
CAPTION_BACKEND=blip \
MODEL_NAME=Salesforce/blip-image-captioning-base \
bash data_processing/video_caption/run_trumans_caption_ego.sh
```

### TASTE-Rob

```bash
ROOT_DIR="$(pwd)/data_refactor/taste_rob_resized" \
CAPTION_BACKEND=blip \
MODEL_NAME=Salesforce/blip-image-captioning-base \
bash data_processing/video_caption/run_taste_rob_caption_ego.sh
```

### Optional higher-quality path: InternVL2

This path requires an additional `vllm` installation compatible with your CUDA / torch environment and a larger GPU budget.

Example extra install:

```bash
pip install vllm
```

### TRUMANS

```bash
ROOT_DIR="$(pwd)/data_refactor/trumans" \
CAPTION_BACKEND=internvl2 \
MODEL_PATH=OpenGVLab/InternVL2-40B-AWQ \
bash data_processing/video_caption/run_trumans_caption_ego.sh
```

### TASTE-Rob

```bash
ROOT_DIR="$(pwd)/data_refactor/taste_rob_resized" \
CAPTION_BACKEND=internvl2 \
MODEL_PATH=OpenGVLab/InternVL2-40B-AWQ \
bash data_processing/video_caption/run_taste_rob_caption_ego.sh
```

The runners write `prompts/*.txt` next to the sample videos.

## 3. Prompt Rewrite

Rewrite is now supported for both TRUMANS and TASTE-Rob using the same `caption_rewrite.py` backend and `prompt/rewrite.txt` template.

### TASTE-Rob

```bash
ROOT_DIR="$(pwd)/data_refactor/taste_rob_resized" \
PROMPT_SUBDIR=prompts \
OUTPUT_FOLDER_NAME=prompts_rewrite \
bash data_processing/video_caption/run_rewrite_taste_rob.sh
```

### TRUMANS

```bash
ROOT_DIR="$(pwd)/data_refactor/trumans" \
PROMPT_SUBDIR=prompts \
OUTPUT_FOLDER_NAME=prompts_rewrite \
bash data_processing/video_caption/run_rewrite_trumans.sh
```

Common environment overrides:

```bash
MODEL_NAME=meta-llama/Meta-Llama-3.1-8B-Instruct
ENGINE=auto
NUM_SPLITS=8
TEMPERATURE=0.7
MAX_TOKENS=1024
```

Smoke-tested lightweight fallback:

```bash
MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct
ENGINE=transformers
NUM_SPLITS=1
MAX_TOKENS=256
```

If `SLURM_ARRAY_TASK_ID` is set, the rewrite runners split work by file automatically.

## 4. Output Contract

After rewrite, each sample root may contain:

- `prompts/<stem>.txt`
- `prompts_rewrite/<stem>.txt`
- optional `prompts_aux/<stem>.json` or `.txt`

The CogVideoX preencoder in this refactor assumes rewritten prompts are stored in `prompts_rewrite` and writes embeddings to `prompt_embeds_prompts_rewrite`.
