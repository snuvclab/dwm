# Video Captioning

All commands below assume you run them from the repository root.

The examples below assume processed data is stored under `data/`.

This pipeline adapts the video caption workflow from [VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun).

`prompts_aux` is optional auxiliary text for caption generation. It is not the main training prompt. Instead, it provides extra context that can be used when generating or refining `prompts/`.

For TRUMANS ego captioning, the pipeline also uses action annotations from `data/trumans/Actions/` to inject clip-level action hints. This does not require `prompts_aux`.

Typical use cases:

- TRUMANS: captions from third-person view videos
- TASTE-Rob: text exported from the provided caption annotations

For TRUMANS, the third-person view videos can be taken from `data/trumans/video_render/`.

The main prompt flow is:

```text
videos/*.mp4 -> prompts/*.txt -> prompts_rewrite/*.txt
```

If `prompts_aux` is available, the captioning pipeline can use it as additional context on top of the TRUMANS action annotations.

## 1. Build `prompts_aux` (optional)

### TRUMANS

```bash
python data_processing/trumans/create_videos_third_trumans.py \
  --root_dir data/trumans \
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
  --data_root data/taste_rob \
  --captions_xlsx /path/to/taste_rob/captions.xlsx \
  --prompt_dir_name prompts_aux
```

## 2. Ego Captioning

InternVL2 captioning requires `vllm`. It is part of the repository dependencies installed by `pip install -r requirements.txt`.

### TRUMANS

TRUMANS ego captioning reads clip videos under `data/trumans/.../videos/` and uses the matching action annotation files under `data/trumans/Actions/`.

```bash
ROOT_DIR="$(pwd)/data/trumans" \
MODEL_PATH=OpenGVLab/InternVL2-40B-AWQ \
bash data_processing/video_caption/run_trumans_caption_ego.sh
```

### TASTE-Rob

```bash
ROOT_DIR="$(pwd)/data/taste_rob" \
MODEL_PATH=OpenGVLab/InternVL2-40B-AWQ \
bash data_processing/video_caption/run_taste_rob_caption_ego.sh
```

## 3. Prompt Rewrite

### TASTE-Rob

```bash
ROOT_DIR="$(pwd)/data/taste_rob" \
PROMPT_SUBDIR=prompts \
OUTPUT_FOLDER_NAME=prompts_rewrite \
bash data_processing/video_caption/run_rewrite_taste_rob.sh
```

### TRUMANS

```bash
ROOT_DIR="$(pwd)/data/trumans" \
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

## 4. Output Contract

```text
<sample>/
├── prompts/
│   └── <stem>.txt
├── prompts_rewrite/
│   └── <stem>.txt
└── prompts_aux/
    └── <stem>.txt
```
