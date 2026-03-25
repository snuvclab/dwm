# DWM

Standalone DWM refactor for the final CogVideoX static-scene + hand-video concat workflow.

## Environment Setup

```bash
git clone https://github.com/snuvclab/dwm
cd dwm

conda create -n dwm python=3.10 -y
conda activate dwm

pip install -r requirements.txt
```

All documented commands assume you run them from the repository root.

## System Prerequisites

- `ffmpeg` for TASTE-Rob resizing, static-video generation, and some TRUMANS post-processing
- `blender` for TRUMANS rendering
- NVIDIA GPU for training / inference
- Optional extras such as original HaMeR and InternVL2 + `vllm` are documented in the task-specific READMEs below

## Recommended Doc Order

1. [`data_processing/README.md`](data_processing/README.md)
2. [`training/cogvideox/README.md`](training/cogvideox/README.md)

The public smoke path is:

1. Render or resize sample videos into `data_refactor/`
2. Generate `videos_static`, `videos_hands`, `prompts`, and `prompts_rewrite`
3. Preencode videos and prompts
4. Create dataset files under `data_refactor/dataset_files/`
5. Train with the public example config in [`training/cogvideox/configs/examples/static_hand_concat_lora_rewrite.yaml`](training/cogvideox/configs/examples/static_hand_concat_lora_rewrite.yaml) or the wrapper in [`training/cogvideox/examples/train_static_hand_concat.sh`](training/cogvideox/examples/train_static_hand_concat.sh)
6. Run single-video or dataset-file inference with [`training/cogvideox/inference.py`](training/cogvideox/inference.py)
