# DWM CogVideoX Static-Hand-Concat

This directory contains the standalone DWM CogVideoX path for the final static-scene + hand-video concat model.

All commands below assume you run them from the repository root.

Scope:

- base model: `CogVideoX-Fun-V1.1-5b-InP`
- conditioning: static scene video + hand video
- supported training modes: `lora`, `full`
- standalone path, without importing `training/cogvideox_static_pose`

## Key Files

- `train_dwm_cogvideox.py`: training entrypoint
- `inference.py`: dataset-file and single-video inference entrypoint
- `static_hand_dataset.py`: dataset loader for the final contract
- `static_hand_utils.py`: shared loading and checkpoint helpers
- `pipeline/static_hand_concat.py`: final inference pipeline
- `models/static_hand_concat_transformer.py`: final transformer wrapper

## Canonical Configs

Recommended public config:

```bash
training/cogvideox/configs/examples/static_hand_concat_lora_rewrite.yaml
training/cogvideox/examples/train_static_hand_concat.sh
training/cogvideox/examples/infer_static_hand_concat.sh
```

The public example config resolves `data_root: data_refactor` relative to the repository root.

Internal / advanced template:

```bash
training/cogvideox/configs/experiments/cogvideox_fun_static_hand_concat_lora_rewrite_prompt.yaml
```

Adapt its dataset paths before using it outside the original internal environment.

## Training

Plain Python smoke run:

```bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate dwm

python training/cogvideox/train_dwm_cogvideox.py \
  --experiment_config training/cogvideox/configs/examples/static_hand_concat_lora_rewrite.yaml \
  --mode debug \
  --override training.max_train_steps=10 data.dataloader_num_workers=0 logging.report_to=none
```

Single-step smoke run that keeps the public 5B path and produces a checkpoint directory that can be loaded by inference:

```bash
python training/cogvideox/train_dwm_cogvideox.py \
  --experiment_config training/cogvideox/configs/examples/static_hand_concat_lora_rewrite.yaml \
  --mode debug \
  --override \
    training.batch_size=1 \
    training.gradient_accumulation_steps=1 \
    training.max_train_steps=1 \
    training.lora_rank=8 \
    training.lora_alpha=8 \
    data.dataloader_num_workers=0 \
    data.num_validation_videos=0 \
    data.max_validation_videos=0 \
    data.init_validation_steps=0 \
    data.validation_steps=0 \
    logging.report_to=none
```

The default 5B backbone typically needs an 80 GB class GPU. The final output directory contains both `pytorch_lora_weights.safetensors` and `non_lora_weights.safetensors`, so inference can load either the output directory itself or a `checkpoint-*` subdirectory.

Torchrun smoke run:

```bash
torchrun --standalone --nproc_per_node=2 \
  training/cogvideox/train_dwm_cogvideox.py \
  --experiment_config training/cogvideox/configs/examples/static_hand_concat_lora_rewrite.yaml \
  --mode slurm_test \
  --override training.max_train_steps=10 data.max_validation_videos=0 logging.report_to=none
```

Example launcher with the same `--debug` and `--slurm_test` flags:

```bash
bash training/cogvideox/examples/train_static_hand_concat.sh --debug
bash training/cogvideox/examples/train_static_hand_concat.sh --slurm_test
```

## Training Contract

The final trainer expects these sibling paths per sample:

- `videos/<stem>.mp4`
- `videos_static/<stem>.mp4`
- `videos_hands/<stem>.mp4`
- `prompts_rewrite/<stem>.txt`
- optional preencoded siblings:
  - `video_latents/<stem>.pt`
  - `static_video_latents/<stem>.pt`
  - `hand_video_latents/<stem>.pt`
  - `prompt_embeds_prompts_rewrite/<stem>.pt`

When `load_tensors: true`, the trainer prefers the `.pt` files.

## Inference

The examples below assume the public config still points at the repository-local `data_refactor/` tree.

`--checkpoint_path` accepts either:

- a training output directory such as `outputs/260219/cogvideox_hand_concat_lora_rewrite_prompt_debug`
- a `checkpoint-*` subdirectory under that output directory

Dataset-file inference:

```bash
python training/cogvideox/inference.py \
  --checkpoint_path outputs/<date>/<experiment> \
  --experiment_config training/cogvideox/configs/examples/static_hand_concat_lora_rewrite.yaml \
  --dataset_file dataset_files/trumans_smoke/test.txt \
  --output_dir outputs_infer/dwm_cogvideox_dataset
```

Single-video inference:

```bash
python training/cogvideox/inference.py \
  --checkpoint_path outputs/<date>/<experiment> \
  --experiment_config training/cogvideox/configs/examples/static_hand_concat_lora_rewrite.yaml \
  --video trumans/<scene>/<action>/videos/00000.mp4 \
  --output_dir outputs_infer/dwm_cogvideox_single
```

Example inference launcher:

```bash
bash training/cogvideox/examples/infer_static_hand_concat.sh \
  --checkpoint_path outputs/<date>/<experiment> \
  --dataset_file dataset_files/trumans_smoke/test.txt \
  --output_dir outputs_infer/dwm_cogvideox_dataset
```

`--dataset_file` and relative `--video` inputs are resolved inside `data_root`. With the public config that means they should be relative to `data_refactor/`, not prefixed with `data_refactor/` again.

Common path overrides:

```bash
training: --override data.data_root=/abs/path/to/data_refactor
inference: --data_root /abs/path/to/data_refactor
```

Other inference overrides:

```bash
--prompt "custom rewritten prompt"
--prompt_file path/to/prompt.txt
--static_video path/to/videos_static/00000.mp4
--hand_video path/to/videos_hands/00000.mp4
--num_inference_steps 50
--guidance_scale 6.0
```
