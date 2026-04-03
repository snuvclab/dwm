# DWM CogVideoX Static-Hand-Concat

This directory contains the standalone DWM CogVideoX path for the final static-scene + hand-video concat model.

All commands below assume you run them from the repository root.

Scope:

- base model: `CogVideoX-Fun-V1.1-5b-InP`
- conditioning: static scene video + hand video
- supported training modes: `lora`, `full`
- standalone path

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
training/cogvideox/configs/examples/dwm_cogvideox_5b_lora.yaml
training/cogvideox/examples/train_static_hand_concat.sh
training/cogvideox/examples/infer_static_hand_concat.sh
```

The public example config resolves `data_root: data` relative to the repository root.

## Base Model

The DWM checkpoint assumes the pretrained `CogVideoX-Fun-V1.1-5b-InP` backbone is available.

Hugging Face download example:

```bash
hf download alibaba-pai/CogVideoX-Fun-V1.1-5b-InP \
  --local-dir ~/.cache/huggingface/hub/models--alibaba-pai--CogVideoX-Fun-V1.1-5b-InP
```

ModelScope download example:

```bash
modelscope download --model PAI/CogVideoX-Fun-V1.1-5b-InP \
  --local_dir ~/.cache/modelscope/hub/models/PAI/CogVideoX-Fun-V1.1-5b-InP
```

If you want to use a downloaded local path instead of the hub id, set `model.base_model_name_or_path` to that directory in the config or via `--override`.

## Example Dataset Files

Example dataset files based on the train and test splits used to train and evaluate the paper models are provided under:

```bash
dataset_files/
```

- [`dataset_files/trumans_train.txt`](../../dataset_files/trumans_train.txt)
- [`dataset_files/taste_rob_train.txt`](../../dataset_files/taste_rob_train.txt)
- [`dataset_files/trumans_test.txt`](../../dataset_files/trumans_test.txt)
- [`dataset_files/taste_rob_test.txt`](../../dataset_files/taste_rob_test.txt)

Their entries remain relative to the actual `data_root`, so use absolute paths if you want to point training or inference to these files directly.

## Training

Plain Python smoke run:

```bash
python training/cogvideox/train_dwm_cogvideox.py \
  --experiment_config training/cogvideox/configs/examples/dwm_cogvideox_5b_lora.yaml \
  --mode debug \
  --override training.max_train_steps=10 data.dataloader_num_workers=0 logging.report_to=none
```

Single-step smoke run that keeps the public 5B path and produces a checkpoint directory that can be loaded by inference:

```bash
python training/cogvideox/train_dwm_cogvideox.py \
  --experiment_config training/cogvideox/configs/examples/dwm_cogvideox_5b_lora.yaml \
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
  --experiment_config training/cogvideox/configs/examples/dwm_cogvideox_5b_lora.yaml \
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
  - `prompt_embeds_rewrite/<stem>.pt`

CogVideoX layout:

```text
<sample>/
├── videos/
├── videos_static/
├── videos_hands/
├── prompts_rewrite/
├── video_latents/
├── static_video_latents/
├── hand_video_latents/
└── prompt_embeds_rewrite/
```

When `load_tensors: true`, the trainer prefers the `.pt` files.

## Inference

The examples below assume the public config still points at the repository-local `data/` tree.

`--checkpoint_path` accepts either:

- a training output directory such as `outputs/<run_name>`
- a `checkpoint-*` subdirectory under that output directory

You can also download the released checkpoint folder from Hugging Face and pass that downloaded directory to `--checkpoint_path`:

- https://huggingface.co/byungjun-kim/DWM-CogVideoX-Fun-5b-LoRA

```bash
hf download byungjun-kim/DWM-CogVideoX-Fun-5b-LoRA --local-dir /path/to/DWM-CogVideoX-Fun-5b-LoRA
```

Dataset-file inference:

```bash
python training/cogvideox/inference.py \
  --checkpoint_path outputs/<date>/<experiment> \
  --experiment_config training/cogvideox/configs/examples/dwm_cogvideox_5b_lora.yaml \
  --dataset_file /abs/path/to/dwm/dataset_files/trumans_test.txt \
  --output_dir outputs_infer/dwm_cogvideox_dataset
```

Single-video inference:

```bash
python training/cogvideox/inference.py \
  --checkpoint_path outputs/<date>/<experiment> \
  --experiment_config training/cogvideox/configs/examples/dwm_cogvideox_5b_lora.yaml \
  --video trumans/<scene>/<action>/videos/00000.mp4 \
  --output_dir outputs_infer/dwm_cogvideox_single
```

Samples prepared with [`data_processing/custom/prepare_custom_hoi_sample.py`](../../data_processing/custom/prepare_custom_hoi_sample.py) can be used directly here, for example:

```bash
--video custom_inputs/videos/<sample>.mp4
```

If you have sufficient VRAM and want better visual quality on custom real-world videos, we recommend DWM WAN 14B first.

Fixed-view raw examples for the custom preprocessing path are provided at:

- [`examples/realworld/realworld_fixed/pour_coke.mp4`](../../examples/realworld/realworld_fixed/pour_coke.mp4)
- [`examples/realworld/realworld_fixed/lift_tumbler.mp4`](../../examples/realworld/realworld_fixed/lift_tumbler.mp4)

A released dynamic-viewpoint example subset is provided under [`examples/realworld/realworld_dynamic/`](../../examples/realworld/realworld_dynamic). The dynamic-viewpoint data was created with ARIA. For details, please refer to the paper.

The same prepared sample also comes with a one-line dataset file under:

```bash
data/dataset_files/custom_inputs/<sample>.txt
```

The released dynamic-viewpoint subset can also be used directly:

```bash
--data_root examples/realworld --dataset_file examples/realworld/realworld_dynamic.txt
```

Example inference launcher:

```bash
bash training/cogvideox/examples/infer_static_hand_concat.sh \
  --checkpoint_path outputs/<date>/<experiment> \
  --dataset_file /abs/path/to/dwm/dataset_files/trumans_test.txt \
  --output_dir outputs_infer/dwm_cogvideox_dataset
```

`--dataset_file` can be either an absolute path or a repo-relative path such as `dataset_files/trumans_test.txt`. Relative `--video` inputs are still resolved inside `data_root`.

Common path overrides:

```bash
training: --override data.data_root=/abs/path/to/data
inference: --data_root /abs/path/to/data
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
