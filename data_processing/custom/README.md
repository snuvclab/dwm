# Custom Videos

Use this path when you want to prepare a single custom hand-object interaction video for DWM inference.

The script below:

- reads an arbitrary input video
- starts from `--start_frame` (default `0`)
- resamples it to `8` FPS
- writes a `49`-frame clip at `720x480`
- creates `videos_static/` by repeating the first frame
- runs HaMeR to create `videos_hands/`
- runs InternVL2 captioning to create `prompts/`
- runs prompt rewrite to create `prompts_rewrite/`
- writes a one-line dataset file under `data/dataset_files/custom_inputs/`

All commands below assume you run them from the repository root.

## Command

A fixed-view raw example is provided at:

```text
examples/realworld/realworld_fixed/pour_coke.mp4
```

```bash
python data_processing/custom/prepare_custom_hoi_sample.py \
  --input_video examples/realworld/realworld_fixed/pour_coke.mp4 \
  --sample_name pour_coke
```

A small processed dynamic-viewpoint subset is also provided under:

```text
examples/realworld/realworld_dynamic/
```

This subset was created with ARIA. For details, please refer to the paper.

The default output layout is:

```text
data/custom_inputs/
├── metadata/
│   └── pour_coke.json
├── prompts/
│   └── pour_coke.txt
├── prompts_rewrite/
│   └── pour_coke.txt
├── videos/
│   └── pour_coke.mp4
├── videos_hands/
│   └── pour_coke.mp4
└── videos_static/
    └── pour_coke.mp4
```

The script also writes:

```text
data/dataset_files/custom_inputs/pour_coke.txt
```

with one line:

```text
custom_inputs/videos/pour_coke.mp4
```

## Useful Options

```bash
--start_frame 0
--target_fps 8
--target_frames 49
--width 720
--height 480
--resize_policy auto
--hand_backend original
--skip_existing
```

`--resize_policy auto` compares center-crop and pad loss, then chooses the lower-loss option.

## Prerequisites

- `ffmpeg` and `ffprobe`
- HaMeR setup from [`data_processing/README.md`](../README.md)
- captioning and rewrite setup from [`data_processing/video_caption/README.md`](../video_caption/README.md)

## Inference

The prepared sample can be used directly with the existing inference paths.

If you have sufficient VRAM and want better visual quality, we recommend DWM WAN 14B for custom real-world videos.

### CogVideoX

```bash
python training/cogvideox/inference.py \
  --checkpoint_path /path/to/checkpoint \
  --experiment_config training/cogvideox/configs/examples/dwm_cogvideox_5b_lora.yaml \
  --video custom_inputs/videos/pour_coke.mp4 \
  --output_dir outputs_infer/custom_cogvideox_demo
```

Dataset-file inference also works:

```bash
python training/cogvideox/inference.py \
  --checkpoint_path /path/to/checkpoint \
  --experiment_config training/cogvideox/configs/examples/dwm_cogvideox_5b_lora.yaml \
  --dataset_file data/dataset_files/custom_inputs/pour_coke.txt \
  --output_dir outputs_infer/custom_cogvideox_demo
```

The released dynamic-viewpoint example can be used directly without preprocessing:

```bash
python training/cogvideox/inference.py \
  --checkpoint_path /path/to/checkpoint \
  --experiment_config training/cogvideox/configs/examples/dwm_cogvideox_5b_lora.yaml \
  --data_root examples/realworld \
  --dataset_file examples/realworld/realworld_dynamic.txt \
  --output_dir outputs_infer/custom_cogvideox_dynamic_demo
```

### WAN

```bash
python training/wan/inference.py \
  --checkpoint_path /path/to/checkpoint \
  --experiment_config training/wan/configs/examples/dwm_wan_14b_lora.yaml \
  --video_path custom_inputs/videos/pour_coke.mp4 \
  --data_root data \
  --output_dir outputs_infer/custom_wan_demo
```

Dataset-file inference also works:

```bash
python training/wan/inference.py \
  --checkpoint_path /path/to/checkpoint \
  --experiment_config training/wan/configs/examples/dwm_wan_14b_lora.yaml \
  --dataset_file data/dataset_files/custom_inputs/pour_coke.txt \
  --data_root data \
  --output_dir outputs_infer/custom_wan_demo
```

The released dynamic-viewpoint example can be used directly without preprocessing:

```bash
python training/wan/inference.py \
  --checkpoint_path /path/to/checkpoint \
  --experiment_config training/wan/configs/examples/dwm_wan_14b_lora.yaml \
  --data_root examples/realworld \
  --dataset_file examples/realworld/realworld_dynamic.txt \
  --output_dir outputs_infer/custom_wan_dynamic_demo
```
