<div align="center">

# Dexterous World Models

### CVPR 2026

[📄 Paper](https://arxiv.org/abs/2512.17907) |
[🌐 Project Page](https://snuvclab.github.io/dwm)

</div>

Official implementation of **Dexterous World Models**.

## Installation

```bash
git clone --recursive https://github.com/snuvclab/dwm
cd dwm

# If you already cloned without submodules, run:
# git submodule update --init --recursive

conda create -n dwm python=3.10 -y

pip install -r requirements.txt
```

All commands below assume you run them from the repository root.

## Getting Started

### Data Preparation

See the preprocessing guides:

- [`data_processing/README.md`](data_processing/README.md)
- [`data_processing/video_caption/README.md`](data_processing/video_caption/README.md)

The expected processed sample structure is:

```text
<processed_root>/
└── <sample>/
    ├── videos/
    │   └── <stem>.mp4
    ├── videos_static/
    │   └── <stem>.mp4
    ├── videos_hands/
    │   └── <stem>.mp4
    ├── prompts/
    │   └── <stem>.txt
    ├── prompts_rewrite/
    │   └── <stem>.txt
    ├── video_latents/
    │   └── <stem>.pt
    ├── static_video_latents/
    │   └── <stem>.pt
    ├── hand_video_latents/
    │   └── <stem>.pt
    └── prompt_embeds_rewrite/
        └── <stem>.pt
```

You may place processed data under any root directory you prefer. Training and inference paths can be configured through the example YAML or CLI overrides.

### Training

The main training guide is:

- [`training/cogvideox/README.md`](training/cogvideox/README.md)

Public example config and launcher:

- [`training/cogvideox/configs/examples/dwm_cogvideox_5b_lora.yaml`](training/cogvideox/configs/examples/dwm_cogvideox_5b_lora.yaml)
- [`training/cogvideox/examples/train_static_hand_concat.sh`](training/cogvideox/examples/train_static_hand_concat.sh)

Example smoke run:

```bash
bash training/cogvideox/examples/train_static_hand_concat.sh \
  --debug \
  --override data.data_root=/path/to/processed_root \
  --override logging.report_to=none
```

### Inference

Inference supports either a dataset file or a single sample. Example launcher:

```bash
bash training/cogvideox/examples/infer_static_hand_concat.sh \
  --checkpoint_path outputs/<date>/<experiment> \
  --data_root /path/to/processed_root \
  --dataset_file dataset_files/trumans_test.txt \
  --output_dir outputs_infer/dwm_cogvideox_dataset
```

Example dataset files based on the train and test splits used for the paper models are available under [`dataset_files/`](dataset_files/):

- [`dataset_files/trumans_train.txt`](dataset_files/trumans_train.txt)
- [`dataset_files/taste_rob_train.txt`](dataset_files/taste_rob_train.txt)
- [`dataset_files/trumans_test.txt`](dataset_files/trumans_test.txt)
- [`dataset_files/taste_rob_test.txt`](dataset_files/taste_rob_test.txt)

Single-sample inference:

```bash
python training/cogvideox/inference.py \
  --checkpoint_path outputs/<date>/<experiment> \
  --experiment_config training/cogvideox/configs/examples/dwm_cogvideox_5b_lora.yaml \
  --data_root /path/to/processed_root \
  --video <relative/path/to/videos/00000.mp4> \
  --output_dir outputs_infer/dwm_cogvideox_single
```

## Notes

- The default 5B training path typically needs an `80 GB`-class GPU.
- Relative dataset paths in training and inference are resolved inside `data_root`.
- If you use a custom processed-data root, update `data.data_root` in the example config or pass it via CLI overrides.

## Acknowledgements

We thank the contributors to [VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun), [finetrainers](https://github.com/huggingface/finetrainers), [CogVideo](https://github.com/zai-org/CogVideo), and [Wan](https://github.com/Wan-Video/Wan2.1) for open-sourcing their work.

## Citation

If you find this repository useful, please cite:

```bibtex
@inproceedings{kim2026dwm,
  title={Dexterous World Models},
  author={Kim, Byungjun and Kim, Taeksoo and Lee, Junyoung and Joo, Hanbyul},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```
