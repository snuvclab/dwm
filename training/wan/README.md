# DWM WAN Static-Hand-Concat

This directory contains the standalone DWM WAN path for the final static-scene + hand-video concat model.

Key files:

- `train_dwm_wan.py`: training entrypoint
- `inference.py`: dataset-file and single-video inference entrypoint
- `models/wan_transformer3d_with_conditions.py`: final concat transformer
- `pipeline/pipeline_wan_fun_inpaint_hand_concat.py`: final hand-concat pipeline

Public example config:

```bash
training/wan/configs/examples/dwm_wan_14b_lora.yaml
```

Example launchers:

```bash
training/wan/examples/train_static_hand_concat.sh
training/wan/examples/infer_static_hand_concat.sh
```

## Base Model

The DWM checkpoint assumes the pretrained `Wan2.1-Fun-V1.1-14B-InP` backbone is available.

Hugging Face download example:

```bash
hf download alibaba-pai/Wan2.1-Fun-V1.1-14B-InP \
  --local-dir ~/.cache/huggingface/hub/models--alibaba-pai--Wan2.1-Fun-V1.1-14B-InP
```

If you prefer ModelScope, the public config defaults to the following cache path:

```bash
~/.cache/modelscope/hub/models/PAI/Wan2.1-Fun-V1.1-14B-InP/
```

ModelScope download example:

```bash
modelscope download --model PAI/Wan2.1-Fun-V1.1-14B-InP \
  --local_dir ~/.cache/modelscope/hub/models/PAI/Wan2.1-Fun-V1.1-14B-InP
```

If you download the backbone to a different location, update `pipeline.base_model_name_or_path` in the config or pass `--base_model_path` during inference.

For dataset preparation and WAN preencoding, see [`data_processing/README.md`](../../data_processing/README.md).

## Training Contract

The WAN trainer expects these sibling paths per sample:

- `videos/<stem>.mp4`
- `videos_static/<stem>.mp4`
- `videos_hands/<stem>.mp4`
- `prompts_rewrite/<stem>.txt`
- optional preencoded siblings:
  - `video_latents_wan/<stem>.pt`
  - `static_video_latents_wan/<stem>.pt`
  - `hand_video_latents_wan/<stem>.pt`
  - `prompt_embeds_rewrite_wan/<stem>.pt`
  - `fun_inp_i2v_latents_wan/<stem>.pt`

WAN layout:

```text
<sample>/
├── videos/
├── videos_static/
├── videos_hands/
├── prompts_rewrite/
├── video_latents_wan/
├── static_video_latents_wan/
├── hand_video_latents_wan/
├── prompt_embeds_rewrite_wan/
└── fun_inp_i2v_latents_wan/
```

When `load_tensors: true`, the trainer prefers these WAN-specific `.pt` files.

Preencoding is optional, but it usually allows larger training batch sizes and faster iterations because the trainer can load cached latents and prompt embeddings instead of re-encoding inputs on the fly.

You can either use a local training output directory or download the released checkpoint folder from Hugging Face and pass that directory to `--checkpoint_path`:

- https://huggingface.co/byungjun-kim/DWM-Wan2.1-Fun-14b-LoRA

```bash
hf download byungjun-kim/DWM-Wan2.1-Fun-14b-LoRA --local-dir /path/to/DWM-Wan2.1-Fun-14b-LoRA
```

Custom samples prepared with [`data_processing/custom/prepare_custom_hoi_sample.py`](../../data_processing/custom/prepare_custom_hoi_sample.py) can be used directly with WAN inference:

```bash
python training/wan/inference.py \
  --checkpoint_path /path/to/checkpoint \
  --experiment_config training/wan/configs/examples/dwm_wan_14b_lora.yaml \
  --video_path custom_inputs/videos/<sample>.mp4 \
  --data_root data \
  --output_dir outputs_infer/custom_wan_demo
```

If you have sufficient VRAM and want better visual quality, we recommend DWM WAN 14B for custom real-world videos.

A fixed-view raw example for the custom preprocessing path is provided at [`examples/realworld/realworld_fixed/pour_coke.mp4`](../../examples/realworld/realworld_fixed/pour_coke.mp4).

A released dynamic-viewpoint example subset is provided under [`examples/realworld/realworld_dynamic/`](../../examples/realworld/realworld_dynamic). The dynamic-viewpoint data was created with ARIA. For details, please refer to the paper.

The same prepared sample also comes with a one-line dataset file under:

```bash
data/dataset_files/custom_inputs/<sample>.txt
```

The released dynamic-viewpoint subset can also be used directly:

```bash
python training/wan/inference.py \
  --checkpoint_path /path/to/checkpoint \
  --experiment_config training/wan/configs/examples/dwm_wan_14b_lora.yaml \
  --data_root examples/realworld \
  --dataset_file examples/realworld/realworld_dynamic.txt \
  --output_dir outputs_infer/custom_wan_dynamic_demo
```
