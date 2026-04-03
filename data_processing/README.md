# Data Processing

All commands below assume you run them from the repository root.

## Output Layout

The examples below assume processed data is stored under `data/`.

Common processed sample:

```text
data/
├── custom/
│   └── Custom/
│       └── <sample>/
│           ├── videos/
│           ├── videos_static/
│           ├── videos_hands/
│           ├── prompts/
│           ├── prompts_rewrite/
│           └── metadata.json
├── taste_rob_orig/
│   └── ...
├── trumans/
│   └── <scene>/<action>/
│       ├── videos/
│       ├── videos_static/
│       ├── videos_hands/
│       ├── prompts/
│       └── prompts_rewrite/
└── taste_rob/
    └── <group>/<scene>/
        ├── videos/
        ├── videos_static/
        ├── videos_hands/
        ├── prompts/
        └── prompts_rewrite/
```

CogVideoX preencoded siblings:

```text
<sample>/
├── video_latents/
├── static_video_latents/
├── hand_video_latents/
└── prompt_embeds_rewrite/
```

WAN preencoded siblings:

```text
<sample>/
├── video_latents_wan/
├── static_video_latents_wan/
├── hand_video_latents_wan/
├── prompt_embeds_rewrite_wan/
└── fun_inp_i2v_latents_wan/
```

## 1. TRUMANS

Download TRUMANS first. We follow the dataset release procedure from [`TRUMANS official repo`](https://github.com/jnnan/trumans_utils).

### 1.1 Blender and HSI addon

TRUMANS rendering requires Blender and the HSI addon from the TRUMANS repository.

Tested version:

- `Blender 4.1.1`

Ubuntu example:

1. Download Blender from [`download.blender.org`](https://download.blender.org/).
2. Download the HSI addon from [`HSI_addon-zzy.zip`](https://github.com/jnnan/trumans_utils/blob/main/HSI_addon-zzy.zip).
3. Extract both and install the addon into the Blender addons directory.
4. Install `torch` and `pytorch3d` into Blender's bundled Python environment.

```bash
sudo tar -xvJf blender-4.1.1-linux-x64.tar.xz -C /opt/
sudo ln -s /opt/blender-4.1.1-linux-x64/blender /usr/local/bin/blender

unzip HSI_addon-zzy.zip
sudo cp -r HSI_addon-zzy /opt/blender-4.1.1-linux-x64/4.1/scripts/addons/

blender -b --python data_processing/trumans/activate_hsi_addon.py
```

Check the Blender Python path first:

```bash
blender --background --python-expr "import sys; print(sys.executable)"
```

Then install `torch` and `pytorch3d` into that Python environment. For example:

```bash
BLENDER_PYTHON_PATH=/opt/blender-4.1.1-linux-x64/4.1/python/bin/python3.11

${BLENDER_PYTHON_PATH} -m pip install \
  torch==2.2.2+cu121 torchvision==0.17.2+cu121 \
  --index-url https://download.pytorch.org/whl/cu121

${BLENDER_PYTHON_PATH} -m pip install \
  --extra-index-url https://miropsota.github.io/torch_packages_builder \
  pytorch3d==0.7.5+pt2.2.2cu121
```

Adjust the CUDA and PyTorch versions to match your environment.

After this, `blender` should be available on `PATH`, the HSI addon should be enabled, and Blender's Python should have the required packages for TRUMANS rendering.

### 1.2 Dataset placement

After download, place the scene `.blend` files under:

```text
data/trumans/Recordings_blend/
```

The rendering scripts below assume this layout.

### 1.3 Rendering

The TRUMANS wrapper uses three renderer scripts:

- `blender_ego_video_render.py`: dynamic ego-scene RGB renderer for `videos/`
- `blender_ego_static.py`: static clip-first RGB renderer for `videos_static/`
- `blender_ego_hand.py`: hand-only renderer for `videos_hands/`

For TRUMANS, `videos_hands/` is produced directly during Blender rendering. No separate hand-mesh estimation step is required.

For a quick test, it is usually better to render a single clip first.

The examples below render one `49`-frame clip with the final training settings:

- `clip_length=49`
- `frame_skip=3`
- `fps=8`

With `start_frame=1` and `end_frame=145`, Blender renders exactly one `49`-frame window.

#### Dynamic RGB

```bash
blender -b data/trumans/Recordings_blend/<scene>/<scene>.blend \
  --python data_processing/trumans/blender_ego_video_render.py -- \
  --save-path ./data/trumans/ego_render_fov90 \
  --animation_index 0 \
  --start_frame 1 --end_frame 145 \
  --frame-skip 3 \
  --width 720 --height 480 \
  --samples 64 \
  --video-output --direct-clips \
  --clip-length 49 --clip-stride 25 --fps 8
```

TRUMANS RGB renders include actor-cast shadows. In some cases these shadows can appear overly strong at inference time. If you do not want this, render training data with `--no-actor-shadow` to keep the actor visible while removing only actor-cast shadows. We did not enable this flag for the paper results.

For batched RGB rendering, you can pass the flag through the launcher:

```bash
bash data_processing/trumans/run_trumans_render_videos.sh --no-actor-shadow
```

#### Static RGB

```bash
blender -b data/trumans/Recordings_blend/<scene>/<scene>.blend \
  --python data_processing/trumans/blender_ego_static.py -- \
  --save-path ./data/trumans/ego_render_fov90 \
  --animation_index 0 \
  --start_frame 1 --end_frame 145 \
  --frame-skip 3 \
  --width 720 --height 480 \
  --samples 64 \
  --video-output --direct-clips \
  --clip-length 49 --clip-stride 25 --fps 8
```

#### Hand Video

```bash
blender -b data/trumans/Recordings_blend/<scene>/<scene>.blend \
  --python data_processing/trumans/blender_ego_hand.py -- \
  --save-path ./data/trumans/ego_render_fov90 \
  --animation_index 0 \
  --start_frame 1 --end_frame 145 \
  --frame-skip 3 \
  --width 720 --height 480 \
  --samples 64 \
  --direct-clips \
  --clip-length 49 --stride 25 --fps 8
```

For full rendering, use the batch launchers:

```bash
bash data_processing/trumans/run_trumans_render_videos.sh
bash data_processing/trumans/run_trumans_render_static.sh
bash data_processing/trumans/run_trumans_render_hands.sh
```

`--auto-split-clips` is mainly for batch rendering. It renders reusable frames once and then splits overlapping clips, which is useful for throughput on dynamic RGB and hand rendering. For quick inspection of a single clip, use `--direct-clips`. Static rendering stays clip-first and should also be run with `--direct-clips`.

The small wrapper below can also be used to generate a few clips end-to-end:

```bash
python data_processing/trumans/render_smoke_trumans.py \
  --recordings_root /path/to/TRUMANS/Recordings_blend \
  --output_root data/trumans \
  --clip_count 6 \
  --gpu 0
```

Useful options:

```bash
--scene <scene_name>
--animation <animation_name>
--clip_length 49
--clip_stride 25
--frame_skip 3
```

## 2. TASTE-Rob

Follow the dataset access instructions from the [official TASTE-Rob repository](https://github.com/GAP-LAB-CUHK-SZ/TASTE-Rob).

For the models reported in the paper, we used the `DoubleHand` subset only.

### 2.1 Download

First, submit the dataset form. After approval, the authors provide a SharePoint or Baidu Netdisk download link.

Use the official TASTE-Rob download tool with the link you receive from the authors.

### 2.2 Raw dataset placement

After download, place the raw dataset under:

```text
data/taste_rob_orig/
```

The preprocessing code expects raw videos to be found under this root, typically with a structure like:

```text
data/taste_rob_orig/
├── SingleHand/
│   └── <scene>/
│       └── *.mp4
└── DoubleHand/
    └── <scene>/
        └── *.mp4
```

### 2.3 Processing

The preprocessing step reads raw videos from `data/taste_rob_orig/` and writes processed data to `data/taste_rob/`.

For each selected video, the code:

- resizes and pads it to `720x480`
- converts it to `8 FPS`
- truncates or samples it to `49` frames
- writes the result to `videos/`
- creates a matching static clip in `videos_static/`

For a quick test on a small subset:

```bash
python data_processing/taste_rob/prepare_smoke_taste_rob.py \
  --input_dir data/taste_rob_orig \
  --output_dir data/taste_rob \
  --clip_count 6
```

For full preprocessing over all videos:

```bash
python data_processing/taste_rob/resize_videos_taste_rob.py \
  --input_dir data/taste_rob_orig \
  --output_dir data/taste_rob \
  --target_width 720 \
  --target_height 480 \
  --target_frames 49 \
  --output_fps 8 \
  --skip_existing
```

## 3. TASTE-Rob HaMeR Pipeline

Unlike TRUMANS, TASTE-Rob does not come with hand-only videos. After resizing the raw videos, run HaMeR to estimate hand meshes and write `videos_hands/`.

### 3.1 Install original HaMeR

This repository includes HaMeR under `third_party/hamer` as a submodule. Clone DWM with `--recursive`, or run `git submodule update --init --recursive` if you already cloned the repository.

Then install HaMeR:

```bash
cd third_party/hamer
pip install --no-build-isolation --no-deps -e .[all]
pip install --no-deps -v -e third-party/ViTPose/
```

Then fetch the demo assets:

```bash
bash fetch_demo_data.sh
```

You also need the MANO right-hand model. Download `MANO_RIGHT.pkl` from the [MANO website](https://mano.is.tue.mpg.de/) and place it at:

```text
third_party/hamer/_DATA/data/mano/MANO_RIGHT.pkl
```

### 3.2 Run HaMeR and generate `videos_hands`

After installation, run:

```bash
bash data_processing/hands/run_render_hands_hamer.sh \
  --backend original \
  --data_root data/taste_rob \
  --skip_existing
```

This command runs HaMeR prediction on `videos/*.mp4` and writes the resulting hand-only videos to `videos_hands/`.

For a quick test on already resized videos, run HaMeR on `data/taste_rob` directly:

```bash
python data_processing/hands/render_videos_hands_hamer_original.py \
  --data_root data/taste_rob \
  --skip_existing
```

### 3.3 Faster mediapipe option

If you want faster processing, you can use the mediapipe-based HaMeR pipeline from [`bjkim95/hamer-mediapipe`](https://github.com/bjkim95/hamer-mediapipe). This repository already includes it under `third_party/hamer-mediapipe` when cloned with `--recursive`.

This backend can be faster, but the output quality may be worse than the original HaMeR backend.

To use it, change the backend flag:

```bash
bash data_processing/hands/run_render_hands_hamer.sh \
  --backend mediapipe \
  --data_root data/taste_rob \
  --skip_existing
```

## 4. Custom Videos

For a single custom hand-object interaction video, use:

- [`data_processing/custom/prepare_custom_hoi_sample.py`](custom/prepare_custom_hoi_sample.py)
- [`data_processing/custom/README.md`](custom/README.md)

This path prepares a single clip under `data/custom_inputs/{videos,videos_static,videos_hands,prompts,prompts_rewrite}/` and writes a one-line dataset file under `data/dataset_files/custom_inputs/`.

## 5. Captions And Rewrite

See [`data_processing/video_caption/README.md`](video_caption/README.md).

## 6. Preencoding

CogVideoX preencoding:

```bash
bash data_processing/encode_with_cogvideox.sh \
  --dataset_type trumans \
  --data_root data/trumans \
  --modalities videos static_videos hand_videos prompts \
  --prompt_subdir prompts_rewrite

bash data_processing/encode_with_cogvideox.sh \
  --dataset_type taste_rob \
  --data_root data/taste_rob \
  --modalities videos static_videos hand_videos prompts \
  --prompt_subdir prompts_rewrite
```

This writes:

- `video_latents/`
- `static_video_latents/`
- `hand_video_latents/`
- `prompt_embeds_rewrite/`

WAN preencoding:

```bash
bash data_processing/encode_with_wan.sh \
  --dataset_type trumans \
  --data_root data/trumans \
  --modalities videos static_videos hand_videos prompts \
  --prompt_subdir prompts_rewrite

bash data_processing/encode_with_wan.sh \
  --dataset_type taste_rob \
  --data_root data/taste_rob \
  --modalities videos static_videos hand_videos prompts \
  --prompt_subdir prompts_rewrite
```

Preencoding is optional for both CogVideoX and WAN, but it usually lets you use larger training batch sizes and improves iteration speed because training can reuse cached latents and prompt embeddings.

This writes WAN-specific siblings:

- `video_latents_wan/`
- `static_video_latents_wan/`
- `hand_video_latents_wan/`
- `prompt_embeds_rewrite_wan/`
- `fun_inp_i2v_latents_wan/`

## 7. Dataset Files

```bash
python data_processing/create_dataset_file.py \
  --dataset_type trumans \
  --data_root data/trumans \
  --output_dir data/dataset_files/trumans \
  --output_base_dir "$(pwd)/data"

python data_processing/create_dataset_file.py \
  --dataset_type taste_rob \
  --data_root data/taste_rob \
  --output_dir data/dataset_files/taste_rob \
  --output_base_dir "$(pwd)/data"
```

After dataset files are created, continue with [`training/cogvideox/README.md`](../training/cogvideox/README.md) or [`training/wan/README.md`](../training/wan/README.md).
