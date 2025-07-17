
## Installation

1. Path.

```shell
cd data_processing/EgoLife/EgoGPT

```

2. Install the dependencies.

```shell
conda create -n egogpt python=3.10
conda activate egogpt
pip install -e .
```



## Quick Start

### Download & Setup

1. Download EgoGPT-7b speech encoder (not used but cannot disentangle easily).
```shell
wget https://huggingface.co/lmms-lab/EgoGPT-7b-EgoIT-EgoLife/resolve/main/speech_encoder/large-v3.pt
```

### Inference

```shell
python inference.py --video_path /virtual_lab/jhb_vclab/world_model/data/trumans/250712_sample/0a761819-05d1-4647-889b-a726747201b1/2023-01-14@22-06-10/sequences/videos/00007.mp4 --query "Please describe the video in detail."
```
