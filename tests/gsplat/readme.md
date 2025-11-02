# 📸 GSplat Data Processing & Training Pipeline

This repository provides a simple pipeline for processing `.vrs` files and training a **GSplat** model using **Nerfstudio**.

---

## 🚀 1. Installation

Follow these steps to set up the environment and install dependencies.

```bash
# Create and activate the Nerfstudio environment
conda create --name nerfstudio -y python=3.10
conda activate nerfstudio

# Install PyTorch (CUDA 11.8)
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# Install CUDA dependencies
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# Clone and install Nerfstudio
git clone https://github.com/nerfstudio-project/nerfstudio.git
cd nerfstudio
pip install --upgrade pip setuptools
pip install -e .

# Install GSplat
pip install gsplat
```


## 🚀 2. Folder structure & Running

```
<vrs_dir>/
  ├── mps/slam/
  ├── file.vrs
```

```bash
bash tests/gsplat/get_static.sh --vrs_file "path/to/file.vrs" --num_frames "frame_num"
```

