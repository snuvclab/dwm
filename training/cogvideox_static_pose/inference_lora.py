import os
import sys
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import (CogVideoXDDIMScheduler, DDIMScheduler,
                       DPMSolverMultistepScheduler, CogVideoXDPMScheduler,
                       EulerAncestralDiscreteScheduler, EulerDiscreteScheduler,
                       PNDMScheduler)
from PIL import Image
import imageio.v3 as iio
from diffusers.utils import convert_unet_state_dict_to_peft

from videox_fun.models import (AutoencoderKLCogVideoX,
                              CogVideoXTransformer3DModel, T5EncoderModel,
                              T5Tokenizer)
from videox_fun.pipeline import (CogVideoXFunPipeline,
                                CogVideoXFunInpaintPipeline)
from videox_fun.utils.utils import get_video_to_video_latent, save_videos_grid
from peft import get_peft_model_state_dict, set_peft_model_state_dict, LoraConfig


def morph2d(x, kernel_size=3, mode="dilate"):
    """
    Morphological operation per frame (2D) on a 5D tensor (b, c, f, h, w)
    mode ∈ {"dilate", "erode"}
    """
    assert mode in {"dilate", "erode"}
    pad = kernel_size // 2
    # Flatten (b*f*c) as batch dimension
    b, c, f, h, w = x.shape
    x_ = x.permute(0, 2, 1, 3, 4).reshape(b * f * c, 1, h, w)

    weight = torch.ones((1, 1, kernel_size, kernel_size), device=x.device)

    if mode == "dilate":
        y = F.conv2d(x_, weight, padding=pad)
        y = (y > 0).float()
    else:  # erosion
        y = F.conv2d(x_, weight, padding=pad)
        y = (y == kernel_size * kernel_size).float()

    # Reshape back
    y = y.reshape(b, f, c, h, w).permute(0, 2, 1, 3, 4)
    return y


parser = argparse.ArgumentParser()
parser.add_argument("--lora_path", type=str, required=True, help="Path to the LoRA weights directory")
parser.add_argument("--txt_path", type=str, required=True, help="Path to the text file containing validation video list")

args = parser.parse_args()

lora_path = args.lora_path
save_path = os.path.join(lora_path, "eval")
os.makedirs(save_path, exist_ok=True)


model_name = "/virtual_lab/jhb_vclab/taeksoo/.cache/huggingface/hub/models--alibaba-pai--CogVideoX-Fun-V1.1-5b-InP/snapshots/b3798d82878e57443314b29d73633a165dd4c008"
weight_dtype = torch.bfloat16

negative_prompt         = "The video is not of a high quality, it has a low resolution. Watermark present in each frame. The background is solid. Strange body and strange trajectory. Distortion. "
guidance_scale          = 6.0
seed                    = 43
num_inference_steps     = 50
device = "cuda"
sample_size         = [480, 720]
video_length        = 49
fps = 8


transformer = CogVideoXTransformer3DModel.from_pretrained(
    model_name, 
    subfolder="transformer",
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
).to(weight_dtype)
vae = AutoencoderKLCogVideoX.from_pretrained(
    model_name, 
    subfolder="vae"
).to(weight_dtype)
tokenizer = T5Tokenizer.from_pretrained(
    model_name, subfolder="tokenizer"
)
text_encoder = T5EncoderModel.from_pretrained(
    model_name, subfolder="text_encoder", torch_dtype=weight_dtype
)
scheduler = CogVideoXDPMScheduler.from_pretrained(
    model_name, 
    subfolder="scheduler"
)

# proj_path = os.path.join(lora_path, "projection_layer_weights.pt")
non_lora_file = os.path.join(lora_path, "non_lora_weights.pt")
proj_path = non_lora_file

if os.path.exists(proj_path) or os.path.exists(non_lora_file):
    with torch.no_grad():
        new_conv_in = torch.nn.Conv2d( # 1 for object mask
            transformer.patch_embed.proj.in_channels + 16, transformer.patch_embed.proj.out_channels, \
                transformer.patch_embed.proj.kernel_size, transformer.patch_embed.proj.stride, transformer.patch_embed.proj.padding
        )
        transformer.patch_embed.proj = new_conv_in
    proj_dict = torch.load(proj_path, map_location="cpu")
    proj_dict['weight'] = proj_dict['patch_embed.proj.weight']
    proj_dict['bias'] = proj_dict['patch_embed.proj.bias']
    model_state_dict = transformer.patch_embed.proj.state_dict()
    loaded_keys = []
    for name, param_data in proj_dict.items():
        if name in model_state_dict:
            model_state_dict[name].copy_(param_data)
            loaded_keys.append(name)
    # transformer.patch_embed.proj.load_state_dict(torch.load(proj_path, weights_only=True))
    transformer.patch_embed.proj.to(dtype=torch.bfloat16, device="cuda")


# Add LoRA to attention layers (same config as training)
transformer_lora_config = LoraConfig(
    r=64,
    lora_alpha=64,
    init_lora_weights=True,
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],
)
transformer.add_adapter(transformer_lora_config)

# Try to load from safetensors first
safetensors_path = os.path.join(lora_path, "pytorch_lora_weights.safetensors")
if os.path.exists(safetensors_path):
    from safetensors.torch import load_file
    lora_state_dict = load_file(safetensors_path)
transformer_state_dict = convert_unet_state_dict_to_peft(lora_state_dict)
set_peft_model_state_dict(transformer, transformer_state_dict, adapter_name="default")

# non_lora_file = os.path.join(lora_path, "non_lora_weights.pt")
if os.path.exists(non_lora_file):
    non_lora_state_dict = torch.load(non_lora_file, map_location="cpu")
    model_state_dict = transformer.state_dict()
    loaded_keys = []
    for name, param_data in non_lora_state_dict.items():
        if name in model_state_dict:
            model_state_dict[name].copy_(param_data)
            loaded_keys.append(name)

pipeline = CogVideoXFunInpaintPipeline(
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        transformer=transformer,
        scheduler=scheduler,
    ).to(device)
generator = torch.Generator(device=device).manual_seed(seed)


# with torch.no_grad():
#     new_conv_in = torch.nn.Conv2d( 
#         pipeline.transformer.patch_embed.proj.in_channels + 16, pipeline.transformer.patch_embed.proj.out_channels, \
#             pipeline.transformer.patch_embed.proj.kernel_size, pipeline.transformer.patch_embed.proj.stride, pipeline.transformer.patch_embed.proj.padding
#     )
#     pipeline.transformer.patch_embed.proj = new_conv_in
# pipeline.transformer.patch_embed.proj.load_state_dict(torch.load(os.path.join(args.lora_path, "patch_embed_conv2d.pth"), weights_only=True))
# pipeline.transformer.patch_embed.proj.to(dtype=torch.bfloat16, device="cuda")

# pipeline.load_lora_weights(args.lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="hand_lora")
# pipeline.set_adapters(["hand_lora"], [1.0])


with open(args.txt_path, "r") as f:
    lines = f.readlines()
validation_video_list = [line.strip() for line in lines]

for validation_video in validation_video_list:

    # validation_video = "inpaint_video.mp4"
    # validation_video_mask = "inpaint_video_mask.mp4"
    static_video_path = validation_video.replace("videos", "videos_static")
    prompt_path = validation_video.replace(".mp4", ".txt").replace("videos", "prompts_ego")
    video_mask_path = validation_video.replace("videos", "videos_hands_mask")
    hand_video_path = validation_video.replace("videos", "videos_hands")
    with open(os.path.join("/virtual_lab/jhb_vclab/world_model/data", prompt_path), "r") as f:
        prompt = f.read().strip()

    input_video,  _, _, _ = get_video_to_video_latent(os.path.join("/virtual_lab/jhb_vclab/world_model/data", static_video_path), video_length=video_length, sample_size=sample_size, validation_video_mask=None, fps=fps)
    # static_video = iio.imread(os.path.join("/virtual_lab/jhb_vclab/world_model/data", static_video_path)).astype(np.float32) / 255.0
    # static_video = static_video.transpose(3, 0, 1, 2)  # [F, H, W, C] -> [C, F, H, W]
    # static_video = static_video[np.newaxis, :]  # Add batch dimension: [C, F, H, W] -> [1, C, F, H, W]
    # static_video = torch.from_numpy(static_video).to(device)

    input_video_mask = torch.zeros_like(input_video[:, :1])

    if proj_path is not None:
        # hand_video = iio.imread(os.path.join("/virtual_lab/jhb_vclab/world_model/data", hand_video_path)).astype(np.float32) / 255.0
        # hand_video = hand_video.transpose(3, 0, 1, 2)  # [F, H, W, C] -> [C, F, H, W]
        # hand_video = hand_video[np.newaxis, :]  # Add batch dimension: [C, F, H, W] -> [1, C, F, H, W]
        # hand_video = torch.from_numpy(hand_video).to(device)
        hand_video,  _, _, _ = get_video_to_video_latent(os.path.join("/virtual_lab/jhb_vclab/world_model/data", hand_video_path), video_length=video_length, sample_size=sample_size, validation_video_mask=None, fps=fps)

    else:
        hand_video = None
    # input_video_mask, _, _, _ = get_video_to_video_latent(os.path.join("/virtual_lab/jhb_vclab/world_model/data", video_mask_path), video_length=video_length, sample_size=sample_size, validation_video_mask=None, fps=fps)
    # input_video_mask = input_video_mask[:, :1]
    # input_video_mask = morph2d(input_video_mask, kernel_size=501, mode="dilate")

    with torch.no_grad():
        sample = pipeline(
            prompt, 
            num_frames = 49,
            negative_prompt = negative_prompt,
            height      = 480,
            width       = 720,
            generator   = generator,
            guidance_scale = guidance_scale,
            num_inference_steps = num_inference_steps,
            video       = input_video,
            mask_video  = input_video_mask,
            hand_video = hand_video,
            strength    = 1.0,
            use_dynamic_cfg = False,
        ).videos

        def save_results():
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)
            name = validation_video.replace("/", "_")

            index = len([path for path in os.listdir(save_path)]) + 1
            prefix = str(index).zfill(8)
            if video_length == 1:
                video_path = os.path.join(save_path, prefix + ".png")

                image = sample[0, :, 0]
                image = image.transpose(0, 1).transpose(1, 2)
                image = (image * 255).numpy().astype(np.uint8)
                image = Image.fromarray(image)
                image.save(video_path)
            else:
                video_path = os.path.join(save_path, name)
                save_videos_grid(sample, video_path, fps=fps)

        save_results()