import argparse
import logging
import os
import random
from glob import glob
from tqdm import tqdm
import imageio.v3 as iio

from easydict import EasyDict
import numpy as np
import torch
from dav.pipelines import DAVPipeline
from dav.models import UNetSpatioTemporalRopeConditionModel
from diffusers import AutoencoderKLTemporalDecoder, FlowMatchEulerDiscreteScheduler
from dav.utils import img_utils


def seed_all(seed: int = 0):
    """
    Set random seeds of all components.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Run video depth estimation using Depth Any Video."
    )

    parser.add_argument(
        "--model_base",
        type=str,
        default="hhyangcs/depth-any-video",
        help="Checkpoint path or hub name.",
    )

    # data setting
    parser.add_argument(
        "--data_path", type=str, required=True, help="input data directory."
    )

    # parser.add_argument(
    #     "--output_dir", type=str, required=True, help="Output directory."
    # )

    # inference setting
    parser.add_argument(
        "--denoise_steps",
        type=int,
        default=3,
        help="Denoising steps, 1-3 steps work fine.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=32,
        help="Number of frames to infer per forward",
    )
    parser.add_argument(
        "--decode_chunk_size",
        type=int,
        default=16,
        help="Number of frames to decode per forward",
    )
    parser.add_argument(
        "--num_interp_frames",
        type=int,
        default=16,
        help="Number of frames for inpaint inference",
    )
    parser.add_argument(
        "--num_overlap_frames",
        type=int,
        default=6,
        help="Number of frames to overlap between windows",
    )
    parser.add_argument(
        "--max_resolution",
        type=int,
        default=720,  # decrease for faster inference and lower memory usage
        help="Maximum resolution for inference.",
    )
    parser.add_argument(
        "--output_type",
        type=str,
        default="mp4",
        choices=["mp4", "npy"],
        help="Output file type.",
    )

    parser.add_argument("--seed", type=int, default=None, help="Random seed.")

    args = parser.parse_args()
    cfg = EasyDict(vars(args))

    if cfg.seed is None:
        import time

        cfg.seed = int(time.time())
    seed_all(cfg.seed)

    device_type = "cuda"
    device = torch.device(device_type)

    output_dir = cfg.data_path.replace("videos", "disparity")
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"output dir = {output_dir}")

    vae = AutoencoderKLTemporalDecoder.from_pretrained(cfg.model_base, subfolder="vae")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        cfg.model_base, subfolder="scheduler"
    )
    unet = UNetSpatioTemporalRopeConditionModel.from_pretrained(
        cfg.model_base, subfolder="unet"
    )
    unet_interp = UNetSpatioTemporalRopeConditionModel.from_pretrained(
        cfg.model_base, subfolder="unet_interp"
    )
    pipe = DAVPipeline(
        vae=vae,
        unet=unet,
        unet_interp=unet_interp,
        scheduler=scheduler,
    )
    pipe = pipe.to(device)

    video_list = sorted(glob(f"{cfg.data_path}/*.mp4"))
    for video in tqdm(video_list):
        print(video)
        file_name = video.split("/")[-1].split(".")[0]
        is_video = video.endswith(".mp4")
        if is_video:
            num_interp_frames = cfg.num_interp_frames
            num_overlap_frames = cfg.num_overlap_frames
            num_frames = cfg.num_frames
            assert num_frames % 2 == 0, "num_frames should be even."
            assert (
                2 <= num_overlap_frames <= (num_interp_frames + 2 + 1) // 2
            ), "Invalid frame overlap."
            max_frames = (num_interp_frames + 2 - num_overlap_frames) * (num_frames // 2)
            image, fps = img_utils.read_video(video, max_frames=max_frames)
        else:
            image = img_utils.read_image(cfg.data_path)

        image = img_utils.imresize_max(image, cfg.max_resolution)
        image = img_utils.imcrop_multi(image)
        image_tensor = np.ascontiguousarray(
            [_img.transpose(2, 0, 1) / 255.0 for _img in image]
        )
        image_tensor = torch.from_numpy(image_tensor).to(device)

        with torch.no_grad(), torch.autocast(device_type=device_type, dtype=torch.float16):
            pipe_out = pipe(
                image_tensor,
                num_frames=cfg.num_frames,
                num_overlap_frames=cfg.num_overlap_frames,
                num_interp_frames=cfg.num_interp_frames,
                decode_chunk_size=cfg.decode_chunk_size,
                num_inference_steps=cfg.denoise_steps,
            )

        disparity = pipe_out.disparity
        # out = {
        #     "disparity": disparity,
        #     "dmax": disparity.max()
        # }
        # with open(f"{output_dir}/{file_name}.pkl", "wb") as f:
        #     pickle.dump(out, f)
        if args.output_type == "mp4":
            iio.imwrite(
                f"{output_dir}/{file_name}.mp4",
                (np.sqrt(disparity) * 255.0).astype(np.uint8),
                fps=fps,
                codec='libx264'
            )
        elif args.output_type == "npy":
            np.savez_compressed(f"{output_dir}/{file_name}.npz", disparity=disparity.astype(np.float16))
        else:
            raise ValueError(f"Unsupported output type: {args.output_type}")
        

        # disparity_mp4 = iio.imread(f"{output_dir}/{file_name}.mp4") / 255.0
        # disparity_mp4 = disparity_mp4[:, :, :, 0]
        # break
    
    # disparity_colored = pipe_out.disparity_colored
    # image = pipe_out.image
    # # (N, H, 2 * W, 3)
    # merged = np.concatenate(
    #     [
    #         image,
    #         disparity_colored,
    #     ],
    #     axis=2,
    # )

    # if is_video:
    #     img_utils.write_video(
    #         os.path.join(cfg.output_dir, f"{file_name}_disp.mp4"),
    #         disparity_colored,
    #         fps,
    #     )
    # else:
    #     img_utils.write_image(
    #         os.path.join(cfg.output_dir, f"{file_name}.png"),
    #         merged[0],
    #     )
