import argparse
import os
import random
from typing import List, Optional, Tuple
import pickle

import imageio.v3 as iio
import numpy as np
import PIL
import rootutils
import torch
import safetensors.torch
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXTransformer3DModel,
)
from diffusers.utils import convert_unet_state_dict_to_peft
from peft import LoraConfig, set_peft_model_state_dict
from transformers import AutoTokenizer, T5EncoderModel


# rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from aetherv1_pipeline_smpl_adaln_zero import (  # noqa: E402
    AetherV1SMPLAdaLNZeroPipelineCogVideoX,
    AetherV1SMPLAdaLNZeroPerFramePipelineCogVideoX,
    AetherV1SMPLAdaLNZeroPipelineOutput,
)
from utils.postprocess_utils import (  # noqa: E402
    align_camera_extrinsics,
    apply_transformation,
    colorize_depth,
    compute_scale,
    get_intrinsics,
    interpolate_poses,
    postprocess_pointmap,
    project,
    raymap_to_poses,
)
from utils.visualize_utils import predictions_to_glb  # noqa: E402


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_all(seed: int = 0) -> None:
    """
    Set random seeds of all components.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="AetherV1-CogvideoX SMPL-Conditioned Inference Demo")

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["reconstruction", "prediction", "planning"],
        help="Task to perform: 'reconstruction', 'prediction' or 'planning'.",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to a video file. Only used for 'reconstruction' task.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to an image file. Only used for 'prediction' and 'planning' tasks.",
    )
    parser.add_argument(
        "--goal",
        type=str,
        default=None,
        help="Path to a goal image file. Only used for 'planning' task.",
    )
    parser.add_argument(
        "--raymap_action",
        type=str,
        default=None,
        help="Path to a raymap action file (.pt or .npy). Should be a tensor/array of shape (num_frame, 6, latent_height, latent_width).",
    )
    parser.add_argument(
        "--pose_params",
        type=str,
        default=None,
        help="Path to SMPL pose parameters file. Should be a numpy array of shape (num_frame, 63) or (num_frame, 72).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Path to save the outputs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=8,
        choices=[8, 10, 12, 15, 24],
        help="Frames per second. Options: 8, 10, 12, 15, 24.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=None,
        help="Number of inference steps. If not specified, will use the default number of steps for the task.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=None,
        help="Guidance scale. If not specified, will use the default guidance scale for the task.",
    )
    parser.add_argument(
        "--use_dynamic_cfg",
        action="store_true",
        default=True,
        help="Use dynamic cfg.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Height of the output video.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=720,
        help="Width of the output video.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=49,
        help="Number of frames to predict.",
    )
    parser.add_argument(
        "--max_depth",
        type=float,
        default=100.0,
        help="Maximum depth of the scene in meters.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=0.2,
        help="Relative tolerance for depth edge detection.",
    )
    parser.add_argument(
        "--cogvideox_pretrained_model_name_or_path",
        type=str,
        default="THUDM/CogVideoX-5b-I2V",
        help="Name or path of the CogVideoX model to use.",
    )
    parser.add_argument(
        "--aether_pretrained_model_name_or_path",
        type=str,
        default="AetherWorldModel/AetherV1",
        help="Name or path of the Aether model to use.",
    )
    parser.add_argument(
        "--smooth_camera",
        action="store_true",
        default=True,
        help="Smooth the camera trajectory.",
    )
    parser.add_argument(
        "--smooth_method",
        type=str,
        default="kalman",
        choices=["kalman", "simple"],
        help="Smooth method.",
    )
    parser.add_argument(
        "--sliding_window_stride",
        type=int,
        default=24,
        help="Sliding window stride (window size equals to num_frames). Only used for 'reconstruction' task.",
    )
    parser.add_argument(
        "--post_reconstruction",
        action="store_true",
        default=True,
        help="Run reconstruction after prediction for better quality. Only used for 'prediction' and 'planning' tasks.",
    )
    parser.add_argument(
        "--pointcloud_save_frame_interval",
        type=int,
        default=10,
        help="Pointcloud save frame interval.",
    )
    parser.add_argument(
        "--align_pointmaps",
        action="store_true",
        default=False,
        help="Align pointmaps.",
    )
    parser.add_argument(
        "--smpl_mode",
        type=str,
        default="global",
        choices=["global", "per_frame"],
        help="SMPL conditioning mode: 'global' for AdaLN-zero with global pose embedding, 'per_frame' for per-frame conditioning.",
    )
    parser.add_argument(
        "--pose_dim",
        type=int,
        default=63,
        choices=[63, 72],
        help="Dimension of SMPL pose parameters (63 for pose only, 72 for pose + shape).",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to the trained checkpoint directory containing LoRA and SMPL parameters.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=64.0,
        help="LoRA alpha parameter for scaling.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=64,
        help="LoRA rank parameter.",
    )
    return parser.parse_args()


def build_pipeline(args: argparse.Namespace):
    """Build the SMPL-conditioned pipeline."""
    
    # Load base components
    tokenizer = AutoTokenizer.from_pretrained(
        args.cogvideox_pretrained_model_name_or_path,
        subfolder="tokenizer",
    )
    
    text_encoder = T5EncoderModel.from_pretrained(
        args.cogvideox_pretrained_model_name_or_path, 
        subfolder="text_encoder"
    )
    
    vae = AutoencoderKLCogVideoX.from_pretrained(
        args.cogvideox_pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=torch.bfloat16,
    )
    
    scheduler = CogVideoXDPMScheduler.from_pretrained(
        args.cogvideox_pretrained_model_name_or_path, 
        subfolder="scheduler"
    )
    
    # Load the SMPL-conditioned transformer
    if args.smpl_mode == "global":
        from aetherv1_pipeline_smpl_adaln_zero import SMPLConditionedTransformer3DAdaLNZero
        transformer = SMPLConditionedTransformer3DAdaLNZero.from_aether_pretrained(
            args.aether_pretrained_model_name_or_path,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
            pose_dim=args.pose_dim,
        )
        pipeline = AetherV1SMPLAdaLNZeroPipelineCogVideoX(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            scheduler=scheduler,
            transformer=transformer,
        )
    else:  # per_frame mode
        from aetherv1_pipeline_smpl_adaln_zero import SMPLConditionedTransformer3DAdaLNZeroPerFrame
        transformer = SMPLConditionedTransformer3DAdaLNZeroPerFrame.from_aether_pretrained(
            args.aether_pretrained_model_name_or_path,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
            pose_dim=args.pose_dim,
        )
        pipeline = AetherV1SMPLAdaLNZeroPerFramePipelineCogVideoX(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            scheduler=scheduler,
            transformer=transformer,
        )
    
    # Add LoRA adapter if checkpoint is provided
    # Note: LoRA is only applied to the original CogVideoX parameters, not to SMPL parameters
    if args.checkpoint_path is not None:
        from peft import LoraConfig
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            init_lora_weights=True,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        transformer.add_adapter(lora_config)
        print(f"✅ Added LoRA adapter with rank={args.lora_rank}, alpha={args.lora_alpha}")
        print(f"   📝 LoRA is applied only to original CogVideoX parameters")
        print(f"   🎭 SMPL parameters are trained from scratch (not LoRA)")
    
    # Load checkpoint if provided
    if args.checkpoint_path is not None:
        print(f"🔄 Loading checkpoint from: {args.checkpoint_path}")
        load_checkpoint(pipeline, transformer, args)
    
    pipeline.vae.enable_slicing()
    pipeline.vae.enable_tiling()
    pipeline.to(device)
    return pipeline


def load_checkpoint(pipeline, transformer, args):
    """Load LoRA and SMPL parameters from checkpoint."""
    checkpoint_path = args.checkpoint_path
    
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")
    
    print(f"📁 Loading checkpoint from: {checkpoint_path}")
    
    # Load LoRA weights (applied only to original CogVideoX parameters)
    lora_weights_path = os.path.join(checkpoint_path, "pytorch_lora_weights.safetensors")
    if os.path.exists(lora_weights_path):
        try:
            lora_state_dict = safetensors.torch.load_file(lora_weights_path)
            print(f"📊 Loaded {len(lora_state_dict)} LoRA parameters from safetensors file")
            
            # Convert state dict for transformer
            transformer_state_dict = {
                f'{k.replace("transformer.", "")}': v for k, v in lora_state_dict.items() if k.startswith("transformer.")
            }
            print(f"📊 Found {len(transformer_state_dict)} transformer LoRA parameters")
            
            transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
            
            # Set LoRA weights (only affects original CogVideoX parameters)
            incompatible_keys = set_peft_model_state_dict(transformer, transformer_state_dict, adapter_name="default")
            if incompatible_keys is not None:
                unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                if unexpected_keys:
                    print(f"⚠️  Warning: Unexpected LoRA keys: {unexpected_keys}")
            
            print(f"✅ Loaded LoRA weights successfully from {lora_weights_path}")
            print(f"   📝 LoRA weights applied to original CogVideoX parameters only")
            
            # Note: LoRA scaling is handled internally by the PEFT adapter
            print(f"ℹ️  LoRA scaling handled internally by PEFT adapter")
            
        except Exception as e:
            print(f"❌ Error loading LoRA weights: {e}")
            print(f"   This might be due to incompatible model architecture or checkpoint format.")
    else:
        print(f"⚠️  Warning: LoRA weights file not found at {lora_weights_path}")
    
    # Load SMPL pose parameters (trained from scratch, not LoRA)
    smpl_pose_file = os.path.join(checkpoint_path, "smpl_pose_parameters.pt")
    if os.path.exists(smpl_pose_file):
        try:
            smpl_state_dict = torch.load(smpl_pose_file, map_location="cpu")
            loaded_smpl_params = 0
            total_smpl_params = 0
            
            # Count total SMPL parameters in the model
            for name, param in transformer.named_parameters():
                if "smpl_pose_embedding" in name or "smpl_linear" in name:
                    total_smpl_params += 1
            
            # Load SMPL pose parameters into the transformer (full parameters, not LoRA)
            for name, param in transformer.named_parameters():
                if "smpl_pose_embedding" in name or "smpl_linear" in name:
                    state_key = f"transformer.{name}"
                    if state_key in smpl_state_dict:
                        param.data.copy_(smpl_state_dict[state_key])
                        loaded_smpl_params += 1
                        print(f"✅ Loaded SMPL parameter: {name}")
                    else:
                        print(f"⚠️  SMPL parameter not found in checkpoint: {name}")
            
            print(f"✅ Loaded {loaded_smpl_params}/{total_smpl_params} SMPL pose parameters successfully")
            print(f"   🎭 SMPL parameters are full parameters (trained from scratch)")
            
            if loaded_smpl_params == 0:
                print(f"⚠️  Warning: No SMPL parameters were loaded. Check if the checkpoint contains SMPL parameters.")
            
        except Exception as e:
            print(f"❌ Error loading SMPL pose parameters: {e}")
            print(f"   This might be due to incompatible model architecture or checkpoint format.")
    else:
        print(f"ℹ️  No SMPL pose parameters found at: {smpl_pose_file}")
        print(f"   This is normal if the model was trained without SMPL conditioning.")
    
    print("=" * 60)
    print("🎯 Checkpoint loading summary:")
    print(f"   📁 Checkpoint path: {checkpoint_path}")
    print(f"   🔧 LoRA weights: {'✅ Loaded' if os.path.exists(lora_weights_path) else '❌ Not found'}")
    print(f"   🤖 SMPL parameters: {'✅ Loaded' if os.path.exists(smpl_pose_file) else '❌ Not found'}")
    print("=" * 60)


def get_window_starts(
    total_frames: int, sliding_window_size: int, temporal_stride: int
) -> List[int]:
    """Calculate window start indices."""
    starts = list(
        range(
            0,
            total_frames - sliding_window_size + 1,
            temporal_stride,
        )
    )
    if (
        total_frames > sliding_window_size
        and (total_frames - sliding_window_size) % temporal_stride != 0
    ):
        starts.append(total_frames - sliding_window_size)
    return starts


def blend_and_merge_window_results(
    window_results: List[AetherV1SMPLAdaLNZeroPipelineOutput],
    window_indices: List[int],
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Blend and merge window results."""
    merged_rgb = None
    merged_disparity = None
    merged_poses = None
    merged_focals = None
    if args.align_pointmaps:
        merged_pointmaps = None

    w1 = window_results[0].disparity

    for idx, (window_result, t_start) in enumerate(zip(window_results, window_indices)):
        t_end = t_start + window_result.rgb.shape[0]
        if idx == 0:
            merged_rgb = window_result.rgb
            merged_disparity = window_result.disparity
            pointmap_dict = postprocess_pointmap(
                window_result.disparity,
                window_result.raymap,
                vae_downsample_scale=8,
                ray_o_scale_inv=0.1,
                smooth_camera=args.smooth_camera,
                smooth_method=args.smooth_method if args.smooth_camera else "none",
            )
            merged_poses = pointmap_dict["camera_pose"]
            merged_focals = (
                pointmap_dict["intrinsics"][:, 0, 0]
                + pointmap_dict["intrinsics"][:, 1, 1]
            ) / 2
            if args.align_pointmaps:
                merged_pointmaps = pointmap_dict["pointmap"]
        else:
            overlap_t = window_indices[idx - 1] + window_result.rgb.shape[0] - t_start

            window_disparity = window_result.disparity

            # Align disparity
            disp_mask = window_disparity[:overlap_t].reshape(1, -1, w1.shape[-1]) > 0.1
            scale = compute_scale(
                window_disparity[:overlap_t].reshape(1, -1, w1.shape[-1]),
                merged_disparity[-overlap_t:].reshape(1, -1, w1.shape[-1]),
                disp_mask.reshape(1, -1, w1.shape[-1]),
            )
            window_disparity = scale * window_disparity

            # Blend disparity
            result_disparity = np.ones((t_end, *w1.shape[1:]))
            result_disparity[:t_start] = merged_disparity[:t_start]
            result_disparity[t_start + overlap_t :] = window_disparity[overlap_t:]
            weight = np.linspace(1, 0, overlap_t)[:, None, None]
            result_disparity[t_start : t_start + overlap_t] = merged_disparity[
                t_start : t_start + overlap_t
            ] * weight + window_disparity[:overlap_t] * (1 - weight)
            merged_disparity = result_disparity

            # Blend RGB
            result_rgb = np.ones((t_end, *w1.shape[1:], 3))
            result_rgb[:t_start] = merged_rgb[:t_start]
            result_rgb[t_start + overlap_t :] = window_result.rgb[overlap_t:]
            weight_rgb = np.linspace(1, 0, overlap_t)[:, None, None, None]
            result_rgb[t_start : t_start + overlap_t] = merged_rgb[
                t_start : t_start + overlap_t
            ] * weight_rgb + window_result.rgb[:overlap_t] * (1 - weight_rgb)
            merged_rgb = result_rgb

            # Align poses
            window_raymap = window_result.raymap
            window_poses, window_Fov_x, window_Fov_y = raymap_to_poses(
                window_raymap, ray_o_scale_inv=0.1
            )
            rel_r, rel_t, rel_s = align_camera_extrinsics(
                torch.from_numpy(window_poses[:overlap_t]),
                torch.from_numpy(merged_poses[-overlap_t:]),
            )
            aligned_window_poses = (
                apply_transformation(
                    torch.from_numpy(window_poses),
                    rel_r,
                    rel_t,
                    rel_s,
                    return_extri=True,
                )
                .cpu()
                .numpy()
            )

            result_poses = np.ones((t_end, 4, 4))
            result_poses[:t_start] = merged_poses[:t_start]
            result_poses[t_start + overlap_t :] = aligned_window_poses[overlap_t:]

            # Interpolate poses in overlap region
            weights = np.linspace(1, 0, overlap_t)
            for t in range(overlap_t):
                weight = weights[t]
                pose1 = merged_poses[t_start + t]
                pose2 = aligned_window_poses[t]
                result_poses[t_start + t] = interpolate_poses(pose1, pose2, weight)

            merged_poses = result_poses

            # Align intrinsics
            window_intrinsics, _ = get_intrinsics(
                batch_size=window_poses.shape[0],
                h=window_result.disparity.shape[1],
                w=window_result.disparity.shape[2],
                fovx=window_Fov_x,
                fovy=window_Fov_y,
            )
            window_focals = (
                window_intrinsics[:, 0, 0] + window_intrinsics[:, 1, 1]
            ) / 2
            scale = (merged_focals[-overlap_t:] / window_focals[:overlap_t]).mean()
            window_focals = scale * window_focals
            result_focals = np.ones((t_end,))
            result_focals[:t_start] = merged_focals[:t_start]
            result_focals[t_start + overlap_t :] = window_focals[overlap_t:]
            weight = np.linspace(1, 0, overlap_t)
            result_focals[t_start : t_start + overlap_t] = merged_focals[
                t_start : t_start + overlap_t
            ] * weight + window_focals[:overlap_t] * (1 - weight)
            merged_focals = result_focals

            if args.align_pointmaps:
                # Align pointmaps
                window_pointmaps = postprocess_pointmap(
                    result_disparity[t_start:],
                    window_raymap,
                    vae_downsample_scale=8,
                    camera_pose=aligned_window_poses,
                    focal=window_focals,
                    ray_o_scale_inv=0.1,
                    smooth_camera=args.smooth_camera,
                    smooth_method=args.smooth_method if args.smooth_camera else "none",
                )
                result_pointmaps = np.ones((t_end, *w1.shape[1:], 3))
                result_pointmaps[:t_start] = merged_pointmaps[:t_start]
                result_pointmaps[t_start + overlap_t :] = window_pointmaps["pointmap"][
                    overlap_t:
                ]
                weight = np.linspace(1, 0, overlap_t)[:, None, None, None]
                result_pointmaps[t_start : t_start + overlap_t] = merged_pointmaps[
                    t_start : t_start + overlap_t
                ] * weight + window_pointmaps["pointmap"][:overlap_t] * (1 - weight)
                merged_pointmaps = result_pointmaps

    # project to pointmaps
    intrinsics = [
        np.array([[f, 0, 0.5 * args.width], [0, f, 0.5 * args.height], [0, 0, 1]])
        for f in merged_focals
    ]
    if args.align_pointmaps:
        pointmaps = merged_pointmaps
    else:
        pointmaps = np.stack(
            [
                project(
                    1 / np.clip(merged_disparity[i], 1e-8, 1e8),
                    intrinsics[i],
                    merged_poses[i],
                )
                for i in range(merged_poses.shape[0])
            ]
        )

    return merged_rgb, merged_disparity, merged_poses, pointmaps, intrinsics


def save_output(
    rgb: np.ndarray,
    disparity: np.ndarray,
    poses: Optional[np.ndarray] = None,
    intrinsics: Optional[List[np.ndarray]] = None,
    raymap: Optional[np.ndarray] = None,
    pointmap: Optional[np.ndarray] = None,
    args: argparse.Namespace = None,
) -> None:
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if pointmap is None:
        assert raymap is not None, "Raymap is required for saving pointmap."
        window_result = AetherV1SMPLAdaLNZeroPipelineOutput(
            rgb=rgb, disparity=disparity, raymap=raymap
        )
        window_results = [window_result]
        window_indices = [0]
        _, _, poses_from_blend, pointmap, intrinsics = blend_and_merge_window_results(
            window_results, window_indices, args
        )

        # Use poses from blend_and_merge_window_results if poses is None
        if poses is None:
            poses = poses_from_blend

    if poses is None:
        assert raymap is not None, "Raymap is required for saving poses."
        poses, fov_x, fov_y = raymap_to_poses(raymap, ray_o_scale_inv=0.1)

    if args.task == "reconstruction":
        filename = f"reconstruction_{args.video.split('/')[-1].split('.')[0]}"
    elif args.task == "prediction":
        filename = f"prediction_{args.image.split('/')[-1].split('.')[0]}"
    elif args.task == "planning":
        filename = f"planning_{args.image.split('/')[-1].split('.')[0]}_{args.goal.split('/')[-1].split('.')[0]}"

    # Add SMPL mode to filename
    filename = f"{filename}_smpl_{args.smpl_mode}"
    filename = os.path.join(output_dir, filename)

    iio.imwrite(
        f"{filename}_rgb.mp4",
        (np.clip(rgb, 0, 1) * 255).astype(np.uint8),
        fps=8,
    )
    iio.imwrite(
        f"{filename}_disparity.mp4",
        (colorize_depth(disparity) * 255).astype(np.uint8),
        fps=8,
    )

    print("Building GLB scene")
    for frame_idx in range(pointmap.shape[0])[:: args.pointcloud_save_frame_interval]:
        predictions = {
            "world_points": pointmap[frame_idx : frame_idx + 1],
            "images": rgb[frame_idx : frame_idx + 1],
            "depths": 1 / np.clip(disparity[frame_idx : frame_idx + 1], 1e-8, 1e8),
            "camera_poses": poses[frame_idx : frame_idx + 1],
            "intrinsics": intrinsics[frame_idx]
        }
        with open(f"{filename}_predictions_{frame_idx:02d}.pkl", "wb") as f:
            pickle.dump(predictions, f)
        # scene_3d = predictions_to_glb(
        #     predictions,
        #     filter_by_frames="all",
        #     show_cam=True,
        #     max_depth=args.max_depth,
        #     rtol=args.rtol,
        #     frame_rel_idx=float(frame_idx) / pointmap.shape[0],
        # )
        # scene_3d.export(f"{filename}_pointcloud_frame_{frame_idx}.glb")
    print("GLB Scene built")


def load_pose_params(pose_path: str, num_frames: int, pose_dim: int) -> np.ndarray:
    """Load and validate SMPL pose parameters."""
    if pose_path is None:
        print("Warning: No pose parameters provided. Using zeros.")
        return np.zeros((1, num_frames, pose_dim))  # Add batch dimension
    
    if pose_path.endswith('.npy'):
        pose_params = np.load(pose_path)
    elif pose_path.endswith('.pt'):
        pose_params = torch.load(pose_path, map_location='cpu').numpy()
    else:
        raise ValueError(f"Unsupported pose file format: {pose_path}")
    
    # Validate shape and add batch dimension if needed
    if len(pose_params.shape) == 2:
        # Shape is (num_frames, pose_dim) - add batch dimension
        pose_params = pose_params[np.newaxis, :, :]  # Add batch dimension
        print(f"📐 Added batch dimension to pose_params: {pose_params.shape}")
    elif len(pose_params.shape) == 3:
        # Shape is already (batch_size, num_frames, pose_dim)
        pass
    else:
        raise ValueError(f"Pose parameters should be 2D or 3D array, got shape {pose_params.shape}")
    
    # Validate dimensions
    if pose_params.shape[2] != pose_dim:
        raise ValueError(f"Expected pose dimension {pose_dim}, got {pose_params.shape[2]}")
    
    # Handle frame count mismatch
    if pose_params.shape[1] != num_frames:
        print(f"Warning: Pose parameters have {pose_params.shape[1]} frames, but video has {num_frames} frames.")
        if pose_params.shape[1] > num_frames:
            pose_params = pose_params[:, :num_frames, :]
        else:
            # Pad with last frame
            last_frame = pose_params[:, -1:, :]
            padding = np.repeat(last_frame, num_frames - pose_params.shape[1], axis=1)
            pose_params = np.concatenate([pose_params, padding], axis=1)
    
    print(f"✅ Loaded pose parameters with shape: {pose_params.shape}")
    return pose_params


def main() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_args()
    seed_all(args.seed)

    if args.num_inference_steps is None:
        args.num_inference_steps = 4 if args.task == "reconstruction" else 50

    if args.guidance_scale is None:
        args.guidance_scale = 1.0 if args.task == "reconstruction" else 3.0

    print("🎯 AetherV1-CogVideoX SMPL-Conditioned Inference Demo")
    print("=" * 60)
    print(f"📋 Task: {args.task}")
    print(f"🤖 SMPL Mode: {args.smpl_mode}")
    print(f"🎭 Pose Dimension: {args.pose_dim}")
    if args.checkpoint_path:
        print(f"📁 Checkpoint: {args.checkpoint_path}")
        print(f"🔧 LoRA Rank: {args.lora_rank}, Alpha: {args.lora_alpha}")
    else:
        print("📁 Checkpoint: None (using base model)")
    if args.raymap_action:
        print(f"📷 Raymap: {args.raymap_action}")
    if args.pose_params:
        print(f"🎭 Pose Params: {args.pose_params}")
    print("=" * 60)

    pipeline = build_pipeline(args)

    if args.task == "reconstruction":
        assert args.video is not None, "Video is required for reconstruction task."
        assert args.image is None, "Image is not required for reconstruction task."
        assert args.goal is None, "Goal is not required for reconstruction task."

        video = iio.imread(args.video).astype(np.float32) / 255.0
        video = video[:49]
        image, goal = None, None
        
        # Load pose parameters for reconstruction
        pose_params = load_pose_params(args.pose_params, len(video), args.pose_dim)
        
    elif args.task == "prediction":
        assert args.image is not None, "Image is required for prediction task."
        assert args.goal is None, "Goal is not required for prediction task."

        image = PIL.Image.open(args.image)
        video, goal = None, None
        
        # Load pose parameters for prediction
        pose_params = load_pose_params(args.pose_params, args.num_frames, args.pose_dim)
        
    elif args.task == "planning":
        assert args.image is not None, "Image is required for planning task."
        assert args.goal is not None, "Goal is required for planning task."

        image = PIL.Image.open(args.image)
        goal = PIL.Image.open(args.goal)
        video = None
        
        # Load pose parameters for planning
        pose_params = load_pose_params(args.pose_params, args.num_frames, args.pose_dim)

    if args.raymap_action is not None:
        # Load raymap from .pt file (same as in training script)
        if args.raymap_action.endswith('.pt'):
            raymap = torch.load(args.raymap_action, map_location="cpu", weights_only=True).numpy()
        else:
            # Fallback to numpy loading for other formats
            raymap = np.load(args.raymap_action)
        print(f"✅ Loaded raymap from: {args.raymap_action}, shape: {raymap.shape}")
    else:
        raymap = None

    if args.task != "reconstruction":
        output = pipeline(
            task=args.task,
            image=image,
            video=video,
            goal=goal,
            raymap=raymap,
            pose_params=pose_params,  # Add SMPL pose parameters
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            fps=args.fps,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            use_dynamic_cfg=args.use_dynamic_cfg,
            generator=torch.Generator(device=device).manual_seed(args.seed),
            return_dict=True,
        )
        if not args.post_reconstruction:
            save_output(
                rgb=output.rgb,
                disparity=output.disparity,
                raymap=output.raymap,
                args=args,
            )
        else:
            recon_output = pipeline(
                task="reconstruction",
                video=output.rgb,
                pose_params=pose_params,  # Add SMPL pose parameters
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                fps=args.fps,
                num_inference_steps=4,
                guidance_scale=1.0,  # we don't need guidance scale for reconstruction task
                use_dynamic_cfg=False,
                generator=torch.Generator(device=device).manual_seed(args.seed),
            )
            save_output(
                rgb=output.rgb,
                disparity=recon_output.disparity,
                raymap=recon_output.raymap,
                args=args,
            )
    else:
        # for reconstruction task, we have to employ sliding window on long videos
        window_results = []
        window_indices = get_window_starts(
            len(video), args.num_frames, args.sliding_window_stride
        )
        for start_idx in window_indices:
            # Extract pose parameters for this window
            window_pose_params = pose_params[:, start_idx : start_idx + args.num_frames, :]
            
            output = pipeline(
                task=args.task,
                image=None,
                goal=None,
                video=video[start_idx : start_idx + args.num_frames],
                raymap=raymap[start_idx : start_idx + args.num_frames]
                if raymap is not None
                else None,
                pose_params=window_pose_params,  # Add SMPL pose parameters
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                fps=args.fps,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=1.0,  # we don't need guidance scale for reconstruction task
                use_dynamic_cfg=False,
                generator=torch.Generator(device=device).manual_seed(args.seed),
            )
            window_results.append(output)

        # merge window results
        (
            merged_rgb,
            merged_disparity,
            merged_poses,
            pointmaps,
            intrinsics
        ) = blend_and_merge_window_results(window_results, window_indices, args)
        save_output(
            rgb=merged_rgb,
            disparity=merged_disparity,
            poses=merged_poses,
            intrinsics=intrinsics,
            pointmap=pointmaps,
            args=args,
        )


if __name__ == "__main__":
    main() 