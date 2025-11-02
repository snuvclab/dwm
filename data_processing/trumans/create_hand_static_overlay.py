#!/usr/bin/env python3
"""
Script to create hand-overlayed static videos using hand masks.

Input structure:
    data/trumans/ego_render_fov90/{scene_name}/{action_name}/
        - videos_hands/*.mp4
        - videos_hands_mask/*.mp4
        - videos_static/*.mp4

Output:
    - videos_hands_static_overlay/*.mp4

Process:
    1. Load static video
    2. Make it gray and blur (0.1 * original + 0.9 * gray)
    3. Load hand mask (255 for hand, 0 otherwise)
    4. Load hand video
    5. Overlay hand video on blurred static using mask
"""

import argparse
import numpy as np
import imageio.v3 as iio
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_video(video_path):
    """Load video from path.
    
    Returns:
        Video array of shape [F, H, W, C] as float32 in [0, 1]
    """
    video = iio.imread(str(video_path))
    return video.astype(np.float32) / 255.0


def blur_with_gray(video, alpha=0.1, gray_value=0.65):
    """Blur video by mixing with single gray color.
    
    Blend original video with a single gray value
    Formula: alpha * original + (1 - alpha) * gray
    
    Args:
        video: [F, H, W, C] float32 in [0, 1]
        alpha: Mixing ratio (0.1 means 0.1 * original + 0.9 * gray)
        gray_value: Gray intensity value in [0, 1] (default: 0.65 for light gray)
    
    Returns:
        Blurred video of same shape
    """
    # Create single gray color for RGB
    # This will mix with the original color video
    if video.shape[-1] == 3:  # RGB
        gray = np.full_like(video, gray_value)  # [F, H, W, 3] with all values = gray_value
    else:
        gray = np.full_like(video, gray_value)  # For grayscale or other
    
    # Mix: alpha * original + (1 - alpha) * gray
    blurred = alpha * video + (1 - alpha) * gray
    
    return blurred


def overlay_hands_on_static(static_video, hand_video, hand_mask, hand_alpha=1.0):
    """Overlay hand video on static video using mask.
    
    Args:
        static_video: [F, H, W, C] background video
        hand_video: [F, H, W, C] hand video to overlay
        hand_mask: [F, H, W, 1] or [F, H, W] mask (1 for hand, 0 for background)
        hand_alpha: Transparency factor for hand (default: 1.0 for fully opaque)
    
    Returns:
        Composite video [F, H, W, C]
    """
    # Normalize mask to [0, 1]
    if hand_mask.max() > 1.0:
        hand_mask = hand_mask / 255.0
    
    # Ensure mask has proper shape
    if hand_mask.ndim == 3:  # [F, H, W]
        hand_mask = hand_mask[:, :, :, np.newaxis]
    elif hand_mask.shape[-1] == 3:  # [F, H, W, 3]
        hand_mask = hand_mask[:, :, :, 0:1]  # Take first channel
    
    # Resize hand_video and mask to match static if needed
    if hand_video.shape[:3] != static_video.shape[:3]:
        # Use simple repeat if frame numbers mismatch
        if hand_video.shape[0] != static_video.shape[0]:
            # Find the first frame that matches in height/width
            if hand_video.shape[1:3] == static_video.shape[1:3]:
                # Repeat hand video frames
                repeat_factor = static_video.shape[0] // hand_video.shape[0]
                remainder = static_video.shape[0] % hand_video.shape[0]
                hand_video = np.concatenate([hand_video] * repeat_factor + [hand_video[:remainder]], axis=0)
                hand_mask = np.concatenate([hand_mask] * repeat_factor + [hand_mask[:remainder]], axis=0)
            else:
                logger.warning(f"Shape mismatch: hand_video {hand_video.shape} vs static {static_video.shape}")
    
    # Apply hand_alpha: blend = alpha * mask
    blend = hand_mask * hand_alpha
    
    # Composite: blend * hand + (1 - blend) * static
    composite = blend * hand_video + (1 - blend) * static_video
    
    return composite


def process_single_video(static_path, hand_path, mask_path, output_path, overlay_alpha=0.1, gray_value=0.65, hand_alpha=1.0, fps=8):
    """Process a single video triplet.
    
    Args:
        static_path: Path to static video
        hand_path: Path to hand video
        mask_path: Path to hand mask video
        output_path: Path to save output
        overlay_alpha: Alpha value for gray overlay blur
        gray_value: Gray intensity value in [0, 1]
        hand_alpha: Transparency factor for hand overlay (default: 1.0)
        fps: FPS for output video
    """
    try:
        # Load videos
        logger.info(f"Loading: {Path(static_path).name}")
        static_video = load_video(static_path)
        hand_video = load_video(hand_path)
        hand_mask = load_video(mask_path)  # Will be [0, 1] after load_video
        
        # Blur static with gray
        static_blurred = blur_with_gray(static_video, alpha=overlay_alpha, gray_value=gray_value)
        
        # Overlay hands
        composite = overlay_hands_on_static(static_blurred, hand_video, hand_mask, hand_alpha=hand_alpha)
        
        # Convert back to uint8
        composite_uint8 = (composite * 255.0).astype(np.uint8)
        
        # Save output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(output_path, composite_uint8, fps=fps)
        logger.info(f"Saved: {output_path}")
        
        return True
    except Exception as e:
        logger.error(f"Error processing {static_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


def find_matching_videos(data_root, scene_name, action_name):
    """Find all matching video triplets.
    
    Returns:
        List of tuples: (static_path, hand_path, mask_path, video_name)
    """
    base_dir = data_root / scene_name / action_name
    
    static_dir = base_dir / 'processed2' / 'videos_static'
    hand_dir = base_dir / 'processed2' / 'videos_hands'
    mask_dir = base_dir / 'processed2' / 'videos_hands_mask'
    
    if not static_dir.exists():
        logger.error(f"Static video directory not found: {static_dir}")
        return []
    
    if not hand_dir.exists():
        logger.error(f"Hand video directory not found: {hand_dir}")
        return []
    
    if not mask_dir.exists():
        logger.error(f"Hand mask directory not found: {mask_dir}")
        return []
    
    # Find all static videos
    static_videos = sorted(static_dir.glob('*.mp4'))
    
    triplets = []
    for static_path in static_videos:
        video_name = static_path.stem
        hand_path = hand_dir / f"{video_name}.mp4"
        mask_path = mask_dir / f"{video_name}.mp4"
        
        if hand_path.exists() and mask_path.exists():
            triplets.append((static_path, hand_path, mask_path, video_name))
        else:
            logger.warning(f"Missing files for {video_name}: hand={hand_path.exists()}, mask={mask_path.exists()}")
    
    return triplets


def main():
    parser = argparse.ArgumentParser(description='Create hand-overlayed static videos')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory containing data/trumans/ego_render_fov90/')
    parser.add_argument('--scene_name', type=str, required=True,
                        help='Scene name directory')
    parser.add_argument('--action_name', type=str, required=True,
                        help='Action name directory')
    parser.add_argument('--overlay_alpha', type=float, default=0.1,
                        help='Alpha for gray overlay (default: 0.1)')
    parser.add_argument('--gray_value', type=float, default=0.5,
                        help='Gray intensity value in [0, 1] (default: 0.5)')
    parser.add_argument('--hand_alpha', type=float, default=1.0,
                        help='Transparency factor for hand overlay [0-1] (default: 1.0 = fully opaque)')
    parser.add_argument('--fps', type=int, default=10,
                        help='FPS for output videos (default: 10)')
    
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    scene_name = args.scene_name
    action_name = args.action_name
    
    # Find all matching videos
    triplets = find_matching_videos(data_root, scene_name, action_name)
    
    if not triplets:
        logger.error("No matching video triplets found!")
        return
    
    logger.info(f"Found {len(triplets)} video triplets to process")
    
    # Create output directory
    output_dir = data_root / scene_name / action_name / 'videos_hands_static_overlay'
    
    # Process each triplet
    success_count = 0
    for static_path, hand_path, mask_path, video_name in tqdm(triplets, desc="Processing"):
        output_path = output_dir / f"{video_name}.mp4"
        success = process_single_video(static_path, hand_path, mask_path, output_path, 
                                       overlay_alpha=args.overlay_alpha, 
                                       gray_value=args.gray_value, 
                                       hand_alpha=args.hand_alpha,
                                       fps=args.fps)
        if success:
            success_count += 1
    
    logger.info(f"✅ Successfully processed {success_count}/{len(triplets)} videos")
    logger.info(f"📁 Output directory: {output_dir}")


if __name__ == '__main__':
    main()
