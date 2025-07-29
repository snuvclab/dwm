#!/usr/bin/env python3
"""
Render a single animation with full Blender output for debugging
Usage: python render_single_animation.py <scene_name> <animation_name>
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

def find_animation_index(blend_file, animation_name, gpu_id=0):
    """
    Find the animation index for a given animation name
    """
    script_path = "data_processing/trumans/blender_ego_rgb_depth_optimized.py"
    
    # Set environment variables
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Build Blender command to list animations (run with any animation index to get the list)
    blender_cmd = [
        'blender',
        '-b',  # Background mode
        blend_file,
        '-P', script_path,
        '--',
        '--animation_index', '0'  # This will show the animation list
    ]
    
    print(f"🔍 Finding animation index for: {animation_name}")
    
    # Run Blender to get animation list
    try:
        result = subprocess.run(
            blender_cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout
        )
        
        # Parse the output to find animation index
        output_lines = result.stdout.split('\n')
        animation_index = None
        
        for line in output_lines:
            if 'Found' in line and 'animation sets:' in line:
                print(f"📋 {line}")
            elif line.strip().startswith(('0:', '1:', '2:', '3:', '4:', '5:', '6:', '7:', '8:', '9:')):
                print(f"📋 {line}")
                # Parse line like "  0: 2023-02-08@21-15-07.pkl"
                parts = line.strip().split(': ', 1)
                if len(parts) == 2:
                    idx = int(parts[0])
                    name = parts[1]
                    # Remove .pkl extension for comparison
                    name_without_ext = name.replace('.pkl', '')
                    if name_without_ext == animation_name:
                        animation_index = idx
                        print(f"✅ Found animation '{animation_name}' at index {idx}")
                        break
        
        return animation_index
        
    except subprocess.TimeoutExpired:
        print("❌ Timeout while finding animation index")
        return None
    except subprocess.CalledProcessError as e:
        print(f"❌ Error finding animation index: {e}")
        return None

def render_single_animation(scene_name, animation_name, gpu_id=0):
    """
    Render a single animation with full Blender output
    """
    # Configuration
    data_path = "../../nas1/public_dataset/trumans/Recordings_blend"
    save_path = "/home/byungjun/workspace/trumans_ego/ego_render_new"
    script_path = "data_processing/trumans/blender_ego_rgb_depth_optimized.py"
    
    # Find the blend file
    blend_file = None
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.blend') and scene_name in root:
                blend_file = os.path.join(root, file)
                break
        if blend_file:
            break
    
    if not blend_file:
        print(f"❌ Error: Could not find blend file for scene {scene_name}")
        return False
    
    print(f"🎬 Rendering animation: {animation_name}")
    print(f"📁 Scene: {scene_name}")
    print(f"📄 Blend file: {blend_file}")
    print(f"🎯 GPU: {gpu_id}")
    print(f"💾 Output: {save_path}")
    print("=" * 80)
    
    # Find the animation index
    animation_index = find_animation_index(blend_file, animation_name, gpu_id)
    if animation_index is None:
        print(f"❌ Error: Could not find animation '{animation_name}' in the blend file")
        return False
    
    # Set environment variables
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Build Blender command
    blender_cmd = [
        'blender',
        '-b',  # Background mode
        blend_file,
        '-P', script_path,
        '--',
        '--animation_index', str(animation_index),
        '--save-path', save_path
    ]
    
    print(f"🚀 Running: {' '.join(blender_cmd)}")
    print("=" * 80)
    
    # Run Blender with full output
    try:
        result = subprocess.run(
            blender_cmd,
            env=env,
            capture_output=False,  # Show all output in real-time
            text=True,
            check=True
        )
        print("=" * 80)
        print("✅ Rendering completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print("=" * 80)
        print(f"❌ Rendering failed with exit code: {e.returncode}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Render a single animation with full Blender output')
    parser.add_argument('scene_name', help='Scene name (directory name)')
    parser.add_argument('animation_name', help='Animation name (e.g., 2023-01-17@00-33-01)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use (default: 0)')
    
    args = parser.parse_args()
    
    success = render_single_animation(args.scene_name, args.animation_name, args.gpu)
    
    if success:
        print("🎉 Animation rendered successfully!")
        sys.exit(0)
    else:
        print("💥 Animation rendering failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 