import os, sys
import cv2
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
from natsort import natsorted


data_root = Path(f"{sys.argv[1]}")
(data_root / "cropped").mkdir(exist_ok=True)

for i, image_path in enumerate(natsorted(data_root.iterdir())):

    if not image_path.name.endswith('.jpg'):
        continue

    undistored_frame_rgb = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    rotated = np.rot90(undistored_frame_rgb, k=3)

    # Step 2: Center crop to 1080x720 (W x H)
    crop_width, crop_height = 720, 480
    h, w = rotated.shape[:2]
    start_x = (w - crop_width) // 2
    start_y = (h - crop_height) // 2
    cropped = rotated[start_y:start_y + crop_height, start_x:start_x + crop_width]

    # Save the processed image
    output_path = data_root / "cropped" / f"{i:05d}.png"
    cv2.imwrite(str(output_path), cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))