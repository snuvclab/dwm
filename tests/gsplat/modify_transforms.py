import os, sys
import cv2
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json

transforms = json.load(open(f"{sys.argv[1]}", 'r'))

for frame in transforms['frames']:
    frame_new = frame.copy()
    frame_new['fl_x'] = 300
    frame_new['fl_y'] = 300
    frame_new['cx'] = 511.5
    frame_new['cy'] = 511.5
    frame_new['distortion_params'] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    frame_new['w'] = 1024
    frame_new['h'] = 1024
    frame.update(frame_new)
with open(f"{sys.argv[1]}", 'w') as f:
    json.dump(transforms, f, indent=4)