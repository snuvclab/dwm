#!/usr/bin/env python3
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from blender_ego_video_render_impl import main as _run_main

_run_main()
raise SystemExit
