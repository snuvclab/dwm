#!/usr/bin/env python3

import sys
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

for path in (SCRIPT_DIR, REPO_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from encode_with_wan_impl import main  # noqa: E402


if __name__ == "__main__":
    if torch.cuda.is_available():
        # Warm up CUDA once before the WAN preencoding entrypoint runs. This
        # avoids intermittent lazy-init crashes seen in the validated fresh-clone
        # environment.
        _ = torch.empty(1, device="cuda:0")
    main()
