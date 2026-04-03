from pathlib import Path
from typing import Iterator, Tuple


TASTE_ROB_HAND_CLASSES = ("SingleHand", "DoubleHand")


def iter_trumans_action_dirs(data_root: Path) -> Iterator[Tuple[Path, Path]]:
    for scene_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        for action_dir in sorted(path for path in scene_dir.iterdir() if path.is_dir()):
            yield scene_dir, action_dir


def iter_taste_rob_sample_dirs(data_root: Path) -> Iterator[Tuple[Path, Path]]:
    for hand_class in TASTE_ROB_HAND_CLASSES:
        hand_dir = data_root / hand_class
        if not hand_dir.is_dir():
            continue
        for sample_dir in sorted(path for path in hand_dir.iterdir() if path.is_dir()):
            yield hand_dir, sample_dir
