from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        normalized = path.expanduser()
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def candidate_hamer_data_dirs(hamer_root: Path) -> list[Path]:
    repo_root = Path(__file__).resolve().parents[2]
    env_override = os.environ.get("HAMER_DATA_DIR")
    candidates: list[Path] = []
    if env_override:
        candidates.append(Path(env_override))
    candidates.extend(
        [
            hamer_root / "_DATA",
            repo_root / "third_party" / "hamer" / "_DATA",
            repo_root / "third_party" / "hamer-mediapipe" / "_DATA",
            repo_root.parent / "large-video-planner" / "video2robot" / "hamer" / "_DATA",
        ]
    )
    return _dedupe_paths(candidates)


def resolve_hamer_data_dir(hamer_root: Path) -> Path:
    required_relpaths = [Path("data/mano_mean_params.npz"), Path("data/mano/MANO_RIGHT.pkl")]
    checked: list[Path] = []
    for candidate in candidate_hamer_data_dirs(hamer_root):
        checked.append(candidate)
        if all((candidate / relpath).exists() for relpath in required_relpaths):
            return candidate.resolve()
    checked_text = "\n".join(f"  - {path}" for path in checked)
    raise FileNotFoundError(
        "No usable HaMeR asset directory found. Expected these files under one of the checked `_DATA` directories:\n"
        "  - data/mano_mean_params.npz\n"
        "  - data/mano/MANO_RIGHT.pkl\n"
        f"Checked:\n{checked_text}"
    )


def resolve_hamer_checkpoint(checkpoint_override: str | None, *, hamer_root: Path, hamer_data_dir: Path) -> Path:
    if checkpoint_override:
        raw_path = Path(checkpoint_override).expanduser()
        candidates = [raw_path if raw_path.is_absolute() else Path.cwd() / raw_path]
        if not raw_path.is_absolute():
            candidates.append(hamer_root / raw_path)
        for candidate in _dedupe_paths(candidates):
            if candidate.exists():
                checkpoint_path = candidate.resolve()
                model_config = checkpoint_path.parent.parent / "model_config.yaml"
                if not model_config.exists():
                    raise FileNotFoundError(f"HaMeR checkpoint override is missing model_config.yaml: {model_config}")
                return checkpoint_path
        searched = "\n".join(f"  - {path}" for path in _dedupe_paths(candidates))
        raise FileNotFoundError(f"Checkpoint override not found: {checkpoint_override}\nSearched:\n{searched}")

    checkpoint_candidates = [
        hamer_data_dir / "hamer_ckpts" / "checkpoints" / "new_hamer_weights.ckpt",
        hamer_data_dir / "hamer_ckpts" / "checkpoints" / "hamer.ckpt",
    ]
    for checkpoint_path in checkpoint_candidates:
        if checkpoint_path.exists():
            return checkpoint_path.resolve()
    searched = "\n".join(f"  - {path}" for path in checkpoint_candidates)
    raise FileNotFoundError(
        "No usable HaMeR checkpoint found. Expected one of:\n"
        f"{searched}\n"
        "You can also pass --checkpoint /abs/path/to/hamer.ckpt."
    )


@dataclass
class HamerHandFrameResult:
    frame_idx: int
    hand_idx: int
    timestamp: float
    hand_type: str
    confidence: float
    bbox_xyxy: np.ndarray
    wrist_position: np.ndarray
    wrist_quaternion: np.ndarray
    keypoints_cam: np.ndarray
    wrist_uv: np.ndarray
    keypoints_uv: np.ndarray
    cam_translation: np.ndarray
    vertices_cam: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["bbox_xyxy"] = self.bbox_xyxy.tolist()
        payload["wrist_position"] = self.wrist_position.tolist()
        payload["wrist_quaternion"] = self.wrist_quaternion.tolist()
        payload["keypoints_cam"] = self.keypoints_cam.tolist()
        payload["wrist_uv"] = self.wrist_uv.tolist()
        payload["keypoints_uv"] = self.keypoints_uv.tolist()
        payload["cam_translation"] = self.cam_translation.tolist()
        payload["vertices_cam"] = self.vertices_cam.tolist()
        return payload


@dataclass
class HamerSequenceResult:
    video_path: Path
    frames: list[HamerHandFrameResult]
    fps: float
    total_frames: int
    frame_size: tuple[int, int]
    intrinsic: np.ndarray

    def to_json(self, output_path: Path) -> None:
        payload = {
            "video_path": str(self.video_path),
            "fps": float(self.fps),
            "total_frames": int(self.total_frames),
            "frame_size": [int(self.frame_size[0]), int(self.frame_size[1])],
            "intrinsic": self.intrinsic.tolist(),
            "frames": [frame.to_dict() for frame in self.frames],
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2))
