from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass
class VideoMetadata:
    path: Path
    fps: float
    total_frames: int
    width: int
    height: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "fps": float(self.fps),
            "total_frames": int(self.total_frames),
            "width": int(self.width),
            "height": int(self.height),
        }


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_video_metadata(video_path: Path) -> VideoMetadata:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Failed to open video: {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or np.isnan(fps):
            fps = 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    return VideoMetadata(
        path=video_path,
        fps=float(fps),
        total_frames=total_frames,
        width=width,
        height=height,
    )


def load_video_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Failed to open video: {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame_bgr = cap.read()
        if not ok:
            raise IndexError(f"Failed to read frame {frame_idx} from {video_path}")
    finally:
        cap.release()
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def build_intrinsic_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    intrinsic = np.eye(3, dtype=np.float32)
    intrinsic[0, 0] = float(fx)
    intrinsic[1, 1] = float(fy)
    intrinsic[0, 2] = float(cx)
    intrinsic[1, 2] = float(cy)
    return intrinsic


def matrix_to_quaternion_wxyz(matrix: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a quaternion in wxyz order."""

    m = np.asarray(matrix, dtype=np.float32)
    if m.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 matrix, got {m.shape}")

    trace = float(np.trace(m))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s

    quat = np.asarray([w, x, y, z], dtype=np.float32)
    norm = np.linalg.norm(quat)
    if norm <= 1e-8:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return quat / norm


def depth_to_point_cloud(
    depth: np.ndarray,
    intrinsic: np.ndarray,
    *,
    image_rgb: np.ndarray | None = None,
    stride: int = 4,
    max_depth: float | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    depth = np.asarray(depth, dtype=np.float32)
    intrinsic = np.asarray(intrinsic, dtype=np.float32)
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])

    ys = np.arange(0, depth.shape[0], max(1, stride), dtype=np.int32)
    xs = np.arange(0, depth.shape[1], max(1, stride), dtype=np.int32)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    sampled_depth = depth[grid_y, grid_x]
    valid = np.isfinite(sampled_depth) & (sampled_depth > 1e-6)
    if max_depth is not None:
        valid &= sampled_depth <= float(max_depth)

    sampled_depth = sampled_depth[valid]
    u = grid_x[valid].astype(np.float32)
    v = grid_y[valid].astype(np.float32)

    x = (u - cx) * sampled_depth / max(fx, 1e-8)
    y = (v - cy) * sampled_depth / max(fy, 1e-8)
    points = np.stack([x, y, sampled_depth], axis=-1).astype(np.float32)

    colors = None
    if image_rgb is not None:
        image_h, image_w = image_rgb.shape[:2]
        color_y = np.clip(
            np.round(grid_y[valid] * image_h / max(depth.shape[0], 1)).astype(np.int32),
            0,
            image_h - 1,
        )
        color_x = np.clip(
            np.round(grid_x[valid] * image_w / max(depth.shape[1], 1)).astype(np.int32),
            0,
            image_w - 1,
        )
        colors = image_rgb[color_y, color_x].astype(np.uint8)
    return points, colors


def estimate_translation_delta(
    hamer_points: np.ndarray,
    depth_points: np.ndarray,
    *,
    fallback: np.ndarray | None = None,
) -> np.ndarray:
    hamer_points = np.asarray(hamer_points, dtype=np.float32)
    depth_points = np.asarray(depth_points, dtype=np.float32)
    if fallback is None:
        fallback_vec = np.zeros((3,), dtype=np.float32)
    else:
        fallback_vec = np.asarray(fallback, dtype=np.float32).reshape(3)

    valid = (
        np.isfinite(hamer_points).all(axis=1)
        & np.isfinite(depth_points).all(axis=1)
        & (depth_points[:, 2] > 1e-6)
    )
    if not np.any(valid):
        return fallback_vec
    return np.median(depth_points[valid] - hamer_points[valid], axis=0).astype(np.float32)


def infer_translation_delta_from_alignment(
    alignment_entry: dict[str, Any] | None,
    *,
    wrist_position: np.ndarray | None = None,
) -> np.ndarray:
    if alignment_entry is None:
        return np.zeros((3,), dtype=np.float32)

    if "translation_delta" in alignment_entry:
        delta = np.asarray(alignment_entry["translation_delta"], dtype=np.float32).reshape(3)
        if np.isfinite(delta).all():
            return delta

    if wrist_position is not None and "wrist_depth_cam" in alignment_entry:
        wrist_depth = np.asarray(alignment_entry["wrist_depth_cam"], dtype=np.float32).reshape(3)
        wrist_position = np.asarray(wrist_position, dtype=np.float32).reshape(3)
        scale = float(alignment_entry.get("depth_scale", alignment_entry.get("hand_scale", 1.0)))
        wrist_position = wrist_position * scale
        if np.isfinite(wrist_depth).all() and np.isfinite(wrist_position).all():
            return (wrist_depth - wrist_position).astype(np.float32)

    if "keypoints_depth_cam" in alignment_entry and "hamer_keypoints_cam" in alignment_entry:
        fallback = None
        return estimate_translation_delta(
            np.asarray(alignment_entry["hamer_keypoints_cam"], dtype=np.float32),
            np.asarray(alignment_entry["keypoints_depth_cam"], dtype=np.float32),
            fallback=fallback,
        )

    return np.zeros((3,), dtype=np.float32)


def translate_hand_vertices(
    vertices_cam: np.ndarray,
    *,
    translation_delta: np.ndarray | None = None,
    depth_scale: float | None = None,
    alignment_entry: dict[str, Any] | None = None,
    wrist_position: np.ndarray | None = None,
) -> np.ndarray:
    vertices_cam = np.asarray(vertices_cam, dtype=np.float32)
    if depth_scale is None and alignment_entry is not None:
        scale = float(alignment_entry.get("depth_scale", alignment_entry.get("hand_scale", 1.0)))
    elif depth_scale is None:
        scale = 1.0
    else:
        scale = float(depth_scale)
    vertices_cam = vertices_cam * scale
    if translation_delta is None:
        delta = infer_translation_delta_from_alignment(
            alignment_entry,
            wrist_position=None if wrist_position is None else np.asarray(wrist_position, dtype=np.float32) * scale,
        )
    else:
        delta = np.asarray(translation_delta, dtype=np.float32).reshape(3)
    return (vertices_cam + delta[None, :]).astype(np.float32)
