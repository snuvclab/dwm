from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from .common import build_intrinsic_matrix, matrix_to_quaternion_wxyz
    from .backend_common import (
        HamerHandFrameResult,
        HamerSequenceResult,
        resolve_hamer_checkpoint,
        resolve_hamer_data_dir,
    )
except ImportError:
    from common import build_intrinsic_matrix, matrix_to_quaternion_wxyz
    from backend_common import (
        HamerHandFrameResult,
        HamerSequenceResult,
        resolve_hamer_checkpoint,
        resolve_hamer_data_dir,
    )


class HamerOriginalExtractor:
    def __init__(
        self,
        *,
        hamer_root: Path,
        checkpoint_path: str | None = None,
        intrinsic_matrix: np.ndarray | None = None,
        device: str | None = None,
        batch_size: int = 8,
    ) -> None:
        self.hamer_root = hamer_root.resolve()
        if not (self.hamer_root / "demo.py").exists():
            raise FileNotFoundError(f"HaMeR root is missing demo.py: {self.hamer_root}")
        if str(self.hamer_root) not in sys.path:
            sys.path.insert(0, str(self.hamer_root))

        import hamer.configs as hamer_configs
        import torch
        from detectron2.config import LazyConfig
        from hamer.datasets.vitdet_dataset import ViTDetDataset
        from hamer.models import load_hamer
        from hamer.utils import recursive_to
        from hamer.utils.renderer import cam_crop_to_full
        from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy

        self.hamer_data_dir = resolve_hamer_data_dir(self.hamer_root)
        hamer_configs.CACHE_DIR_HAMER = str(self.hamer_data_dir)
        resolved_checkpoint = resolve_hamer_checkpoint(
            checkpoint_override=checkpoint_path,
            hamer_root=self.hamer_root,
            hamer_data_dir=self.hamer_data_dir,
        )
        print(f"[HaMeR] Using asset directory: {self.hamer_data_dir}")
        print(f"[HaMeR] Using checkpoint: {resolved_checkpoint}")

        self._torch = torch
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        if self.device.type != "cuda":
            raise RuntimeError(
                "Original HaMeR backend currently requires CUDA because the Detectron2 demo predictor runs on CUDA."
            )
        self.batch_size = int(batch_size)
        self._ViTDetDataset = ViTDetDataset
        self._recursive_to = recursive_to
        self._cam_crop_to_full = cam_crop_to_full

        self._vitpose_module = self._configure_vitpose_module()
        self._vitpose = self._vitpose_module.ViTPoseModel(self.device)

        import hamer

        cfg_path = Path(hamer.__file__).parent / "configs" / "cascade_mask_rcnn_vitdet_h_75ep.py"
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = (
            "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/"
            "cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        )
        for idx in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[idx].test_score_thresh = 0.25
        self._detector = DefaultPredictor_Lazy(detectron2_cfg)

        self.model, self.model_cfg = load_hamer(str(resolved_checkpoint))
        self.model = self.model.to(self.device)
        self.model.eval()
        self.mano_faces = np.asarray(self.model.mano.faces, dtype=np.int32)

        if intrinsic_matrix is None:
            self.intrinsic_matrix = None
        else:
            matrix = np.asarray(intrinsic_matrix, dtype=np.float32)
            if matrix.shape != (3, 3):
                raise ValueError(f"intrinsic_matrix must be 3x3, got {matrix.shape}")
            self.intrinsic_matrix = matrix

    def _configure_vitpose_module(self):
        vitpose_module = importlib.import_module("vitpose_model")
        root_dir = self.hamer_root.resolve()

        def _resolve_repo_relative(path_str: str) -> str:
            raw = Path(path_str)
            if raw.is_absolute():
                return str(raw)
            return str(root_dir / str(raw).lstrip("./"))

        vitpose_module.ROOT_DIR = str(root_dir)
        vitpose_module.VIT_DIR = str(root_dir / "third-party" / "ViTPose")
        for model_entry in vitpose_module.ViTPoseModel.MODEL_DICT.values():
            model_entry["config"] = _resolve_repo_relative(model_entry["config"])
            model_entry["model"] = _resolve_repo_relative(model_entry["model"])
        return vitpose_module

    def _frame_intrinsic(self, frame_width: int, frame_height: int) -> np.ndarray:
        if self.intrinsic_matrix is not None:
            return self.intrinsic_matrix.copy()
        focal = (
            float(self.model_cfg.EXTRA.FOCAL_LENGTH)
            / float(self.model_cfg.MODEL.IMAGE_SIZE)
            * float(max(frame_width, frame_height))
        )
        return build_intrinsic_matrix(
            fx=focal,
            fy=focal,
            cx=frame_width / 2.0,
            cy=frame_height / 2.0,
        )

    @staticmethod
    def _bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
        xa1, ya1, xa2, ya2 = [float(v) for v in box_a]
        xb1, yb1, xb2, yb2 = [float(v) for v in box_b]
        inter_x1 = max(xa1, xb1)
        inter_y1 = max(ya1, yb1)
        inter_x2 = min(xa2, xb2)
        inter_y2 = min(ya2, yb2)
        inter_w = max(inter_x2 - inter_x1, 0.0)
        inter_h = max(inter_y2 - inter_y1, 0.0)
        inter_area = inter_w * inter_h
        area_a = max(xa2 - xa1, 0.0) * max(ya2 - ya1, 0.0)
        area_b = max(xb2 - xb1, 0.0) * max(yb2 - yb1, 0.0)
        union = area_a + area_b - inter_area
        if union <= 1e-6:
            return 0.0
        return inter_area / union

    @staticmethod
    def _is_duplicate_left_right_pair(left_entry: dict[str, Any], right_entry: dict[str, Any]) -> bool:
        left_box = np.asarray(left_entry["bbox_xyxy"], dtype=np.float32)
        right_box = np.asarray(right_entry["bbox_xyxy"], dtype=np.float32)
        iou = HamerOriginalExtractor._bbox_iou(left_box, right_box)
        if iou < 0.6:
            return False

        left_size = np.asarray([left_box[2] - left_box[0], left_box[3] - left_box[1]], dtype=np.float32)
        right_size = np.asarray([right_box[2] - right_box[0], right_box[3] - right_box[1]], dtype=np.float32)
        scale = max(float(np.max(left_size)), float(np.max(right_size)), 1.0)
        keypoint_delta = (
            np.asarray(left_entry["keypoints_uv"], dtype=np.float32)
            - np.asarray(right_entry["keypoints_uv"], dtype=np.float32)
        )
        mean_distance = float(np.linalg.norm(keypoint_delta, axis=1).mean())
        return mean_distance <= 0.35 * scale

    def _deduplicate_entries(self, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if len(entries) != 2:
            return entries
        left_entry = next((entry for entry in entries if not entry["is_right"]), None)
        right_entry = next((entry for entry in entries if entry["is_right"]), None)
        if left_entry is None or right_entry is None:
            return entries
        if not self._is_duplicate_left_right_pair(left_entry, right_entry):
            return entries
        winner = right_entry if float(right_entry["score"]) >= float(left_entry["score"]) else left_entry
        return [winner]

    def _extract_hands(self, frame_bgr: np.ndarray) -> list[dict[str, Any]]:
        det_out = self._detector(frame_bgr)
        det_instances = det_out["instances"]
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
        pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].detach().cpu().numpy()
        pred_scores = det_instances.scores[valid_idx].detach().cpu().numpy()
        if pred_bboxes.size == 0:
            return []

        frame_rgb = frame_bgr[:, :, ::-1]
        pose_inputs = [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)]
        vitposes_out = self._vitpose.predict_pose(frame_rgb, pose_inputs)

        entries: list[dict[str, Any]] = []
        for vitposes in vitposes_out:
            hand_candidates = [
                ("left", False, vitposes["keypoints"][-42:-21]),
                ("right", True, vitposes["keypoints"][-21:]),
            ]
            person_entries: list[dict[str, Any]] = []
            for hand_type, is_right, keypoints in hand_candidates:
                keypoints = np.asarray(keypoints, dtype=np.float32)
                valid = keypoints[:, 2] > 0.5
                if int(np.sum(valid)) <= 3:
                    continue
                bbox = np.asarray(
                    [
                        float(np.min(keypoints[valid, 0])),
                        float(np.min(keypoints[valid, 1])),
                        float(np.max(keypoints[valid, 0])),
                        float(np.max(keypoints[valid, 1])),
                    ],
                    dtype=np.float32,
                )
                person_entries.append(
                    {
                        "hand_type": hand_type,
                        "is_right": is_right,
                        "score": float(np.mean(keypoints[valid, 2])),
                        "bbox_xyxy": bbox,
                        "keypoints_uv": keypoints[:, :2].astype(np.float32),
                    }
                )
            entries.extend(self._deduplicate_entries(person_entries))
        return entries

    def run(self, video_path: Path) -> HamerSequenceResult:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise OSError(f"Failed to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or np.isnan(fps):
            fps = 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        intrinsic = self._frame_intrinsic(frame_width, frame_height)

        sequence: list[HamerHandFrameResult] = []
        frame_idx = 0
        flip_mat = np.diag([-1.0, 1.0, 1.0]).astype(np.float32)

        try:
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break

                entries = self._extract_hands(frame_bgr)
                if not entries:
                    frame_idx += 1
                    continue

                boxes_np = np.stack([entry["bbox_xyxy"] for entry in entries], axis=0)
                right_np = np.asarray([1.0 if entry["is_right"] else 0.0 for entry in entries], dtype=np.float32)
                dataset = self._ViTDetDataset(
                    self.model_cfg,
                    frame_bgr,
                    boxes_np,
                    right_np,
                    rescale_factor=2.0,
                )
                dataloader = self._torch.utils.data.DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=0,
                )

                timestamp = frame_idx / fps
                for batch in dataloader:
                    batch = self._recursive_to(batch, self.device)
                    with self._torch.no_grad():
                        out = self.model(batch)

                    multiplier = 2.0 * batch["right"] - 1.0
                    pred_cam = out["pred_cam"].clone()
                    pred_cam[:, 1] = multiplier * pred_cam[:, 1]
                    focal_tensor = self._torch.as_tensor(
                        intrinsic[0, 0],
                        dtype=pred_cam.dtype,
                        device=pred_cam.device,
                    )
                    pred_cam_t_full = self._cam_crop_to_full(
                        pred_cam,
                        batch["box_center"].float(),
                        batch["box_size"].float(),
                        batch["img_size"].float(),
                        focal_tensor,
                    ).detach().cpu().numpy()

                    global_orients = out["pred_mano_params"]["global_orient"].detach().cpu().numpy()
                    pred_vertices = out["pred_vertices"].detach().cpu().numpy()
                    if "pred_keypoints_3d" in out:
                        pred_keypoints = out["pred_keypoints_3d"].detach().cpu().numpy()
                    else:
                        pred_keypoints = pred_vertices[:, :21]

                    batch_size = batch["img"].shape[0]
                    for item_idx in range(batch_size):
                        person_id = int(batch["personid"][item_idx])
                        entry = entries[person_id]
                        is_right = bool(entry["is_right"])
                        cam_t = pred_cam_t_full[item_idx].astype(np.float32)
                        verts = pred_vertices[item_idx].copy()
                        keypoints = pred_keypoints[item_idx].copy()
                        if not is_right:
                            verts[:, 0] *= -1.0
                            keypoints[:, 0] *= -1.0

                        vertices_cam = verts + cam_t[None, :]
                        keypoints_cam = keypoints + cam_t[None, :]
                        wrist_position = keypoints_cam[0].astype(np.float32)

                        orient_mat = global_orients[item_idx]
                        if orient_mat.ndim == 3:
                            orient_mat = orient_mat[0]
                        orient_mat = orient_mat.astype(np.float32)
                        if not is_right:
                            orient_mat = flip_mat @ orient_mat @ flip_mat
                        wrist_quat = matrix_to_quaternion_wxyz(orient_mat)

                        sequence.append(
                            HamerHandFrameResult(
                                frame_idx=frame_idx,
                                hand_idx=person_id,
                                timestamp=timestamp,
                                hand_type=str(entry["hand_type"]),
                                confidence=float(entry["score"]),
                                bbox_xyxy=entry["bbox_xyxy"].astype(np.float32),
                                wrist_position=wrist_position,
                                wrist_quaternion=wrist_quat,
                                keypoints_cam=keypoints_cam.astype(np.float32),
                                wrist_uv=entry["keypoints_uv"][0].astype(np.float32),
                                keypoints_uv=entry["keypoints_uv"].astype(np.float32),
                                cam_translation=cam_t,
                                vertices_cam=vertices_cam.astype(np.float32),
                            )
                        )

                frame_idx += 1
        finally:
            cap.release()

        return HamerSequenceResult(
            video_path=video_path,
            frames=sequence,
            fps=float(fps),
            total_frames=total_frames,
            frame_size=(frame_width, frame_height),
            intrinsic=intrinsic.astype(np.float32),
        )
