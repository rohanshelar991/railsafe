from __future__ import annotations

import cv2
import numpy as np


class TrackSegmenter:
    """Common contract for a track-region implementation."""

    name = "unknown"

    def predict(self, frame: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class GeometricCorridorSegmenter(TrackSegmenter):
    """Conservative perspective corridor fallback, not a learned segmentation model.

    A validated rail-segmentation checkpoint can replace this component through the
    Pipeline interface. The fallback is deliberately named and rendered as an
    estimate so it cannot be mistaken for U-Net output.
    """

    name = "geometric_corridor"

    def __init__(self, config: dict) -> None:
        self.top_width_ratio = float(config["fallback_top_width_ratio"])
        self.bottom_width_ratio = float(config["fallback_bottom_width_ratio"])
        self.horizon_ratio = float(config["fallback_horizon_ratio"])

    def predict(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        cx = width // 2
        top_y = int(height * self.horizon_ratio)
        top_half = int(width * self.top_width_ratio / 2)
        bottom_half = int(width * self.bottom_width_ratio / 2)
        polygon = np.array(
            [[cx - top_half, top_y], [cx + top_half, top_y],
             [cx + bottom_half, height - 1], [cx - bottom_half, height - 1]],
            dtype=np.int32,
        )
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(mask, polygon, 255)
        return mask

    @staticmethod
    def polygon_from_mask(mask: np.ndarray) -> np.ndarray:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.empty((0, 2), dtype=np.int32)
        return max(contours, key=cv2.contourArea)


class LearnedTrackSegmenter(TrackSegmenter):
    """TorchScript segmentation adapter; never substitutes a geometric result silently."""

    name = "learned"

    def __init__(self, config: dict) -> None:
        from pathlib import Path

        self.path = Path(config["learned_model_path"])
        self.threshold = float(config.get("threshold", 0.5))
        self.error: str | None = None
        self.model = None
        if not self.path.exists():
            self.error = f"Learned track checkpoint not found: {self.path}"
            return
        try:
            import torch

            self.torch = torch
            self.model = torch.jit.load(str(self.path), map_location="cpu").eval()
        except Exception as exc:
            self.error = f"Could not load learned track checkpoint: {exc}"

    def predict(self, frame: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError(self.error or "Learned track model unavailable")
        height, width = frame.shape[:2]
        resized = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA)
        tensor = self.torch.from_numpy(resized[..., ::-1].copy()).permute(2, 0, 1).float().div(255).unsqueeze(0)
        with self.torch.no_grad():
            prediction = self.model(tensor)
        if isinstance(prediction, (tuple, list)):
            prediction = prediction[0]
        mask = prediction.squeeze().detach().cpu().numpy()
        mask = (mask >= self.threshold).astype(np.uint8) * 255
        return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)


class YOLOTrackSegmenter(TrackSegmenter):
    """Optional Ultralytics segmentation adapter inspired by the modern reference.

    The reference repository ships a separate YOLO track model. RailSafe keeps
    that model optional because its class map and training data are not part of
    the preserved baseline. When present, polygons labelled ``track`` become
    the track mask; no model is silently replaced by synthetic detections.
    """

    name = "yolo_segmentation"

    def __init__(self, config: dict) -> None:
        from pathlib import Path

        self.path = Path(config.get("model_path", "models/track.pt"))
        self.confidence = float(config.get("confidence_threshold", 0.25))
        self.image_size = int(config.get("image_size", 640))
        self.track_class_name = str(config.get("track_class_name", "track"))
        self.error: str | None = None
        self.model = None
        if not self.path.exists():
            self.error = f"YOLO track checkpoint not found: {self.path}"
            return
        try:
            from ultralytics import YOLO

            self.model = YOLO(str(self.path))
        except Exception as exc:
            self.error = f"Could not load YOLO track checkpoint: {exc}"

    def predict(self, frame: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError(self.error or "YOLO track model unavailable")
        height, width = frame.shape[:2]
        result = self.model.predict(
            frame, conf=self.confidence, imgsz=self.image_size, device="cpu", verbose=False
        )[0]
        mask = np.zeros((height, width), dtype=np.uint8)
        if result.masks is None or result.boxes is None:
            return mask
        names = result.names
        classes = result.boxes.cls.cpu().numpy().astype(int)
        for polygon, class_id in zip(result.masks.xy, classes):
            if str(names[int(class_id)]) != self.track_class_name:
                continue
            cv2.fillPoly(mask, [np.asarray(polygon, dtype=np.int32)], 255)
        return mask


# Backward-compatible import name used by the initial prototype.
GeometricRailMask = GeometricCorridorSegmenter


def build_track_segmenter(config: dict) -> TrackSegmenter:
    method = str(config.get("method", "geometric")).lower()
    if method == "auto":
        candidate = YOLOTrackSegmenter(config)
        if candidate.model is not None:
            return candidate
        geometric = GeometricCorridorSegmenter(config)
        geometric.error = candidate.error
        return geometric
    if method in {"segmentation", "yolo", "yolo_segmentation"}:
        return YOLOTrackSegmenter(config)
    if method == "geometric":
        return GeometricCorridorSegmenter(config)
    if method == "learned":
        return LearnedTrackSegmenter(config)
    raise ValueError(f"Unsupported track segmentation method: {method}")
