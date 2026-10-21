from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from typing import Optional

import numpy as np

from enhanced.types import Detection


class YOLODetector:
    """YOLO11n + Ultralytics ByteTrack adapter with explicit unavailable state."""

    def __init__(self, config: dict, model_path: Optional[str] = None) -> None:
        self.config = config
        self.model_path = model_path or config["model"]
        self.model = None
        self.error: Optional[str] = None
        self.trajectories: dict[int, deque[tuple[int, int]]] = defaultdict(lambda: deque(maxlen=24))

    def load(self) -> bool:
        try:
            from ultralytics import YOLO

            self.model = YOLO(self.model_path)
            return True
        except Exception as exc:  # Model download/network/runtime errors are user-facing state.
            self.error = f"YOLO unavailable: {exc}"
            return False

    @property
    def available(self) -> bool:
        return self.model is not None

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if self.model is None:
            return []
        try:
            result = self.model.track(
                frame,
                persist=True,
                tracker=self.config["tracker"],
                conf=float(self.config["confidence_threshold"]),
                imgsz=int(self.config["image_size"]),
                verbose=False,
            )[0]
        except Exception as exc:
            self.error = f"YOLO inference failed: {exc}"
            return []
        if result.boxes is None:
            return []

        allowed = set(self.config["allowed_classes"])
        names = result.names
        detections: list[Detection] = []
        ids = result.boxes.id.int().cpu().tolist() if result.boxes.id is not None else [None] * len(result.boxes)
        for box, track_id in zip(result.boxes, ids):
            class_id = int(box.cls.item())
            label = str(names[class_id])
            if label not in allowed:
                continue
            x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].cpu().tolist())
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            if track_id is not None:
                self.trajectories[track_id].append(center)
            detections.append(Detection(
                bbox=(x1, y1, x2, y2),
                confidence=float(box.conf.item()),
                label=label,
                track_id=track_id,
                trajectory=list(self.trajectories[track_id]) if track_id is not None else [],
            ))
        return detections

    def reset(self) -> None:
        self.trajectories.clear()
        if self.model is not None:
            # A fresh track call will establish new ByteTrack identities.
            try:
                self.model.predictor = None
            except Exception:
                pass
