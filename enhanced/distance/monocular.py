from __future__ import annotations

from typing import Optional


class MonocularDistanceEstimator:
    """Approximate distance from assumed class height and calibrated focal length."""

    def __init__(self, config: dict) -> None:
        self.focal_length_px = float(config["focal_length_px"])
        self.heights = {str(key): float(value) for key, value in config["class_heights_m"].items()}
        self.minimum_height_px = int(config["minimum_bbox_height_px"])

    def estimate(self, label: str, bbox: tuple[int, int, int, int]) -> Optional[float]:
        pixel_height = bbox[3] - bbox[1]
        object_height = self.heights.get(label)
        if object_height is None or pixel_height < self.minimum_height_px:
            return None
        return round(self.focal_length_px * object_height / pixel_height, 2)
