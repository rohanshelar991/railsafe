from __future__ import annotations

import numpy as np

from enhanced.types import TrackStatus


def classify_track_overlap(overlap: float, config: dict) -> TrackStatus:
    if overlap >= float(config["overlap_threshold"]):
        return TrackStatus.ON_TRACK
    if overlap >= float(config["partial_overlap_threshold"]):
        return TrackStatus.PARTIALLY_ON_TRACK
    return TrackStatus.OFF_TRACK


def lateral_position(mask: np.ndarray, bbox: tuple[int, int, int, int], center_threshold: float) -> tuple[float | None, str]:
    """Return an estimated lateral coordinate relative to track width at the contact row."""
    _, y1, _, y2 = bbox
    row_index = max(0, min(mask.shape[0] - 1, y2 - 1))
    columns = np.flatnonzero(mask[row_index] > 0)
    if len(columns) < 2:
        return None, "UNKNOWN"
    center_x = (bbox[0] + bbox[2]) / 2
    midpoint = (columns[0] + columns[-1]) / 2
    half_width = max(1.0, (columns[-1] - columns[0]) / 2)
    normalized = float(np.clip((center_x - midpoint) / half_width, -1.0, 1.0))
    if normalized < -center_threshold:
        position = "LEFT_OF_TRACK"
    elif normalized > center_threshold:
        position = "RIGHT_OF_TRACK"
    else:
        position = "CENTER_OF_TRACK"
    return round(normalized, 3), position
