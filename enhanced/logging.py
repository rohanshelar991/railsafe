from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

from enhanced.pipeline import FrameResult


class SafetyCSVLogger:
    fields = [
        "timestamp_utc", "timestamp", "frame", "frame_id", "object_id", "class", "confidence", "track_overlap",
        "track_status", "temporal_status", "lateral_position", "lateral_class", "trajectory_status",
        "estimated_velocity_px_s", "estimated_approach_rate_mps", "estimated_distance_m", "train_speed_mps",
        "train_speed_kmh", "ttc_s", "reaction_distance_m", "braking_distance_m", "stopping_distance_m",
        "safety_margin_m", "risk", "decision", "train_state", "braking", "braking_triggered", "latency_ms",
        "segmentation_ms", "detection_ms", "safety_ms",
    ]

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            with self.path.open(encoding="utf-8") as handle:
                existing_header = handle.readline().strip().split(",")
            if existing_header and existing_header != self.fields:
                self.path = self.path.with_stem(f"{self.path.stem}_research")
        self.new_file = not self.path.exists()

    def write(self, frame_id: int, result: FrameResult, speed_kmh: float, braking: bool) -> None:
        rows = result.detections or [None]
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fields)
            if self.new_file:
                writer.writeheader()
                self.new_file = False
            for detection in rows:
                writer.writerow({
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "timestamp": result.detections[0].timestamp_s if result.detections else None,
                    "frame": frame_id,
                    "frame_id": frame_id,
                    "object_id": detection.track_id if detection else None,
                    "class": detection.label if detection else None,
                    "confidence": detection.confidence if detection else None,
                    "track_overlap": detection.track_overlap if detection else None,
                    "track_status": detection.track_status.value if detection else None,
                    "temporal_status": detection.temporal_status.value if detection else None,
                    "lateral_position": detection.lateral_position if detection else None,
                    "lateral_class": detection.lateral_class if detection else None,
                    "trajectory_status": detection.trajectory_status.value if detection else None,
                    "estimated_velocity_px_s": detection.estimated_velocity_px_s if detection else None,
                    "estimated_approach_rate_mps": detection.estimated_approach_rate_mps if detection else None,
                    "estimated_distance_m": detection.estimated_distance_m if detection else None,
                    "train_speed_mps": round(speed_kmh / 3.6, 3),
                    "train_speed_kmh": speed_kmh,
                    "ttc_s": result.assessment.ttc_s,
                    "reaction_distance_m": result.assessment.reaction_distance_m,
                    "braking_distance_m": result.assessment.braking_distance_m,
                    "stopping_distance_m": result.assessment.stopping_distance_m,
                    "safety_margin_m": result.assessment.safety_margin_m,
                    "risk": result.assessment.risk.value,
                    "decision": result.assessment.decision,
                    "train_state": result.train_state,
                    "braking": braking,
                    "braking_triggered": braking,
                    "latency_ms": result.latency_ms,
                    "segmentation_ms": result.segmentation_ms,
                    "detection_ms": result.detection_ms,
                    "safety_ms": result.safety_ms,
                })
