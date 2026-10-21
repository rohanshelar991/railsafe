from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from enhanced.detection import YOLODetector
from enhanced.distance import MonocularDistanceEstimator
from enhanced.safety import SafetyDecisionEngine
from enhanced.segmentation import build_track_segmenter
from enhanced.track_geometry import classify_track_overlap, lateral_position
from enhanced.tracking import TemporalConfirmation
from enhanced.trajectory import TrajectoryEstimator
from enhanced.types import Detection, RiskLevel, SafetyAssessment, TrackStatus
from simulation import TrainSimulator


@dataclass
class FrameResult:
    frame: np.ndarray
    mask: np.ndarray
    detections: list[Detection]
    assessment: SafetyAssessment
    assessments: list[SafetyAssessment]
    latency_ms: float
    segmentation_method: str
    segmentation_error: str | None = None
    segmentation_ms: float = 0.0
    detection_ms: float = 0.0
    safety_ms: float = 0.0
    train_state: str = "STOPPED"


class RailSafetyPipeline:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.segmenter = build_track_segmenter(config.get("track_detection", config["track_segmentation"]))
        self.detector = YOLODetector(config["detection"])
        self.distance = MonocularDistanceEstimator(config["distance"])
        self.temporal = TemporalConfirmation(config["temporal"])
        self.trajectory = TrajectoryEstimator()
        self.safety = SafetyDecisionEngine(config["risk"], config["train"])
        self.train = TrainSimulator(config["train"])
        self.elapsed_s = 0.0
        self.frame_index = 0
        self.segmentation_error: str | None = getattr(self.segmenter, "error", None)

    def load_models(self) -> bool:
        return self.detector.load()

    def reset(self) -> None:
        self.train.reset()
        self.elapsed_s = 0.0
        self.frame_index = 0
        self.temporal.reset()
        self.trajectory.reset()
        self.detector.reset()

    def process(self, frame: np.ndarray, dt_s: float = 1 / 30) -> FrameResult:
        started = time.perf_counter()
        self.elapsed_s += max(0.0, dt_s)
        self.frame_index += 1
        segmentation_started = time.perf_counter()
        try:
            mask = self.segmenter.predict(frame)
            self.segmentation_error = None
        except RuntimeError as exc:
            self.segmentation_error = str(exc)
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        segmentation_ms = (time.perf_counter() - segmentation_started) * 1000
        detection_started = time.perf_counter()
        detections = self.detector.detect(frame)
        detection_ms = (time.perf_counter() - detection_started) * 1000
        safety_started = time.perf_counter()
        observed_ids: set[int] = set()
        assessments: list[SafetyAssessment] = []
        for detection in detections:
            detection.timestamp_s = self.elapsed_s
            detection.frame_index = self.frame_index
            x1, y1, x2, y2 = detection.bbox
            clipped = mask[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            box_area = max(1, (x2 - x1) * (y2 - y1))
            detection.track_overlap = float(np.count_nonzero(clipped)) / box_area
            detection.track_status = classify_track_overlap(detection.track_overlap, self.config["on_track"])
            detection.on_track = detection.track_status == TrackStatus.ON_TRACK
            detection.lateral_position, detection.lateral_class = lateral_position(
                mask, detection.bbox, float(self.config["on_track"]["lateral_center_threshold"])
            )
            detection.estimated_distance_m = self.distance.estimate(detection.label, detection.bbox)
            eligible = detection.track_status in {TrackStatus.PARTIALLY_ON_TRACK, TrackStatus.ON_TRACK}
            detection.temporal_status = self.temporal.update(detection.track_id, eligible, detection.confidence)
            if detection.track_id is not None:
                observed_ids.add(detection.track_id)
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            trajectory = self.trajectory.update(detection.track_id, center, detection.estimated_distance_m, self.elapsed_s)
            detection.estimated_velocity_px_s = trajectory.velocity_px_s
            detection.estimated_approach_rate_mps = trajectory.approach_rate_mps
            detection.trajectory_status = trajectory.status
            assessments.append(self.safety.assess(detection, self.train.speed_kmh))
        self.temporal.advance_missing(observed_ids)
        if assessments:
            rank = {RiskLevel.SAFE: 0, RiskLevel.LOW: 1, RiskLevel.CAUTION: 2, RiskLevel.WARNING: 3, RiskLevel.HIGH: 4, RiskLevel.CRITICAL: 5}
            # Risk is ranked first; for ties, the smallest valid margin/TTC wins.
            assessment = max(
                assessments,
                key=lambda item: (
                    rank[item.risk],
                    -(item.safety_margin_m if item.safety_margin_m is not None else float("inf")),
                    -(item.ttc_s if item.ttc_s is not None else float("inf")),
                ),
            )
        else:
            assessment = self.safety.assess(None, self.train.speed_kmh)
        self.train.update(assessment.risk, dt_s)
        safety_ms = (time.perf_counter() - safety_started) * 1000
        latency_ms = (time.perf_counter() - started) * 1000
        return FrameResult(
            frame, mask, detections, assessment, assessments, latency_ms, self.segmenter.name,
            self.segmentation_error, segmentation_ms, detection_ms, safety_ms,
            self.train.state.value,
        )

    def render(self, result: FrameResult) -> np.ndarray:
        output = result.frame.copy()
        overlay = output.copy()
        overlay[result.mask > 0] = (40, 180, 40)
        output = cv2.addWeighted(output, 0.78, overlay, 0.22, 0)
        for detection in result.detections:
            x1, y1, x2, y2 = detection.bbox
            color = (0, 0, 255) if detection.track_status == TrackStatus.ON_TRACK else (0, 190, 255)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            identity = f"{detection.label} #{detection.track_id if detection.track_id is not None else '-'} {detection.confidence:.0%}"
            details = f"{detection.track_status.value} {detection.track_overlap:.0%} | est. {detection.estimated_distance_m if detection.estimated_distance_m else 'n/a'} m"
            cv2.putText(output, identity, (x1, max(22, y1 - 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2)
            cv2.putText(output, details, (x1, max(42, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1)
            if len(detection.trajectory) > 1:
                cv2.polylines(output, [np.array(detection.trajectory, dtype=np.int32)], False, color, 2)
        status = f"{self.train.state.value} | {self.train.speed_kmh:.1f} km/h | {result.assessment.risk.value}"
        cv2.rectangle(output, (10, 10), (min(output.shape[1] - 10, 590), 42), (20, 20, 20), -1)
        cv2.putText(output, status, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
        return output
