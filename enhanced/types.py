from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class RiskLevel(str, Enum):
    SAFE = "SAFE"
    LOW = "LOW"
    CAUTION = "CAUTION"
    WARNING = "WARNING"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class TrainState(str, Enum):
    STOPPED = "STOPPED"
    ACCELERATING = "ACCELERATING"
    MOVING = "MOVING"
    CAUTION = "CAUTION"
    WARNING = "WARNING"
    SERVICE_BRAKING = "SERVICE_BRAKING"
    EMERGENCY_BRAKING = "EMERGENCY_BRAKING"


class TrackStatus(str, Enum):
    OFF_TRACK = "OFF_TRACK"
    PARTIALLY_ON_TRACK = "PARTIALLY_ON_TRACK"
    ON_TRACK = "ON_TRACK"


class TemporalStatus(str, Enum):
    DETECTED = "DETECTED"
    CANDIDATE = "CANDIDATE"
    CONFIRMED = "CONFIRMED"
    LOST = "LOST"


class TrajectoryStatus(str, Enum):
    STATIONARY = "STATIONARY"
    MOVING_AWAY = "MOVING_AWAY"
    APPROACHING = "APPROACHING"
    CROSSING = "CROSSING"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class Detection:
    bbox: tuple[int, int, int, int]
    confidence: float
    label: str
    track_id: Optional[int] = None
    trajectory: list[tuple[int, int]] = field(default_factory=list)
    track_overlap: float = 0.0
    on_track: bool = False
    estimated_distance_m: Optional[float] = None
    track_status: TrackStatus = TrackStatus.OFF_TRACK
    temporal_status: TemporalStatus = TemporalStatus.DETECTED
    lateral_position: Optional[float] = None
    lateral_class: str = "UNKNOWN"
    estimated_velocity_px_s: Optional[tuple[float, float]] = None
    estimated_approach_rate_mps: Optional[float] = None
    trajectory_status: TrajectoryStatus = TrajectoryStatus.UNCERTAIN
    timestamp_s: Optional[float] = None
    frame_index: Optional[int] = None


@dataclass
class SafetyAssessment:
    risk: RiskLevel
    ttc_s: Optional[float]
    reason: str
    braking_distance_m: Optional[float] = None
    reaction_distance_m: Optional[float] = None
    stopping_distance_m: Optional[float] = None
    safety_margin_m: Optional[float] = None
    decision: str = "SAFE"
    object_id: Optional[int] = None
