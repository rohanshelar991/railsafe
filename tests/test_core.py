import numpy as np

from enhanced.config import load_config
from enhanced.distance import MonocularDistanceEstimator
from enhanced.risk import RiskEngine
from enhanced.safety import SafetyDecisionEngine
from enhanced.segmentation import GeometricRailMask
from enhanced.segmentation.rail_mask import build_track_segmenter
from enhanced.tracking import TemporalConfirmation
from enhanced.track_geometry import classify_track_overlap
from enhanced.types import Detection, RiskLevel, TemporalStatus, TrackStatus, TrainState
from simulation import TrainPhysics, TrainSimulator
from safety.ttc import time_to_collision


def config() -> dict:
    return load_config()


def test_ttc_uses_consistent_units() -> None:
    assert RiskEngine.ttc(100, 36) == 10.0
    assert RiskEngine.ttc(100, 0) is None
    assert RiskEngine.ttc(None, 36) is None


def test_risk_ignores_off_track_obstacle() -> None:
    assert RiskEngine(config()["risk"]).assess(False, 1, 60).risk == RiskLevel.LOW


def test_no_detection_is_safe() -> None:
    assert RiskEngine(config()["risk"]).assess(False, None, 60, obstacle_detected=False).risk == RiskLevel.SAFE


def test_geometric_mask_contains_lower_center() -> None:
    mask = GeometricRailMask(config()["track_segmentation"]).predict(np.zeros((100, 200, 3), dtype=np.uint8))
    assert mask[99, 100] == 255


def test_track_overlap_has_three_explicit_states() -> None:
    track_config = config()["on_track"]
    assert classify_track_overlap(0.0, track_config) == TrackStatus.OFF_TRACK
    assert classify_track_overlap(0.10, track_config) == TrackStatus.PARTIALLY_ON_TRACK
    assert classify_track_overlap(0.31, track_config) == TrackStatus.ON_TRACK


def test_partial_track_overlap_cannot_trigger_automatic_braking() -> None:
    cfg = config()
    detection = Detection((0, 0, 100, 100), 0.9, "person", track_id=4, estimated_distance_m=1)
    detection.track_status = TrackStatus.PARTIALLY_ON_TRACK
    detection.temporal_status = TemporalStatus.CONFIRMED
    assessment = SafetyDecisionEngine(cfg["risk"], cfg["train"]).assess(detection, 60)
    assert assessment.risk == RiskLevel.CAUTION
    assert assessment.decision == "MONITOR"


def test_ttc_helper_rejects_invalid_inputs() -> None:
    assert time_to_collision(20, 10) == 2
    assert time_to_collision(None, 10) is None
    assert time_to_collision(-1, 10) is None
    assert time_to_collision(20, 0) is None


def test_auto_track_detection_falls_back_without_reference_checkpoint() -> None:
    segmenter = build_track_segmenter({
        "method": "auto",
        "model_path": "/tmp/railsafe-missing-track-model.pt",
        "fallback_top_width_ratio": 0.1,
        "fallback_bottom_width_ratio": 0.8,
        "fallback_horizon_ratio": 0.4,
    })
    assert segmenter.name == "geometric_corridor"


def test_temporal_confirmation_requires_consecutive_track_evidence() -> None:
    temporal = TemporalConfirmation({"confirmation_frames": 3, "lost_track_frames": 2, "minimum_confidence": 0.4})
    assert temporal.update(7, True, 0.9) == TemporalStatus.CANDIDATE
    assert temporal.update(7, True, 0.9) == TemporalStatus.CANDIDATE
    assert temporal.update(7, True, 0.9) == TemporalStatus.CONFIRMED
    assert temporal.update(7, False, 0.9) == TemporalStatus.CONFIRMED
    assert temporal.update(7, False, 0.9) == TemporalStatus.LOST


def test_distance_is_unknown_without_known_class_or_usable_box() -> None:
    estimator = MonocularDistanceEstimator(config()["distance"])
    assert estimator.estimate("unknown", (0, 0, 10, 100)) is None
    assert estimator.estimate("car", (0, 0, 10, 2)) is None
    assert estimator.estimate("person", (0, 0, 10, 70)) == 17.0


def test_stopping_distance_and_margin() -> None:
    metrics = TrainPhysics.stopping_metrics(36, emergency_deceleration_mps2=2.5, reaction_time_s=1.5, obstacle_distance_m=30)
    assert metrics.reaction_distance_m == 15.0
    assert metrics.braking_distance_m == 20.0
    assert metrics.stopping_distance_m == 35.0
    assert metrics.safety_margin_m == -5.0


def test_confirmed_on_track_negative_margin_triggers_critical() -> None:
    cfg = config()
    detection = Detection((0, 0, 100, 100), 0.9, "person", track_id=4, estimated_distance_m=5)
    detection.track_status = TrackStatus.ON_TRACK
    detection.temporal_status = TemporalStatus.CONFIRMED
    assessment = SafetyDecisionEngine(cfg["risk"], cfg["train"]).assess(detection, 60)
    assert assessment.risk == RiskLevel.CRITICAL
    assert assessment.decision == "EMERGENCY_BRAKING"


def test_candidate_is_not_allowed_to_trigger_emergency_braking() -> None:
    cfg = config()
    detection = Detection((0, 0, 100, 100), 0.9, "person", track_id=4, estimated_distance_m=1)
    detection.track_status = TrackStatus.ON_TRACK
    detection.temporal_status = TemporalStatus.CANDIDATE
    assessment = SafetyDecisionEngine(cfg["risk"], cfg["train"]).assess(detection, 60)
    assert assessment.risk == RiskLevel.CAUTION
    assert assessment.decision == "SAFE"


def test_critical_risk_brakes_to_stop_over_time() -> None:
    train = TrainSimulator(config()["train"])
    train.speed_kmh = 10
    for _ in range(4):
        train.update(RiskLevel.CRITICAL, 1.0)
    assert train.speed_kmh == 0
    assert train.state == TrainState.STOPPED


def test_reset_restores_a_non_braking_state() -> None:
    train = TrainSimulator(config()["train"])
    train.speed_kmh = 20
    train.emergency_stop()
    train.update(RiskLevel.CRITICAL, 1.0)
    train.reset()
    assert not train.brake_active
    assert train.state == TrainState.STOPPED


def test_keyboard_control_and_automatic_critical_stop_are_independent() -> None:
    train = TrainSimulator(config()["train"])

    # UP: stopped -> accelerating, then the normal update settles at moving.
    train.accelerate(3.0)
    assert train.state == TrainState.ACCELERATING
    assert train.speed_kmh > 0
    train.update(RiskLevel.SAFE, 0.1)
    assert train.state == TrainState.MOVING

    # DOWN: service braking only; repeated service intervals reach STOPPED.
    train.decelerate(1.0)
    assert train.state == TrainState.SERVICE_BRAKING
    while train.speed_kmh > 0:
        train.decelerate(1.0)
    assert train.state == TrainState.STOPPED
    assert not train.brake_active

    # Restart, then inject a confirmed on-track critical assessment. No DOWN.
    train.reset()
    train.accelerate(10.0)
    train.update(RiskLevel.SAFE, 0.1)
    starting_speed = train.speed_kmh
    assert train.state == TrainState.MOVING
    train.update(RiskLevel.CRITICAL, 1.0)
    assert train.brake_active
    assert train.state == TrainState.EMERGENCY_BRAKING
    assert train.speed_kmh < starting_speed
    while train.speed_kmh > 0:
        previous_speed = train.speed_kmh
        train.update(RiskLevel.CRITICAL, 1.0)
        assert train.speed_kmh <= previous_speed
    assert train.state == TrainState.STOPPED
    assert train.speed_kmh == 0
