from __future__ import annotations

from enhanced.risk import RiskEngine
from enhanced.types import Detection, RiskLevel, SafetyAssessment, TemporalStatus, TrackStatus, TrajectoryStatus
from simulation.train import TrainPhysics


class SafetyDecisionEngine:
    """Transparent, configurable virtual-safety decision logic."""

    def __init__(self, risk_config: dict, train_config: dict) -> None:
        self.risk = RiskEngine(risk_config)
        self.config = risk_config
        self.train_config = train_config

    def assess(self, detection: Detection | None, speed_kmh: float) -> SafetyAssessment:
        if detection is None:
            return SafetyAssessment(RiskLevel.SAFE, None, "No eligible obstacle detected")
        if detection.track_status == TrackStatus.OFF_TRACK:
            return SafetyAssessment(RiskLevel.LOW, None, "Obstacle is outside the estimated track region", object_id=detection.track_id)
        if detection.track_status == TrackStatus.PARTIALLY_ON_TRACK:
            return SafetyAssessment(
                RiskLevel.CAUTION, None,
                "Obstacle partially overlaps the estimated track region; automatic braking is withheld",
                decision="MONITOR", object_id=detection.track_id,
            )
        if detection.temporal_status != TemporalStatus.CONFIRMED:
            return SafetyAssessment(RiskLevel.CAUTION, None, "On-track candidate awaits temporal confirmation", object_id=detection.track_id)
        metrics = TrainPhysics.stopping_metrics(
            speed_kmh=speed_kmh,
            emergency_deceleration_mps2=float(self.train_config["emergency_brake_mps2"]),
            reaction_time_s=float(self.train_config["reaction_time_s"]),
            obstacle_distance_m=detection.estimated_distance_m,
        )
        ttc_s = self.risk.ttc(detection.estimated_distance_m, speed_kmh)
        if detection.estimated_distance_m is None:
            return SafetyAssessment(
                RiskLevel.CAUTION, None, "Confirmed on-track obstacle lacks a valid distance estimate",
                decision="CAUTION", object_id=detection.track_id,
            )
        if detection.trajectory_status == TrajectoryStatus.MOVING_AWAY:
            return SafetyAssessment(
                RiskLevel.LOW, ttc_s, "Confirmed obstacle is estimated to be moving away",
                braking_distance_m=metrics.braking_distance_m,
                reaction_distance_m=metrics.reaction_distance_m,
                stopping_distance_m=metrics.stopping_distance_m,
                safety_margin_m=metrics.safety_margin_m,
                decision="MONITOR", object_id=detection.track_id,
            )
        margin = metrics.safety_margin_m
        if (ttc_s is not None and ttc_s < float(self.config["critical_ttc_s"])) or (margin is not None and margin < float(self.config["critical_margin_m"])):
            risk, decision, reason = RiskLevel.CRITICAL, "EMERGENCY_BRAKING", "Confirmed on-track obstacle has critical estimated TTC or negative stopping margin"
        elif (ttc_s is not None and ttc_s < float(self.config["high_ttc_s"])) or (margin is not None and margin < float(self.config["warning_margin_m"])):
            risk, decision, reason = RiskLevel.HIGH, "WARNING", "Confirmed on-track obstacle has high estimated collision risk"
        elif (ttc_s is not None and ttc_s < float(self.config["warning_ttc_s"])) or (margin is not None and margin < float(self.config["caution_margin_m"])):
            risk, decision, reason = RiskLevel.WARNING, "WARNING", "Confirmed on-track obstacle is inside caution stopping margin"
        else:
            risk, decision, reason = RiskLevel.LOW, "MONITOR", "Confirmed on-track obstacle currently has positive estimated margin"
        return SafetyAssessment(
            risk=risk, ttc_s=ttc_s, reason=reason, braking_distance_m=metrics.braking_distance_m,
            reaction_distance_m=metrics.reaction_distance_m, stopping_distance_m=metrics.stopping_distance_m,
            safety_margin_m=metrics.safety_margin_m, decision=decision, object_id=detection.track_id,
        )
