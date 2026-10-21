from __future__ import annotations

from typing import Optional

from enhanced.types import RiskLevel, SafetyAssessment


class RiskEngine:
    def __init__(self, config: dict) -> None:
        self.config = config

    @staticmethod
    def ttc(distance_m: Optional[float], speed_kmh: float) -> Optional[float]:
        if distance_m is None or distance_m < 0 or speed_kmh <= 0:
            return None
        return round(distance_m / (speed_kmh / 3.6), 2)

    def assess(
        self,
        on_track: bool,
        distance_m: Optional[float],
        speed_kmh: float,
        obstacle_detected: bool = True,
    ) -> SafetyAssessment:
        if not obstacle_detected:
            return SafetyAssessment(RiskLevel.SAFE, None, "No eligible obstacle detected")
        if not on_track:
            return SafetyAssessment(RiskLevel.LOW, None, "No confirmed obstacle overlap with track region")
        ttc_s = self.ttc(distance_m, speed_kmh)
        if ttc_s is None:
            return SafetyAssessment(RiskLevel.CAUTION, None, "On-track obstacle; distance or motion estimate unavailable")
        if ttc_s < float(self.config["critical_ttc_s"]):
            return SafetyAssessment(RiskLevel.CRITICAL, ttc_s, "Estimated TTC below critical threshold")
        if ttc_s < float(self.config["high_ttc_s"]):
            return SafetyAssessment(RiskLevel.HIGH, ttc_s, "Estimated TTC below high threshold")
        if ttc_s < float(self.config["warning_ttc_s"]):
            return SafetyAssessment(RiskLevel.WARNING, ttc_s, "Estimated TTC below warning threshold")
        if ttc_s < float(self.config["caution_ttc_s"]):
            return SafetyAssessment(RiskLevel.CAUTION, ttc_s, "Estimated TTC below caution threshold")
        return SafetyAssessment(RiskLevel.LOW, ttc_s, "On-track obstacle is beyond caution TTC threshold")
