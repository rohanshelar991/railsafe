from __future__ import annotations

from enhanced.types import RiskLevel, TrainState


class TrainPhysics:
    """Unit-consistent virtual stopping-distance calculations."""

    @staticmethod
    def stopping_metrics(
        speed_kmh: float,
        emergency_deceleration_mps2: float,
        reaction_time_s: float,
        obstacle_distance_m: float | None,
    ) -> "StoppingMetrics":
        speed_mps = max(0.0, speed_kmh / 3.6)
        if emergency_deceleration_mps2 <= 0:
            raise ValueError("Emergency deceleration must be positive")
        reaction_distance = speed_mps * max(0.0, reaction_time_s)
        braking_distance = speed_mps ** 2 / (2 * emergency_deceleration_mps2)
        stopping_distance = reaction_distance + braking_distance
        safety_margin = None if obstacle_distance_m is None else obstacle_distance_m - stopping_distance
        return StoppingMetrics(reaction_distance, braking_distance, stopping_distance, safety_margin)


class StoppingMetrics:
    def __init__(self, reaction_distance_m: float, braking_distance_m: float, stopping_distance_m: float, safety_margin_m: float | None) -> None:
        self.reaction_distance_m = round(reaction_distance_m, 2)
        self.braking_distance_m = round(braking_distance_m, 2)
        self.stopping_distance_m = round(stopping_distance_m, 2)
        self.safety_margin_m = None if safety_margin_m is None else round(safety_margin_m, 2)


class TrainSimulator:
    """Software-only longitudinal train state machine. It controls no hardware."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.speed_kmh = float(config["initial_speed_kmh"])
        self.acceleration_mps2 = 0.0
        self.brake_active = False
        self.state = TrainState.STOPPED if self.speed_kmh == 0 else TrainState.MOVING

    def reset(self) -> None:
        self.speed_kmh = float(self.config["initial_speed_kmh"])
        self.acceleration_mps2 = 0.0
        self.brake_active = False
        self.state = TrainState.STOPPED if self.speed_kmh == 0 else TrainState.MOVING

    def accelerate(self, dt_s: float = 1.0) -> None:
        if self.brake_active:
            return
        delta = float(self.config["acceleration_mps2"]) * dt_s * 3.6
        self.speed_kmh = min(float(self.config["max_speed_kmh"]), self.speed_kmh + delta)
        self.acceleration_mps2 = float(self.config["acceleration_mps2"])
        self.state = TrainState.ACCELERATING if self.speed_kmh > 0 else TrainState.STOPPED

    def decelerate(self, dt_s: float = 1.0, emergency: bool = False) -> None:
        rate = float(self.config["emergency_brake_mps2"] if emergency else self.config["service_brake_mps2"])
        self.speed_kmh = max(0.0, self.speed_kmh - rate * dt_s * 3.6)
        self.acceleration_mps2 = -rate
        self.state = (TrainState.EMERGENCY_BRAKING if emergency else TrainState.SERVICE_BRAKING) if self.speed_kmh > 0 else TrainState.STOPPED

    def emergency_stop(self) -> None:
        """Engage emergency braking; `update` reduces virtual speed over time."""
        self.brake_active = True
        self.acceleration_mps2 = -float(self.config["emergency_brake_mps2"])
        self.state = TrainState.EMERGENCY_BRAKING if self.speed_kmh > 0 else TrainState.STOPPED

    def update(self, risk: RiskLevel, dt_s: float) -> None:
        if risk == RiskLevel.CRITICAL:
            self.brake_active = True
        if self.brake_active:
            self.decelerate(dt_s, emergency=True)
        elif risk == RiskLevel.HIGH:
            self.state = TrainState.WARNING
        elif risk in {RiskLevel.CAUTION, RiskLevel.WARNING}:
            self.state = TrainState.CAUTION
        else:
            self.state = TrainState.MOVING if self.speed_kmh > 0 else TrainState.STOPPED
