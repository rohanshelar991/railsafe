"""Braking-physics helpers used by the virtual train and risk engine."""

from simulation.train import TrainPhysics, StoppingMetrics


def braking_distance(speed_mps: float, deceleration_mps2: float) -> float:
    if speed_mps < 0 or deceleration_mps2 <= 0:
        raise ValueError("speed must be non-negative and deceleration must be positive")
    return speed_mps ** 2 / (2 * deceleration_mps2)


def stopping_distance(speed_mps: float, deceleration_mps2: float, reaction_time_s: float) -> float:
    if speed_mps < 0 or reaction_time_s < 0:
        raise ValueError("speed and reaction time must be non-negative")
    return speed_mps * reaction_time_s + braking_distance(speed_mps, deceleration_mps2)


__all__ = ["TrainPhysics", "StoppingMetrics", "braking_distance", "stopping_distance"]
