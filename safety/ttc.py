"""Unit-safe time-to-collision helper."""


def time_to_collision(distance_m: float | None, train_speed_mps: float) -> float | None:
    if distance_m is None or distance_m < 0 or train_speed_mps <= 0:
        return None
    return distance_m / train_speed_mps


__all__ = ["time_to_collision"]
