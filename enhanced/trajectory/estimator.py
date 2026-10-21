from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Optional

from enhanced.types import TrajectoryStatus


@dataclass
class TrajectoryEstimate:
    velocity_px_s: Optional[tuple[float, float]]
    approach_rate_mps: Optional[float]
    status: TrajectoryStatus


class TrajectoryEstimator:
    """Finite-difference screen-space trajectory; physical motion is never asserted."""

    def __init__(self, history_length: int = 12) -> None:
        self.history: dict[int, deque[tuple[float, float, float | None, float]]] = defaultdict(
            lambda: deque(maxlen=history_length)
        )

    def update(
        self,
        track_id: int | None,
        center: tuple[int, int],
        distance_m: float | None,
        timestamp_s: float,
    ) -> TrajectoryEstimate:
        if track_id is None:
            return TrajectoryEstimate(None, None, TrajectoryStatus.UNCERTAIN)
        points = self.history[track_id]
        points.append((float(center[0]), float(center[1]), distance_m, timestamp_s))
        if len(points) < 3:
            return TrajectoryEstimate(None, None, TrajectoryStatus.UNCERTAIN)
        first, last = points[0], points[-1]
        dt = last[3] - first[3]
        if dt <= 0:
            return TrajectoryEstimate(None, None, TrajectoryStatus.UNCERTAIN)
        velocity = ((last[0] - first[0]) / dt, (last[1] - first[1]) / dt)
        approach_rate = None
        if first[2] is not None and last[2] is not None:
            approach_rate = (first[2] - last[2]) / dt
        magnitude = abs(velocity[0]) + abs(velocity[1])
        if magnitude < 3:
            status = TrajectoryStatus.STATIONARY
        elif abs(velocity[0]) > abs(velocity[1]) * 1.5:
            status = TrajectoryStatus.CROSSING
        elif approach_rate is not None and approach_rate > 0.15:
            status = TrajectoryStatus.APPROACHING
        elif approach_rate is not None and approach_rate < -0.15:
            status = TrajectoryStatus.MOVING_AWAY
        else:
            status = TrajectoryStatus.UNCERTAIN
        return TrajectoryEstimate(velocity, approach_rate, status)

    def reset(self) -> None:
        self.history.clear()
