from __future__ import annotations

from dataclasses import dataclass

from enhanced.types import TemporalStatus


@dataclass
class _TrackEvidence:
    consecutive_frames: int = 0
    missed_frames: int = 0
    confirmed: bool = False


class TemporalConfirmation:
    """Per-track confirmation/hysteresis; untracked boxes cannot become safety events."""

    def __init__(self, config: dict) -> None:
        self.confirmation_frames = int(config["confirmation_frames"])
        self.lost_track_frames = int(config["lost_track_frames"])
        self.minimum_confidence = float(config["minimum_confidence"])
        self.tracks: dict[int, _TrackEvidence] = {}

    def update(self, track_id: int | None, eligible: bool, confidence: float) -> TemporalStatus:
        if track_id is None:
            return TemporalStatus.DETECTED
        state = self.tracks.setdefault(track_id, _TrackEvidence())
        if eligible and confidence >= self.minimum_confidence:
            state.consecutive_frames += 1
            state.missed_frames = 0
            state.confirmed = state.confirmed or state.consecutive_frames >= self.confirmation_frames
            return TemporalStatus.CONFIRMED if state.confirmed else TemporalStatus.CANDIDATE
        state.consecutive_frames = 0
        state.missed_frames += 1
        if state.missed_frames >= self.lost_track_frames:
            state.confirmed = False
            return TemporalStatus.LOST
        return TemporalStatus.CONFIRMED if state.confirmed else TemporalStatus.DETECTED

    def advance_missing(self, observed_ids: set[int]) -> None:
        for track_id, state in self.tracks.items():
            if track_id not in observed_ids:
                state.consecutive_frames = 0
                state.missed_frames += 1
                if state.missed_frames >= self.lost_track_frames:
                    state.confirmed = False

    def reset(self) -> None:
        self.tracks.clear()
