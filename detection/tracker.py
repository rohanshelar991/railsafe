"""ByteTrack-backed tracking entry point.

Ultralytics owns the ByteTrack implementation; the detector invokes it with
``persist=True`` and exposes persistent IDs and histories as Detection records.
"""

from enhanced.detection.yolo_detector import YOLODetector

ByteTrackTracker = YOLODetector

__all__ = ["ByteTrackTracker"]
