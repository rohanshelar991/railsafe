# AI Pipeline

YOLO11n runs on CPU or Apple Silicon-compatible PyTorch and filters to configured obstacle classes. ByteTrack persistence is provided by Ultralytics. Detection confidence, bounding boxes, IDs, timestamps, trajectories, and track status are retained for each observation. No synthetic detections are generated when inference is unavailable.

Track awareness supports three modes through `track_detection.method`: `segmentation` loads a supplied Ultralytics segmentation checkpoint, `geometric` uses the transparent perspective corridor, and `auto` tries the learned checkpoint first and falls back to the geometric corridor when it is missing or cannot load. A reference checkpoint is not treated as validated without matching labels.
