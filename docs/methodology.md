# Methodology and assumptions

YOLO11n uses COCO labels and Ultralytics `model.track(..., tracker="bytetrack.yaml")`. Detections outside `allowed_classes` are ignored. A persistent ID is shown only when ByteTrack yields one; an absent ID is shown as unavailable, not invented.

The fallback rail mask is a trapezoid based on the camera centre, horizon ratio, and top/bottom width ratios. It is an explicit interim geometry assumption, not railway segmentation performance. Track overlap is `nonzero(mask inside bbox) / bbox area`; on-track requires overlap >= configured threshold.

Monocular distance assumes configured focal length and nominal object heights. Perspective, occlusion, detector boxes, unknown object scale, camera mounting, and slope make it unsuitable for real-world distance claims. TTC inherits those limitations.
