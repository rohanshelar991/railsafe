# Repository Comparison

RailSafe remains the main project. The comparison was performed against the preserved `full_git_repo/` baseline and a read-only checkout of `Ramameiyappan/railway_obstacle_detection`. No files in `full_git_repo/` were changed.

| Component | Tamasino52 baseline | Ramameiyappan reference | Current RailSafe | Best usable approach | Action |
|---|---|---|---|---|---|
| Track detection | TensorFlow/Keras U-Net workflow with external checkpoint assumptions | Separate Ultralytics YOLO segmentation model at `model/track.pt`; polygons labelled `track` | Geometric perspective corridor plus replaceable segmenter contract | Optional YOLO polygon mask with explicit geometric fallback | Integrated adapter; default `auto` |
| Obstacle detection | Faster R-CNN Inception V2, TensorFlow 1-era setup | Separate Ultralytics obstacle model, image inference | YOLO11n | YOLO11n for CPU/Apple Silicon and current tracking API | Retained YOLO11n |
| Tracking | No persistent tracker found in the runnable baseline | Frame-level detection only | Ultralytics ByteTrack with persistent IDs | ByteTrack | Retained |
| On-track filtering | Heuristic colour/risk scoring | Bounding-box overlap with segmentation mask at 10% | Configurable three-state mask overlap | Three states plus configurable thresholds | Retained and documented |
| Video processing | OpenCV/Windows-oriented runner | Streamlit image upload, not continuous video | OpenCV video capture and Streamlit fragment playback | RailSafe continuous frame pipeline | Retained |
| Model loading | External TensorFlow graph/checkpoint and Windows dependencies | Bundled YOLO weights, no CPU safety model | Explicit model availability/error state and local YOLO11n | Explicit failure and no synthetic detections | Retained |
| UI | OpenCV display | Simple Streamlit image comparison and alerts | Streamlit control room with live frame, controls, trends, explainability | RailSafe UI plus reference polygon-mask idea | Retained and extended |
| Alerts | Visual warning/risk colour | gTTS audio alert requiring network access | Explainable textual decision and virtual braking | Deterministic, offline safety decision | Audio not placed in safety path |
| Distance | Not implemented | Not implemented | Configurable monocular estimate | RailSafe estimator | Retained |
| TTC | Not implemented | Not implemented | Unit-safe estimated TTC | RailSafe risk pipeline | Retained |
| Braking/simulation | Demonstration material, no current physics contract | No train simulator | Configurable virtual train and stopping physics | RailSafe | Retained |
| Logging/evaluation | Legacy evaluation code tied to TensorFlow | No research evaluation harness | CSV logs, scenario runner, ablation status, ground-truth guardrails | RailSafe | Retained |
| Data | 12 local baseline MP4 files | Model files and image-oriented app; no compatible labelled evaluation set identified | Baseline manifest and explicit ground-truth-unavailable status | Keep datasets separate and claims bounded | Retained |
| Mac compatibility | CUDA/TensorFlow 1/Windows assumptions | Modern Python stack but bundled weights/classes are reference-specific | CPU-oriented PyTorch/Ultralytics/OpenCV/Streamlit | RailSafe dependencies | Retained |
| Maintainability | Monolithic legacy scripts | Small image demo with logic in `utils/detector.py` | Modular detector, tracker, track, safety, simulation, dashboard packages | RailSafe modular boundaries | Retained |

## Integration decision

The useful reference contribution is the idea of a separate learned track mask and polygon-overlap classification. RailSafe now supports `track_detection.method: auto`: it attempts `models/track.pt` when a compatible checkpoint is supplied and otherwise uses the transparent geometric corridor. The reference model is not copied or treated as validated on the baseline footage.

The reference image-only Streamlit flow, gTTS network audio, model-specific class names, and bundled weights were not copied into the safety loop. They would add runtime assumptions without improving temporal confirmation, TTC, braking physics, or reproducibility on Apple Silicon.
