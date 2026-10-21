# RailSafe: Track-Aware Railway Obstacle Safety Prototype

RailSafe is a local, software-only research prototype for front-camera railway video analysis. It combines YOLO11n detection, ByteTrack IDs, a track-region filter, monocular estimated distance, estimated TTC, risk classification, and a virtual train braking state machine. It does not control, connect to, or certify any real railway equipment.

## Problem statement

Obstacle detection from a forward-facing railway camera must distinguish objects that are actually within the rail corridor from objects beside the track, then communicate an interpretable risk estimate. A student-scale prototype needs to demonstrate this pipeline without pretending that uncalibrated monocular video is a certified signalling or braking system.

## Objectives

- Detect and track relevant obstacles in railway video.
- Filter detections by an explicit, inspectable track corridor.
- Estimate distance and time to collision (TTC) with clearly stated assumptions.
- Convert evidence into risk levels and a software-only emergency-braking state.
- Provide repeatable tests and deterministic evaluation evidence.

## Technologies

Python 3.10+, Ultralytics YOLO11n, ByteTrack, OpenCV, NumPy, PyTorch, Streamlit, Pandas, Matplotlib, PyYAML, and pytest. The implementation runs locally and does not require a cloud service.

## Current baseline and attribution

The preserved upstream reference is [Railroad_and_Obstacle_detection](https://github.com/tamasino52/Railroad_and_Obstacle_detection), checked out under `full_git_repo/` at commit `8317428c5314a7010d02314bf7da158739cea96e`. The original README attributes the work to Kim Minseok and Lee Juhee at Soongsil University.

| Capability | Upstream baseline | Proposed implementation |
| --- | --- | --- |
| Track perception | U-Net | Configurable geometric corridor fallback; replaceable by validated segmentation model |
| Detector | Faster R-CNN Inception V2 | YOLO11n (`yolo11n.pt`) |
| Tracking | None found | Ultralytics ByteTrack |
| On-track test | Heuristic colour scoring | Mask-overlap threshold |
| Distance / TTC | Not present | Explicit monocular estimate / estimated TTC |
| Response | Visual warning | Software-only train state machine and auto-braking |
| UI | OpenCV window | Streamlit dashboard and keyboard CLI |

The original system assumes TensorFlow 1, CUDA 10, the TensorFlow Models repository, Windows-only `win32*` modules, an external U-Net checkpoint, and a trained Faster R-CNN graph. It is retained as a reference, not forced onto current macOS/Python.

## Quick start

Install into a Python 3.10+ environment (the current workspace was verified with Python 3.14):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Then launch the dashboard:

```bash
source .venv/bin/activate
python run.py
```

Or launch Streamlit directly:

```bash
.venv/bin/streamlit run dashboard/app.py
```

The canonical configuration is `configs/config.yaml`; `configs/default.yaml` remains supported for compatibility.

For the OpenCV runner with direct keyboard control:

```bash
.venv/bin/python run.py --cli --video full_git_repo/video/complex_crash.mp4
```

Keyboard controls in the OpenCV runner: Up = accelerate/start, Down = service deceleration, Space = emergency stop, R = reset, Q or Esc = quit. The dashboard provides matching buttons.

## Integrated implementation status

| Component | Current implementation | Verification status |
| --- | --- | --- |
| Repository footage | All 12 upstream MP4s are checked out locally | Every file is selectable from the dashboard |
| Input sources | Repository video, uploaded video, or webcam | Implemented |
| Obstacle detection | Official Ultralytics YOLO11n weights | Weight loaded and inference ran on `complex_crash.mp4` |
| Object tracking | Ultralytics ByteTrack | Persistent IDs observed on contiguous source frames |
| Track region | Configurable geometric perspective corridor | Mask rendered for every processed frame; not a learned-model evaluation |
| On-track filter | Mask area intersecting each bounding box | Implemented with configurable overlap threshold |
| Distance / TTC | Monocular class-height estimate / distance ÷ simulated speed | Implemented and explicitly labelled as estimates |
| Risk | SAFE, LOW, CAUTION, WARNING, HIGH, CRITICAL | Unit tested |
| Train response | Independent software-only state machine and CRITICAL auto-brake | Unit tested and exercised in deterministic scenario generation |
| Interface | Streamlit dashboard plus OpenCV keyboard runner | Implemented |
| Evidence | CSV event logs and speed/distance/TTC charts | Scenario CSV and three figures generated |

The `complex_crash.mp4` smoke test did not produce a confirmed critical on-track obstacle with the present conservative corridor; it therefore stayed out of automatic braking. That is correct behavior, not a failed attempt to manufacture a demonstration. The deterministic scenario validates the simulated braking state transition only; it is not video-based perception evidence.

## How the system works

1. Open a baseline video and load YOLO11n. The Streamlit fragment continuously advances real source frames; `Process next frame` remains available for debugging.
2. YOLO11n detects configured obstacle classes and ByteTrack maintains IDs and histories.
3. Track awareness uses an optional YOLO polygon model when `track_detection.method: auto` finds `models/track.pt`; otherwise it uses the labelled geometric corridor fallback. Objects are classified as `ON_TRACK`, `PARTIALLY_ON_TRACK`, or `OFF_TRACK`.
4. A tracked object must persist for the configured confirmation window before it can become a safety event.
5. RailSafe estimates distance, trajectory, approach rate, TTC, braking distance, stopping distance, and safety margin from runtime values.
6. Press `Up` in the dashboard or the CLI to accelerate the virtual train. `Down` is manual service braking; `Space` is manual emergency stop; `R` resets the simulation.
7. Automatic braking requires `CONFIRMED + ON_TRACK + CRITICAL`. No `Down` key is required. The virtual train decelerates until its speed reaches zero and its state is `STOPPED`.

The video is a camera source and can continue playing after the virtual train stops. A prerecorded crash in a source video is not a physical consequence of the virtual train state.

## Models and inputs

YOLO11n is selected because it provides current object detection plus ByteTrack integration while remaining practical on laptop CPU/Apple Silicon. At first run, Ultralytics obtains the official `yolo11n.pt` weight if it is not already in the working directory. When this cannot happen, model loading fails visibly and no synthetic detection is used.

The original repository contains these locally checked-out videos: `complex1`–`complex7`, `complex_crash`, `complex_left`, `complex_right`, `normal_left`, and `normal_right` (MP4). The initial dashboard/CLI source is `complex_crash.mp4`, verified at 1920×1080, 30 FPS, 268 frames (8.93 seconds). The dashboard also supports uploads and webcam input.

The current track region is a conservative perspective corridor marked in green. It is a transparent fallback, not a claim of learned segmentation. A U-Net or other segmentation checkpoint must be validated on comparable ground truth before it replaces this component.

## Estimation and risk

Distance uses `focal_length_px * nominal_class_height_m / bounding_box_height_px`; all values are configurable in `configs/default.yaml` and are displayed as **estimated**. TTC is `estimated_distance_m / train_speed_mps`. No distance, TTC, or accuracy claim is made without calibration and ground truth.

An obstacle counts as on-track when the overlap of its bounding-box area and the track-region mask meets `track.overlap_threshold`. Risk is `SAFE` when there is no eligible detection, `LOW` for off-track detections, then `CAUTION`, `WARNING`, `HIGH`, and `CRITICAL` by configured estimated-TTC thresholds. Only `CRITICAL` engages automatic simulated emergency braking.

The editable controls are in `configs/default.yaml`: YOLO confidence and classes, ByteTrack configuration, track overlap, camera/nominal-height assumptions, TTC thresholds, and virtual train acceleration/deceleration. Change an assumption there rather than burying it in code.

## Evaluation

```bash
.venv/bin/python -m pytest -q
.venv/bin/python evaluation/run_scenarios.py
.venv/bin/python evaluation/ablation.py
.venv/bin/python evaluation/evaluate.py
```

The scenario generator emits `outputs/scenario_critical.csv` and speed/distance/TTC charts. It is a deterministic software state-machine scenario, not an AI benchmark and not a measured real-world result. Dataset-backed detector, segmentation, tracking, and safety metrics remain intentionally unreported until labels and ground truth are supplied.

## Methodology

Each frame is processed by YOLO11n and ByteTrack. A configurable trapezoidal rail corridor (or an optional validated segmentation checkpoint) is intersected with each bounding box. Consecutive on-track observations are temporally confirmed before they become safety events. Distance is estimated from focal length, an assumed object height, and detected box height; TTC is estimated distance divided by virtual-train speed. Risk thresholds then drive an independent virtual-train state machine. See [docs/methodology.md](docs/methodology.md) for assumptions and [docs/system_architecture.md](docs/system_architecture.md) for the component view.

## Results

The reproducible baseline currently reports 16 passing deterministic tests. The generated scenario demonstrates the intended critical-risk-to-stop transition and writes CSV/plot evidence under `outputs/`. A video smoke run exercised YOLO11n and ByteTrack on `complex_crash.mp4`, but did not confirm a critical on-track obstacle with the conservative fallback corridor. This is reported as an observation rather than converted into an accuracy claim.

## Reproducibility and input media

The source code, configuration, tests, and deterministic scenario are versioned. Large upstream videos are intentionally excluded from GitHub to keep the repository practical; place compatible `.mp4` files in `full_git_repo/video/` or upload a file through the dashboard. The default configuration and tests do not require video media.

From a fresh checkout:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q
.venv/bin/python evaluation/run_scenarios.py
```

The expected test result is `16 passed`. The scenario command is deterministic and recreates its CSV and figures; generated outputs and runtime logs are ignored by Git.

## Limitations

This is not a safety-certified system. Monocular distance, TTC, the geometric corridor, nominal object sizes, detector confidence, tracking IDs, and train dynamics are assumptions. Performance can degrade with occlusion, lighting, weather, camera mounting, perspective, domain shift, CPU latency, and detector or tracker failures. There is no labelled dataset or ground truth in this repository, so no real-world precision, recall, braking-distance, or safety guarantee is claimed. See [docs/limitations.md](docs/limitations.md).

Generated evidence currently includes:

- `outputs/scenario_critical.csv`
- `outputs/figures/scenario_speed_kmh.png`
- `outputs/figures/scenario_distance_m.png`
- `outputs/figures/scenario_ttc_s.png`
- `outputs/figures/complex_crash_detection.jpg`

The dashboard and CLI append runtime observations to `outputs/logs/`. These outputs capture what was actually observed; they are not ground truth.

## Project layout

```text
full_git_repo/       Preserved sparse upstream baseline checkout
enhanced/            Detector, track region, distance, TTC, risk, logging, CLI
simulation/          Independent virtual train state machine
dashboard/           Streamlit control room
evaluation/          Deterministic scenario evidence generator
configs/             Editable runtime thresholds and assumptions
docs/                Research preparation and limitations
tests/               Core deterministic tests
outputs/             Logs and generated figures (ignored by Git)
```

See `docs/` for methods, assumptions, experiments, limitations, and a paper outline. Do not use this research prototype for operational safety decisions.
