from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import pandas as pd
import psutil
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from enhanced.config import load_config
from enhanced.logging import SafetyCSVLogger
from enhanced.pipeline import RailSafetyPipeline


st.set_page_config(page_title="RailSafe", page_icon="RS", layout="wide")


def discover_videos(config: dict) -> list[Path]:
    video_dir = ROOT / config["project"]["source_video_dir"]
    return sorted(path for path in video_dir.glob("*.*") if path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"})


def initialize() -> None:
    if "pipeline" not in st.session_state:
        config = load_config(ROOT / "configs/default.yaml")
        st.session_state.config = config
        st.session_state.pipeline = RailSafetyPipeline(config)
        st.session_state.model_loaded = False
        st.session_state.capture = None
        st.session_state.source = None
        st.session_state.frame_id = 0
        st.session_state.logger = SafetyCSVLogger(ROOT / "outputs/logs/session.csv")
        st.session_state.history = []


def open_capture(source: str | int) -> None:
    current = st.session_state.capture
    if current is not None:
        current.release()
    capture = cv2.VideoCapture(source)
    st.session_state.capture = capture if capture.isOpened() else None
    st.session_state.source = source
    st.session_state.frame_id = 0
    st.session_state.autoplay = False


def reset_simulation() -> None:
    pipeline.reset()
    st.session_state.frame_id = 0
    st.session_state.result = None
    st.session_state.history = []
    st.session_state.autoplay = False
    capture = st.session_state.get("capture")
    if capture is not None:
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)


initialize()
st.session_state.setdefault("history", [])
st.session_state.setdefault("autoplay", False)
st.session_state.setdefault("result", None)
pipeline: RailSafetyPipeline = st.session_state.pipeline
config = st.session_state.config

st.title("RailSafe Control Room")
st.caption("Research prototype. Perception, distance, TTC, and braking are software simulation only; this system controls no real train.")

with st.sidebar:
    st.subheader("Input")
    repository_videos = discover_videos(config)
    source_kind = st.radio("Source", ["Repository video", "Upload", "Webcam"], horizontal=True)
    selected_path: str | int | None = None
    if source_kind == "Repository video":
        if repository_videos:
            selected_path = str(st.selectbox("Available repository footage", repository_videos, format_func=lambda value: value.name))
        else:
            st.warning("The upstream video catalog is known but its large MP4 files are not checked out locally yet.")
    elif source_kind == "Upload":
        upload = st.file_uploader("Video", type=["mp4", "avi", "mov", "mkv"])
        if upload is not None:
            upload_dir = ROOT / "data/uploads"
            upload_dir.mkdir(parents=True, exist_ok=True)
            target = upload_dir / upload.name
            target.write_bytes(upload.getbuffer())
            selected_path = str(target)
    else:
        selected_path = 0
    if st.button("Open source", width="stretch", disabled=selected_path is None):
        open_capture(selected_path)
    if st.button("Pause video" if st.session_state.get("autoplay", False) else "Play video", width="stretch", disabled=st.session_state.get("capture") is None):
        st.session_state.autoplay = not st.session_state.get("autoplay", False)

    st.subheader("Simulation")
    control_cols = st.columns(3)
    if control_cols[0].button("Up", help="Accelerate simulated train"):
        pipeline.train.accelerate()
    if control_cols[1].button("Down", help="Service deceleration"):
        pipeline.train.decelerate()
    if control_cols[2].button("Stop", help="Emergency stop"):
        pipeline.train.emergency_stop()
    if st.button("Reset simulation", width="stretch"):
        reset_simulation()
        st.session_state.last_action = "Simulation reset: train STOPPED, speed 0 km/h, video rewound."
    st.caption("Dashboard controls: Up, Down, Stop, and Reset. Keyboard shortcuts are available in the CLI runner.")

    if st.button("Load YOLO11n", width="stretch"):
        st.session_state.model_loaded = pipeline.load_models()
        if st.session_state.model_loaded and st.session_state.capture is not None:
            st.session_state.autoplay = True
    if pipeline.detector.error:
        st.error(pipeline.detector.error)

st.caption(f"Video playback: {'PLAYING' if st.session_state.get('autoplay', False) else 'PAUSED'}")
if st.session_state.get("last_action"):
    st.info(st.session_state.pop("last_action"))


def advance_frame() -> None:
    capture = st.session_state.capture
    if capture is None:
        return
    ok, frame = capture.read()
    if not ok:
        st.session_state.autoplay = False
        return
    result = pipeline.process(frame)
    st.session_state.result = result
    st.session_state.frame_id += 1
    st.session_state.history.append({
        "frame": st.session_state.frame_id,
        "time_s": pipeline.elapsed_s,
        "speed_kmh": pipeline.train.speed_kmh,
        "distance_m": result.detections[0].estimated_distance_m if result.detections else None,
        "ttc_s": result.assessment.ttc_s,
        "stopping_distance_m": result.assessment.stopping_distance_m,
        "safety_margin_m": result.assessment.safety_margin_m,
        "risk": result.assessment.risk.value,
    })
    st.session_state.logger.write(st.session_state.frame_id, result, pipeline.train.speed_kmh, pipeline.train.brake_active)


@st.fragment(run_every=0.15)
def playback() -> None:
    if st.session_state.autoplay:
        advance_frame()
    result = st.session_state.get("result")
    left, right = st.columns([1.7, 1])
    with left:
        st.subheader("Front Camera")
        if result is not None:
            st.image(cv2.cvtColor(pipeline.render(result), cv2.COLOR_BGR2RGB), width="stretch")
            st.caption(f"Frame {st.session_state.frame_id} | Source time {pipeline.elapsed_s:.2f} s")
        else:
            st.info("Open a source, load YOLO11n, and process a frame. Enable Auto-play video to advance continuously.")
    with right:
        st.subheader("Train Status")
        a, b = st.columns(2)
        a.metric("State", pipeline.train.state.value)
        b.metric("Speed", f"{pipeline.train.speed_kmh:.1f} km/h")
        a, b = st.columns(2)
        a.metric("Acceleration", f"{pipeline.train.acceleration_mps2:.1f} m/s2")
        b.metric("Brake", "ACTIVE" if pipeline.train.brake_active else "NORMAL")
        st.subheader("Safety Analysis")
        if result is None:
            st.metric("Risk", "SAFE")
        else:
            st.metric("Risk", result.assessment.risk.value)
            st.metric("Estimated TTC", f"{result.assessment.ttc_s:.2f} s" if result.assessment.ttc_s is not None else "N/A")
            st.caption(result.assessment.reason)
            if result.assessment.risk.value in {"CRITICAL", "HIGH"} or pipeline.train.brake_active:
                st.subheader("Why is the system braking?")
                threat = next((item for item in result.assessments if item.object_id == result.assessment.object_id), result.assessment)
                detection = next((item for item in result.detections if item.track_id == result.assessment.object_id), None)
                st.write({
                    "obstacle_id": result.assessment.object_id,
                    "class": detection.label if detection else None,
                    "confidence": detection.confidence if detection else None,
                    "track_status": detection.track_status.value if detection else None,
                    "lateral_position": detection.lateral_position if detection else None,
                    "estimated_distance_m": detection.estimated_distance_m if detection else None,
                    "train_speed_kmh": pipeline.train.speed_kmh,
                    "ttc_s": threat.ttc_s,
                    "braking_distance_m": threat.braking_distance_m,
                    "stopping_distance_m": threat.stopping_distance_m,
                    "safety_margin_m": threat.safety_margin_m,
                    "trajectory": detection.trajectory_status.value if detection else None,
                    "risk": threat.risk.value,
                    "decision": threat.decision,
                })
            st.metric("Pipeline latency", f"{result.latency_ms:.1f} ms")
            st.metric("Pipeline FPS", f"{1000 / result.latency_ms:.1f}" if result.latency_ms else "N/A")
            st.metric("CPU utilization", f"{psutil.cpu_percent(interval=None):.0f}%")
            st.caption(f"Detector: YOLO11n / ByteTrack | Device: {pipeline.detector.model.device if pipeline.detector.available else 'not loaded'}")
            if result.detections:
                detection = result.detections[0]
                st.subheader("Primary Detection")
                st.write({"class": detection.label, "confidence": f"{detection.confidence:.0%}", "object_id": detection.track_id, "on_track": detection.on_track, "track_overlap": f"{detection.track_overlap:.0%}", "estimated_distance_m": detection.estimated_distance_m})
            elif st.session_state.model_loaded:
                st.info("No eligible obstacle was detected in this frame.")
    st.divider()
    if st.session_state.history:
        st.subheader("Safety trends")
        chart = pd.DataFrame(st.session_state.history).set_index("time_s")
        st.line_chart(chart[["speed_kmh", "distance_m", "ttc_s", "stopping_distance_m", "safety_margin_m"]])
        st.dataframe(chart[["frame", "risk", "speed_kmh", "distance_m", "ttc_s", "safety_margin_m"]].tail(20), width="stretch")
    st.caption("Distance and TTC are estimates based on configured monocular assumptions. Verify against calibrated ground truth before using any quantitative research claim.")


if st.button("Process next frame", type="primary", width="stretch"):
    if st.session_state.capture is None:
        st.warning("Open a valid video or webcam source first.")
    else:
        advance_frame()

playback()
