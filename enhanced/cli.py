from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from enhanced.config import load_config
from enhanced.logging import SafetyCSVLogger
from enhanced.pipeline import RailSafetyPipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="RailSafe keyboard-controlled local video runner")
    parser.add_argument("--video", default="full_git_repo/video/complex_crash.mp4")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    pipeline = RailSafetyPipeline(config)
    if not pipeline.load_models():
        print(pipeline.detector.error)
        return 1
    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        print(f"Cannot open video: {args.video}")
        return 1
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    logger = SafetyCSVLogger(Path(config["project"]["output_dir"]) / "logs/cli_session.csv")
    frame_id = 0
    paused = False
    print("Controls: Up accelerate, Down service brake, Space emergency stop, R reset, Q/Esc quit")
    while True:
        if not paused:
            ok, frame = capture.read()
            if not ok:
                break
            result = pipeline.process(frame, 1 / fps)
            rendered = pipeline.render(result)
            primary = result.assessment
            print(
                f"frame={frame_id} fps={fps:.2f} detections={len(result.detections)} "
                f"risk={primary.risk.value} ttc={primary.ttc_s if primary.ttc_s is not None else 'N/A'} "
                f"speed={pipeline.train.speed_kmh:.2f}km/h "
                f"braking_distance={primary.braking_distance_m if primary.braking_distance_m is not None else 'N/A'}m "
                f"stopping_distance={primary.stopping_distance_m if primary.stopping_distance_m is not None else 'N/A'}m "
                f"margin={primary.safety_margin_m if primary.safety_margin_m is not None else 'N/A'}m "
                f"state={pipeline.train.state.value} braking={pipeline.train.brake_active}",
                flush=True,
            )
            cv2.putText(rendered, "SOFTWARE SIMULATION ONLY", (18, rendered.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 230), 2)
            cv2.imshow("RailSafe", rendered)
            logger.write(frame_id, result, pipeline.train.speed_kmh, pipeline.train.brake_active)
            frame_id += 1
        key = cv2.waitKeyEx(max(1, int(1000 / fps)))
        # macOS/OpenCV commonly reports arrows as extended key codes.
        key8 = key & 0xFF
        if key8 in (27, ord("q")):
            break
        if key in (2490368, 65362, 82) or key8 == 82:
            pipeline.train.accelerate()
        elif key in (2621440, 65364, 84) or key8 == 84:
            pipeline.train.decelerate()
        elif key8 == ord(" "):
            pipeline.train.emergency_stop()
        elif key8 in (ord("r"), ord("R")):
            pipeline.train.reset()
        elif key8 == ord("p"):
            paused = not paused
    capture.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
