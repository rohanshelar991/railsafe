# Reproducibility record

| ID | Input | Model | Purpose | Result status |
| --- | --- | --- | --- | --- |
| E-001 | `complex_crash.mp4` | YOLO11n + ByteTrack | Pipeline smoke test | Source, inference, and IDs verified; not a metric experiment |
| E-002 | Deterministic risk inputs | No detector | Virtual braking state machine | CSV and charts generated; simulation only |

Configuration: `configs/default.yaml`. Hardware/software details and non-results are recorded in `docs/experiments.md` and `docs/results.md`.
