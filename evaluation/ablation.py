"""Deterministic software ablation harness.

Stages measure the simulator output only. They are not detector or real-video
benchmarks and are labelled accordingly in the generated report.
"""

from __future__ import annotations

import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from enhanced.config import load_config
from enhanced.risk import RiskEngine
from simulation.train import TrainSimulator


STAGES = ["A_YOLO_ONLY", "B_YOLO_BYTETRACK", "C_TRACK_FILTERING", "D_TEMPORAL_CONFIRMATION", "E_DISTANCE_TTC", "F_BRAKING_PHYSICS", "G_FULL_RAILSAFE"]


def main() -> None:
    config = load_config()
    output = Path("outputs/reports")
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for stage in STAGES:
        train = TrainSimulator(config["train"])
        train.speed_kmh = 60.0
        risk = RiskEngine(config["risk"])
        assessment = risk.assess(stage in {"C_TRACK_FILTERING", "D_TEMPORAL_CONFIRMATION", "E_DISTANCE_TTC", "F_BRAKING_PHYSICS", "G_FULL_RAILSAFE"}, 25.0, train.speed_kmh)
        if stage in {"F_BRAKING_PHYSICS", "G_FULL_RAILSAFE"}:
            train.update(assessment.risk, 1.0)
        rows.append({"stage": stage, "experiment_type": "SIMULATED", "risk": assessment.risk.value, "speed_after_1s_kmh": train.speed_kmh, "braking": train.brake_active})
    with (output / "ablation_simulated.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote simulated ablation evidence to {output / 'ablation_simulated.csv'}")


if __name__ == "__main__":
    main()
