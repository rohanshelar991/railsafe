from __future__ import annotations

import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from enhanced.config import load_config
from enhanced.risk import RiskEngine
from simulation import TrainSimulator


def main() -> None:
    root = ROOT
    config = load_config(root / "configs/default.yaml")
    train = TrainSimulator(config["train"])
    risk = RiskEngine(config["risk"])
    train.speed_kmh = 60.0
    rows = []
    for second in range(20):
        distance = max(0.5, 85 - second * 5)
        assessment = risk.assess(True, distance, train.speed_kmh)
        train.update(assessment.risk, 1.0)
        rows.append({"time_s": second, "distance_m": distance, "speed_kmh": train.speed_kmh,
                     "ttc_s": assessment.ttc_s, "risk": assessment.risk.value,
                     "braking": train.brake_active})
    output = root / "outputs"
    output.mkdir(exist_ok=True)
    with (output / "scenario_critical.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    for key, label in [("speed_kmh", "Speed (km/h)"), ("distance_m", "Estimated distance (m)"), ("ttc_s", "Estimated TTC (s)")]:
        plt.figure(figsize=(7, 3.5))
        plt.plot([row["time_s"] for row in rows], [row[key] for row in rows], marker="o")
        plt.xlabel("Simulation time (s)")
        plt.ylabel(label)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures / f"scenario_{key}.png", dpi=180)
        plt.close()
    print(f"Wrote deterministic simulation evidence to {output}")
    braking_rows = [row for row in rows if row["braking"]]
    first_braking = braking_rows[0]["time_s"] if braking_rows else "never"
    print(f"Simulation: speed starts at {rows[0]['speed_kmh']:.1f} km/h")
    print(f"Automatic emergency braking starts at t={first_braking}s")
    print(f"Final result: speed={rows[-1]['speed_kmh']:.1f} km/h, braking={rows[-1]['braking']}")


if __name__ == "__main__":
    main()
