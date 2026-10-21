"""Ground-truth-aware evaluation entry point.

This intentionally refuses to invent metrics. It reports unavailable metrics
until annotations and a matching prediction export are supplied.
"""

from __future__ import annotations

import csv
from pathlib import Path


METRICS = [
    "Precision", "Recall", "F1", "mAP50", "mAP50-95", "ID consistency",
    "ID switches", "track persistence", "false alarm rate", "missed obstacle rate",
    "warning lead time", "confirmation delay", "emergency braking accuracy",
    "MAE", "RMSE", "MAPE", "FPS", "inference latency", "CPU usage", "memory usage",
]


def main() -> None:
    output = Path("outputs/reports")
    output.mkdir(parents=True, exist_ok=True)
    path = output / "evaluation_status.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value", "status"])
        writer.writeheader()
        for metric in METRICS:
            writer.writerow({"metric": metric, "value": "", "status": "Not evaluated - ground truth unavailable."})
    print(f"Wrote {path}; no metrics were calculated without ground truth.")


if __name__ == "__main__":
    main()
