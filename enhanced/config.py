from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    """Load the editable runtime configuration."""
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)
