"""One-command launcher for the local RailSafe Streamlit dashboard."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch RailSafe")
    parser.add_argument("--cli", action="store_true", help="Use the OpenCV keyboard-controlled video runner")
    parser.add_argument("--video", default="full_git_repo/video/complex_crash.mp4")
    args = parser.parse_args()
    if args.cli:
        return subprocess.call([sys.executable, "-m", "enhanced.cli", "--video", args.video])
    streamlit = shutil.which("streamlit")
    if streamlit is None:
        print("Streamlit is missing. Install dependencies with: python -m pip install -r requirements.txt")
        return 1
    return subprocess.call([streamlit, "run", str(ROOT / "dashboard/app.py")])


if __name__ == "__main__":
    raise SystemExit(main())
