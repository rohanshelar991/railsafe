# Optional input data

The deterministic tests and scenario evaluation do not require input video.
For video inference, place compatible local `.mp4` files in
`full_git_repo/video/` or upload a file through the Streamlit dashboard.

Video files are excluded from Git because the upstream collection is large.
The expected filenames are listed in `configs/video_manifest.csv`; filenames
are only metadata and do not download or fabricate media.
