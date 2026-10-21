# Results status

## Reproducible baseline

On the submission environment (Python 3.14, installed dependencies, no
video input), the deterministic test suite completed with **16 passed**.
`evaluation/run_scenarios.py` recreated the critical-risk braking CSV and
figures. These checks validate software behavior and repeatability; they are
not measurements of railway perception accuracy.

Measured claims currently available:

- Source video opens successfully.
- YOLO11n weight loads successfully.
- YOLO inference runs on source frames.
- ByteTrack IDs occur in contiguous detected tracks.
- The geometric track-region module produces a mask on each frame.
- CSV logging and deterministic virtual braking scenario output run successfully.

Not measured: mAP, precision, recall, F1, segmentation IoU/Dice, MOT metrics, distance error, safety false-alarm rate, warning lead time, or baseline-versus-proposed comparison. Required ground-truth labels and a defined experimental protocol are absent.
