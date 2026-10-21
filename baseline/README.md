# Original baseline reference

`../full_git_repo` is the sparse upstream checkout of `tamasino52/Railroad_and_Obstacle_detection` at commit `8317428c5314a7010d02314bf7da158739cea96e`. No upstream code has been edited.

Observed implementation:

- `Railtracking_module.py`: Keras U-Net architecture at 256×256 with BCE + Dice loss.
- `ObstacleDetection_module.py`: TensorFlow 1 frozen-graph loader for Faster R-CNN Inception V2.
- `Run.py`: runs segmentation then detection in an OpenCV loop; danger styling comes from `Evaluation_module.py`.
- `training/labelmap.pbtxt`: `people`, `web`, `tree`, `box` labels.
- `video/`: twelve MP4 demonstration videos, including `complex_crash.mp4`.

The supplied README requires an external trained U-Net, an external TensorFlow Models checkout, a Faster R-CNN model setup, TensorFlow GPU, CUDA 10, cuDNN 7.3, and Windows-oriented dependencies. Consequently, it is documented as a baseline rather than run in this current macOS/Python 3.14 environment.
