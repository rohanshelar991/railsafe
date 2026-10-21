# RailSafe System Architecture

RailSafe is a software-only research prototype. The flow is video -> YOLO11n detection -> Ultralytics ByteTrack IDs -> configurable geometric track corridor -> track-aware filtering -> temporal confirmation -> trajectory and monocular distance estimates -> TTC and stopping physics -> explainable risk decision -> virtual train state machine.

The geometric corridor is explicitly a perspective heuristic, not learned railway segmentation. Every component is replaceable through the pipeline interfaces.
