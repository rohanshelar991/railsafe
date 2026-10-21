# Limitations

Known limitations include monocular distance uncertainty, assumed object
dimensions, camera-perspective dependence, geometric corridor error,
unavailable labels for the baseline footage, CPU latency, occlusion, lighting
and weather sensitivity, detector failures, ByteTrack ID switches, and
simplified virtual-train parameters. The repository therefore does not report
precision, recall, mAP, MOT accuracy, distance error, false-alarm rate, or
warning lead time. Those require labelled data, calibration, and a defined
evaluation protocol.

This is not a certified railway safety system and does not control a real
train. The virtual braking state machine is a deterministic demonstration of
software behavior, not evidence that a physical train could stop safely.
