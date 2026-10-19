# SDS pose test frames

Run `tools/train_synthetic_pose.py` to reproduce the three PNG previews, their
64 x 64 RGB888 payloads, and `model/pose_weights.pt`. Then run
`tools/create_pose_sds.py` to create `Pose_In.0.sds`. In the CMSIS SDS view,
open `pose.sdsio.yml` and play the `Synthetic pose frames` step. The control
file connects to RTT channel 1 and uses this directory for recordings and
metadata. `Pose_In.sds.yml` describes the RGB888 frame contents.

The colored synthetic skeletons make this sample deterministic. They validate
the complete SDS -> RTT -> ExecuTorch -> Ethos-U55 -> keypoint pipeline; they
are not a replacement for training on a photographic human-pose dataset.
