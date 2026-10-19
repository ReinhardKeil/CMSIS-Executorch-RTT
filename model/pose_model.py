# Copyright 2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tiny heat-map pose estimator trained on the synthetic SDS test poses."""

from pathlib import Path

import torch
from torch import nn

INPUT_SHAPE = (1, 64, 64, 3)
JOINT_COUNT = 17
WEIGHTS = Path(__file__).with_name("pose_weights.pt")


class TinyPoseNet(nn.Module):
    def __init__(self, input_shape: tuple[int, ...] = INPUT_SHAPE) -> None:
        super().__init__()
        channels = input_shape[-1]
        self.net = nn.Sequential(
            nn.Conv2d(channels, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, JOINT_COUNT, 1),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image.permute(0, 3, 1, 2))


def get_model(input_shape: tuple[int, ...] = INPUT_SHAPE) -> nn.Module:
    if not WEIGHTS.exists():
        raise FileNotFoundError(f"{WEIGHTS} is missing; run tools/train_synthetic_pose.py")
    model = TinyPoseNet(input_shape)
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
    return model.eval()


def get_calibration_inputs(
    input_shape: tuple[int, ...] = INPUT_SHAPE, calibration_samples: int = 8
) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(20261018)
    return [torch.rand(input_shape, generator=generator) for _ in range(calibration_samples)]
