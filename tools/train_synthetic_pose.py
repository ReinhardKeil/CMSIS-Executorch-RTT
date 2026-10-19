#!/usr/bin/env python3
# Copyright 2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Train TinyPoseNet and create reproducible RGB test frames for SDS playback."""

from pathlib import Path
import sys

import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
from pose_model import JOINT_COUNT, TinyPoseNet, WEIGHTS  # noqa: E402

SIZE = 64
EDGES = ((0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9),
         (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13),
         (13, 15), (12, 14), (14, 16))


def pose(seed: int):
    g = torch.Generator().manual_seed(seed)
    cx = int(torch.randint(25, 40, (1,), generator=g))
    top = int(torch.randint(7, 13, (1,), generator=g))
    lean = int(torch.randint(-5, 6, (1,), generator=g))
    pts = [(cx, top), (cx-3, top-1), (cx+3, top-1), (cx-6, top), (cx+6, top),
           (cx-8, top+13), (cx+8, top+13), (cx-13+lean, top+25),
           (cx+13+lean, top+25), (cx-18+lean, top+35), (cx+18+lean, top+35),
           (cx-6+lean, top+31), (cx+6+lean, top+31), (cx-9, top+44),
           (cx+9, top+44), (cx-12, top+54), (cx+12, top+54)]
    return [(max(2, min(61, x)), max(2, min(61, y))) for x, y in pts]


def render(points):
    image = Image.new("RGB", (SIZE, SIZE), (24, 28, 36))
    draw = ImageDraw.Draw(image)
    for a, b in EDGES:
        draw.line((points[a], points[b]), fill=(185, 190, 200), width=2)
    for i, (x, y) in enumerate(points):
        color = ((53*i+80) % 220 + 35, (97*i+30) % 220 + 35, (151*i+10) % 220 + 35)
        draw.ellipse((x-2, y-2, x+2, y+2), fill=color)
    return image


def tensors(seed: int):
    pts = pose(seed)
    image = torch.tensor(list(render(pts).getdata()), dtype=torch.float32).reshape(SIZE, SIZE, 3) / 255.0
    heat = torch.zeros(JOINT_COUNT, SIZE // 4, SIZE // 4)
    for joint, (x, y) in enumerate(pts):
        heat[joint, y // 4, x // 4] = 1.0
    return image, heat


def main():
    torch.manual_seed(7)
    model = TinyPoseNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(30.0))
    for step in range(600):
        batch = [tensors(step * 16 + i) for i in range(16)]
        images = torch.stack([x[0] for x in batch])
        labels = torch.stack([x[1] for x in batch])
        loss = loss_fn(model(images), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print(f"step {step}: loss={loss.item():.4f}")
    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), WEIGHTS)

    out = ROOT / "test_images" / "pose"
    out.mkdir(parents=True, exist_ok=True)
    for index, seed in enumerate((10001, 10002, 10003), 1):
        render(pose(seed)).resize((512, 512), Image.Resampling.NEAREST).save(out / f"pose_{index}.png")
        (out / f"pose_{index}_64x64.rgb").write_bytes(render(pose(seed)).tobytes())
    print(f"wrote {WEIGHTS} and test frames in {out}")


if __name__ == "__main__":
    main()
