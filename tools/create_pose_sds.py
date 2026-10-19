#!/usr/bin/env python3
"""Pack generated 64x64 RGB frames into an SDS Pose_In recording."""

from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[1]
FRAMES = ROOT / "test_images" / "pose"
OUTPUT = FRAMES / "Pose_In.0.sds"

with OUTPUT.open("wb") as stream:
    for index, path in enumerate(sorted(FRAMES.glob("pose_*_64x64.rgb"))):
        payload = path.read_bytes()
        if len(payload) != 64 * 64 * 3:
            raise ValueError(f"{path}: expected 12288 bytes, got {len(payload)}")
        stream.write(struct.pack("<II", index * 500_000, len(payload)))
        stream.write(payload)

print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")
