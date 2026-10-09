"""Encode captured RGB pixels as PNG using the standard library."""

import struct
import zlib

import numpy as np


def encode_frame_png(frame: np.ndarray) -> bytes:
    image = np.asarray(frame)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("PNG encoder requires an RGB uint8 image")
    height, width, _ = image.shape
    if not 1 <= height <= 1080 or not 1 <= width <= 1920:
        raise ValueError("PNG image exceeds the capture budget")

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xffffffff)

    raw = b"".join(b"\x00" + row.tobytes() for row in image)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + chunk(
        b"IDAT", zlib.compress(raw, level=6)) + chunk(b"IEND", b"")
