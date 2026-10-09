"""Validate bounded RGB PNG uploads without loading an optional image package."""
import base64
import binascii
import struct
import zlib

import numpy as np


def decode_frame_png(encoded: str) -> np.ndarray:
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("Frame must be a base64-encoded PNG") from error
    if len(data) > 1024 * 1024 or not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("Frame must be a PNG of at most one MiB")
    offset, width, height = 8, 0, 0
    compressed = bytearray()
    saw_end = False
    while offset < len(data):
        if offset + 12 > len(data):
            raise ValueError("Truncated PNG chunk")
        length = struct.unpack_from(">I", data, offset)[0]
        kind = data[offset + 4:offset + 8]
        end = offset + 8 + length
        if end + 4 > len(data):
            raise ValueError("Truncated PNG chunk payload")
        payload = data[offset + 8:end]
        crc = struct.unpack_from(">I", data, end)[0]
        if zlib.crc32(kind + payload) & 0xffffffff != crc:
            raise ValueError("PNG chunk checksum does not match")
        if not width and kind != b"IHDR":
            raise ValueError("PNG must start with its image header")
        if kind == b"IHDR":
            if width or length != 13:
                raise ValueError("Invalid PNG image header")
            width, height, depth, color, compression, filtering, interlace = struct.unpack(">IIBBBBB", payload)
            if not 1 <= width <= 640 or not 1 <= height <= 480:
                raise ValueError("Arena frame resolution is limited to 640 by 480")
            if (depth, color, compression, filtering, interlace) != (8, 2, 0, 0, 0):
                raise ValueError("Arena PNGs must use noninterlaced eight-bit RGB")
        elif kind == b"IDAT":
            compressed.extend(payload)
        elif kind == b"IEND":
            if length or end + 4 != len(data):
                raise ValueError("PNG has trailing data or an invalid terminator")
            saw_end = True
        elif kind[0] & 32 == 0:
            raise ValueError("Unsupported critical PNG chunk")
        offset = end + 4
    if not saw_end or not compressed:
        raise ValueError("PNG is missing its compressed pixels or terminator")
    row_bytes = width * 3
    expected = height * (row_bytes + 1)
    try:
        inflater = zlib.decompressobj()
        pixels = inflater.decompress(bytes(compressed), expected + 1)
    except zlib.error as error:
        raise ValueError("PNG pixel compression is invalid") from error
    if len(pixels) != expected or not inflater.eof or inflater.unused_data or inflater.unconsumed_tail:
        raise ValueError("PNG decompressed pixel length does not match its bounded header")
    output = np.empty((height, row_bytes), dtype=np.uint8)
    for row in range(height):
        start = row * (row_bytes + 1)
        method = pixels[start]
        values = bytearray(pixels[start + 1:start + 1 + row_bytes])
        if method > 4:
            raise ValueError("PNG row filter is invalid")
        for column in range(row_bytes):
            left = values[column - 3] if column >= 3 else 0
            up = int(output[row - 1, column]) if row else 0
            upper_left = int(output[row - 1, column - 3]) if row and column >= 3 else 0
            if method == 1:
                predictor = left
            elif method == 2:
                predictor = up
            elif method == 3:
                predictor = (left + up) // 2
            elif method == 4:
                estimate = left + up - upper_left
                distances = (abs(estimate - left), abs(estimate - up), abs(estimate - upper_left))
                predictor = (left, up, upper_left)[distances.index(min(distances))]
            else:
                predictor = 0
            values[column] = (values[column] + predictor) & 255
        output[row] = np.frombuffer(values, dtype=np.uint8)
    return output.reshape(height, width, 3)
