"""Invoke a supplied llama.cpp converter, never pretend a toy model is GGUF."""

import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from threading import Thread


def inspect_gguf(path: str | Path) -> dict:
    source = Path(path).resolve()
    if source.suffix.lower() != ".gguf" or not source.is_file():
        raise ValueError("Supply an existing .gguf model file")
    size = source.stat().st_size
    if size < 24:
        raise ValueError("GGUF file is too short to contain a valid header")
    with source.open("rb") as handle:
        def unpack(format):
            length = struct.calcsize(format)
            data = handle.read(length)
            if len(data) != length:
                raise ValueError("GGUF metadata or tensor directory is truncated")
            return struct.unpack(format, data)[0]

        def string():
            length = unpack("<Q")
            if length > 1024 * 1024 or handle.tell() + length > 64 * 1024 * 1024:
                raise ValueError("GGUF string exceeds the metadata validation budget")
            value = handle.read(length)
            if len(value) != length:
                raise ValueError("GGUF string is truncated")
            try:
                return value.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError("GGUF metadata strings must be UTF-8") from error

        scalar = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
                  10: "<Q", 11: "<q", 12: "<d"}

        def value(kind, *, nested=False):
            if kind in scalar:
                return unpack(scalar[kind])
            if kind == 8:
                return string()
            if kind == 9 and not nested:
                subtype, count = unpack("<I"), unpack("<Q")
                if count > 100000:
                    raise ValueError("GGUF metadata array exceeds the validation budget")
                for _ in range(count):
                    value(subtype, nested=True)
                return None
            raise ValueError("GGUF metadata contains an unsupported value type")

        magic = handle.read(4)
        version, tensors, metadata = unpack("<I"), unpack("<Q"), unpack("<Q")
        if magic != b"GGUF" or version not in {2, 3} or not 1 <= tensors <= 100000 or metadata > 100000:
            raise ValueError("File does not contain a supported populated GGUF header")
        alignment, architecture = 32, None
        keys = set()
        for _ in range(metadata):
            key = string()
            if key in keys:
                raise ValueError("GGUF metadata contains a duplicate key")
            keys.add(key)
            item = value(unpack("<I"))
            if key == "general.alignment":
                alignment = item
            if key == "general.architecture":
                architecture = item
            if handle.tell() > 64 * 1024 * 1024:
                raise ValueError("GGUF metadata exceeds the 64 MiB validation budget")
        if not isinstance(architecture, str) or not architecture:
            raise ValueError("GGUF must declare its architecture")
        if not isinstance(alignment, int) or not 1 <= alignment <= 4096 or alignment & (alignment - 1):
            raise ValueError("GGUF alignment is invalid")
        # Supported GGML types: (elements per block, bytes per block).
        layouts = {0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22),
                   7: (32, 24), 8: (32, 34), 9: (32, 40), 10: (256, 84), 11: (256, 110),
                   12: (256, 144), 13: (256, 176), 14: (256, 210), 15: (256, 292),
                   16: (256, 66), 17: (256, 74), 18: (256, 98), 19: (256, 50),
                   20: (32, 18), 21: (256, 110), 22: (256, 82), 23: (256, 136),
                   24: (1, 1), 25: (1, 2), 26: (1, 4), 27: (1, 8), 28: (1, 8),
                   29: (256, 56), 30: (1, 2), 34: (256, 54), 35: (256, 66)}
        ranges, names = [], set()
        for _ in range(tensors):
            name = string()
            dimensions = unpack("<I")
            if not name or name in names or not 1 <= dimensions <= 4:
                raise ValueError("GGUF tensor name or dimensions are invalid")
            names.add(name)
            shape = [unpack("<Q") for _ in range(dimensions)]
            kind, offset = unpack("<I"), unpack("<Q")
            if kind not in layouts:
                raise ValueError(f"GGUF tensor type {kind} requires a newer validator/runtime")
            block, block_bytes = layouts[kind]
            elements = math.prod(shape)
            if not all(shape) or elements > 2**48 or shape[0] % block or offset % alignment:
                raise ValueError("GGUF tensor shape, quantization block, or offset is invalid")
            ranges.append((offset, offset + elements // block * block_bytes))
        data_start = (handle.tell() + alignment - 1) // alignment * alignment
        previous = 0
        for start, end in sorted(ranges):
            if start < previous or data_start + end > size:
                raise ValueError("GGUF tensor data overlaps or is truncated")
            previous = end
    return {"path": str(source), "bytes": size, "version": version,
            "tensor_count": tensors, "metadata_count": metadata,
            "architecture": architecture, "alignment": alignment,
            "validation": "metadata, tensor directory and data bounds; target runtime must verify model compatibility"}


class GGUFConverter:
    """The caller deliberately supplies a trusted local converter script.

    Conversion supports HF architectures implemented by that llama.cpp version.
    Neuralis' toy classification networks are not convertible with this adapter.
    No model or converter download is performed and no shell is invoked.
    """

    def __init__(self, converter_path: str | Path, *, python_executable: str | Path = sys.executable,
                 timeout_seconds: float = 300):
        self.converter = Path(converter_path).resolve()
        self.python = Path(python_executable).resolve()
        if not self.converter.is_file() or self.converter.suffix != ".py":
            raise ValueError("Supply an existing trusted llama.cpp convert_hf_to_gguf.py script")
        if not self.python.is_file():
            raise ValueError("Python executable must exist")
        if not 1 <= timeout_seconds <= 3600:
            raise ValueError("Conversion timeout must fit one to 3600 seconds")
        self.timeout_seconds = timeout_seconds

    def convert(self, model_directory: str | Path, output_path: str | Path, *, outtype: str = "f16") -> dict:
        source, destination = Path(model_directory).resolve(), Path(output_path).resolve()
        if not source.is_dir() or not (source / "config.json").is_file():
            raise ValueError("Source must be a local Hugging Face model directory with config.json")
        config_path = source / "config.json"
        if config_path.stat().st_size > 1024 * 1024:
            raise ValueError("Model configuration exceeds one MiB")
        config = json.loads(config_path.read_text())
        if not isinstance(config, dict) or not config.get("architectures"):
            raise ValueError("HF config must declare an architecture supported by the supplied converter")
        if not any(source.glob("*.safetensors")) and not any(source.glob("pytorch_model*.bin")):
            raise ValueError("Source contains no local Hugging Face weight files")
        if destination.suffix.lower() != ".gguf" or destination.exists():
            raise ValueError("Output must be a new .gguf path; existing files are preserved")
        if outtype not in {"f32", "f16", "bf16", "q8_0"}:
            raise ValueError("Unsupported conversion output type")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="neuralis-gguf-", dir=destination.parent) as temporary:
            staged = Path(temporary) / "converted.gguf"
            args = [str(self.python), str(self.converter), str(source), "--outfile", str(staged), "--outtype", outtype]
            # Drain output continuously while retaining only a four-KiB tail.
            # Avoid communicate()'s unbounded memory and persistent log files.
            tail = bytearray()
            process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=False)

            def drain():
                while True:
                    chunk = process.stdout.read(4096)
                    if not chunk:
                        break
                    tail.extend(chunk)
                    del tail[:-4096]

            reader = Thread(target=drain, name="neuralis-converter-output", daemon=True)
            reader.start()
            try:
                returncode = process.wait(timeout=self.timeout_seconds)
            except subprocess.TimeoutExpired as error:
                process.kill()
                process.wait()
                reader.join(timeout=2)
                raise RuntimeError("GGUF conversion exceeded its configured timeout") from error
            finally:
                reader.join(timeout=2)
                if not reader.is_alive():
                    process.stdout.close()
            if returncode:
                raise RuntimeError(f"llama.cpp converter exited with status {returncode}: {tail.decode(errors='replace')}")
            if staged.is_symlink():
                raise ValueError("Converter output must be a regular staged GGUF file")
            metadata = inspect_gguf(staged)
            # Stage beside the destination; a hard link atomically publishes the
            # complete file and fails if a destination appeared concurrently.
            os.link(staged, destination)
        metadata.update(path=str(destination), outtype=outtype, source_architectures=config["architectures"],
                        converter=str(self.converter))
        return metadata


def write_ollama_modelfile(gguf_path: str | Path, output_path: str | Path | None = None,
                          *, temperature: float = 0.7, context_length: int = 2048) -> dict:
    metadata = inspect_gguf(gguf_path)
    if not 0 <= temperature <= 2 or not 128 <= context_length <= 131072:
        raise ValueError("Ollama temperature or context length is outside supported bounds")
    path = metadata["path"]
    if any(x in path for x in ['\n', '\r', '"']):
        raise ValueError("Model path contains characters unsupported by the Modelfile reference")
    content = f'FROM "{path}"\nPARAMETER temperature {temperature}\nPARAMETER num_ctx {context_length}\n'
    if output_path is not None:
        destination = Path(output_path)
        with destination.open("x", encoding="utf-8") as output:
            output.write(content)
    return {"modelfile": content, "model": metadata,
            "runtime_status": "Not imported into Ollama; run ollama create with this Modelfile to validate compatibility"}
