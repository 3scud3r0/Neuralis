"""Dispatch to the installed PyTorch runtime, without emulating accelerators."""

from contextlib import nullcontext
from dataclasses import dataclass
import importlib.util

import torch


def _installed(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ValueError, ImportError):
        return False


def backend_capabilities() -> dict:
    hip = bool(torch.version.hip)
    cuda_api = torch.cuda.is_available()
    mps = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    result = {
        "cpu": {"available": True, "devices": [{"index": 0, "name": "CPU"}],
                "precision": ["float32", "bfloat16"], "validation": "CPU numerical tests"},
        "cuda": {"available": cuda_api and not hip, "devices": [],
                 "precision": ["float32", "float16"], "validation": "hardware-dependent; not validated here"},
        "rocm": {"available": cuda_api and hip, "devices": [],
                 "precision": ["float32", "float16"], "validation": "hardware-dependent; not validated here"},
        "mps": {"available": mps, "devices": [{"index": 0, "name": "Apple Metal"}] if mps else [],
                "precision": ["float32", "float16"], "validation": "hardware-dependent; not validated here"},
    }
    if cuda_api:
        name = "rocm" if hip else "cuda"
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            result[name]["devices"].append({"index": index, "name": properties.name,
                                           "total_memory_bytes": properties.total_memory})
        if torch.cuda.is_bf16_supported():
            result[name]["precision"].append("bfloat16")
    return {"backends": result, "torch_version": str(torch.__version__),
            "cuda_runtime": torch.version.cuda, "hip_runtime": torch.version.hip,
            "optional": {name: _installed(name) for name in ("bitsandbytes", "transformers", "peft", "zeroconf")}}


@dataclass(frozen=True)
class BackendSelection:
    name: str
    device: torch.device
    precision: str = "float32"

    def autocast(self):
        if self.precision == "float32":
            return nullcontext()
        dtype = torch.bfloat16 if self.precision == "bfloat16" else torch.float16
        return torch.autocast(self.device.type, dtype=dtype)

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            torch.mps.synchronize()

    def memory(self) -> dict:
        if self.device.type == "cuda":
            return {"allocated_bytes": torch.cuda.memory_allocated(self.device),
                    "reserved_bytes": torch.cuda.memory_reserved(self.device),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device)}
        if self.device.type == "mps":
            return {"allocated_bytes": torch.mps.current_allocated_memory(),
                    "driver_allocated_bytes": torch.mps.driver_allocated_memory()}
        return {"allocated_bytes": None, "reason": "PyTorch does not expose CPU allocator accounting"}


def resolve_backend(requested: str = "auto", precision: str = "float32") -> BackendSelection:
    capabilities = backend_capabilities()["backends"]
    if requested == "auto":
        requested = next(name for name in ("cuda", "rocm", "mps", "cpu") if capabilities[name]["available"])
    parts = requested.split(":")
    name = parts[0]
    if len(parts) > 2 or name not in capabilities:
        raise ValueError("Backend must be auto, cpu, cuda[:index], rocm[:index], or mps")
    if not capabilities[name]["available"]:
        raise ValueError(f"{name} is unavailable in this PyTorch installation")
    try:
        index = int(parts[1]) if len(parts) == 2 else 0
    except ValueError as error:
        raise ValueError("Accelerator index must be an integer") from error
    if index < 0 or index >= len(capabilities[name]["devices"]):
        raise ValueError("Backend device index is out of range")
    if name in {"cpu", "mps"} and len(parts) == 2:
        raise ValueError(f"{name} does not accept an indexed device")
    if precision not in capabilities[name]["precision"]:
        raise ValueError(f"{precision} is unsupported for {name}")
    # ROCm uses PyTorch's CUDA API and device namespace.
    device = torch.device(f"cuda:{index}" if name in {"cuda", "rocm"} else name)
    return BackendSelection(name, device, precision)
