"""Selective inference weight offload and differentiable activation offload."""

from contextlib import contextmanager
from dataclasses import dataclass
import threading

import torch
from torch import nn

from neuralis.runtime.acceleration.backends import resolve_backend


def module_bytes(module: nn.Module) -> int:
    return sum(tensor.numel() * tensor.element_size()
               for tensor in list(module.parameters()) + list(module.buffers()))


@dataclass
class OffloadStats:
    selected_bytes: int = 0
    loads: int = 0
    evictions: int = 0


class SelectiveCPUOffload:
    """Load selected inference modules for execution, then evict their weights.

    Requires torch.no_grad()/inference_mode at call time. Moving parameters after
    forward would invalidate autograd saved tensors, so training uses the separate
    activation_offload context. Keep this object alive for the model's lifetime,
    or use it as a context manager. Calls to the same model must be serialized.
    """

    def __init__(self, model: nn.Module, module_names: list[str], compute_device: str = "cpu"):
        if not module_names or len(module_names) > 128 or len(set(module_names)) != len(module_names):
            raise ValueError("Select between 1 and 128 unique modules to offload")
        if any(not name or any(other.startswith(name + ".") for other in module_names if other != name)
               for name in module_names):
            raise ValueError("Offloaded modules must be nonoverlapping named children")
        self.model = model
        self.device = resolve_backend(compute_device).device
        self.modules = [(name, model.get_submodule(name)) for name in module_names]
        ownership: dict[int, set[str]] = {}
        for name, tensor in list(model.named_parameters(remove_duplicate=False)) + list(model.named_buffers(remove_duplicate=False)):
            ownership.setdefault(id(tensor), set()).add(name)
        self.original_devices = {}
        for name, module in self.modules:
            devices = {tensor.device for tensor in list(module.parameters()) + list(module.buffers())}
            if len(devices) > 1:
                raise ValueError(f"Module {name} spans devices; selective offload requires one device per module")
            self.original_devices[name] = next(iter(devices), torch.device("cpu"))
            for tensor in list(module.parameters()) + list(module.buffers()):
                if any(not path.startswith(name + ".") for path in ownership[id(tensor)]):
                    raise ValueError(f"Module {name} shares tensors outside its subtree")
        self.stats = OffloadStats(sum(module_bytes(module) for _, module in self.modules))
        self._lock = threading.RLock()
        self._local = threading.local()
        self._handles = []
        self._closed = False
        try:
            for _, module in self.modules:
                module.to("cpu")
                self._handles.append(module.register_forward_pre_hook(self._load))
                self._handles.append(module.register_forward_hook(self._evict, always_call=True))
        except BaseException:
            self.close()
            raise

    def _load(self, module, inputs):
        if torch.is_grad_enabled():
            raise RuntimeError("Selective parameter offload requires inference_mode/no_grad; use activation_offload for training")
        self._lock.acquire()
        try:
            module.to(self.device)
            self.stats.loads += 1
            active = getattr(self._local, "active", {})
            active[id(module)] = active.get(id(module), 0) + 1
            self._local.active = active
        except BaseException:
            self._lock.release()
            raise

    def _evict(self, module, inputs, output):
        # always_call hooks run even when the pre-hook rejects grad-enabled use.
        active = getattr(self._local, "active", {})
        if not active.get(id(module), 0):
            return
        try:
            module.to("cpu")
            self.stats.evictions += 1
        finally:
            active[id(module)] -= 1
            self._lock.release()

    def close(self):
        if self._closed:
            return
        with self._lock:
            for handle in self._handles:
                handle.remove()
            self._handles.clear()
            for name, module in self.modules:
                module.to(self.original_devices[name])
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def selective_cpu_offload(model: nn.Module, module_names: list[str], compute_device: str = "cpu") -> SelectiveCPUOffload:
    return SelectiveCPUOffload(model, module_names, compute_device)


@contextmanager
def activation_offload(*, min_bytes: int = 4096, max_bytes: int = 256 * 1024 * 1024):
    """Move saved accelerator tensors to CPU and restore for backward on demand.

    Tensors below min_bytes and CPU tensors stay in place. A live saved-tensor
    per-context byte budget limits host offload; overflow leaves tensors on their original
    device rather than dropping gradients. Statistics describe actual transfers.
    """
    if min_bytes < 0 or max_bytes < 0:
        raise ValueError("Activation offload budgets must be nonnegative")
    stats = {"offloaded_bytes": 0, "offloaded_tensors": 0, "restores": 0, "budget_bypasses": 0}

    def pack(tensor):
        size = tensor.numel() * tensor.element_size()
        if tensor.device.type == "cpu" or size < min_bytes:
            return tensor.detach()
        if stats["offloaded_bytes"] + size > max_bytes:
            stats["budget_bypasses"] += 1
            return tensor.detach()
        stats["offloaded_bytes"] += size
        stats["offloaded_tensors"] += 1
        return (tensor.device, tensor.detach().to("cpu"))

    def unpack(saved):
        if isinstance(saved, tuple):
            stats["restores"] += 1
            return saved[1].to(saved[0])
        return saved

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        yield stats
