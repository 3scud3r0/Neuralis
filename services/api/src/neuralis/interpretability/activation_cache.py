"""Selective, bounded activation snapshots without retaining autograd graphs."""

from dataclasses import dataclass
from collections.abc import Iterable
import torch
from torch import nn

from ._tensors import output_tensor, sequence_positions


@dataclass(frozen=True)
class ActivationRecord:
    module: str
    tensor: torch.Tensor
    positions: tuple[int, ...] | None
    invocation: int

    @property
    def bytes(self):
        return self.tensor.numel() * self.tensor.element_size()


class ActivationCache:
    """A context owns hooks; closing releases hooks while retaining CPU records.

    Budget failures skip whole tensors; values are never silently downsampled.
    Use clear() to release retained snapshots. Repeated module invocations produce
    separate records. Switching selected modules does not erase existing records.
    """

    def __init__(self, model: nn.Module, modules: Iterable[str], *,
                 max_bytes: int = 8_388_608, max_records: int = 64, positions=None):
        if not 1 <= max_bytes <= 67_108_864 or not 1 <= max_records <= 1024:
            raise ValueError("Activation cache budget must be 1..64 MiB and 1..1024 records")
        self.model = model
        self.max_bytes = max_bytes
        self.max_records = max_records
        self.positions = tuple(positions) if positions is not None else None
        self.records: list[ActivationRecord] = []
        self.dropped = 0
        self.retained_bytes = 0
        self.invocations = 0
        self._handles = []
        self.active = False
        self.modules = self._validate_modules(modules)

    def _validate_modules(self, modules):
        raw = tuple(modules)
        if not raw or len(raw) > 128 or any(not isinstance(n, str) for n in raw):
            raise ValueError("Select 1..128 explicit module names")
        selected = tuple(dict.fromkeys(raw))
        unknown = set(selected).difference(dict(self.model.named_modules()))
        if unknown:
            raise ValueError(f"Unknown activation modules: {sorted(unknown)}")
        return selected

    def _capture(self, name, output):
        self.invocations += 1
        tensor = output_tensor(output)
        positions = sequence_positions(tensor, self.positions)
        elements = tensor.numel()
        if positions is not None:
            elements = elements // tensor.shape[-2] * len(positions)
        size = elements * tensor.element_size()
        if len(self.records) >= self.max_records or self.retained_bytes + size > self.max_bytes:
            self.dropped += 1
            return
        detached = tensor.detach()
        if positions is not None:
            detached = detached.index_select(-2, torch.tensor(positions, device=tensor.device))
        # clone is needed even for CPU tensors: later in-place operations must not
        # alter the experimental baseline retained by this cache.
        copied = detached.to(device="cpu").clone()
        self.records.append(ActivationRecord(name, copied,
                                            tuple(positions) if positions is not None else None,
                                            self.invocations))
        self.retained_bytes += size

    def start(self):
        if self.active:
            raise RuntimeError("Activation cache is already active")
        self.active = True
        try:
            known = dict(self.model.named_modules())
            for name in self.modules:
                self._handles.append(known[name].register_forward_hook(
                    lambda module, inputs, output, n=name: self._capture(n, output)))
        except BaseException:
            self.close()
            raise
        return self

    def set_modules(self, modules):
        selected = self._validate_modules(modules)
        was_active = self.active
        self.close()
        self.modules = selected
        if was_active:
            self.start()

    def latest(self, module):
        for record in reversed(self.records):
            if record.module == module:
                return record.tensor
        raise KeyError(f"No retained activation for {module}; inspect dropped/budget")

    def clear(self):
        self.records.clear()
        self.retained_bytes = 0
        self.dropped = 0
        self.invocations = 0

    def snapshot(self):
        return {"records": [{"module": r.module, "shape": list(r.tensor.shape),
                              "dtype": str(r.tensor.dtype).removeprefix("torch."),
                              "positions": list(r.positions) if r.positions is not None else None,
                              "invocation": r.invocation, "values": r.tensor.tolist()}
                             for r in self.records],
                "retained_bytes": self.retained_bytes, "dropped": self.dropped,
                "complete": self.dropped == 0,
                "budget": {"max_bytes": self.max_bytes, "max_records": self.max_records}}

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.active = False

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, traceback):
        self.close()
