import time

import torch
from torch import nn

from neuralis.core.schemas import CaptureConfig


def summarize_tensor(tensor: torch.Tensor, max_elements: int = 1024, bins: int = 32) -> dict:
    detached = tensor.detach().float().cpu()
    flat = detached.reshape(-1)
    finite = flat[torch.isfinite(flat)]
    count = flat.numel()
    stride = max(1, (count + max_elements - 1) // max_elements)
    sample = flat[::stride][:max_elements]
    safe_sample = torch.nan_to_num(sample, nan=0.0, posinf=0.0, neginf=0.0)
    result = {"shape": list(tensor.shape), "dtype": str(tensor.dtype).removeprefix("torch."),
              "elements": count, "captured_elements": len(sample), "sampled": len(sample) < count,
              "sampling": {"method": "uniform-stride", "stride": stride}, "values": safe_sample.tolist(),
              "nonfinite": int((~torch.isfinite(flat)).sum()), "bytes": tensor.numel() * tensor.element_size()}
    if len(finite):
        minimum, maximum = float(finite.min()), float(finite.max())
        low, high = (minimum, maximum) if minimum != maximum else (minimum - 0.5, maximum + 0.5)
        histogram = torch.histc(finite, bins=bins, min=low, max=high)
        result.update(mean=float(finite.mean()), std=float(finite.std(unbiased=False)), minimum=minimum,
                      maximum=maximum, norm=float(torch.linalg.vector_norm(finite)),
                      sparsity=float((finite.abs() < 1e-8).float().mean()),
                      histogram={"counts": histogram.int().tolist(), "minimum": low, "maximum": high},
                      quantiles=torch.quantile(finite, torch.tensor([0.01, 0.25, 0.5, 0.75, 0.99])).tolist())
    else:
        result.update(mean=None, std=None, minimum=None, maximum=None, norm=None, sparsity=None,
                      histogram={"counts": [0] * bins, "minimum": 0, "maximum": 1}, quantiles=[])
    return result


class CaptureSession:
    """Hooks own no autograd graphs and are removed even on failed execution."""

    def __init__(self, model: nn.Module, config: CaptureConfig):
        self.model = model
        self.config = config
        self.handles = []
        self.records: list[dict] = []
        self.enabled = False
        self.step = 0
        self.dropped = 0
        self.elapsed_ms = 0.0
        self._attach()

    def _record(self, module: str, kind: str, tensor: torch.Tensor):
        if not self.enabled:
            return
        if len(self.records) >= self.config.max_tensors:
            self.dropped += 1
            return
        start = time.perf_counter()
        summary = summarize_tensor(tensor, self.config.max_elements, self.config.histogram_bins)
        self.elapsed_ms += (time.perf_counter() - start) * 1000
        self.records.append({"module": module, "kind": kind, "step": self.step, **summary})

    def _attach(self):
        selected = set(self.config.modules)
        known = dict(self.model.named_modules())
        unknown = selected.difference(known)
        if unknown:
            raise ValueError(f"Unknown capture modules: {sorted(unknown)}")
        for name, module in self.model.named_modules():
            if not name or (selected and name not in selected) or (not selected and any(module.children())):
                continue
            def forward_hook(mod, inputs, output, module_name=name):
                if not self.enabled or not isinstance(output, torch.Tensor):
                    return
                if self.config.activations:
                    self._record(module_name, "activation", output)
                if self.config.gradients and output.requires_grad:
                    output.register_hook(lambda gradient, n=module_name: self._record(n, "activation-gradient", gradient))
            self.handles.append(module.register_forward_hook(forward_hook))

    def begin(self, step: int):
        self.step = step
        self.enabled = step == 1 or step % self.config.every == 0
        self.records = []
        self.dropped = 0
        self.elapsed_ms = 0

    def finish(self):
        if self.enabled:
            for name, parameter in self.model.named_parameters():
                module = name.rsplit(".", 1)[0]
                if self.config.modules and module not in self.config.modules:
                    continue
                if self.config.weights:
                    self._record(name, "weight", parameter)
                if self.config.gradients and parameter.grad is not None:
                    self._record(name, "parameter-gradient", parameter.grad)
        self.enabled = False
        return {"tensors": self.records, "dropped": self.dropped, "capture_ms": self.elapsed_ms,
                "complete": self.dropped == 0, "budget": self.config.model_dump()}

    def close(self):
        self.enabled = False
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.records.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
