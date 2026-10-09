"""Clean/corrupt activation replacement with measured scalar effects."""

import math
import torch

from ..activation_cache import ActivationCache
from .._tensors import (evaluating, module_by_name, output_tensor,
                       replace_output, sequence_positions)


class ActivationPatcher:
    def __init__(self, model, module: str, clean_activation: torch.Tensor, *,
                 positions=None, source_positions=None):
        self.module = module_by_name(model, module)
        if clean_activation.ndim < 2:
            raise ValueError("Clean activation requires batch and channel axes")
        if clean_activation.numel() * clean_activation.element_size() > 67_108_864:
            raise ValueError("Clean activation exceeds the 64 MiB patch budget")
        if not torch.isfinite(clean_activation).all():
            raise ValueError("Clean activation must be finite")
        self.clean_activation = clean_activation.detach().cpu().clone()
        self.positions = tuple(positions) if positions is not None else None
        self.source_positions = tuple(source_positions) if source_positions is not None else self.positions
        if positions is None and source_positions is not None:
            raise ValueError("Source positions require corresponding target positions")
        self._handle = None
        self.applications = 0

    def _hook(self, module, inputs, output):
        tensor = output_tensor(output)
        source = self.clean_activation.to(device=tensor.device, dtype=tensor.dtype)
        if not torch.isfinite(source).all():
            raise ValueError("Clean activation overflows the selected output dtype")
        if source.ndim != tensor.ndim or source.shape[-1] != tensor.shape[-1]:
            raise ValueError("Clean and corrupt activation ranks/channel widths differ")
        targets = sequence_positions(tensor, self.positions)
        if targets is None:
            if source.shape != tensor.shape:
                raise ValueError("Full activation patching requires identical shapes")
            changed = source.clone()
        else:
            sources = sequence_positions(source, self.source_positions)
            if len(targets) != len(sources):
                raise ValueError("Source and target token selections must have equal lengths")
            if source.shape[:-2] != tensor.shape[:-2]:
                raise ValueError("Clean and corrupt batch shapes differ")
            values = source.index_select(-2, torch.tensor(sources, device=tensor.device))
            changed = tensor.clone().index_copy(-2, torch.tensor(targets, device=tensor.device), values)
        self.applications += 1
        return replace_output(output, changed)

    def start(self):
        if self._handle is not None:
            raise RuntimeError("Activation patcher is already active")
        self._handle = self.module.register_forward_hook(self._hook)
        return self

    def close(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, traceback):
        self.close()


def _score(metric, output):
    value = metric(output)
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("Patching metric must return one scalar")
        value = value.detach().item()
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Patching metric must return a finite scalar")
    return value


def run_patching_experiment(model, clean_tokens, corrupt_tokens, module, metric, *,
                            positions=None, source_positions=None):
    """Recovery is (patched-corrupt)/(clean-corrupt), and can exceed [0,1].

    A negligible clean/corrupt score difference gives recovery=None, because
    dividing by it would fabricate evidence of causal recovery. One patching
    experiment measures sensitivity at the supplied site, not a whole circuit.
    """
    with evaluating(model):
        with ActivationCache(model, [module], max_bytes=67_108_864, max_records=1) as cache:
            clean_output = model(clean_tokens)
        if cache.invocations != 1 or cache.dropped:
            raise ValueError("Patching experiment requires one bounded invocation of the selected site")
        clean = cache.latest(module)
        clean_score = _score(metric, clean_output)
        corrupt_score = _score(metric, model(corrupt_tokens))
        with ActivationPatcher(model, module, clean, positions=positions,
                               source_positions=source_positions) as patcher:
            patched_score = _score(metric, model(corrupt_tokens))
        restored_score = _score(metric, model(corrupt_tokens))
    difference = clean_score - corrupt_score
    return {"module": module, "positions": list(positions) if positions is not None else None,
            "clean_score": clean_score, "corrupt_score": corrupt_score,
            "patched_score": patched_score, "restored_score": restored_score,
            "score_change": patched_score - corrupt_score,
            "recovery": (patched_score - corrupt_score) / difference if abs(difference) > 1e-10 else None,
            "applications": patcher.applications,
            "interpretation": "Measured effect of one activation intervention; not proof of a semantic circuit."}
