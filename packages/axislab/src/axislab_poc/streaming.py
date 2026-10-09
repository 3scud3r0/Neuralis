"""Bounded-memory activation capture and numerically stable online moments."""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor, nn
from .core import _tensor_from_output


@dataclass
class ActivationMoments:
    """Online [features] mean and variance from flattened token/sample vectors."""
    count: int = 0
    mean: Tensor | None = None
    m2: Tensor | None = None

    def update(self, activations: Tensor) -> None:
        if activations.ndim < 1:
            raise ValueError("Activations must have feature dimension")
        x = activations.detach().to(device="cpu", dtype=torch.float64).reshape(-1, activations.shape[-1])
        if not x.numel() or not torch.isfinite(x).all():
            raise ValueError("Empty/non-finite activations")
        n = len(x)
        batch_mean = x.mean(0)
        batch_m2 = (x - batch_mean).square().sum(0)
        if self.count == 0:
            self.mean, self.m2, self.count = batch_mean, batch_m2, n
            return
        if batch_mean.shape != self.mean.shape:
            raise ValueError("Hidden width changed")
        delta = batch_mean - self.mean
        new_count = self.count + n
        self.m2 += batch_m2 + delta.square() * self.count * n / new_count
        self.mean += delta * (n / new_count)
        self.count = new_count

    @property
    def variance(self) -> Tensor:
        if self.count < 2:
            raise ValueError("At least 2 observations required")
        return self.m2 / (self.count - 1)


def forward_moments(model: nn.Module, module_name: str, batches: list[Tensor],
                    *, position: str = "last_token") -> ActivationMoments:
    """Stream hidden vectors into moments; never cache the full activation tensors.

    This local reference takes a list. Production data sources can be generators.
    """
    mods = dict(model.named_modules())
    if module_name not in mods:
        raise KeyError(module_name)
    if position not in ("all", "last_token"):
        raise ValueError("position must be all/last_token")
    result = ActivationMoments()
    def hook(_mod, _inputs, output):
        v = _tensor_from_output(output)
        if position == "last_token":
            if v.ndim != 3:
                raise ValueError("last_token expects [batch,seq,features]")
            v = v[:, -1]
        result.update(v)
    handle = mods[module_name].register_forward_hook(hook)
    try:
        with torch.no_grad():
            for batch in batches:
                model(batch)
    finally:
        handle.remove()
    return result
