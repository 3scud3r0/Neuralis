"""Transparent linear key/value editors inspired by ROME and MEMIT mathematics.

These are exact single-layer editing primitives, NOT full ROME/MEMIT algorithms:
no key localization, target optimization, covariance collection or multi-layer
transformer propagation is implemented here.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch
from torch import Tensor, nn


def _validate(layer: nn.Linear, keys: Tensor, targets: Tensor):
    if not isinstance(layer, nn.Linear):
        raise TypeError("Expected nn.Linear layer")
    if keys.ndim != 2 or targets.ndim != 2 or keys.shape[1] != targets.shape[1]:
        raise ValueError("keys [in_dim,n] and targets [out_dim,n] must match n")
    if keys.shape[0] != layer.in_features or targets.shape[0] != layer.out_features:
        raise ValueError("Editing dimensions mismatch linear layer")
    if keys.shape[1] < 1 or not torch.isfinite(keys).all() or not torch.isfinite(targets).all():
        raise ValueError("Non-finite or empty editing data")


def rank_one_update(layer: nn.Linear, key: Tensor, target: Tensor) -> Tensor:
    """Minimum-Frobenius-norm delta W satisfying (W+delta W)key=target.

    Bias is respected but not modified. Supports rectangular dense layers.
    """
    key = key.detach().flatten().to(layer.weight.device, torch.float64)
    target = target.detach().flatten().to(layer.weight.device, torch.float64)
    _validate(layer, key[:, None], target[:, None])
    denom = torch.dot(key, key)
    if denom <= 1e-12:
        raise ValueError("Key is zero")
    W = layer.weight.detach().double()
    bias = layer.bias.detach().double() if layer.bias is not None else torch.zeros(layer.out_features, dtype=W.dtype, device=W.device)
    err = target - (W @ key + bias)
    return (err[:, None] @ key[None, :] / denom).to(layer.weight.dtype)


def batched_ridge_update(layer: nn.Linear, keys: Tensor, targets: Tensor,
                         *, ridge: float = 1e-4) -> Tensor:
    """Ridge-regularized batched key/value editing on one nn.Linear.

    deltaW = (Targets - W Keys - bias) (Keys^T Keys + ridge I)^-1 Keys^T.
    This is a MEMIT-*inspired* local solve, not a MEMIT implementation.
    """
    if ridge <= 0:
        raise ValueError("ridge must be positive")
    keys = keys.detach().to(layer.weight.device, torch.float64)
    targets = targets.detach().to(layer.weight.device, torch.float64)
    _validate(layer, keys, targets)
    W = layer.weight.detach().double()
    bias = layer.bias.detach().double()[:, None] if layer.bias is not None else 0
    residual = targets - (W @ keys + bias)
    n = keys.shape[1]
    gram = keys.T @ keys + ridge * torch.eye(n, device=keys.device, dtype=keys.dtype)
    return torch.linalg.solve(gram, residual.T).T.matmul(keys.T).to(layer.weight.dtype)


@dataclass
class EditStats:
    update_norm: float
    original_norm: float
    relative_update: float


@contextmanager
def weight_edit(layer: nn.Linear, delta_weight: Tensor, *, max_relative_norm: float = 0.5,
                keep: bool = False) -> Iterator[EditStats]:
    """Apply computed delta in-place, rollback on exit unless keep=True.

    Backups consume one copy of the layer weight: not suitable for trillion-scale
    layers without sharded storage. Keep=True still rolls back on exceptions.
    """
    if not isinstance(layer, nn.Linear):
        raise TypeError("Only nn.Linear layers supported")
    if delta_weight.shape != layer.weight.shape:
        raise ValueError("Weight update shape mismatch")
    if not torch.isfinite(delta_weight).all() or max_relative_norm <= 0:
        raise ValueError("Invalid update or norm budget")
    delta = delta_weight.detach().to(layer.weight.device, layer.weight.dtype)
    old_norm = layer.weight.detach().float().norm().item()
    delta_norm = delta.float().norm().item()
    rel = delta_norm / max(old_norm, 1e-12)
    if rel > max_relative_norm:
        raise ValueError(f"Edit relative norm {rel:.5g} exceeds budget {max_relative_norm}")
    original = layer.weight.detach().clone()
    succeeded = False
    try:
        with torch.no_grad():
            layer.weight.add_(delta)
        yield EditStats(delta_norm, old_norm, rel)
        succeeded = True
    finally:
        if not (keep and succeeded):
            with torch.no_grad():
                layer.weight.copy_(original)


def preservation_drift(layer: nn.Linear, held_out_keys: Tensor,
                       baseline_output: Tensor) -> float:
    """Mean absolute change on unrelated keys; always measure separately from success."""
    with torch.no_grad():
        output = layer(held_out_keys)
        if output.shape != baseline_output.shape:
            raise ValueError("Preservation data shape mismatch")
        return float((output - baseline_output.to(output.device)).abs().mean())
