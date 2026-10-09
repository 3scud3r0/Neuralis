"""Causal interventions on PyTorch module outputs; reversible by context managers."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterator, Literal

import torch
from torch import Tensor, nn
from .core import _replace_output, _tensor_from_output


def _module(model: nn.Module, name: str) -> nn.Module:
    modules = dict(model.named_modules())
    if name not in modules:
        raise KeyError(f"Unknown module: {name}")
    return modules[name]


@contextmanager
def transform_activation(model: nn.Module, module_name: str,
                         transform: Callable[[Tensor], Tensor],
                         *, position: Literal["all", "last_token"] = "all") -> Iterator[None]:
    """Temporarily transform an activation. No in-place mutation of original output."""
    if position not in ("all", "last_token"):
        raise ValueError("position must be all or last_token")

    def hook(_module, _inputs, output):
        before = _tensor_from_output(output)
        if position == "last_token":
            if before.ndim != 3:
                raise ValueError("last_token requires [batch,tokens,features]")
            result = before.clone()
            altered = transform(before[:, -1, :])
            if altered.shape != before[:, -1, :].shape:
                raise ValueError("Transform shape mismatch")
            result[:, -1, :] = altered
        else:
            result = transform(before)
            if result.shape != before.shape:
                raise ValueError("Transform shape mismatch")
        return _replace_output(output, result)

    handle = _module(model, module_name).register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def activation_patch(model: nn.Module, module_name: str, replacement: Tensor,
                     *, position: Literal["all", "last_token"] = "all") -> Iterator[None]:
    """Patch a layer with a captured activation, checking shape and dtype."""
    if not isinstance(replacement, Tensor):
        raise TypeError("replacement must be a tensor")

    def patch(h):
        if h.shape != replacement.shape:
            raise ValueError(f"Patch shape mismatch: {tuple(h.shape)} != {tuple(replacement.shape)}")
        return replacement.to(device=h.device, dtype=h.dtype).clone()

    with transform_activation(model, module_name, patch, position=position):
        yield


@contextmanager
def projection_ablation(model: nn.Module, module_name: str, direction: Tensor,
                        *, baseline: Tensor | None = None,
                        position: Literal["all", "last_token"] = "all") -> Iterator[None]:
    """Zero the signed projection along one feature direction."""
    if direction.ndim != 1 or direction.norm() <= 1e-12:
        raise ValueError("direction must be a nonzero 1D tensor")
    def ablate(h):
        v = direction.to(h.device, h.dtype)
        if h.shape[-1] != len(v):
            raise ValueError("Direction width mismatch")
        v = v / v.norm()
        ref = torch.zeros_like(v) if baseline is None else baseline.to(h.device,h.dtype)
        if ref.shape != v.shape:
            raise ValueError("Baseline width mismatch")
        return h - ((h - ref) @ v).unsqueeze(-1) * v
    with transform_activation(model, module_name, ablate, position=position):
        yield


@contextmanager
def multi_axis_steering(model: nn.Module, module_name: str, directions: Tensor,
                        coefficients: Tensor, *, position: Literal["all", "last_token"] = "all") -> Iterator[None]:
    """Inject k directions simultaneously: [k,features] x [k]."""
    if directions.ndim != 2 or coefficients.ndim != 1 or len(directions) != len(coefficients):
        raise ValueError("Expected [k,d] directions and [k] coefficients")
    if not torch.isfinite(directions).all() or not torch.isfinite(coefficients).all():
        raise ValueError("Non-finite interventions")
    def steer(h):
        if h.shape[-1] != directions.shape[-1]:
            raise ValueError("Activation width mismatch")
        delta = coefficients.to(h.device, h.dtype) @ directions.to(h.device, h.dtype)
        return h + delta
    with transform_activation(model, module_name, steer, position=position):
        yield
