"""Prototype: discover and intervene on activation directions in PyTorch models."""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Literal

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class Axis:
    """A normalized direction and the control mean, both stored on CPU."""

    direction: Tensor
    control_mean: Tensor
    removed_components: int

    def score(self, activations: Tensor) -> Tensor:
        """Project activations of shape (..., features) onto this axis."""
        direction = self.direction.to(device=activations.device, dtype=activations.dtype)
        reference = self.control_mean.to(device=activations.device, dtype=activations.dtype)
        return (activations - reference) @ direction


def fit_axis(
    positive: Tensor,
    control: Tensor,
    denoise_variance: float = 0.5,
) -> Axis:
    """Fit a denoised difference-in-means axis using control PCA.

    Both inputs must be matrices [examples, features]. Denoising projects out
    the leading PCs of the control set that explain `denoise_variance` variance.
    """
    if positive.ndim != 2 or control.ndim != 2:
        raise ValueError("Expected [examples, features] tensors.")
    if positive.shape[0] < 2 or control.shape[0] < 2:
        raise ValueError("At least two examples per group are required.")
    if positive.shape[1] != control.shape[1]:
        raise ValueError("Positive and control sets must have equal feature counts.")
    if not 0.0 <= denoise_variance < 1.0:
        raise ValueError("denoise_variance must be in [0, 1).")

    # Float64 reduces cancellation errors during PCA; keep data on CPU for this demo.
    pos = positive.detach().to(device="cpu", dtype=torch.float64)
    neg = control.detach().to(device="cpu", dtype=torch.float64)
    if not torch.isfinite(pos).all() or not torch.isfinite(neg).all():
        raise ValueError("Activations contain non-finite values.")

    mean_control = neg.mean(dim=0)
    difference = pos.mean(dim=0) - mean_control
    removed = 0

    if denoise_variance > 0:
        centered = neg - mean_control
        # Vh rows are orthonormal feature-space principal directions.
        _, singular, vh = torch.linalg.svd(centered, full_matrices=False)
        variance = singular.square()
        total = variance.sum()
        if total.item() > 1e-20:
            cumulative = variance.cumsum(0) / total
            removed = int(torch.searchsorted(cumulative, denoise_variance).item()) + 1
            basis = vh[:removed]
            difference = difference - (difference @ basis.T) @ basis

    norm = torch.linalg.vector_norm(difference)
    if norm.item() <= 1e-12:
        raise ValueError("The contrast direction is zero after denoising.")

    return Axis(
        direction=(difference / norm).to(torch.float32),
        control_mean=mean_control.to(torch.float32),
        removed_components=removed,
    )


def _tensor_from_output(output):
    if isinstance(output, Tensor):
        return output
    if isinstance(output, tuple) and output and isinstance(output[0], Tensor):
        return output[0]
    raise TypeError("The hooked module must output a tensor or a tuple starting with a tensor.")


def _replace_output(output, replacement: Tensor):
    if isinstance(output, tuple):
        return (replacement, *output[1:])
    return replacement


def capture_activations(model: nn.Module, module_name: str, *args, **kwargs):
    """Return (model output, activation) for one named PyTorch module."""
    modules = dict(model.named_modules())
    if module_name not in modules:
        raise KeyError(f"Unknown module: {module_name}")
    activations = []

    def remember(_module, _inputs, output):
        activations.append(_tensor_from_output(output).detach().cpu().clone())

    handle = modules[module_name].register_forward_hook(remember)
    try:
        with torch.no_grad():
            result = model(*args, **kwargs)
    finally:
        handle.remove()

    if not activations:
        raise RuntimeError(f"Module {module_name} was not executed.")
    return result, activations[-1]


@contextmanager
def temporary_steering(
    model: nn.Module,
    module_name: str,
    axis: Axis,
    coefficient: float,
    position: Literal["all", "last_token"] = "all",
) -> Iterator[None]:
    """Add coefficient * direction while in this context, then remove the hook."""
    modules = dict(model.named_modules())
    if module_name not in modules:
        raise KeyError(f"Unknown module: {module_name}")
    if position not in ("all", "last_token"):
        raise ValueError("position must be 'all' or 'last_token'.")

    def inject(_module, _inputs, output):
        activation = _tensor_from_output(output)
        if activation.shape[-1] != axis.direction.numel():
            raise ValueError("Activation width and axis width do not match.")
        delta = axis.direction.to(device=activation.device, dtype=activation.dtype)
        if position == "all":
            modified = activation + float(coefficient) * delta
        else:
            if activation.ndim != 3:
                raise ValueError("last_token requires [batch, tokens, features].")
            # Clone avoids changing tensors shared with other parts of the graph.
            modified = activation.clone()
            modified[:, -1, :] = modified[:, -1, :] + float(coefficient) * delta
        return _replace_output(output, modified)

    handle = modules[module_name].register_forward_hook(inject)
    try:
        yield
    finally:
        handle.remove()
