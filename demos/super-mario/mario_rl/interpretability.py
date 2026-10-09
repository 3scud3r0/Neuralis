"""Policy attribution and bounded causal activation ablations.

Attributions explain a selected output score at an observation; they are not
proof of semantic understanding. Ablations modify activations during evaluation
and measure effects using the same seeded starts as the unmodified policy.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn


@dataclass
class Attribution:
    method: str
    action: int
    attribution: np.ndarray
    score: float
    baseline_score: float | None = None
    completeness_residual: float | None = None


def selected_score(model, observations, action):
    logits, _ = model(observations)
    if not 0 <= action < logits.shape[1]:
        raise ValueError("Selected action is outside the policy output")
    return logits[:, action].sum()


def gradient_attribution(model, observation: np.ndarray, action: int, multiply_input: bool = True):
    device = next(model.parameters()).device
    inputs = torch.as_tensor(observation[None], device=device).float().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    score = selected_score(model, inputs, action)
    gradient = torch.autograd.grad(score, inputs)[0]
    values = gradient * inputs if multiply_input else gradient
    return Attribution("gradient-times-input" if multiply_input else "gradient", action,
                       values[0].detach().cpu().numpy(), float(score.item()))


def integrated_gradients(model, observation: np.ndarray, action: int, steps: int = 32):
    if not 2 <= steps <= 256:
        raise ValueError("Integrated-gradient steps must be between 2 and 256")
    device = next(model.parameters()).device
    inputs = torch.as_tensor(observation[None], device=device).float()
    baseline = torch.zeros_like(inputs)
    average_gradient = torch.zeros_like(inputs)
    # Trapezoidal quadrature of the gradient along a black-frame baseline path.
    for index in range(steps + 1):
        alpha = index / steps
        interpolated = (baseline + alpha * (inputs - baseline)).requires_grad_(True)
        output = selected_score(model, interpolated, action)
        gradient = torch.autograd.grad(output, interpolated)[0]
        weight = 0.5 if index in {0, steps} else 1.0
        average_gradient += weight * gradient / steps
    attribution = (inputs - baseline) * average_gradient
    with torch.no_grad():
        output_score = float(selected_score(model, inputs, action).item())
        baseline_score = float(selected_score(model, baseline, action).item())
    residual = output_score - baseline_score - float(attribution.sum().item())
    return Attribution("integrated-gradients", action, attribution[0].detach().cpu().numpy(),
                       output_score, baseline_score, residual)


def grad_cam(model, observation: np.ndarray, action: int, layer_name: str = "encoder.4"):
    layers = dict(model.named_modules())
    layer = layers.get(layer_name)
    if not isinstance(layer, nn.Conv2d):
        raise ValueError("Grad-CAM layer must be a Conv2d module")
    activation = None
    def capture(module, inputs, output):
        nonlocal activation
        activation = output
    handle = layer.register_forward_hook(capture)
    try:
        device = next(model.parameters()).device
        inputs = torch.as_tensor(observation[None], device=device).float().requires_grad_(True)
        score = selected_score(model, inputs, action)
        if activation is None:
            raise RuntimeError("Selected Grad-CAM module did not execute")
        gradients = torch.autograd.grad(score, activation)[0]
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        heatmap = torch.relu((weights * activation).sum(dim=1, keepdim=True))
        heatmap = torch.nn.functional.interpolate(heatmap, size=observation.shape[-2:],
                                                 mode="bilinear", align_corners=False)
        return Attribution("grad-cam", action, heatmap[0].detach().cpu().numpy(), float(score.item()))
    finally:
        handle.remove()


@contextmanager
def activation_intervention(model, layer_name: str, channels: tuple[int, ...] = (), scale: float = 0.0):
    if not np.isfinite(scale) or abs(scale) > 10:
        raise ValueError("Intervention scale must be finite and between -10 and 10")
    layer = dict(model.named_modules()).get(layer_name)
    if not isinstance(layer, (nn.Conv2d, nn.Linear, nn.ReLU)):
        raise ValueError("Interventions require a Conv2d, Linear, or ReLU layer")
    def intervene(module, inputs, output):
        if output.ndim < 2:
            raise ValueError("Selected layer has no channel dimension")
        if any(channel < 0 or channel >= output.shape[1] for channel in channels):
            raise ValueError("Selected intervention channels are outside the layer")
        changed = output.clone()
        if channels:
            changed[:, list(channels)] *= scale
        else:
            changed *= scale
        return changed
    handle = layer.register_forward_hook(intervene)
    try:
        yield
    finally:
        handle.remove()


def heatmap_rgb(values: np.ndarray) -> np.ndarray:
    """A bounded diverging numerical heatmap, independent of plotting packages."""
    if values.ndim == 3:
        values = values.sum(axis=0)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Attribution must be a finite [channel,height,width] or [height,width] array")
    denominator = max(float(np.percentile(np.abs(values), 99)), 1e-12)
    scaled = np.clip(values / denominator, -1, 1)
    positive = np.maximum(scaled, 0)
    negative = np.maximum(-scaled, 0)
    red = 12 + 213 * negative + 43 * positive
    green = 20 + 208 * positive + 31 * negative
    blue = 35 + 186 * positive + 183 * negative
    return np.stack((red, green, blue), axis=-1).clip(0, 255).astype(np.uint8)
