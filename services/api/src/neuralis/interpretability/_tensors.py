"""Shared tensor and lifecycle adapters for local instrumented models."""

from contextlib import contextmanager
import torch


def output_tensor(output):
    """Resolve tensor or conventional tuple output without guessing hidden states."""
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError("Selected module must return a tensor or tuple beginning with a tensor")


def replace_output(output, tensor):
    if isinstance(output, torch.Tensor):
        return tensor
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return (tensor, *output[1:])
    raise TypeError("Selected module has no supported tensor output")


def sequence_positions(tensor, positions):
    """Token positions always address the penultimate [sequence, channel] axis."""
    if positions is None:
        return None
    if tensor.ndim < 3:
        raise ValueError("Token position selection requires [batch,sequence,channel] output")
    length = tensor.shape[-2]
    result = []
    for value in positions:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("Token positions must be integers")
        position = value + length if value < 0 else value
        if not 0 <= position < length:
            raise ValueError(f"Token position {value} is outside sequence length {length}")
        if position in result:
            raise ValueError("Duplicate token positions are ambiguous")
        result.append(position)
    if not result:
        raise ValueError("At least one token position must be selected")
    return result


def module_by_name(model, name):
    modules = dict(model.named_modules())
    if name not in modules:
        raise ValueError(f"Unknown module: {name}")
    return modules[name]


@contextmanager
def evaluating(model):
    """Restore each submodule's mode, including mixed train/eval configurations."""
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        with torch.no_grad():
            yield
    finally:
        for module, training in modes:
            module.training = training


def logits_tensor(output):
    value = output.logits if hasattr(output, "logits") else output_tensor(output)
    if value.ndim != 3 or not torch.isfinite(value).all():
        raise ValueError("Inference requires finite [batch,sequence,vocabulary] logits")
    if value.numel() > 8_000_000:
        raise ValueError("Logit output exceeds the local inference budget")
    return value
