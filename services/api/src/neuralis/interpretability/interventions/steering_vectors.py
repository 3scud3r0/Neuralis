"""Reversible additive interventions on explicitly selected module outputs."""

import math
import torch

from .._tensors import module_by_name, output_tensor, replace_output, sequence_positions


class SteeringSession:
    """Add a weighted sum of fixed feature directions during the session only."""

    def __init__(self, model, module: str, *, vectors, strengths, positions=None):
        self.module = module_by_name(model, module)
        self.module_name = module
        if not vectors or len(vectors) > 256 or not strengths:
            raise ValueError("Select 1..256 steering directions and at least one strength")
        if set(strengths).difference(vectors):
            raise ValueError("Every steering strength requires a supplied direction")
        self.vectors = {}
        width = None
        for feature, vector in vectors.items():
            if not isinstance(vector, torch.Tensor) or vector.ndim != 1 or not 1 <= vector.numel() <= 4096:
                raise ValueError("Steering directions must be bounded channel vectors")
            if not torch.isfinite(vector).all():
                raise ValueError("Steering directions must be finite")
            if width is not None and width != len(vector):
                raise ValueError("All steering directions must share a channel width")
            width = len(vector)
            self.vectors[feature] = vector.detach().float().cpu().clone()
            if not torch.isfinite(self.vectors[feature]).all():
                raise ValueError("Steering directions exceed finite float32 range")
        self.strengths = {}
        self.update_strengths(strengths)
        self.positions = tuple(positions) if positions is not None else None
        self._handle = None
        self.applications = 0

    def update_strengths(self, strengths):
        validated = {}
        if set(strengths).difference(self.vectors):
            raise ValueError("Unknown steering feature")
        for feature, strength in strengths.items():
            strength = float(strength)
            if not math.isfinite(strength) or abs(strength) > 100:
                raise ValueError("Steering strengths must be finite and in [-100,100]")
            validated[feature] = strength
        if not validated:
            raise ValueError("Provide at least one steering strength")
        self.strengths = validated

    def _hook(self, module, inputs, output):
        tensor = output_tensor(output)
        if tensor.ndim < 2 or tensor.shape[-1] != next(iter(self.vectors.values())).numel():
            raise ValueError("Steering direction width differs from module output")
        positions = sequence_positions(tensor, self.positions)
        delta = torch.zeros(tensor.shape[-1], device=tensor.device, dtype=tensor.dtype)
        for feature, strength in self.strengths.items():
            delta = delta + self.vectors[feature].to(tensor.device, tensor.dtype) * strength
        if not torch.isfinite(delta).all():
            raise ValueError("Steering values overflow the selected module output dtype")
        changed = tensor.clone()
        if positions is None:
            changed = changed + delta
        else:
            indices = torch.tensor(positions, device=tensor.device)
            selected = changed.index_select(-2, indices) + delta
            changed = changed.index_copy(-2, indices, selected)
        self.applications += 1
        return replace_output(output, changed)

    def start(self):
        if self._handle is not None:
            raise RuntimeError("Steering session is already active")
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
