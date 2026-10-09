"""Interventions and component readouts for local, owned or permitted models.

Attribution is an accounting identity only *before nonlinear normalizations*;
causal patching evaluates actual model outputs and is not an explanation by itself.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Any
import torch
from torch import Tensor, nn
from .core import capture_activations
from .interventions import activation_patch, transform_activation


@dataclass(frozen=True)
class PatchingResult:
    clean: float
    corrupted: float
    patched: float
    recovered_fraction: float | None
    module: str


def causal_patch_effect(model: nn.Module, module_name: str,
                        clean_inputs: Tensor, corrupted_inputs: Tensor,
                        metric: Callable[[Tensor], Tensor | float], *,
                        position: str = "all") -> PatchingResult:
    """Capture a clean activation and replace the corrupt activation with it.

    `metric` returns a scalar score where *higher means better*, e.g. the
    target logit or negative MSE. Inputs must align in batch and sequence.
    """
    if position not in ("all", "last_token"):
        raise ValueError("position must be all/last_token")
    model.eval()
    with torch.no_grad():
        y_clean, full_acts = capture_activations(model, module_name, clean_inputs)
        y_corrupt = model(corrupted_inputs)
        selected = full_acts if position == "all" else full_acts[:, -1, :]
        with activation_patch(model, module_name, selected, position=position):
            y_patched = model(corrupted_inputs)
        def val(out):
            value = torch.as_tensor(metric(out)).detach()
            if value.numel() != 1 or not torch.isfinite(value).all():
                raise ValueError("metric must return a finite scalar")
            return float(value.item())
        a, b, c = val(y_clean), val(y_corrupt), val(y_patched)
    difference = a - b
    return PatchingResult(a, b, c, None if abs(difference) < 1e-9 else (c - b) / difference,
                          module_name)


def sae_feature_patch(base_activation: Tensor, clean_activation: Tensor, sae: Any,
                      feature_ids: list[int]) -> Tensor:
    """Patch selected SAE features, preserving the reconstruction residual of base.

    The SAE must be shared/fixed between both conditions. Using feature IDs from
    independently trained dictionaries is NOT valid alignment.
    """
    if base_activation.shape != clean_activation.shape:
        raise ValueError("base and clean activation shape mismatch")
    h_base = sae.encode(base_activation)
    h_clean = sae.encode(clean_activation)
    if h_base.shape != h_clean.shape or h_base.shape[:-1] != base_activation.shape[:-1]:
        raise ValueError("SAE feature shape mismatch")
    if not feature_ids or len(feature_ids) != len(set(feature_ids)):
        raise ValueError("feature_ids must be unique and nonempty")
    if min(feature_ids) < 0 or max(feature_ids) >= h_base.shape[-1]:
        raise ValueError("feature index out of bounds")
    edit = h_base.clone()
    edit[..., feature_ids] = h_clean[..., feature_ids]
    return base_activation + sae.decode(edit) - sae.decode(h_base)


def component_logit_contributions(parts: dict[str, Tensor],
                                  unembed_weight: Tensor, *,
                                  target_token_id: int) -> dict[str, Tensor]:
    """Project additive residual-stream components to one token's logit.

    Only valid as linear accounting when components share a coordinate system and
    no nonlinear final LayerNorm/RMSNorm is inserted after summing them. For such
    models use frozen-normalization approximation and label it accordingly.
    """
    if unembed_weight.ndim != 2 or not 0 <= target_token_id < unembed_weight.shape[0]:
        raise ValueError("Invalid unembedding or target_token_id")
    if not parts:
        raise ValueError("Need named components")
    vec = unembed_weight[target_token_id]
    res = {}
    for name, tensor in parts.items():
        if tensor.shape[-1] != vec.numel():
            raise ValueError(f"Width mismatch in {name}")
        res[name] = (tensor.float() * vec.to(tensor.device).float()).sum(dim=-1)
    return res
