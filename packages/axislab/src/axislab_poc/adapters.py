"""Explicit optional adapters to upstream libraries; nothing downloaded implicitly.

Not all external adapters can be integration-tested offline. Use upstream pins,
model-specific hook names and test a tiny model before remote execution.
"""
from __future__ import annotations

from typing import Any
import torch
from torch import Tensor


def transformer_lens_capture(bridge: Any, inputs: Any, hook_name: str):
    """Capture one named activation using TransformerBridge.run_with_cache.

    Uses names_filter to avoid caching all activations on large models.
    Requires TransformerBridge compatible with TransformerLens 4.
    """
    logits, cache = bridge.run_with_cache(inputs, names_filter=hook_name)
    try:
        activation = cache[hook_name]
    except KeyError as e:
        raise KeyError(f"Hook {hook_name!r} missing; inspect your bridge cache keys") from e
    return logits, activation


def nnsight_capture(model: Any, inputs: Any, module_path: str,
                    *, remote: bool = False):
    """NNsight tracer capture (remote requires externally configured NDIF account).

    module_path uses dot-separated attributes, e.g. 'transformer.h.0'.
    This is a local or remote *single forward* trace, not autoregressive generation.
    """
    current = model
    for name in module_path.split("."):
        current = current[int(name)] if name.isdigit() else getattr(current, name)
    with model.trace(inputs, remote=remote):
        captured = current.output.save()
    return captured.value


def saelens_encode_decode(sae: Any, activation: Tensor) -> tuple[Tensor, Tensor]:
    """Interoperate with SAE Lens' encode/decode methods and local TopKSAE."""
    feature = sae.encode(activation)
    return feature, sae.decode(feature)


def repe_registry():
    """Register RepE's official Hugging Face 'rep-reading'/'rep-control' tasks.

    Requires external `repe` installed separately from its upstream repository.
    """
    try:
        from repe import repe_pipeline_registry
    except ImportError as e:
        raise ImportError("RepE not installed. Install the upstream representation-engineering repository.") from e
    repe_pipeline_registry()
    return ("rep-reading", "rep-control")


def upstream_model_edit(model: Any, tokenizer: Any, requests: list[dict],
                        hparams: Any, *, algorithm: str, **kwargs):
    """Invoke *actual* upstream ROME/MEMIT implementation (optional dependency).

    Unlike local rank-one/ridge primitives, these upstream methods run model-
    specific procedures that may need precomputed stats, VRAM and disk cache.
    `requests` and hparams must follow upstream API; caller handles evaluations.
    """
    if algorithm == "ROME":
        try:
            from rome.rome_main import apply_rome_to_model
        except ImportError as e:
            raise ImportError("Install kmeng01/rome separately") from e
        return apply_rome_to_model(model, tokenizer, requests, hparams, **kwargs)
    if algorithm == "MEMIT":
        try:
            from memit.memit_main import apply_memit_to_model
        except ImportError as e:
            raise ImportError("Install kmeng01/memit separately") from e
        return apply_memit_to_model(model, tokenizer, requests, hparams, **kwargs)
    raise ValueError("algorithm must be 'ROME' or 'MEMIT'")


def transformer_lens_steer(bridge: Any, inputs: Any, hook_name: str,
                           direction: Tensor, *, coefficient: float = 1.0,
                           position: str = "last_token"):
    """Single forward activation steering through *local* TransformerBridge.

    RemoteTransformerBridge's run_with_hooks is read-only; intervention specs
    are required on that backend. Explicitly reject a detected remote bridge.
    """
    if "remote" in type(bridge).__name__.lower():
        raise NotImplementedError("Remote Bridge requires declarative intervention specs")
    if position not in ("all", "last_token"):
        raise ValueError("position must be all or last_token")
    if direction.ndim != 1:
        raise ValueError("direction must be [features]")
    def hook(activation, _hook=None, **kwargs):
        if activation.shape[-1] != direction.numel():
            raise ValueError("Hook activation dimension mismatch")
        delta = coefficient * direction.to(activation.device, activation.dtype)
        if position == "last_token":
            if activation.ndim != 3:
                raise ValueError("last_token expects [batch,tokens,features]")
            edited = activation.clone()
            edited[:, -1, :] = edited[:, -1, :] + delta
            return edited
        return activation + delta
    return bridge.run_with_hooks(inputs, fwd_hooks=[(hook_name, hook)])


def nnsight_steer(model: Any, inputs: Any, module_path: str,
                  direction: Tensor, *, coefficient: float = 1.0,
                  remote: bool = False):
    """Experimentally steer the full activation of an NNsight-traced module.

    Remote mode is NOT verified here. Never transmit secrets in trace payloads;
    check NDIF access, versions, resource usage and hook tensor shapes first.
    """
    if direction.ndim != 1:
        raise ValueError("direction must be [features]")
    target = model
    for name in module_path.split("."):
        target = target[int(name)] if name.isdigit() else getattr(target, name)
    with model.trace(inputs, remote=remote):
        target.output = target.output + coefficient * direction
        saved = target.output.save()
    return saved.value
