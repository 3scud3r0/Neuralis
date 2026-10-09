"""Actual token inference, bounded generation, and explicit output projections."""

import math
import torch

from .activation_cache import ActivationCache
from ._tensors import evaluating, logits_tensor


def _device(model):
    parameter = next(model.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(model.buffers(), None)
    return buffer.device if buffer is not None else torch.device("cpu")


def _tokens(model, tokens):
    value = torch.as_tensor(tokens)
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if value.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError("Token IDs must be integers")
    if value.ndim != 2 or not 1 <= value.shape[0] <= 16 or not 1 <= value.shape[1] <= 512:
        raise ValueError("Token inference accepts 1..16 sequences of 1..512 tokens")
    if (value < 0).any():
        raise ValueError("Token IDs must be nonnegative")
    return value.to(device=_device(model), dtype=torch.long)


def infer_tokens(model, tokens, *, modules=None, positions=None, max_bytes=8_388_608):
    """Return CPU tensor logits/probabilities and the exact selected activations.

    Supplied local models may return a tensor, tuple beginning with logits, or
    an object with .logits. No tokenizer, checkpoint, or remote model is loaded.
    """
    inputs = _tokens(model, tokens)
    with evaluating(model):
        if modules:
            with ActivationCache(model, modules, positions=positions, max_bytes=max_bytes) as cache:
                logits = logits_tensor(model(inputs)).detach().float().cpu().clone()
            activations = cache.snapshot()
        else:
            logits = logits_tensor(model(inputs)).detach().float().cpu().clone()
            activations = {"records": [], "retained_bytes": 0, "dropped": 0, "complete": True}
    return {"token_ids": inputs.cpu().tolist(), "logits": logits,
            "probabilities": logits.softmax(-1), "activations": activations,
            "interpretation": "Measured local model outputs and tensors; these do not expose hidden thoughts."}


def logit_lens(model, activation: torch.Tensor, *, normalized: bool = True, top_k: int = 5):
    """Project a selected residual tensor through the supplied model's final head.

    The default includes its final normalization, matching TinyTransformer's
    readout. Intermediate residual projections are diagnostics, not predictions
    made by the model at those layers and not validated semantic explanations.
    """
    head = getattr(model, "output", None)
    if head is None and hasattr(model, "get_output_embeddings"):
        head = model.get_output_embeddings()
    norm = getattr(model, "norm", None)
    if head is None:
        raise ValueError("Supply a local model with output or get_output_embeddings()")
    if normalized and norm is None:
        raise ValueError("Normalized lens requires an explicit model.norm; select normalized=False otherwise")
    if activation.ndim < 2 or activation.numel() > 8_000_000 or not torch.isfinite(activation).all():
        raise ValueError("Lens activation must be finite, bounded, and have a channel axis")
    if not 1 <= top_k <= 128:
        raise ValueError("Lens top_k must be 1..128")
    parameter = next(head.parameters(), None)
    dtype = parameter.dtype if parameter is not None else activation.dtype
    with evaluating(model):
        inputs = activation.detach().to(device=_device(model), dtype=dtype)
        logits = head(norm(inputs) if normalized else inputs).float().cpu()
        if logits.numel() > 8_000_000 or not torch.isfinite(logits).all():
            raise ValueError("Lens output exceeds the budget or contains nonfinite logits")
        probabilities = logits.softmax(-1)
        values, indices = probabilities.topk(min(top_k, probabilities.shape[-1]), -1)
    return {"logits": logits, "probabilities": probabilities,
            "top_token_ids": indices.tolist(), "top_probabilities": values.tolist(),
            "normalized": normalized,
            "interpretation": "Final readout applied to measured residuals; no hidden-thought or semantic claim."}


def _sample_distribution(logits, temperature, top_k, top_p):
    scaled = logits / temperature if temperature else logits.clone()
    if top_k:
        indices = scaled.topk(min(top_k, scaled.numel())).indices
        filtered = torch.full_like(scaled, -float("inf"))
        filtered[indices] = scaled[indices]
        scaled = filtered
    if top_p < 1:
        ordered, indices = scaled.sort(descending=True)
        remove = ordered.softmax(-1).cumsum(-1) > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        scaled[indices[remove]] = -float("inf")
    return scaled.softmax(-1)


def axislab_logit_lens(model, activation: torch.Tensor, *, top_k=5, target_ids=None,
                       normalized=True):
    """Explicit optional bridge to the extracted AxisLab public lens API.

    This keeps the API's required dependencies independent of the optional local
    research package. AxisLab supplies entropy and target-token ranks in addition
    to its top-token projection. No model weights are downloaded.
    """
    try:
        from axislab_poc.lenses import logit_lens as project
    except ImportError as error:
        raise ImportError("AxisLab bridge requires packages/axislab/src on PYTHONPATH or its local package installed") from error
    head = getattr(model, "output", None)
    if head is None and hasattr(model, "get_output_embeddings"):
        head = model.get_output_embeddings()
    if head is None or not isinstance(getattr(head, "weight", None), torch.Tensor):
        raise ValueError("AxisLab projection requires a linear readout weight [vocabulary,channels]")
    norm = getattr(model, "norm", None)
    if normalized and norm is None:
        raise ValueError("Normalized AxisLab lens requires an explicit model.norm")
    if not 1 <= top_k <= 128 or activation.numel() > 8_000_000:
        raise ValueError("AxisLab readout exceeds the bounded local projection profile")
    with evaluating(model):
        inputs = activation.detach().to(device=_device(model), dtype=head.weight.dtype)
        result = project(inputs, head.weight, bias=getattr(head, "bias", None),
                         norm=norm if normalized else None, top_k=min(top_k, head.weight.shape[0]),
                         target_ids=target_ids)
    return result.as_dict()


def generate_tokens(model, prompt_ids, *, max_new_tokens: int = 16, modules=None,
                    temperature: float = 0.8, top_k: int = 0, top_p: float = 1,
                    seed: int = 42, context_length=None, eos_token_id=None,
                    max_capture_bytes: int = 8_388_608):
    """Autoregressive inference with measured last-position activations per step.

    Capture bytes are bounded over the whole generation, rather than per token.
    A generated token's record describes the preceding context used to sample
    it. Full base and sampling distributions are retained for local vocabularies
    of at most 4096 entries; this utility deliberately avoids expensive jobs.
    """
    prompt = _tokens(model, prompt_ids)
    if prompt.shape[0] != 1 or not 1 <= max_new_tokens <= 256:
        raise ValueError("Generation accepts one prompt and 1..256 generated tokens")
    if not math.isfinite(temperature) or not 0 <= temperature <= 5:
        raise ValueError("Temperature must be in [0,5]")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or not 0 <= top_k <= 4096:
        raise ValueError("Sampling top_k must be 0..4096")
    if not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("Sampling top_p must be in (0,1]")
    if not 1 <= max_capture_bytes <= 67_108_864:
        raise ValueError("Generation capture budget must be 1..64 MiB")
    native_context = getattr(getattr(model, "position", None), "num_embeddings", None)
    if context_length is None:
        context_length = native_context or 512
    if not isinstance(context_length, int) or not 1 <= context_length <= 512:
        raise ValueError("Context length must be 1..512")
    if native_context is not None and context_length > native_context:
        raise ValueError("Requested context exceeds the supplied model's positional capacity")
    tokens = prompt[0].cpu().tolist()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    steps = []
    remaining = max_capture_bytes
    dropped = 0
    with evaluating(model):
        for step in range(max_new_tokens):
            inputs = torch.tensor([tokens[-context_length:]], device=_device(model), dtype=torch.long)
            if modules and remaining:
                with ActivationCache(model, modules, positions=[-1], max_bytes=remaining) as cache:
                    logits = logits_tensor(model(inputs))[0, -1].float().cpu()
                activation_snapshot = cache.snapshot()
                remaining -= cache.retained_bytes
                dropped += cache.dropped
            else:
                logits = logits_tensor(model(inputs))[0, -1].float().cpu()
                activation_snapshot = {"records": [], "retained_bytes": 0, "dropped": 0, "complete": not bool(modules)}
                if modules:
                    dropped += len(modules)
                    activation_snapshot["dropped"] = len(modules)
            if logits.numel() > 4096:
                raise ValueError("Local generation vocabulary exceeds the 4096-token budget")
            probabilities = logits.softmax(-1)
            if temperature == 0:
                token = int(logits.argmax())
                sample_probabilities = torch.zeros_like(probabilities)
                sample_probabilities[token] = 1
            else:
                sample_probabilities = _sample_distribution(logits, temperature, top_k, top_p)
                token = int(torch.multinomial(sample_probabilities, 1, generator=generator))
            entropy = float(-(probabilities * probabilities.clamp_min(1e-12).log()).sum())
            steps.append({"step": step, "context_token_ids": inputs[0].cpu().tolist(),
                          "token_id": token, "model_probability": float(probabilities[token]),
                          "sample_probability": float(sample_probabilities[token]),
                          "probabilities": probabilities.tolist(),
                          "sampling_probabilities": sample_probabilities.tolist(),
                          "entropy": entropy, "activations": activation_snapshot})
            tokens.append(token)
            if token == eos_token_id:
                break
    return {"prompt_token_ids": prompt[0].cpu().tolist(), "token_ids": tokens,
            "generated_token_ids": [step["token_id"] for step in steps], "steps": steps,
            "capture": {"max_bytes": max_capture_bytes, "retained_bytes": max_capture_bytes - remaining,
                        "dropped": dropped, "complete": dropped == 0},
            "settings": {"max_new_tokens": max_new_tokens, "temperature": temperature,
                         "top_k": top_k, "top_p": top_p, "seed": seed, "context_length": context_length},
            "interpretation": "Actual local token inference and sampled continuation; activations do not reveal hidden thoughts."}
