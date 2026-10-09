"""Non-faithful intermediate-token readouts for inspectable LOCAL neural networks.

Logit Lens is a projection, not a recovery of internal natural-language thoughts.
For serious work use appropriate output normalization and evaluate a Tuned Lens on
held-out examples. All shapes and vocab orientation are explicit.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Sequence
import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class LensReadout:
    """Data from a vocabulary projection; last axis of all fields is top-k, except entropy."""
    token_ids: Tensor
    probabilities: Tensor
    entropy: Tensor
    target_rank: Tensor | None
    source: str = "logit_lens:intermediate_prediction_not_reasoning_transcript"

    def as_dict(self) -> dict:
        return {
            "token_ids": self.token_ids.detach().cpu().tolist(),
            "probabilities": self.probabilities.detach().cpu().tolist(),
            "entropy": self.entropy.detach().cpu().tolist(),
            "target_rank": None if self.target_rank is None else self.target_rank.detach().cpu().tolist(),
            "source": self.source,
        }


def _project(hidden: Tensor, unembed_weight: Tensor, *,
             bias: Tensor | None = None, norm: Callable[[Tensor], Tensor] | None = None) -> Tensor:
    """Explicit unembedding convention: weight [vocab_size, hidden_dim]."""
    if hidden.ndim < 1 or unembed_weight.ndim != 2 or hidden.shape[-1] != unembed_weight.shape[1]:
        raise ValueError("Expected hidden [...,d], unembedding [vocab,d].")
    if not hidden.is_floating_point() or not unembed_weight.is_floating_point():
        raise TypeError("Hidden states and unembedding must be floating-point tensors")
    if bias is not None and bias.shape != (unembed_weight.shape[0],):
        raise ValueError("Unembedding bias must have shape [vocab]")
    prepared = hidden if norm is None else norm(hidden)
    if prepared.shape != hidden.shape:
        raise ValueError("Norm changed hidden shape")
    logits = F.linear(prepared.float(), unembed_weight.detach().float(),
                      None if bias is None else bias.detach().float())
    if not torch.isfinite(logits).all():
        raise ValueError("Nonfinite projected logits")
    return logits


def _readout(logits: Tensor, *, top_k: int = 5, target_ids: Tensor | None = None,
             source: str = "logit_lens:intermediate_prediction_not_reasoning_transcript") -> LensReadout:
    if not 1 <= top_k <= logits.shape[-1]:
        raise ValueError("top_k must be between 1 and vocabulary size")
    probs = logits.softmax(-1)
    top_probs, ids = probs.topk(top_k, dim=-1)
    entropy = -(probs * probs.clamp_min(1e-30).log()).sum(-1)
    ranks = None
    if target_ids is not None:
        t = torch.as_tensor(target_ids, device=logits.device, dtype=torch.long)
        if t.shape != logits.shape[:-1]:
            raise ValueError(f"target_ids must have shape {tuple(logits.shape[:-1])}")
        if (t < 0).any() or (t >= logits.shape[-1]).any():
            raise ValueError("target_ids outside vocabulary")
        target_logits = logits.gather(-1, t.unsqueeze(-1))
        # Rank 1 means highest score. Ties share rank: deterministic and cheap.
        ranks = (logits > target_logits).sum(-1) + 1
    return LensReadout(ids.detach(), top_probs.detach(), entropy.detach(),
                       None if ranks is None else ranks.detach(), source)


def logit_lens(hidden: Tensor, unembed_weight: Tensor, *, bias: Tensor | None = None,
               norm: Callable[[Tensor], Tensor] | None = None,
               top_k: int = 5, target_ids: Tensor | None = None) -> LensReadout:
    """Project hidden states to token logits. Not a transcript or causal explanation."""
    return _readout(_project(hidden, unembed_weight, bias=bias, norm=norm),
                    top_k=top_k, target_ids=target_ids)


class TunedLens(nn.Module):
    """Small affine translator for a specific layer, initialized to identity.

    Training minimizes KL(teacher distribution || projected translation). The
    teacher must be produced by the same accessible local model.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        self.affine = nn.Linear(hidden_size, hidden_size)
        nn.init.eye_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.affine(hidden)

    @torch.no_grad()
    def read(self, hidden: Tensor, unembed_weight: Tensor, *, bias: Tensor | None = None,
             norm: Callable[[Tensor], Tensor] | None = None, top_k: int = 5,
             target_ids: Tensor | None = None) -> LensReadout:
        self.eval()
        logits = _project(self(hidden), unembed_weight, bias=bias, norm=norm)
        return _readout(logits, top_k=top_k, target_ids=target_ids,
                        source="tuned_lens:calibrated_prediction_not_reasoning_transcript")


def fit_tuned_lens(hidden_train: Tensor, final_logits_train: Tensor,
                   unembed_weight: Tensor, *, steps: int = 120, lr: float = 1e-2,
                   regularization: float = 1e-4,
                   norm: Callable[[Tensor], Tensor] | None = None,
                   seed: int = 0) -> tuple[TunedLens, dict[str, float]]:
    """Train an affine translator on paired intermediate hidden states and final logits.

    Input tensors: hidden_train [N,d], final_logits_train [N,vocab].
    Returns TRAINING loss; assess generalization with separate held-out data.
    No model weights are trained or modified.
    """
    if hidden_train.ndim != 2 or final_logits_train.ndim != 2 or hidden_train.shape[0] < 2:
        raise ValueError("Expected training data [N,d] and final logits [N,vocab], N>=2")
    if final_logits_train.shape != (hidden_train.shape[0], unembed_weight.shape[0]):
        raise ValueError("Train labels/hidden batch or vocabulary mismatch")
    if steps < 1 or lr <= 0 or regularization < 0:
        raise ValueError("Invalid lens optimization parameters")
    if not torch.isfinite(hidden_train).all() or not torch.isfinite(final_logits_train).all():
        raise ValueError("Nonfinite inputs")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        lens = TunedLens(hidden_train.shape[1]).to(hidden_train.device)
    h = hidden_train.detach().float()
    target = final_logits_train.detach().float().softmax(-1)
    w = unembed_weight.detach().to(h.device).float()
    optimizer = torch.optim.Adam(lens.parameters(), lr=lr)
    eye = torch.eye(h.shape[-1], device=h.device)
    for _ in range(steps):
        output = lens(h)
        if norm is not None:
            output = norm(output)
        predicted = F.linear(output, w)
        loss = F.kl_div(F.log_softmax(predicted, -1), target, reduction="batchmean")
        loss = loss + regularization * (lens.affine.weight - eye).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    lens.eval()
    with torch.no_grad():
        read = lens(h)
        if norm is not None:
            read = norm(read)
        final_kl = float(F.kl_div(F.log_softmax(F.linear(read, w), -1), target,
                                  reduction="batchmean"))
    return lens, {"training_KL": final_kl, "steps": float(steps)}


def decode_states(hidden_by_stage: Sequence[Tensor], unembed_weight: Tensor,
                  *, lens_by_stage: Sequence[TunedLens] | None = None,
                  norm: Callable[[Tensor], Tensor] | None = None,
                  target_ids: Tensor | None = None,
                  top_k: int = 5) -> list[LensReadout]:
    """Decode a sequence of named stages/recurrence iterations; one result per stage."""
    if lens_by_stage is not None and len(hidden_by_stage) != len(lens_by_stage):
        raise ValueError("Need one tuned translator per stage")
    result: list[LensReadout] = []
    for i, hidden in enumerate(hidden_by_stage):
        kwargs = dict(norm=norm, target_ids=target_ids, top_k=top_k)
        read = logit_lens(hidden, unembed_weight, **kwargs) if lens_by_stage is None \
            else lens_by_stage[i].read(hidden, unembed_weight, **kwargs)
        result.append(read)
    return result


def trace_token_ranks(readouts: Sequence[LensReadout]) -> list:
    """Return target-token rank per stage when targets were supplied."""
    if any(r.target_rank is None for r in readouts):
        raise ValueError("All readouts must include target_ids")
    return [r.target_rank.detach().cpu().tolist() for r in readouts]

@torch.no_grad()
def coda_lens(hidden: Tensor, coda_fn: Callable[[Tensor], Tensor],
              unembed_weight: Tensor, *, norm: Callable[[Tensor], Tensor] | None = None,
              top_k: int = 5, target_ids: Tensor | None = None) -> LensReadout:
    """Architecture-aware readout: apply the caller's own Coda/suffix network.

    Generic interface for recurrent networks such as Huginn; NOT a validated
    reproduction of the Huginn-specific published Coda Lens. `coda_fn` must
    match the trained model's actual architecture, attention masks and state.
    """
    suffix_hidden = coda_fn(hidden)
    if not isinstance(suffix_hidden, Tensor) or suffix_hidden.shape != hidden.shape:
        raise ValueError("coda_fn must return a tensor with same shape as hidden")
    return _readout(_project(suffix_hidden, unembed_weight, norm=norm),
                    top_k=top_k, target_ids=target_ids,
                    source="coda_like_suffix_readout_not_reasoning_transcript")
