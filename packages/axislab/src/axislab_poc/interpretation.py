"""Hypothesis labeling with provenance and descriptive feature statistics.

Human-readable descriptions are hypotheses, not an automatic faithful translation
of hidden computation, and they must be independently validated.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import torch
from torch import Tensor


@dataclass(frozen=True)
class FeatureHypothesis:
    feature_index: int
    description: str
    provenance: str
    evidence: str = ""

    def __post_init__(self):
        if self.feature_index < 0 or not self.description.strip():
            raise ValueError("Feature index and description required")
        if self.provenance not in ("human_label", "heuristic_guess", "heldout_validated"):
            raise ValueError("Explicit provenance required")


def inspect_sae_features(sae: Any, activations: Tensor,
                         *, labels: dict[int, FeatureHypothesis] | None = None,
                         top_k: int = 8) -> list[dict]:
    """Rank SAE features by mean activation with activation density and labels."""
    with torch.no_grad():
        z = sae.encode(activations).detach().float()
    if z.ndim < 2:
        raise ValueError("Expected batch/sequence axis and last feature axis")
    flat = z.reshape(-1, z.shape[-1]); means = flat.abs().mean(0)
    if not 1 <= top_k <= z.shape[-1]:
        raise ValueError("Invalid top_k")
    ranking = means.topk(top_k).indices.tolist()
    out = []
    for idx in ranking:
        lab = (labels or {}).get(idx)
        if lab is not None and (lab.feature_index != idx):
            raise ValueError("Incorrect label mapping")
        out.append({"feature_index": int(idx), "mean_absolute_activation": float(means[idx]),
                    "density": float((flat[:, idx] != 0).float().mean()),
                    "label": None if lab is None else lab.description,
                    "provenance": "unlabeled" if lab is None else lab.provenance,
                    "evidence": "" if lab is None else lab.evidence})
    return out


def inspect_axis(axis: Any, activations: Tensor, *, description: str,
                 provenance: str = "heuristic_guess") -> dict:
    """Score a known axis with an explicitly supplied tentative human description."""
    scores = axis.score(activations).detach().float()
    if not torch.isfinite(scores).all():
        raise ValueError("Non-finite axis scores")
    if provenance not in ("human_label", "heuristic_guess", "heldout_validated"):
        raise ValueError("Invalid provenance")
    return {"description": description, "provenance": provenance,
            "mean_score": float(scores.mean()), "std_score": float(scores.std(unbiased=False)),
            "warning": "Concept description is an interpretation hypothesis, not a transcript"}


def vocabulary_partition(token_ids: Tensor, *, visible_vocab_size: int,
                         latent_vocab_size: int = 0) -> list:
    """Label model-owned dual-vocabulary token IDs without inventing visible text."""
    if visible_vocab_size <= 0 or latent_vocab_size < 0:
        raise ValueError("Invalid vocabulary sizes")
    t = torch.as_tensor(token_ids, dtype=torch.long)
    labels = []
    for x in t.flatten().tolist():
        if x < 0:
            raise ValueError("Negative token ID")
        if x < visible_vocab_size:
            labels.append({"id": x, "kind": "visible", "base_id": x})
        elif x < visible_vocab_size + latent_vocab_size:
            labels.append({"id": x, "kind": "latent", "base_id": x - visible_vocab_size})
        else:
            labels.append({"id": x, "kind": "special_or_unknown", "base_id": None})
    return labels
