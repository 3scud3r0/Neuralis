"""Representation engineering: contrastive axes, PCA, probes and validation.

Independent implementations of standard linear algebra; no external project code copied.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor
from .core import Axis, fit_axis


def fit_paired_axis(positive: Tensor, negative: Tensor, *, center: bool = True) -> Axis:
    """Recover a mean paired contrast (positive[i] and negative[i] match).

    Pair matching is established by the caller; this function does not infer it.
    """
    if positive.ndim != 2 or negative.shape != positive.shape or positive.shape[0] < 2:
        raise ValueError("Expected two matching [N,D] matrices with N>=2")
    if not torch.isfinite(positive).all() or not torch.isfinite(negative).all():
        raise ValueError("Non-finite activations")
    p = positive.detach().double().cpu()
    n = negative.detach().double().cpu()
    diff = (p - n).mean(0)
    norm = diff.norm()
    if norm <= 1e-12:
        raise ValueError("Zero paired contrast")
    return Axis(direction=(diff / norm).float(),
                control_mean=n.mean(0).float() if center else torch.zeros(p.shape[1]),
                removed_components=0)


def fit_pca_axis(positive: Tensor, negative: Tensor, *, n_components: int = 1) -> Axis:
    """RepE-inspired PCA of paired differences; orientation follows mean difference.

    PCA directions can be poor if an unrelated high-variance feature dominates;
    evaluate on held-out data before applying any intervention.
    """
    if positive.ndim != 2 or positive.shape != negative.shape or positive.shape[0] < 2:
        raise ValueError("Expected two matching [N,D] matrices")
    if n_components != 1:
        raise ValueError("fit_pca_axis returns one principal direction; use pca_subspace for k>1")
    if not torch.isfinite(positive).all() or not torch.isfinite(negative).all():
        raise ValueError("Non-finite activations")
    diffs = (positive - negative).detach().double().cpu()
    _, _, vh = torch.linalg.svd(diffs, full_matrices=False)
    vector = vh[0]
    if torch.dot(vector, diffs.mean(0)) < 0:
        vector = -vector
    return Axis(vector.float(), negative.detach().double().cpu().mean(0).float(), 0)


def fit_ridge_probe(positive: Tensor, negative: Tensor, *, ridge: float = 1.0) -> Axis:
    """Fit a regularized linear discrimination direction, not a probability model."""
    if positive.ndim != 2 or negative.ndim != 2 or positive.shape[1] != negative.shape[1]:
        raise ValueError("Expected matrices with matching feature widths")
    if min(positive.shape[0], negative.shape[0]) < 2 or ridge <= 0:
        raise ValueError("Need >=2 rows per class and positive regularization")
    X = torch.cat([positive, negative]).detach().double().cpu()
    if not torch.isfinite(X).all():
        raise ValueError("Non-finite activations")
    y = torch.cat([torch.ones(len(positive)), -torch.ones(len(negative))]).double()
    mean = X.mean(0)
    xc = X - mean
    # Dual ridge is efficient if examples << hidden dimension.
    if X.shape[0] < X.shape[1]:
        w = xc.T @ torch.linalg.solve(xc @ xc.T + ridge * torch.eye(len(X)), y)
    else:
        w = torch.linalg.solve(xc.T @ xc + ridge * torch.eye(X.shape[1]), xc.T @ y)
    if w.norm() <= 1e-12:
        raise ValueError("Degenerate ridge probe")
    return Axis((w / w.norm()).float(), negative.detach().double().cpu().mean(0).float(), 0)


def roc_auc(scores: Tensor, labels: Tensor) -> float:
    """AUC with average ranks for ties, calculated without scikit-learn."""
    x = scores.detach().flatten().double().cpu()
    y = labels.detach().flatten().long().cpu()
    if x.numel() != y.numel() or not x.numel() or not torch.isfinite(x).all():
        raise ValueError("Invalid AUC input")
    if not torch.all((y == 0) | (y == 1)):
        raise ValueError("Binary labels must be 0 or 1")
    n_pos = int(y.sum().item())
    n_neg = len(y) - n_pos
    if not n_pos or not n_neg:
        raise ValueError("AUC requires both classes")
    order = torch.argsort(x, stable=True)
    sorted_scores = x[order]
    ranks = torch.arange(1, len(x) + 1, dtype=torch.float64)
    _, inverse, counts = torch.unique_consecutive(sorted_scores, return_inverse=True, return_counts=True)
    starts = torch.cumsum(counts, 0) - counts
    average_ranks = (2 * starts + counts + 1).double() / 2
    ranks[order] = average_ranks[inverse]
    return float(((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)).item())


def evaluate_axis(axis: Axis, positive: Tensor, negative: Tensor) -> dict[str, float]:
    """Evaluate separation of held-out activation sets; no fitting occurs here."""
    if positive.ndim != 2 or negative.ndim != 2:
        raise ValueError("Expected [N,D] held-out activations")
    a = axis.score(positive).float().cpu().flatten()
    b = axis.score(negative).float().cpu().flatten()
    return {
        "auc": roc_auc(torch.cat([a,b]), torch.cat([torch.ones(len(a)),torch.zeros(len(b))])),
        "positive_mean": float(a.mean()),
        "negative_mean": float(b.mean()),
        "mean_gap": float(a.mean()-b.mean()),
    }


@dataclass
class AxisBank:
    """Named directions, with Gram/cosine diagnostics and composition."""
    axes: Mapping[str, Axis]

    def __post_init__(self):
        if not self.axes:
            raise ValueError("AxisBank cannot be empty")
        widths = {a.direction.numel() for a in self.axes.values()}
        if len(widths) != 1:
            raise ValueError("All axes must have the same width")

    def cosine_matrix(self) -> Tensor:
        a = torch.stack([v.direction.float().cpu() for v in self.axes.values()])
        a = torch.nn.functional.normalize(a, dim=-1)
        return a @ a.T

    def compose(self, weights: Mapping[str, float], *, orthogonalize: bool = False) -> Tensor:
        """Weighted vector, optionally remove overlap between selected basis vectors."""
        unknown = set(weights) - set(self.axes)
        if unknown:
            raise KeyError(f"Unknown axes: {sorted(unknown)}")
        if not weights:
            raise ValueError("No axes selected")
        vs = torch.stack([self.axes[name].direction.float().cpu() for name in weights])
        coeff = torch.tensor([weights[name] for name in weights], dtype=torch.float32)
        if orthogonalize:
            # QR works with dependent directions too; the rank check catches instability.
            basis = []
            for v in vs:
                w = v.clone()
                for b in basis:
                    w = w - (w @ b) * b
                if w.norm() > 1e-6:
                    basis.append(w / w.norm())
                else:
                    basis.append(torch.zeros_like(w))
            vs = torch.stack(basis)
        return coeff @ vs


def pca_subspace(activations: Tensor, k: int, *, center: bool = True) -> Tensor:
    """Orthonormal feature-space PCA basis [k,d], from N examples.

    Does not assign semantics to components; evaluate their causal role separately.
    """
    if activations.ndim != 2 or not 1 <= k <= min(activations.shape):
        raise ValueError("Expected [N,D] and 1 <= k <= min(N,D)")
    if not torch.isfinite(activations).all():
        raise ValueError("Non-finite activations")
    x = activations.detach().double().cpu()
    if center:
        x = x - x.mean(0)
    _, _, vh = torch.linalg.svd(x, full_matrices=False)
    return vh[:k].float()
