"""Compact TopK SAE and an adapter for SAE-Lens-style encode/decode APIs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class TopKSAE(nn.Module):
    """Trainable TopK ReLU sparse autoencoder for activations [...,d_model].

    A teaching/research baseline, not a substitute for SAE Lens's mature trainer.
    """
    def __init__(self, d_model: int, d_sae: int, k: int):
        super().__init__()
        if d_model < 1 or d_sae < 1 or not 1 <= k <= d_sae:
            raise ValueError("Invalid SAE dimensions or k")
        self.d_model, self.d_sae, self.k = d_model, d_sae, k
        self.encoder = nn.Linear(d_model, d_sae)
        self.decoder = nn.Linear(d_sae, d_model)

    def encode(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.d_model:
            raise ValueError("SAE input width mismatch")
        pre = F.relu(self.encoder(x))
        if self.k == self.d_sae:
            return pre
        values, indexes = pre.topk(self.k, dim=-1)
        features = torch.zeros_like(pre)
        return features.scatter(-1, indexes, values)

    def decode(self, features: Tensor) -> Tensor:
        if features.shape[-1] != self.d_sae:
            raise ValueError("SAE feature width mismatch")
        return self.decoder(features)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        feat = self.encode(x)
        return self.decode(feat), feat


@dataclass(frozen=True)
class SAETrainMetrics:
    final_mse: float
    feature_density: float
    steps: int


def train_topk_sae(sae: TopKSAE, activations: Tensor, *, steps: int = 80,
                   batch_size: int = 64, lr: float = 1e-3,
                   seed: int = 0) -> SAETrainMetrics:
    """Small local baseline trainer; full datasets require streaming/distributed training."""
    if activations.ndim != 2 or activations.shape[1] != sae.d_model or len(activations) < 2:
        raise ValueError("Expected [N,d_model] activation matrix")
    if steps < 1 or batch_size < 1 or lr <= 0:
        raise ValueError("Invalid training hyperparameters")
    if not torch.isfinite(activations).all():
        raise ValueError("Non-finite activations")
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    device = next(sae.parameters()).device
    sae.train()
    for _ in range(steps):
        ids = torch.randint(len(activations), (batch_size,), generator=generator)
        batch = activations[ids].to(device)
        reconstruction, _ = sae(batch)
        loss = (batch - reconstruction).square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    sae.eval()
    with torch.no_grad():
        rec, feats = sae(activations.to(device))
        return SAETrainMetrics(float((rec - activations.to(device)).square().mean()),
                               float((feats != 0).float().mean()), steps)


def sae_features(sae: Any, activation: Tensor, *, keep: list[int] | None = None,
                 amplification: float = 1.0) -> tuple[Tensor, Tensor]:
    """Works with local TopKSAE or SAELens SAE using their encode/decode contract.

    Returns (reconstructed_or_modified_activation, feature_activations). A SAE
    reconstruction is not exactly the input; use residual-preservation in interventions.
    """
    features = sae.encode(activation)
    if not isinstance(features, Tensor):
        raise TypeError("sae.encode must return a Tensor")
    edited = features.clone()
    if keep is not None:
        if any(i < 0 or i >= edited.shape[-1] for i in keep):
            raise ValueError("Feature index out of bounds")
        mask = torch.zeros_like(edited)
        mask[..., keep] = 1
        edited = edited * mask
    edited = edited * amplification
    return sae.decode(edited), features


def sae_delta_intervention(sae: Any, activation: Tensor,
                           *, feature_index: int, coefficient: float) -> Tensor:
    """Intervene in feature space and preserve SAE reconstruction residual.

    Result = x + (decode(edited_features) - decode(original_features)).
    Avoids accidentally replacing the original with a lossy SAE reconstruction.
    """
    features = sae.encode(activation)
    if feature_index < 0 or feature_index >= features.shape[-1]:
        raise ValueError("Feature index out of bounds")
    modified = features.clone()
    modified[..., feature_index] = modified[..., feature_index] + coefficient
    return activation + (sae.decode(modified) - sae.decode(features))
