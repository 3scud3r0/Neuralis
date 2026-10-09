"""A bounded experimental sparse autoencoder; feature semantics are not inferred."""

import torch
from torch import nn


class SparseAutoencoder(nn.Module):
    def __init__(self, input_width: int, features: int, seed: int = 42):
        super().__init__()
        if input_width < 1 or input_width > 4096 or features < 1 or features > 8192:
            raise ValueError("SAE dimensions exceed the bounded experimental profile")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.encoder = nn.Linear(input_width, features)
            self.decoder = nn.Linear(features, input_width)
        self.normalize_decoder()

    def normalize_decoder(self):
        with torch.no_grad():
            self.decoder.weight.div_(self.decoder.weight.norm(dim=0, keepdim=True).clamp_min(1e-8))

    def forward(self, inputs):
        features = self.encoder(inputs).relu()
        return self.decoder(features), features


def train_sae(activations: torch.Tensor, features: int = 64, steps: int = 100, l1_coefficient: float = 0.005, seed: int = 42):
    if activations.ndim != 2 or len(activations) < 2 or not torch.isfinite(activations).all():
        raise ValueError("SAE expects finite [observations,channels] activations")
    if not 1 <= steps <= 2000 or len(activations) > 10000:
        raise ValueError("Experimental SAE work budget exceeded")
    model = SparseAutoencoder(activations.shape[-1], features, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    history = []
    generator = torch.Generator().manual_seed(seed)
    normalized = activations.detach().float()
    mean = normalized.mean(0)
    scale = normalized.std(0, unbiased=False).clamp_min(1e-5)
    normalized = (normalized - mean) / scale
    split = max(1, int(len(normalized) * 0.8))
    training, validation = normalized[:split], normalized[split:]
    for step in range(steps):
        indices = torch.randint(len(training), (min(128, len(training)),), generator=generator)
        inputs = training[indices]
        reconstruction, encoded = model(inputs)
        reconstruction_loss = (reconstruction - inputs).square().mean()
        penalty = encoded.abs().mean()
        loss = reconstruction_loss + l1_coefficient * penalty
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        model.normalize_decoder()
        if step % 10 == 0 or step == steps - 1:
            history.append({"step": step + 1, "loss": float(loss.detach()),
                            "reconstruction_loss": float(reconstruction_loss.detach()), "l1": float(penalty.detach())})
    with torch.no_grad():
        reconstructed, encoded = model(validation)
        metrics = {"validation_mse": float((reconstructed - validation).square().mean()),
                   "active_feature_fraction": float((encoded > 0).float().mean()),
                   "dead_features": int((encoded.sum(0) == 0).sum()), "features": features,
                   "training_observations": len(training), "validation_observations": len(validation)}
    return model, {"history": history, "metrics": metrics, "mean": mean.tolist(), "scale": scale.tolist(),
                   "interpretation": "Experimental dictionary; feature semantics require independent validation."}
