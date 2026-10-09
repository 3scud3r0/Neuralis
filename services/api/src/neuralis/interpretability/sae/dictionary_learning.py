"""Trainable TopK dictionaries with user-authored labels and measured fit quality."""

from collections.abc import Mapping
import math
import torch
from torch import nn


class TopKSAE(nn.Module):
    def __init__(self, input_width: int, features: int, k: int, seed: int = 42):
        super().__init__()
        if not 1 <= input_width <= 4096 or not 1 <= features <= 8192:
            raise ValueError("Dictionary dimensions exceed the CPU profile")
        if input_width * features > 8_000_000 or not 1 <= k <= features:
            raise ValueError("Dictionary parameter budget exceeded or invalid TopK")
        self.input_width, self.features, self.k = input_width, features, k
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.encoder = nn.Linear(input_width, features)
            self.decoder = nn.Linear(features, input_width, bias=False)
        self.register_buffer("center", torch.zeros(input_width))
        self.user_labels: dict[int, str] = {}
        with torch.no_grad():
            self.encoder.bias.zero_()
        self.normalize_decoder()

    def normalize_decoder(self):
        with torch.no_grad():
            self.decoder.weight.div_(self.decoder.weight.norm(dim=0, keepdim=True).clamp_min(1e-8))

    def encode(self, inputs):
        if inputs.shape[-1] != self.input_width:
            raise ValueError("Activation width does not match dictionary")
        dense = self.encoder(inputs - self.center).relu()
        values, indices = dense.topk(self.k, dim=-1)
        return torch.zeros_like(dense).scatter(-1, indices, values)

    def decode(self, encoded):
        return self.decoder(encoded) + self.center

    def forward(self, inputs):
        encoded = self.encode(inputs)
        return self.decode(encoded), encoded

    def direction(self, feature: int):
        self._validate_feature(feature)
        return self.decoder.weight[:, feature].detach().cpu().clone()

    def _validate_feature(self, feature):
        if isinstance(feature, bool) or not isinstance(feature, int) or not 0 <= feature < self.features:
            raise ValueError("Feature index is outside this dictionary")

    def set_user_labels(self, labels: Mapping[int, str]):
        validated = {}
        for feature, label in labels.items():
            self._validate_feature(feature)
            if not isinstance(label, str) or not label.strip() or len(label) > 128:
                raise ValueError("User labels must contain 1..128 characters")
            validated[feature] = label.strip()
        self.user_labels = validated

    def feature_catalog(self):
        return [{"id": index, "name": f"Feature {index}",
                 "user_label": self.user_labels.get(index),
                 "label_source": "user" if index in self.user_labels else None}
                for index in range(self.features)]


def fit_dictionary(activations: torch.Tensor, *, features: int = 64, k: int = 8,
                   steps: int = 100, batch_size: int = 128, learning_rate: float = 0.005,
                   seed: int = 42, user_labels=None):
    """Fit a CPU dictionary; held-out metrics use the same observed activation units.

    Flatten token/batch dimensions before passing [observations,channels]. A
    deterministic shuffled 80/20 partition prevents contiguous sequence order
    from choosing the validation split. No claim of semantic identification is
    produced by this routine.
    """
    if activations.ndim != 2 or not 2 <= len(activations) <= 10_000:
        raise ValueError("Expected 2..10000 observations in a [observations,channels] tensor")
    if activations.numel() > 8_000_000 or not torch.isfinite(activations).all():
        raise ValueError("Activation data must be finite and within the CPU budget")
    if not 1 <= steps <= 2000 or not 1 <= batch_size <= 512:
        raise ValueError("Training step or batch budget exceeded")
    if not math.isfinite(learning_rate) or not 0 < learning_rate <= 0.1:
        raise ValueError("Learning rate must be in (0,0.1]")
    data = activations.detach().to(device="cpu", dtype=torch.float32).clone()
    if not torch.isfinite(data).all():
        raise ValueError("Activation values exceed finite float32 training range")
    model = TopKSAE(data.shape[-1], features, k, seed)
    if user_labels is not None:
        model.set_user_labels(user_labels)
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(data), generator=generator)
    split = min(len(data) - 1, max(1, int(len(data) * 0.8)))
    if data.shape[-1] * features * steps * min(batch_size, split) > 2_000_000_000:
        raise ValueError("Dictionary training exceeds the bounded CPU work budget")
    training, validation = data[order[:split]], data[order[split:]]
    with torch.no_grad():
        model.center.copy_(training.mean(0))
        initial_reconstruction, _ = model(validation)
        initial_mse = float((initial_reconstruction - validation).square().mean())
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    for step in range(steps):
        indices = torch.randint(len(training), (min(batch_size, len(training)),), generator=generator)
        inputs = training[indices]
        reconstructed, encoded = model(inputs)
        loss = (reconstructed - inputs).square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Dictionary optimization produced nonfinite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
        model.normalize_decoder()
        if step == 0 or (step + 1) % 10 == 0 or step == steps - 1:
            history.append({"step": step + 1, "training_mse": float(loss.detach()),
                            "mean_active_features": float((encoded > 0).float().sum(-1).mean())})
    model.eval()
    with torch.no_grad():
        reconstructed, encoded = model(validation)
        mse = float((reconstructed - validation).square().mean())
        centered_variance = float((validation - model.center).square().mean())
        variance_explained = 1 - mse / centered_variance if centered_variance > 1e-12 else None
        frequency = (encoded > 0).float().mean(0)
        magnitude = encoded.mean(0)
    report = {"history": history,
              "metrics": {"initial_validation_mse": initial_mse, "validation_mse": mse,
                          "variance_explained": variance_explained,
                          "mean_active_features": float((encoded > 0).float().sum(-1).mean()),
                          "dead_features": int((frequency == 0).sum()),
                          "training_observations": len(training),
                          "validation_observations": len(validation)},
              "features": [{**entry, "activation_frequency": float(frequency[entry["id"]]),
                            "mean_activation": float(magnitude[entry["id"]])}
                           for entry in model.feature_catalog()],
              "settings": {"features": features, "k": k, "steps": steps,
                           "seed": seed, "learning_rate": learning_rate},
              "interpretation": "Learned numerical directions; user labels are hypotheses, not discovered semantics."}
    return model, report
