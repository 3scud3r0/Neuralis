"""A convolutional actor-critic with directly inspectable features and logits."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


def initialize(layer: nn.Module, gain: float = np.sqrt(2), bias: float = 0.0):
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.constant_(layer.bias, bias)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, observation_shape: tuple[int, int, int], actions: int):
        super().__init__()
        self.observation_shape = tuple(observation_shape)
        self.actions = actions
        channels, height, width = observation_shape
        self.encoder = nn.Sequential(
            initialize(nn.Conv2d(channels, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            initialize(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            initialize(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            encoded_size = self.encoder(torch.zeros(1, channels, height, width)).shape[1]
        self.trunk = nn.Sequential(initialize(nn.Linear(encoded_size, 512)), nn.ReLU())
        self.actor = initialize(nn.Linear(512, actions), 0.01)
        self.critic = initialize(nn.Linear(512, 1), 1.0)

    def features(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim != 4 or tuple(observations.shape[1:]) != self.observation_shape:
            raise ValueError(f"Expected [batch, {self.observation_shape}], got {tuple(observations.shape)}")
        return self.trunk(self.encoder(observations.float() / 255.0))

    def forward(self, observations: torch.Tensor):
        features = self.features(observations)
        return self.actor(features), self.critic(features).squeeze(-1)

    def value(self, observations: torch.Tensor) -> torch.Tensor:
        return self.forward(observations)[1]

    def act(self, observations: torch.Tensor, actions: torch.Tensor | None = None, deterministic: bool = False):
        logits, values = self.forward(observations)
        distribution = Categorical(logits=logits)
        if actions is None:
            actions = logits.argmax(dim=-1) if deterministic else distribution.sample()
        return actions, distribution.log_prob(actions), distribution.entropy(), values

    def architecture(self) -> dict:
        layers = []
        handles = []
        def hook(name):
            def record(module, inputs, output):
                layers.append({
                    "name": name,
                    "type": type(module).__name__,
                    "inputShape": list(inputs[0].shape[1:]),
                    "outputShape": list(output.shape[1:]),
                    "parameters": sum(parameter.numel() for parameter in module.parameters()),
                })
            return record
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                handles.append(module.register_forward_hook(hook(name)))
        try:
            with torch.no_grad():
                self.forward(torch.zeros(1, *self.observation_shape, device=next(self.parameters()).device))
        finally:
            for handle in handles:
                handle.remove()
        return {"layers": layers, "parameters": sum(p.numel() for p in self.parameters())}

    def inspect(self, observations: torch.Tensor) -> dict:
        """Bounded layer summaries; never export full high-dimensional tensors."""
        records = []
        handles = []
        def hook(name):
            def record(module, inputs, output):
                tensor = output.detach().float()
                records.append({
                    "layer": name,
                    "shape": list(tensor.shape),
                    "mean": tensor.mean().item(),
                    "std": tensor.std(unbiased=False).item(),
                    "min": tensor.min().item(),
                    "max": tensor.max().item(),
                    "zeroFraction": (tensor == 0).float().mean().item(),
                    "sample": tensor.flatten()[:128].cpu().tolist(),
                })
            return record
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.ReLU)):
                handles.append(module.register_forward_hook(hook(name)))
        try:
            with torch.no_grad():
                logits, value = self.forward(observations)
        finally:
            for handle in handles:
                handle.remove()
        return {"layers": records, "logits": logits.cpu().tolist(), "values": value.cpu().tolist()}


def choose_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no usable CUDA device is available")
    if name == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("MPS requested but not available on this machine")
    return torch.device(name)
