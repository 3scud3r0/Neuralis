import math
from dataclasses import dataclass

import torch
from torch import nn

from neuralis.datasets.synthetic import VOCABULARY
from neuralis.core.schemas import ArchitectureConfig
from neuralis.runtime.devices import module_device


class MLP(nn.Module):
    def __init__(self, output: int, width: int):
        super().__init__()
        self.input = nn.Linear(2, width)
        self.activation1 = nn.Tanh()
        self.hidden = nn.Linear(width, width)
        self.activation2 = nn.Tanh()
        self.output = nn.Linear(width, output)

    def forward(self, inputs):
        return self.output(self.activation2(self.hidden(self.activation1(self.input(inputs)))))


class ShapeCNN(nn.Module):
    def __init__(self, channels=(12, 24), hidden_width=48):
        super().__init__()
        self.conv1 = nn.Conv2d(1, channels[0], 3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.hidden = nn.Linear(channels[1] * 4 * 4, hidden_width)
        self.relu3 = nn.ReLU()
        self.output = nn.Linear(hidden_width, 3)

    def forward(self, inputs):
        x = self.pool1(self.relu1(self.conv1(inputs)))
        x = self.pool2(self.relu2(self.conv2(x)))
        return self.output(self.relu3(self.hidden(self.flatten(x))))


class CausalAttention(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_width = width // heads
        self.qkv = nn.Linear(width, width * 3)
        self.projection = nn.Linear(width, width)
        self.attention_probs = nn.Identity()

    def forward(self, inputs):
        batch, length, width = inputs.shape
        qkv = self.qkv(inputs).reshape(batch, length, 3, self.heads, self.head_width)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_width)
        mask = torch.ones(length, length, dtype=torch.bool, device=inputs.device).triu(1)
        probabilities = scores.masked_fill(mask, float("-inf")).softmax(-1)
        probabilities = self.attention_probs(probabilities)
        attended = (probabilities @ v).transpose(1, 2).reshape(batch, length, width)
        return self.projection(attended)


class TransformerBlock(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = CausalAttention(width, heads)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, width * 3), nn.GELU(), nn.Linear(width * 3, width))

    def forward(self, inputs):
        residual = inputs + self.attention(self.norm1(inputs))
        return residual + self.mlp(self.norm2(residual))


class TinyTransformer(nn.Module):
    def __init__(self, vocabulary: int = len(VOCABULARY), width: int = 48, depth: int = 2, heads=4, context=32):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary, width)
        self.position = nn.Embedding(context, width)
        self.blocks = nn.ModuleList([TransformerBlock(width, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, vocabulary)

    def forward(self, tokens):
        if tokens.shape[1] > self.position.num_embeddings:
            raise ValueError(f"Transformer context is limited to {self.position.num_embeddings} tokens")
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        residual = self.embedding(tokens) + self.position(positions)
        for block in self.blocks:
            residual = block(residual)
        return self.output(self.norm(residual))


@dataclass(frozen=True)
class ModelDefinition:
    id: str
    name: str
    family: str
    dataset: str
    description: str
    input_shape: tuple[int, ...]
    output_classes: int


DEFINITIONS = {
    "mlp-xor": ModelDefinition("mlp-xor", "XOR multilayer perceptron", "mlp", "xor",
                                "Two hidden tanh layers on a noisy XOR truth table", (2,), 2),
    "mlp-spiral": ModelDefinition("mlp-spiral", "Spiral classifier", "mlp", "spiral",
                                   "Three-class nonlinear spiral classification", (2,), 3),
    "cnn-shapes": ModelDefinition("cnn-shapes", "Geometric vision network", "cnn", "shapes",
                                   "Two convolutional blocks trained on 16×16 geometric shapes", (1, 16, 16), 3),
    "tiny-transformer": ModelDefinition("tiny-transformer", "Tiny causal transformer", "transformer", "text",
                                         "Two blocks, four heads, character-level next-token prediction", (32,), len(VOCABULARY)),
}


def build_model(model_id: str, seed: int = 42, architecture: ArchitectureConfig | None = None) -> nn.Module:
    # fork_rng restores global state: creating an inspection model does not perturb jobs.
    architecture = architecture or ArchitectureConfig()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if model_id == "mlp-xor":
            if architecture.hidden_widths:
                return configurable_mlp(2, architecture)
            return MLP(2, 24)
        if model_id == "mlp-spiral":
            if architecture.hidden_widths:
                return configurable_mlp(3, architecture)
            return MLP(3, 64)
        if model_id == "cnn-shapes":
            return ShapeCNN(architecture.convolution_channels, architecture.hidden_widths[0] if architecture.hidden_widths else 48)
        if model_id == "tiny-transformer":
            return TinyTransformer(width=architecture.transformer_width, depth=architecture.transformer_depth,
                                   heads=architecture.transformer_heads, context=architecture.context_length)
    raise ValueError(f"Unsupported architecture: {model_id}")


def configurable_mlp(output_classes, architecture):
    activations = {"tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}
    widths = [2, *architecture.hidden_widths, output_classes]
    layers = []
    for index, (source, target) in enumerate(zip(widths, widths[1:])):
        layers.append(nn.Linear(source, target))
        if index < len(widths) - 2:
            layers.append(activations[architecture.activation]())
    return nn.Sequential(*layers)


def model_graph(model: nn.Module, sample: torch.Tensor) -> dict:
    nodes, edges, observed, handles = [], [], [], []
    for name, module in model.named_modules():
        if not name:
            continue
        parameters = sum(parameter.numel() for parameter in module.parameters(recurse=False))
        nodes.append({"id": name, "label": name, "type": type(module).__name__,
                      "parameters": parameters, "parent": name.rsplit(".", 1)[0] if "." in name else None,
                      "leaf": not any(module.children()), "instrumentable": True})
        if not any(module.children()):
            def hook(mod, inputs, output, module_name=name):
                shape = list(output.shape) if isinstance(output, torch.Tensor) else None
                observed.append({"module": module_name, "shape": shape})
            handles.append(module.register_forward_hook(hook))
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            model(sample.to(module_device(model)))
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    # Sequence is explicitly execution order, not a falsely complete computational graph.
    for source, target in zip(observed, observed[1:]):
        edges.append({"source": source["module"], "target": target["module"], "kind": "observed-order"})
    shape_lookup = {item["module"]: item["shape"] for item in observed}
    for node in nodes:
        node["output_shape"] = shape_lookup.get(node["id"])
    return {"nodes": nodes, "edges": edges, "coverage": "module hierarchy and observed leaf execution order",
            "parameters": sum(parameter.numel() for parameter in model.parameters()), "observations": observed}
