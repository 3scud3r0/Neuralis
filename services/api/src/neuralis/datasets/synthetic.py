from dataclasses import dataclass
import math

import torch


CORPUS = (
    "neural networks learn patterns from examples. attention connects tokens. "
    "research requires evidence and careful experiments. a model transforms inputs into outputs. "
    "weights change during training. gradients describe local sensitivity. "
    "a small experiment can reveal a useful pattern. compare the baseline with an intervention. "
    "the cat sat on the mat. the bird flew over the tree. the scientist measured the signal. "
) * 12
VOCABULARY = sorted(set(CORPUS))
TOKEN_TO_ID = {character: index for index, character in enumerate(VOCABULARY)}


@dataclass(frozen=True)
class DataSplit:
    inputs: torch.Tensor
    targets: torch.Tensor

    def __len__(self):
        return self.inputs.shape[0]

    def batch(self, indices: torch.Tensor):
        return self.inputs[indices], self.targets[indices]


@dataclass(frozen=True)
class DatasetBundle:
    id: str
    train: DataSplit
    validation: DataSplit
    labels: tuple[str, ...]
    task: str
    description: str


def xor_data(seed: int) -> DatasetBundle:
    # Exact truth-table validation, noisy repeated training observations.
    generator = torch.Generator().manual_seed(seed)
    base = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    targets = torch.tensor([0, 1, 1, 0], dtype=torch.long)
    inputs = base.repeat(128, 1)
    inputs = inputs + torch.randn(inputs.shape, generator=generator) * 0.035
    return DatasetBundle("xor", DataSplit(inputs, targets.repeat(128)), DataSplit(base, targets),
                         ("false", "true"), "classification", "Noisy XOR; validation is the exact truth table.")


def spiral_split(seed: int, count: int) -> DataSplit:
    generator = torch.Generator().manual_seed(seed)
    points, labels = [], []
    for label in range(3):
        radius = torch.rand(count, generator=generator).sqrt()
        angle = radius * 4.8 + label * (2 * math.pi / 3)
        angle += torch.randn(count, generator=generator) * 0.12
        points.append(torch.stack([radius * angle.sin(), radius * angle.cos()], dim=1))
        labels.append(torch.full((count,), label, dtype=torch.long))
    return DataSplit(torch.cat(points), torch.cat(labels))


def spiral_data(seed: int) -> DatasetBundle:
    return DatasetBundle("spiral", spiral_split(seed, 256), spiral_split(seed + 1, 128),
                         ("amber", "cyan", "violet"), "classification", "Three interleaved spirals with independent splits.")


def shapes_split(seed: int, count: int) -> DataSplit:
    generator = torch.Generator().manual_seed(seed)
    images = torch.randn((count, 1, 16, 16), generator=generator) * 0.04
    targets = torch.arange(count) % 3
    grid_y, grid_x = torch.meshgrid(torch.arange(16), torch.arange(16), indexing="ij")
    for index in range(count):
        cx = int(torch.randint(5, 11, (1,), generator=generator))
        cy = int(torch.randint(5, 11, (1,), generator=generator))
        size = int(torch.randint(2, 5, (1,), generator=generator))
        x, y = grid_x - cx, grid_y - cy
        if targets[index] == 0:
            mask = x.square() + y.square() <= size ** 2
        elif targets[index] == 1:
            mask = torch.maximum(x.abs(), y.abs()) <= size
        else:
            mask = (y >= -size) & (y <= size) & (x.abs() <= (y + size) / 2)
        images[index, 0, mask] = 1
    return DataSplit(images.clamp(0, 1), targets.long())


def shapes_data(seed: int) -> DatasetBundle:
    return DatasetBundle("shapes", shapes_split(seed, 768), shapes_split(seed + 1, 192),
                         ("circle", "square", "triangle"), "classification", "Translated geometric shapes, generated locally.")


def text_split(text: str, stride: int = 3, context: int = 32) -> DataSplit:
    tokens = torch.tensor([TOKEN_TO_ID[character] for character in text], dtype=torch.long)
    starts = range(0, len(tokens) - context - 1, stride)
    return DataSplit(torch.stack([tokens[start:start + context] for start in starts]),
                     torch.stack([tokens[start + 1:start + context + 1] for start in starts]))


def text_data(seed: int, context: int = 32) -> DatasetBundle:
    # Contiguous split prevents windows from crossing train/validation boundaries.
    split = int(len(CORPUS) * 0.8)
    return DatasetBundle("text", text_split(CORPUS[:split], context=context), text_split(CORPUS[split:], context=context),
                         tuple(VOCABULARY), "language-modeling", "Character-level local corpus; repeated demonstration text.")


FACTORIES = {"mlp-xor": xor_data, "mlp-spiral": spiral_data, "cnn-shapes": shapes_data,
             "tiny-transformer": text_data}


def dataset_for(model_id: str, seed: int, context: int = 32) -> DatasetBundle:
    try:
        if model_id == "tiny-transformer":
            return text_data(seed, context)
        return FACTORIES[model_id](seed)
    except KeyError as error:
        raise ValueError(f"Unsupported model: {model_id}") from error
