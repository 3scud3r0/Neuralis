from contextlib import contextmanager

import torch

from neuralis.analysis.experiments import prepare_input
from neuralis.runtime.devices import module_device


@contextmanager
def collect_activations(model, module_names):
    modules = dict(model.named_modules())
    values, handles = {}, []
    try:
        for name in module_names:
            if name not in modules or not name:
                raise ValueError(f"Unknown representation module: {name}")
            def hook(module, arguments, output, target=name):
                if not isinstance(output, torch.Tensor):
                    raise ValueError("Representation analysis requires tensor module outputs")
                values[target] = output.detach().cpu()
            handles.append(modules[name].register_forward_hook(hook))
        yield values
    finally:
        for handle in handles:
            handle.remove()


def principal_components(values, dimensions=3):
    if values.ndim != 2 or len(values) < 2 or not torch.isfinite(values).all():
        raise ValueError("PCA expects at least two finite observations in a matrix")
    if dimensions < 1 or dimensions > min(values.shape):
        raise ValueError("Projection dimensionality exceeds the matrix rank bound")
    values = values.float()
    mean = values.mean(0)
    centered = values - mean
    # Exact SVD for bounded demonstrations, avoiding stochastic projection artifacts.
    _, singular_values, components = torch.linalg.svd(centered, full_matrices=False)
    selected = components[:dimensions]
    scores = centered @ selected.T
    variance = singular_values.square() / max(1, len(values) - 1)
    total = variance.sum()
    explained = variance[:dimensions] / total if total > 0 else torch.zeros(dimensions)
    return {"coordinates": scores.tolist(), "components": selected.tolist(), "mean": mean.tolist(),
            "explained_variance": variance[:dimensions].tolist(), "explained_variance_ratio": explained.tolist(),
            "method": "centered exact PCA/SVD", "observations": len(values), "features": values.shape[1]}


def linear_cka(first, second):
    if first.ndim != 2 or second.ndim != 2 or len(first) != len(second) or len(first) < 2:
        raise ValueError("CKA requires two matrices with matching observation counts")
    first = first.float() - first.float().mean(0)
    second = second.float() - second.float().mean(0)
    cross = (first.T @ second).square().sum()
    denominator = (first.T @ first).square().sum().sqrt() * (second.T @ second).square().sum().sqrt()
    if denominator <= 1e-12:
        return {"value": None, "reason": "At least one centered representation has zero variance"}
    return {"value": float((cross / denominator).clamp(0, 1)), "method": "linear centered kernel alignment"}


def compare_representations(model, dataset, request, module_names):
    inputs, targets = prepare_input(request, dataset)
    if len(inputs) < 2:
        raise ValueError("Representation comparison requires at least two examples")
    with collect_activations(model, module_names) as values, torch.no_grad():
        model(inputs.to(module_device(model)))
    flattened = {}
    for name in module_names:
        if name not in values:
            raise ValueError(f"Module {name} was not executed")
        tensor = values[name]
        if tensor.shape[0] != len(inputs):
            raise ValueError("Selected module does not preserve the batch observation axis")
        matrix = tensor.reshape(len(inputs), -1)
        if matrix.shape[1] > 8192:
            stride = (matrix.shape[1] + 8191) // 8192
            matrix = matrix[:, ::stride]
        flattened[name] = matrix
    comparisons = []
    for first_index, first_name in enumerate(module_names):
        for second_name in module_names[first_index + 1:]:
            comparisons.append({"first": first_name, "second": second_name,
                                **linear_cka(flattened[first_name], flattened[second_name])})
    projections = {name: principal_components(matrix, min(3, min(matrix.shape))) for name, matrix in flattened.items()}
    return {"comparisons": comparisons, "projections": projections,
            "labels": targets.tolist() if targets is not None else None,
            "interpretation": "Representation similarity on identical examples; it does not establish equivalent computation.",
            "maximum_features_per_module": 8192}


def attention_details(model, dataset, request, layer=0, head=0, example=0, token=0):
    module_name = f"blocks.{layer}.attention.attention_probs"
    inputs, _ = prepare_input(request, dataset)
    with collect_activations(model, [module_name]) as values, torch.no_grad():
        model(inputs.to(module_device(model)))
    tensor = values[module_name]
    if not 0 <= head < tensor.shape[1] or not 0 <= example < tensor.shape[0] or not 0 <= token < tensor.shape[2]:
        raise ValueError("Attention example, head, or token selection is out of range")
    matrix = tensor[example, head]
    tokens = inputs[example].tolist()
    row = matrix[token]
    top_values, top_indices = row.topk(min(10, len(row)))
    entropy = -(matrix * matrix.clamp_min(1e-12).log()).sum(-1)
    return {"module": module_name, "layer": layer, "head": head, "example": example,
            "token": token, "matrix": matrix.tolist(), "shape": list(matrix.shape),
            "tokens": [{"id": identifier, "label": dataset.labels[identifier]} for identifier in tokens],
            "top_connections": [{"position": int(index), "token": dataset.labels[tokens[index]], "weight": float(value)}
                                for value, index in zip(top_values, top_indices)],
            "entropy_by_query": entropy.tolist(), "causal_mask_maximum": float(matrix.triu(1).max()),
            "interpretation": "Exact attention probabilities from the explicit demonstrator attention implementation."}
