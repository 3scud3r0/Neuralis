from contextlib import contextmanager
import time

import torch
from torch import nn

from neuralis.core.schemas import AttributionRequest, ExperimentRequest, InferenceRequest
from neuralis.runtime.capture import summarize_tensor
from neuralis.runtime.supervisor import loss_for
from neuralis.runtime.devices import module_device


def prepare_input(request: InferenceRequest, dataset):
    split = dataset.train if request.split == "train" else dataset.validation
    if request.inputs is not None:
        inputs = torch.tensor(request.inputs, dtype=split.inputs.dtype)
        if inputs.ndim != split.inputs.ndim or tuple(inputs.shape[1:]) != tuple(split.inputs.shape[1:]):
            raise ValueError(f"Expected batched shape [N,{','.join(map(str,split.inputs.shape[1:]))}]")
        if inputs.shape[0] < 1 or inputs.shape[0] > 64:
            raise ValueError("Inference batch must contain 1 to 64 examples")
        if inputs.is_floating_point() and not torch.isfinite(inputs).all():
            raise ValueError("Inference inputs must be finite")
        if not inputs.is_floating_point() and (inputs.min() < 0 or inputs.max() >= len(dataset.labels)):
            raise ValueError("Token ID outside the registered vocabulary")
        return inputs, None
    if not request.sample_indices:
        raise ValueError("Select at least one sample")
    indices = torch.tensor(request.sample_indices, dtype=torch.long)
    if indices.min() < 0 or indices.max() >= len(split):
        raise ValueError("Dataset sample index is out of range")
    return split.inputs[indices].clone(), split.targets[indices].clone()


def output_summary(logits, targets=None, labels=()):
    probabilities = logits.softmax(-1)
    selected = logits[:, -1] if logits.ndim == 3 else logits
    selected_probabilities = selected.softmax(-1)
    top_count = min(5, selected.shape[-1])
    values, indices = selected_probabilities.topk(top_count, -1)
    result = {"logits": logits.detach().tolist(), "probabilities": probabilities.detach().tolist(),
              "predictions": logits.argmax(-1).detach().tolist(), "shape": list(logits.shape),
              "top_predictions": [[{"index": int(index), "label": labels[index] if labels else str(index),
                                    "probability": float(value)} for value, index in zip(row_values, row_indices)]
                                  for row_values, row_indices in zip(values, indices)]}
    if targets is not None:
        result["loss"] = float(loss_for(logits, targets))
        result["accuracy"] = float((logits.argmax(-1) == targets).float().mean())
    return result


def infer(model, dataset, request: InferenceRequest):
    inputs, targets = prepare_input(request, dataset)
    device = module_device(model)
    inputs = inputs.to(device)
    targets = targets.to(device) if targets is not None else None
    with torch.no_grad():
        logits = model(inputs)
        return {"input": inputs.tolist(), "targets": targets.tolist() if targets is not None else None,
                "output": output_summary(logits, targets, dataset.labels)}


@contextmanager
def intervention_hooks(model, specifications, clean_activations=None):
    modules = dict(model.named_modules())
    handles = []
    clean_activations = clean_activations or {}
    applied = []
    try:
        for specification in specifications:
            if specification.module not in modules or not specification.module:
                raise ValueError(f"Unknown intervention module: {specification.module}")
            def hook(module, inputs, output, spec=specification):
                if not isinstance(output, torch.Tensor):
                    raise ValueError(f"Module {spec.module} has an unsupported non-tensor output")
                replacement = output.clone()
                indices = spec.indices
                if indices is not None and (not indices or min(indices) < 0 or max(indices) >= output.shape[-1]):
                    raise ValueError("Intervention indices select the last tensor axis and must be in range")
                selection = (..., indices) if indices is not None else (...,)
                selected = replacement[selection]
                if spec.operation == "ablate":
                    replacement[selection] = 0
                elif spec.operation == "scale":
                    replacement[selection] = selected * spec.strength
                elif spec.operation == "steer":
                    if spec.vector is None:
                        raise ValueError("Steering requires an explicit vector with recorded provenance")
                    vector = torch.tensor(spec.vector, dtype=output.dtype, device=output.device)
                    if vector.numel() != selected.shape[-1]:
                        raise ValueError(f"Steering vector must have {selected.shape[-1]} elements")
                    replacement[selection] = selected + spec.strength * vector
                else:
                    if spec.module not in clean_activations:
                        raise ValueError("Patching requires a clean input and a captured clean activation")
                    clean = clean_activations[spec.module]
                    if clean.shape != output.shape:
                        raise ValueError("Clean/corrupted activation shapes must match exactly")
                    replacement[selection] = clean[selection]
                applied.append({"module": spec.module, "operation": spec.operation, "shape": list(output.shape),
                                "axis": "last", "indices": indices, "strength": spec.strength,
                                "change_norm": float(torch.linalg.vector_norm((replacement - output).float()))})
                return replacement
            handles.append(modules[specification.module].register_forward_hook(hook))
        yield applied
    finally:
        for handle in handles:
            handle.remove()


def capture_clean(model, inputs, names):
    modules = dict(model.named_modules())
    activations, handles = {}, []
    try:
        for name in names:
            if name not in modules:
                raise ValueError(f"Unknown patch module: {name}")
            def capture(module, arguments, output, target=name):
                if not isinstance(output, torch.Tensor):
                    raise ValueError("Patching supports tensor outputs only")
                activations[target] = output.detach().clone()
            handles.append(modules[name].register_forward_hook(capture))
        with torch.no_grad():
            logits = model(inputs)
        return activations, logits
    finally:
        for handle in handles:
            handle.remove()


def execute_experiment(model, dataset, request: ExperimentRequest):
    start = time.perf_counter()
    inputs, targets = prepare_input(request.input, dataset)
    device = module_device(model)
    inputs = inputs.to(device)
    targets = targets.to(device) if targets is not None else None
    with torch.no_grad():
        baseline = model(inputs)
    patch_names = {spec.module for spec in request.interventions if spec.operation == "patch"}
    clean_activations, clean_logits = {}, None
    if patch_names:
        if request.clean_input is None:
            raise ValueError("Activation patching requires clean_input")
        clean_inputs, _ = prepare_input(request.clean_input, dataset)
        clean_inputs = clean_inputs.to(device)
        clean_activations, clean_logits = capture_clean(model, clean_inputs, patch_names)
    with torch.no_grad(), intervention_hooks(model, request.interventions, clean_activations) as applied:
        changed = model(inputs)
    base_probabilities = baseline.softmax(-1)
    changed_probabilities = changed.softmax(-1)
    # KL(base || changed), computed from log_softmax to avoid probability underflow.
    kl = (base_probabilities * (baseline.log_softmax(-1) - changed.log_softmax(-1))).sum(-1).mean()
    metrics = {"kl_baseline_to_intervention": float(kl),
               "logit_l2": float(torch.linalg.vector_norm(changed - baseline)),
               "mean_probability_change": float((changed_probabilities - base_probabilities).abs().mean()),
               "prediction_change_fraction": float((changed.argmax(-1) != baseline.argmax(-1)).float().mean())}
    if targets is not None:
        metrics["delta_loss"] = float(loss_for(changed, targets) - loss_for(baseline, targets))
    if request.target_class is not None:
        if request.target_class >= baseline.shape[-1]:
            raise ValueError("Target class is out of range")
        metrics["target_logit_change"] = float((changed[..., request.target_class] - baseline[..., request.target_class]).mean())
    if clean_logits is not None:
        corrupted_distance = torch.linalg.vector_norm(baseline - clean_logits)
        patched_distance = torch.linalg.vector_norm(changed - clean_logits)
        metrics["clean_logit_distance_before"] = float(corrupted_distance)
        metrics["clean_logit_distance_after"] = float(patched_distance)
        metrics["patch_recovery"] = float(1 - patched_distance / corrupted_distance) if corrupted_distance > 1e-12 else None
    return {"baseline": output_summary(baseline, targets, dataset.labels),
            "intervention": output_summary(changed, targets, dataset.labels),
            "clean": output_summary(clean_logits, None, dataset.labels) if clean_logits is not None else None,
            "metrics": metrics, "applied": applied, "elapsed_ms": (time.perf_counter() - start) * 1000,
            "interpretation": "Measured effects on this input; attention and attribution alone do not establish causality."}


def attribution(model, dataset, request: AttributionRequest):
    inputs, _ = prepare_input(request.input, dataset)
    inputs = inputs.to(module_device(model))
    if not inputs.is_floating_point():
        raise ValueError("Input attribution requires floating inputs; token IDs need an embedding-specific adapter")
    with torch.no_grad():
        logits = model(inputs)
    target_indices = logits.argmax(-1) if request.target_class is None else torch.full(logits.shape[:-1], request.target_class, dtype=torch.long, device=inputs.device)
    if target_indices.max() >= logits.shape[-1]:
        raise ValueError("Target class is out of range")

    def score(x):
        return model(x).gather(-1, target_indices.unsqueeze(-1)).squeeze(-1)

    baseline = torch.zeros_like(inputs)
    if request.method == "gradient-input":
        variable = inputs.detach().requires_grad_(True)
        gradient = torch.autograd.grad(score(variable).sum(), variable)[0]
        values = gradient * inputs
        completeness = None
    else:
        # Trapezoidal quadrature includes both endpoints and reports completeness residual.
        total_gradient = torch.zeros_like(inputs)
        for index in range(request.integration_steps + 1):
            alpha = index / request.integration_steps
            variable = (baseline + alpha * (inputs - baseline)).detach().requires_grad_(True)
            gradient = torch.autograd.grad(score(variable).sum(), variable)[0]
            total_gradient += gradient * (0.5 if index in (0, request.integration_steps) else 1.)
        values = (inputs - baseline) * total_gradient / request.integration_steps
        with torch.no_grad():
            output_difference = score(inputs) - score(baseline)
        attribution_sum = values.reshape(len(values), -1).sum(-1)
        completeness = {"output_difference": output_difference.tolist(), "attribution_sum": attribution_sum.tolist(),
                        "residual": (attribution_sum - output_difference).tolist()}
    return {"method": request.method, "target_classes": target_indices.tolist(), "baseline": "zero input",
            "integration_steps": request.integration_steps if request.method == "integrated-gradients" else None,
            "attribution": summarize_tensor(values, 16384), "values": values.detach().tolist(),
            "completeness": completeness, "interpretation": "Local/path sensitivity, not a causal circuit proof."}


def circuit_candidates(model, dataset, request: InferenceRequest, target_class=None, limit=32):
    inputs, _ = prepare_input(request, dataset)
    inputs = inputs.to(module_device(model))
    observations, handles = [], []
    try:
        for name, module in model.named_modules():
            if not name or any(module.children()):
                continue
            def capture(mod, arguments, output, module_name=name):
                if isinstance(output, torch.Tensor) and output.requires_grad:
                    output.retain_grad()
                    observations.append((module_name, output))
            handles.append(module.register_forward_hook(capture))
        logits = model(inputs)
        selected = logits[:, -1] if logits.ndim == 3 else logits
        classes = selected.argmax(-1) if target_class is None else torch.full((len(selected),), target_class, dtype=torch.long, device=inputs.device)
        if classes.max() >= selected.shape[-1]:
            raise ValueError("Target class out of range")
        selected.gather(-1, classes[:, None]).sum().backward()
        nodes = []
        for name, output in observations:
            if output.grad is not None:
                product = output.detach() * output.grad.detach()
                nodes.append({"id": name, "score": float(product.abs().mean()),
                              "signed_score": float(product.mean()), "shape": list(output.shape)})
        edges = [{"source": source["id"], "target": target["id"],
                  "score": min(source["score"], target["score"]), "kind": "observed-order-candidate"}
                 for source, target in zip(nodes, nodes[1:])]
        ranked = sorted(nodes, key=lambda node: node["score"], reverse=True)[:limit]
        selected_names = {node["id"] for node in ranked}
        return {"nodes": ranked, "edges": [edge for edge in edges if edge["source"] in selected_names and edge["target"] in selected_names],
                "method": "mean absolute gradient × activation", "target_classes": classes.tolist(),
                "causally_verified": False, "coverage": "Observed leaf execution order; residual dependencies are not fully represented.",
                "interpretation": "Hypotheses for intervention testing; scores do not prove causal edges."}
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)
