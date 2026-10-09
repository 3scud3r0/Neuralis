import math

import numpy as np
from pydantic import BaseModel, Field
import torch

from neuralis.core.schemas import RunConfig
from neuralis.datasets.synthetic import DataSplit
from neuralis.models.registry import build_model
from neuralis.runtime.devices import optimizer_for, resolve_device, scheduled_learning_rate
from neuralis.runtime.supervisor import evaluate, loss_for


class CrossValidationRequest(BaseModel):
    dataset_id: str
    config: RunConfig = Field(default_factory=lambda: RunConfig(steps=100))
    folds: int = Field(default=3, ge=2, le=10)
    standardize: bool = False


def stratified_folds(targets, folds, seed):
    array = np.asarray(targets, dtype=np.int64)
    generator = np.random.default_rng(seed)
    partitions = [[] for _ in range(folds)]
    for label in np.unique(array):
        indices = np.flatnonzero(array == label)
        if len(indices) < folds:
            raise ValueError("Every class must contain at least one observation per fold")
        generator.shuffle(indices)
        for fold, subset in enumerate(np.array_split(indices, folds)):
            partitions[fold].extend(subset.tolist())
    result = [np.array(sorted(partition), dtype=np.int64) for partition in partitions]
    if len(set(np.concatenate(result).tolist())) != len(array):
        raise ValueError("Fold partition did not preserve unique observations")
    return result


def cross_validate(store, request):
    config = request.config
    if config.model_id not in {"mlp-xor", "mlp-spiral", "cnn-shapes"} or config.steps > 500:
        raise ValueError("Cross-validation is limited to classification models and 500 steps per fold")
    metadata = store.metadata(request.dataset_id)
    if metadata["project_id"] != config.project_id or metadata["model_id"] != config.model_id:
        raise ValueError("Cross-validation dataset must match the project and model task")
    if metadata["preprocessing"]["method"] != "none":
        raise ValueError("Cross-validation requires an unprocessed import; standardization must be fitted independently in each fold")
    bundle = store.load(request.dataset_id, config.model_id)
    inputs = torch.cat([bundle.train.inputs, bundle.validation.inputs])
    targets = torch.cat([bundle.train.targets, bundle.validation.targets])
    if len(inputs) > 5000:
        raise ValueError("Cross-validation observation budget is 5,000 examples")
    partitions = stratified_folds(targets.numpy(), request.folds, config.seed)
    reports = []
    device = resolve_device(config.device)
    for fold, held_out in enumerate(partitions):
        training_indices = np.concatenate([partition for index, partition in enumerate(partitions) if index != fold])
        train_x, valid_x = inputs[training_indices].clone(), inputs[held_out].clone()
        if request.standardize:
            mean = train_x.mean(0)
            scale = train_x.std(0, unbiased=False).clamp_min(1e-6)
            train_x = (train_x - mean) / scale
            valid_x = (valid_x - mean) / scale
        training = DataSplit(train_x, targets[training_indices])
        validation = DataSplit(valid_x, targets[held_out])
        model = build_model(config.model_id, config.seed + fold, config.architecture).to(device)
        optimizer = optimizer_for(model, config)
        generator = torch.Generator().manual_seed(config.seed + fold + 1000)
        baseline = evaluate(model, validation)
        for step in range(config.steps):
            indices = torch.randint(len(training), (config.batch_size,), generator=generator)
            x, y = training.batch(indices)
            logits = model(x.to(device))
            loss = loss_for(logits, y.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip or float("inf"))
            for group in optimizer.param_groups:
                group["lr"] = scheduled_learning_rate(config, step)
            optimizer.step()
        final = evaluate(model, validation)
        reports.append({"fold": fold, "seed": config.seed + fold, "training_examples": len(training),
                        "validation_examples": len(validation), "baseline": baseline, "final": final,
                        "held_out_indices": held_out.tolist()})
    losses = [report["final"]["validation_loss"] for report in reports]
    accuracies = [report["final"]["validation_accuracy"] for report in reports]
    return {"folds": reports, "aggregate": {"mean_validation_loss": float(np.mean(losses)),
                                            "standard_deviation_loss": float(np.std(losses, ddof=1)),
                                            "mean_validation_accuracy": float(np.mean(accuracies)),
                                            "standard_deviation_accuracy": float(np.std(accuracies, ddof=1))},
            "seed": config.seed, "standardize": request.standardize,
            "interpretation": "Stratified held-out folds; hyperparameter selection on these folds requires a separate final test set."}
