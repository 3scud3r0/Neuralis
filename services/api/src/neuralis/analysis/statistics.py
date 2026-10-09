import math

import numpy as np


def finite_vector(values, name="values"):
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or len(result) < 2 or len(result) > 100000 or not np.isfinite(result).all():
        raise ValueError(f"{name} requires 2 to 100,000 finite scalar observations")
    return result


def descriptive_statistics(values):
    array = finite_vector(values)
    return {"count": len(array), "mean": float(array.mean()), "standard_deviation": float(array.std(ddof=1)),
            "standard_error": float(array.std(ddof=1) / np.sqrt(len(array))), "minimum": float(array.min()),
            "maximum": float(array.max()), "median": float(np.median(array)),
            "quantiles": np.quantile(array, [0.05, 0.25, 0.75, 0.95]).tolist()}


def paired_bootstrap(baseline, intervention, repetitions=2000, confidence=0.95, seed=42):
    first, second = finite_vector(baseline, "baseline"), finite_vector(intervention, "intervention")
    if first.shape != second.shape:
        raise ValueError("Paired effect estimates require matching observation counts")
    if not 100 <= repetitions <= 10000 or not 0.5 <= confidence < 1:
        raise ValueError("Bootstrap repetition/confidence budget is invalid")
    generator = np.random.default_rng(seed)
    differences = second - first
    sampled_means = np.empty(repetitions, dtype=np.float64)
    # Chunked resampling bounds temporary arrays for large experiment cohorts.
    chunk_size = max(1, min(100, 1000000 // len(differences)))
    for start in range(0, repetitions, chunk_size):
        count = min(chunk_size, repetitions - start)
        indices = generator.integers(len(differences), size=(count, len(differences)))
        sampled_means[start:start + count] = differences[indices].mean(axis=1)
    alpha = (1 - confidence) / 2
    interval = np.quantile(sampled_means, [alpha, 1 - alpha])
    deviation = differences.std(ddof=1)
    effect = float(differences.mean() / deviation) if deviation > 0 else None
    return {"mean_paired_effect": float(differences.mean()), "confidence_interval": interval.tolist(),
            "confidence": confidence, "paired_standardized_effect": effect, "observations": len(first),
            "repetitions": repetitions, "seed": seed, "method": "paired percentile bootstrap",
            "interpretation": "Exploratory uncertainty for these paired inputs; independence and sampling design matter."}


def paired_permutation_test(baseline, intervention, repetitions=5000, seed=42):
    first, second = finite_vector(baseline), finite_vector(intervention)
    if first.shape != second.shape or not 100 <= repetitions <= 10000:
        raise ValueError("Invalid paired permutation data or work budget")
    differences = second - first
    observed = abs(differences.mean())
    generator = np.random.default_rng(seed)
    extreme = 0
    chunk_size = max(1, min(100, 1000000 // len(differences)))
    for start in range(0, repetitions, chunk_size):
        count = min(chunk_size, repetitions - start)
        signs = generator.integers(0, 2, size=(count, len(differences))) * 2 - 1
        simulated = np.abs((signs * differences).mean(axis=1))
        extreme += int(np.count_nonzero(simulated >= observed - 1e-15))
    return {"statistic": float(observed), "p_value": (extreme + 1) / (repetitions + 1),
            "repetitions": repetitions, "seed": seed, "method": "two-sided paired sign permutation",
            "assumption": "Exchangeability of paired signs under the null"}


def holm_adjust(p_values):
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("P-values must be finite numbers between zero and one")
    order = np.argsort(values, kind="stable")
    adjusted = np.empty_like(values)
    running = 0
    for rank, index in enumerate(order):
        running = max(running, float(values[index]) * (len(values) - rank))
        adjusted[index] = min(1, running)
    return adjusted.tolist()


def classification_metrics(probabilities, targets):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    if probabilities.ndim != 2 or targets.shape != (len(probabilities),) or not len(targets):
        raise ValueError("Classification metrics require [examples,classes] probabilities and one label per example")
    classes = probabilities.shape[1]
    if classes < 2 or (targets < 0).any() or (targets >= classes).any():
        raise ValueError("Classification target label outside class range")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or not np.allclose(probabilities.sum(1), 1, atol=1e-5):
        raise ValueError("Class probabilities must be finite, nonnegative, and sum to one")
    predictions = probabilities.argmax(1)
    confusion = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(confusion, (targets, predictions), 1)
    precision, recall, f1 = [], [], []
    for index in range(classes):
        true_positive = confusion[index, index]
        predicted_positive = confusion[:, index].sum()
        actual_positive = confusion[index].sum()
        p = true_positive / predicted_positive if predicted_positive else 0
        r = true_positive / actual_positive if actual_positive else 0
        precision.append(float(p))
        recall.append(float(r))
        f1.append(float(2 * p * r / (p + r)) if p + r else 0)
    one_hot = np.eye(classes)[targets]
    confidence = probabilities.max(1)
    correctness = predictions == targets
    calibration, ece = [], 0
    for index in range(10):
        low, high = index / 10, (index + 1) / 10
        selected = (confidence >= low) & ((confidence <= high) if index == 9 else (confidence < high))
        count = int(selected.sum())
        if count:
            accuracy = float(correctness[selected].mean())
            mean_confidence = float(confidence[selected].mean())
            ece += count / len(targets) * abs(accuracy - mean_confidence)
        else:
            accuracy = mean_confidence = None
        calibration.append({"minimum": low, "maximum": high, "count": count, "accuracy": accuracy, "confidence": mean_confidence})
    return {"accuracy": float(correctness.mean()), "confusion_matrix": confusion.tolist(), "precision": precision,
            "recall": recall, "f1": f1, "macro_f1": float(np.mean(f1)), "balanced_accuracy": float(np.mean(recall)),
            "negative_log_likelihood": float(-np.log(probabilities[np.arange(len(targets)), targets].clip(1e-12)).mean()),
            "brier_score": float(np.square(probabilities - one_hot).sum(1).mean()),
            "expected_calibration_error": float(ece), "calibration_bins": calibration}
