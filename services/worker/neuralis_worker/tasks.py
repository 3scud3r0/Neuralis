from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import math
import random
import time


@dataclass(frozen=True)
class TaskConfig:
    task: str = "xor"
    seed: int = 42
    steps: int = 1000
    learning_rate: float = 0.08
    hidden_size: int = 8
    capture_every: int = 25
    cpu_threads: int | None = None
    memory_mb: int | None = None
    world_size: int | None = None
    project_id: str = "local"

    @classmethod
    def parse(cls, value: dict[str, Any]) -> "TaskConfig":
        allowed = {"task", "seed", "steps", "learning_rate", "hidden_size", "capture_every", "cpu_threads", "memory_mb", "world_size", "project_id"}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"Unsupported task fields: {', '.join(sorted(unknown))}")
        result = cls(**value)
        if not isinstance(result.project_id, str) or not 1 <= len(result.project_id) <= 128:
            raise ValueError("Project identifier exceeds metadata bounds")
        if result.task not in {"xor", "ddp-xor"}:
            raise ValueError("Only registered xor and ddp-xor tasks can be executed")
        for name, lower, upper in [("seed", 0, 2**31 - 1), ("steps", 1, 100000), ("hidden_size", 2, 256), ("capture_every", 1, 10000)]:
            item = getattr(result, name)
            if isinstance(item, bool) or not isinstance(item, int) or not lower <= item <= upper:
                raise ValueError(f"{name} must be an integer between {lower} and {upper}")
        for name, lower, upper in [("cpu_threads", 1, 64), ("memory_mb", 128, 131072), ("world_size", 1, 8)]:
            item = getattr(result, name)
            if item is not None and (isinstance(item, bool) or not isinstance(item, int) or not lower <= item <= upper):
                raise ValueError(f"{name} must be an integer between {lower} and {upper}")
        if isinstance(result.learning_rate, bool) or not isinstance(result.learning_rate, (int, float)) or not math.isfinite(result.learning_rate) or not 1e-6 <= result.learning_rate <= 2:
            raise ValueError("Learning rate must be finite and between 1e-6 and 2")
        return result


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1 + exponential)


def train_xor(config: TaskConfig, emit: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    rng = random.Random(config.seed)
    hidden = config.hidden_size
    w1 = [[rng.gauss(0, 0.6) for _ in range(2)] for _ in range(hidden)]
    b1 = [rng.gauss(0, 0.1) for _ in range(hidden)]
    w2 = [rng.gauss(0, 0.6) for _ in range(hidden)]
    b2 = 0.0
    dataset = [([0.0, 0.0], 0.0), ([0.0, 1.0], 1.0), ([1.0, 0.0], 1.0), ([1.0, 1.0], 0.0)]
    started = time.monotonic()
    metrics: list[dict[str, Any]] = []
    for step in range(1, config.steps + 1):
        gw1 = [[0.0, 0.0] for _ in range(hidden)]
        gb1 = [0.0] * hidden
        gw2 = [0.0] * hidden
        gb2 = 0.0
        loss = 0.0
        correct = 0
        for inputs, target in dataset:
            activation = [math.tanh(sum(w * x for w, x in zip(row, inputs)) + bias) for row, bias in zip(w1, b1)]
            prediction = sigmoid(sum(w * a for w, a in zip(w2, activation)) + b2)
            loss += -(target * math.log(max(prediction, 1e-12)) + (1 - target) * math.log(max(1 - prediction, 1e-12))) / 4
            correct += (prediction >= 0.5) == (target >= 0.5)
            delta = (prediction - target) / 4
            for j in range(hidden):
                gw2[j] += delta * activation[j]
                hidden_delta = delta * w2[j] * (1 - activation[j] ** 2)
                gb1[j] += hidden_delta
                for i in range(2):
                    gw1[j][i] += hidden_delta * inputs[i]
            gb2 += delta
        for j in range(hidden):
            for i in range(2):
                w1[j][i] -= config.learning_rate * gw1[j][i]
            b1[j] -= config.learning_rate * gb1[j]
            w2[j] -= config.learning_rate * gw2[j]
        b2 -= config.learning_rate * gb2
        if step == 1 or step == config.steps or step % config.capture_every == 0:
            metric = {"type": "metric", "step": step, "loss": loss, "accuracy": correct / 4, "elapsed_seconds": time.monotonic() - started}
            emit(metric)
            # Keep result metrics bounded independently from the total number of steps.
            metrics.append(metric)
            if len(metrics) > 1000:
                metrics.pop(0)
    predictions = []
    for inputs, target in dataset:
        activation = [math.tanh(sum(w * x for w, x in zip(row, inputs)) + bias) for row, bias in zip(w1, b1)]
        prediction = sigmoid(sum(w * a for w, a in zip(w2, activation)) + b2)
        predictions.append({"input": inputs, "target": target, "prediction": prediction})
    accuracy = sum((row["prediction"] >= 0.5) == (row["target"] >= 0.5) for row in predictions) / 4
    final_loss = sum(-(row["target"] * math.log(max(row["prediction"], 1e-12)) + (1 - row["target"]) * math.log(max(1 - row["prediction"], 1e-12))) for row in predictions) / 4
    return {"task": "xor", "seed": config.seed, "steps": config.steps, "loss": final_loss, "accuracy": accuracy,
        "elapsed_seconds": time.monotonic() - started, "metrics": metrics, "predictions": predictions,
        "checkpoint": {"architecture": [2, hidden, 1], "w1": w1, "b1": b1, "w2": w2, "b2": b2}}
