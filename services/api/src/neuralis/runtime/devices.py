import math
import os

import torch


def device_capabilities():
    cuda = torch.cuda.is_available()
    mps = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    result = {"cpu": {"available": True, "validated_in_this_build": True},
              "cuda": {"available": cuda, "validated_in_this_build": False},
              "mps": {"available": mps, "validated_in_this_build": False}}
    if cuda:
        result["cuda"]["devices"] = [{"index": index, "name": torch.cuda.get_device_name(index),
                                      "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory}
                                     for index in range(torch.cuda.device_count())]
    return result


def resolve_device(requested: str) -> torch.device:
    available = device_capabilities()
    if requested == "auto":
        requested = "cuda" if available["cuda"]["available"] else "mps" if available["mps"]["available"] else "cpu"
    if requested not in available or not available[requested]["available"]:
        raise ValueError(f"Requested device {requested} is unavailable; use CPU or install a compatible accelerator runtime")
    return torch.device(requested)


def module_device(model):
    return next(model.parameters()).device


def optimizer_for(model, config):
    common = {"lr": config.learning_rate, "weight_decay": config.weight_decay}
    options = config.optimizer
    if options.name == "adamw":
        return torch.optim.AdamW(model.parameters(), betas=(options.beta1, options.beta2), eps=options.epsilon, **common)
    if options.name == "adam":
        return torch.optim.Adam(model.parameters(), betas=(options.beta1, options.beta2), eps=options.epsilon, **common)
    if options.name == "sgd":
        return torch.optim.SGD(model.parameters(), momentum=options.momentum, nesterov=options.nesterov, **common)
    return torch.optim.RMSprop(model.parameters(), momentum=options.momentum, eps=options.epsilon, **common)


def scheduled_learning_rate(config, completed_steps: int):
    schedule = config.schedule
    if completed_steps < schedule.warmup_steps:
        return config.learning_rate * (completed_steps + 1) / schedule.warmup_steps
    elapsed = max(0, completed_steps - schedule.warmup_steps)
    total = max(1, config.steps - schedule.warmup_steps)
    if schedule.name == "cosine":
        progress = min(1, elapsed / total)
        ratio = schedule.minimum_ratio + (1 - schedule.minimum_ratio) * (1 + math.cos(math.pi * progress)) / 2
    elif schedule.name == "exponential":
        ratio = max(schedule.minimum_ratio, schedule.gamma ** elapsed)
    elif schedule.name == "step":
        ratio = max(schedule.minimum_ratio, schedule.gamma ** (elapsed // schedule.step_size))
    else:
        ratio = 1
    return config.learning_rate * ratio


def container_resources():
    """Prefer cgroup v2 limits over misleading host resources when available."""
    result = {"cpu_quota": None, "memory_limit_bytes": None, "memory_current_bytes": None}
    paths = {"cpu_quota": "/sys/fs/cgroup/cpu.max", "memory_limit_bytes": "/sys/fs/cgroup/memory.max",
             "memory_current_bytes": "/sys/fs/cgroup/memory.current"}
    for key, path in paths.items():
        try:
            with open(path, encoding="utf-8") as stream:
                value = stream.read().strip()
            if value == "max" or value.startswith("max "):
                continue
            if key == "cpu_quota":
                quota, period = value.split()
                result[key] = int(quota) / int(period)
            else:
                result[key] = int(value)
        except (OSError, ValueError):
            continue
    result["logical_cpu_count"] = os.cpu_count()
    return result
