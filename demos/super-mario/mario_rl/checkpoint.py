"""Versioned PyTorch checkpoints restricted to tensors and primitive metadata."""
from __future__ import annotations

import os
from pathlib import Path
import random
import tempfile
import numpy as np
import torch

from .config import Config
from .model import ActorCritic, choose_device


FORMAT = "neuralis.mario.ppo/v1"


def save_checkpoint(path: Path, model: ActorCritic, optimizer: torch.optim.Optimizer,
                    config: Config, step: int, update: int, stage: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    numpy_state = np.random.get_state()
    data = {
        "format": FORMAT,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config.to_dict(),
        "observation_shape": model.observation_shape,
        "action_count": model.actions,
        "step": step,
        "update": update,
        "stage": stage,
        "torch_rng": torch.get_rng_state(),
        "python_rng": random.getstate(),
        "numpy_rng": {"kind": numpy_state[0], "keys": numpy_state[1].tolist(),
                      "position": numpy_state[2], "has_gauss": numpy_state[3],
                      "cached_gaussian": numpy_state[4]},
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(data, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_checkpoint(path: str | Path) -> dict:
    # Never use unrestricted pickle for imported checkpoints.
    data = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        raise ValueError("Not a supported Neuralis Mario PPO checkpoint")
    Config.from_dict(data["config"])
    if tuple(data["observation_shape"]) != (
        data["config"]["frame_stack"], data["config"]["image_size"], data["config"]["image_size"]
    ):
        raise ValueError("Checkpoint observation shape conflicts with its configuration")
    return data


def load_policy(path: str | Path, device_name: str = "cpu"):
    data = read_checkpoint(path)
    config = Config.from_dict(data["config"])
    device = choose_device(device_name)
    model = ActorCritic(tuple(data["observation_shape"]), data["action_count"]).to(device)
    model.load_state_dict(data["model"], strict=True)
    model.eval()
    return model, config, data


def restore_rng(data: dict) -> None:
    torch.set_rng_state(data["torch_rng"])
    random.setstate(data["python_rng"])
    state = data["numpy_rng"]
    np.random.set_state((state["kind"], np.asarray(state["keys"], dtype=np.uint32),
                         state["position"], state["has_gauss"], state["cached_gaussian"]))
    if torch.cuda.is_available() and data.get("cuda_rng"):
        torch.cuda.set_rng_state_all(data["cuda_rng"])
