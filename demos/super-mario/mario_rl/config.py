"""Typed, validated experiment configuration with no silent unknown keys."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
from typing import Any


@dataclass
class Config:
    env_id: str = "SuperMarioBros-1-1-v0"
    stages: tuple[str, ...] = ()
    seed: int = 42
    device: str = "cpu"
    num_envs: int = 4
    vector_mode: str = "process"
    worker_timeout: float = 120.0
    total_steps: int = 1_000_000
    rollout_steps: int = 128
    minibatch_size: int = 128
    update_epochs: int = 4
    learning_rate: float = 0.00025
    anneal_learning_rate: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    target_kl: float = 0.03
    frame_skip: int = 4
    frame_stack: int = 4
    image_size: int = 84
    action_set: str = "simple"
    reward_scale: float = 0.1
    max_episode_steps: int = 4500
    noop_max: int = 10
    torch_threads: int = 2
    checkpoint_interval: int = 10
    evaluate_interval: int = 0
    evaluation_episodes: int = 5
    curriculum_window: int = 20
    curriculum_success_threshold: float = 0.8
    output_dir: str = "runs/mario-ppo"

    def validate(self) -> "Config":
        positive_ints = (
            "num_envs", "total_steps", "rollout_steps", "minibatch_size",
            "update_epochs", "frame_skip", "frame_stack", "image_size",
            "max_episode_steps", "torch_threads", "checkpoint_interval",
            "evaluation_episodes", "curriculum_window",
        )
        for name in positive_ints:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("learning_rate", "gamma", "gae_lambda", "clip_coef", "value_coef",
                     "entropy_coef", "max_grad_norm", "target_kl", "reward_scale",
                     "worker_timeout", "curriculum_success_threshold"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or not 0 <= self.seed < 2 ** 32:
            raise ValueError("seed must be an integer in [0, 2^32)")
        if self.image_size < 36:
            raise ValueError("image_size must be at least 36 for the CNN")
        if self.vector_mode not in {"process", "sync"}:
            raise ValueError("vector_mode must be process or sync")
        if self.action_set not in {"simple", "right", "complex"}:
            raise ValueError("action_set must be simple, right, or complex")
        if self.device not in {"cpu", "cuda", "mps", "auto"}:
            raise ValueError("device must be cpu, cuda, mps, or auto")
        if self.noop_max < 0 or self.evaluate_interval < 0:
            raise ValueError("noop_max and evaluate_interval cannot be negative")
        for name in ("gamma", "gae_lambda", "curriculum_success_threshold"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in ("learning_rate", "reward_scale", "max_grad_norm", "worker_timeout"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 < self.clip_coef < 1:
            raise ValueError("clip_coef must be between 0 and 1")
        if self.value_coef < 0 or self.entropy_coef < 0 or self.target_kl < 0:
            raise ValueError("Loss coefficients and target_kl cannot be negative")
        if not isinstance(self.env_id, str) or not self.env_id.startswith("SuperMarioBros"):
            raise ValueError("env_id must be a registered SuperMarioBros environment")
        if not all(isinstance(stage, str) and stage.startswith("SuperMarioBros") for stage in self.stages):
            raise ValueError("stages must contain SuperMarioBros environment IDs")
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        known = {field.name for field in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown configuration keys: {', '.join(sorted(unknown))}")
        data = dict(data)
        if "stages" in data:
            if not isinstance(data["stages"], (list, tuple)):
                raise ValueError("stages must be an array")
            data["stages"] = tuple(data["stages"])
        return cls(**data).validate()

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        with Path(path).open(encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError("Configuration root must be an object")
        return cls.from_dict(data)
