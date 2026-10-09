"""Dueling Double-DQN with proportional prioritized, n-step experience replay.

The online network selects the next action; the target network estimates it.
Importance weights correct prioritized sampling bias. Replay snapshots are paired
with checkpoint metadata by SHA-256, allowing resumable off-policy training.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
import math
import os
from pathlib import Path
import tempfile
import time
import numpy as np
import torch
from torch.nn import functional as F

from .config import Config
from .model import ActorCritic, choose_device, initialize
from .replay import NStepAccumulator, PrioritizedReplay
from .telemetry import Telemetry, atomic_json
from .training import set_reproducibility
from .vector import VectorMario


@dataclass
class DQNConfig:
    replay_capacity: int = 10000
    replay_memory_mb: int = 768
    batch_size: int = 32
    learning_starts: int = 1000
    train_every: int = 4
    gradient_steps: int = 1
    target_update_every: int = 1000
    target_tau: float = 1.0
    learning_rate: float = 0.0001
    n_steps: int = 3
    priority_alpha: float = 0.6
    priority_beta_start: float = 0.4
    priority_beta_steps: int = 1000000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 500000
    max_gradient_norm: float = 10.0
    checkpoint_every: int = 10000
    log_every: int = 100
    save_replay: bool = True

    def validate(self):
        for name in ("replay_capacity", "replay_memory_mb", "batch_size", "learning_starts",
                     "train_every", "gradient_steps", "target_update_every", "n_steps",
                     "priority_beta_steps", "epsilon_decay_steps", "checkpoint_every", "log_every"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("target_tau", "learning_rate", "priority_alpha", "priority_beta_start",
                     "epsilon_start", "epsilon_end", "max_gradient_norm"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if self.batch_size > self.replay_capacity:
            raise ValueError("batch_size cannot exceed replay_capacity")
        if not 0 <= self.priority_alpha <= 1 or not 0 <= self.priority_beta_start <= 1:
            raise ValueError("Priority alpha and beta must be in [0,1]")
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError("Require 0 <= epsilon_end <= epsilon_start <= 1")
        if not 0 < self.target_tau <= 1:
            raise ValueError("target_tau must be in (0,1]")
        if self.learning_rate <= 0 or self.max_gradient_norm <= 0:
            raise ValueError("Learning rate and gradient norm must be positive")
        return self

    @classmethod
    def from_dict(cls, data: dict):
        unknown = set(data) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown Double-DQN configuration keys: {sorted(unknown)}")
        return cls(**data).validate()


class DuelingQNetwork(ActorCritic):
    def __init__(self, observation_shape, actions):
        super().__init__(observation_shape, actions)
        # The actor/critic module names preserve Neuralis layer-inspection tools.
        initialize(self.actor, 1.0)

    def forward(self, observations):
        features = self.features(observations)
        advantage = self.actor(features)
        value = self.critic(features).squeeze(-1)
        q_values = value[:, None] + advantage - advantage.mean(dim=-1, keepdim=True)
        return q_values, value

    def act(self, observations, actions=None, deterministic=False):
        q_values, value = self.forward(observations)
        if actions is None:
            # Evaluation uses the greedy action. Exploration belongs to training.
            actions = q_values.argmax(dim=-1)
        zeros = torch.zeros_like(value)
        return actions, zeros, zeros, value


def linear_schedule(start: float, end: float, duration: int, step: int):
    if duration <= 0:
        raise ValueError("Schedule duration must be positive")
    fraction = min(1.0, max(0.0, step / duration))
    return start + fraction * (end - start)


def double_dqn_targets(online_next_q, target_next_q, rewards, discounts, terminated):
    actions = online_next_q.argmax(dim=1)
    selected = target_next_q.gather(1, actions[:, None]).squeeze(1)
    return rewards + discounts * (~terminated).float() * selected


def update_q(online, target, optimizer, replay, config: DQNConfig, beta: float, device):
    batch = replay.sample(config.batch_size, beta)
    tensors = {name: torch.as_tensor(value, device=device) for name, value in batch.items() if name != "indices"}
    with torch.no_grad():
        next_online, _ = online(tensors["next_observations"])
        next_target, _ = target(tensors["next_observations"])
        targets = double_dqn_targets(next_online, next_target, tensors["rewards"],
                                     tensors["discounts"], tensors["terminated"])
    q_values, _ = online(tensors["observations"])
    predictions = q_values.gather(1, tensors["actions"][:, None]).squeeze(1)
    td_errors = targets - predictions
    elementwise_loss = F.smooth_l1_loss(predictions, targets, reduction="none")
    loss = (tensors["weights"] * elementwise_loss).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("Double-DQN loss became non-finite")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient = torch.nn.utils.clip_grad_norm_(online.parameters(), config.max_gradient_norm)
    if not torch.isfinite(gradient):
        raise FloatingPointError("Double-DQN gradient became non-finite")
    optimizer.step()
    replay.update_priorities(batch["indices"], td_errors.detach().cpu().numpy())
    return {"loss": float(loss.item()), "meanAbsoluteTD": float(td_errors.abs().mean().item()),
            "meanQ": float(predictions.mean().item()), "meanTarget": float(targets.mean().item()),
            "gradientNorm": float(gradient.item()), "importanceWeight": float(tensors["weights"].mean().item())}


def update_target(online, target, tau: float):
    with torch.no_grad():
        for online_parameter, target_parameter in zip(online.parameters(), target.parameters()):
            target_parameter.lerp_(online_parameter, tau)


def save_dqn(path: Path, online, target, optimizer, replay, env_config, dqn_config,
             step: int, updates: int, rng: np.random.Generator):
    path.parent.mkdir(parents=True, exist_ok=True)
    replay_reference = None
    if dqn_config.save_replay:
        # Immutable step-specific replay artifact; latest.pt is replaced last.
        replay_path = path.parent / f"replay-{step:012d}.npz"
        checksum = replay.save(replay_path)
        replay_reference = {"file": replay_path.name, "sha256": checksum}
    data = {
        "format": "neuralis.mario.double-dqn/v1",
        "model": online.state_dict(), "target": target.state_dict(),
        "optimizer": optimizer.state_dict(), "config": env_config.to_dict(),
        "dqn_config": asdict(dqn_config), "observation_shape": online.observation_shape,
        "action_count": online.actions, "step": step, "updates": updates,
        "rng": json.dumps(rng.bit_generator.state), "torch_rng": torch.get_rng_state(),
        "replay": replay_reference,
        "note": "Emulator episodes and pending n-step prefixes restart on resume.",
    }
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(data, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_dqn(path: Path):
    data = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(data, dict) or data.get("format") != "neuralis.mario.double-dqn/v1":
        raise ValueError("Not a supported Neuralis Double-DQN checkpoint")
    Config.from_dict(data["config"])
    DQNConfig.from_dict(data["dqn_config"])
    return data


def load_dqn_policy(path: Path, device_name: str = "cpu"):
    data = read_dqn(path)
    config = Config.from_dict(data["config"])
    model = DuelingQNetwork(tuple(data["observation_shape"]), data["action_count"]).to(choose_device(device_name))
    model.load_state_dict(data["model"], strict=True)
    model.eval()
    return model, config, data


def train_dqn(env_config: Config, dqn_config: DQNConfig, resume: Path | None = None):
    env_config.validate()
    dqn_config.validate()
    if env_config.stages:
        raise ValueError("Double-DQN trains one stage at a time; use PPO for the automatic stage curriculum")
    set_reproducibility(env_config.seed, env_config.torch_threads)
    device = choose_device(env_config.device)
    output = Path(env_config.output_dir)
    if (output / "latest-dqn.pt").exists() and resume is None:
        raise FileExistsError("Run already exists; resume it or select a new output directory")
    data = read_dqn(resume) if resume else None
    step = int(data["step"]) if data else 0
    updates = int(data["updates"]) if data else 0
    if step >= env_config.total_steps:
        raise ValueError("total_steps must exceed the resumed step count")
    rng = np.random.default_rng(env_config.seed)
    start_time = time.monotonic()
    initial_step = step
    status = "completed"
    with VectorMario(env_config) as envs:
        online = DuelingQNetwork(envs.observation_shape, envs.action_count).to(device)
        target = DuelingQNetwork(envs.observation_shape, envs.action_count).to(device)
        target.load_state_dict(online.state_dict())
        target.eval()
        for parameter in target.parameters():
            parameter.requires_grad_(False)
        optimizer = torch.optim.Adam(online.parameters(), lr=dqn_config.learning_rate, eps=1e-5)
        replay = PrioritizedReplay(dqn_config.replay_capacity, envs.observation_shape, env_config.seed,
                                   dqn_config.priority_alpha, max_memory_mb=dqn_config.replay_memory_mb)
        if data:
            saved = Config.from_dict(data["config"])
            keys = ("env_id", "frame_stack", "frame_skip", "image_size", "action_set", "reward_scale")
            if any(getattr(saved, key) != getattr(env_config, key) for key in keys):
                raise ValueError("Resume environment/preprocessing configuration differs from checkpoint")
            if DQNConfig.from_dict(data["dqn_config"]) != dqn_config:
                raise ValueError("Resume Double-DQN settings must match the checkpoint")
            online.load_state_dict(data["model"], strict=True)
            target.load_state_dict(data["target"], strict=True)
            optimizer.load_state_dict(data["optimizer"])
            rng.bit_generator.state = json.loads(data["rng"])
            torch.set_rng_state(data["torch_rng"])
            if data["replay"]:
                reference = data["replay"]
                if Path(reference["file"]).name != reference["file"]:
                    raise ValueError("Replay reference must be a filename in the checkpoint directory")
                replay = PrioritizedReplay.load(resume.parent / reference["file"], reference["sha256"],
                                                 dqn_config.replay_memory_mb)
        accumulators = [NStepAccumulator(dqn_config.n_steps, env_config.gamma) for _ in range(env_config.num_envs)]
        observations = envs.reset()
        telemetry = Telemetry(output, env_config.to_dict(), online.architecture(), resume=bool(resume))
        telemetry.data["run"]["algorithm"] = "Dueling Double-DQN + prioritized n-step replay"
        telemetry.data["run"]["device"] = str(device)
        telemetry.data["dqnConfig"] = asdict(dqn_config)
        atomic_json(output / "config-dqn.json", {"environment": env_config.to_dict(), "algorithm": asdict(dqn_config)})
        last_train = step
        last_target = step
        last_checkpoint = step
        last_log = step
        latest_metrics = {}
        telemetry.event("startup", step, {"algorithm": "double-dqn", "resume": bool(resume),
                        "replayEntries": len(replay), "replayIntegrity": "SHA-256 verified" if data and data["replay"] else "new buffer"})
        try:
            while step < env_config.total_steps:
                epsilon = linear_schedule(dqn_config.epsilon_start, dqn_config.epsilon_end,
                                          dqn_config.epsilon_decay_steps, step)
                with torch.no_grad():
                    q_values, _ = online(torch.as_tensor(observations, device=device))
                    actions = q_values.argmax(dim=1).cpu().numpy()
                explore = rng.random(env_config.num_envs) < epsilon
                random_actions = rng.integers(0, envs.action_count, size=env_config.num_envs)
                actions = np.where(explore, random_actions, actions)
                next_observations, rewards, terminated, truncated, infos = envs.step(actions)
                step += env_config.num_envs
                for index, info in enumerate(infos):
                    final = info.get("final_observation", next_observations[index])
                    ready = accumulators[index].append(observations[index], actions[index], rewards[index],
                                                        final, terminated[index], truncated[index])
                    for transition in ready:
                        replay.add(transition)
                    if "episode" in info:
                        telemetry.event("episode", step, {"worker": index, **info["episode"]})
                observations = next_observations
                beta = linear_schedule(dqn_config.priority_beta_start, 1.0, dqn_config.priority_beta_steps, step)
                if step >= dqn_config.learning_starts and len(replay) >= dqn_config.batch_size and step - last_train >= dqn_config.train_every:
                    for _ in range(dqn_config.gradient_steps):
                        latest_metrics = update_q(online, target, optimizer, replay, dqn_config, beta, device)
                        updates += 1
                    last_train = step
                if step - last_target >= dqn_config.target_update_every:
                    update_target(online, target, dqn_config.target_tau)
                    last_target = step
                if step - last_log >= dqn_config.log_every or step >= env_config.total_steps:
                    elapsed = time.monotonic() - start_time
                    metrics = {**latest_metrics, "updates": updates, "epsilon": epsilon, "beta": beta,
                               "replayEntries": len(replay), "fps": (step - initial_step) / max(elapsed, 1e-6),
                               "elapsedSeconds": elapsed, "learningRate": dqn_config.learning_rate}
                    telemetry.event("metrics", step, metrics)
                    telemetry.flush()
                    last_log = step
                    print(f"double-dqn transitions={step} updates={updates} epsilon={epsilon:.3f} "
                          f"replay={len(replay)} loss={latest_metrics.get('loss', 0):.4f}", flush=True)
                if step - last_checkpoint >= dqn_config.checkpoint_every:
                    save_dqn(output / "latest-dqn.pt", online, target, optimizer, replay,
                             env_config, dqn_config, step, updates, rng)
                    last_checkpoint = step
        except KeyboardInterrupt:
            status = "interrupted"
        except BaseException as error:
            status = "failed"
            telemetry.event("error", step, {"type": type(error).__name__, "message": str(error)})
            raise
        finally:
            save_dqn(output / "latest-dqn.pt", online, target, optimizer, replay,
                     env_config, dqn_config, step, updates, rng)
            telemetry.data["run"]["totalTransitions"] = step
            telemetry.finish(status)
    return {"status": status, "step": step, "updates": updates, "output": str(output),
            "checkpoint": str(output / "latest-dqn.pt")}
