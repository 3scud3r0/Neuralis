"""Bounded rollout/optimization loop with checkpoint resume and stage curriculum."""
from __future__ import annotations

from collections import deque
from dataclasses import replace
import math
from pathlib import Path
import random
import time
import numpy as np
import torch

from .checkpoint import read_checkpoint, restore_rng, save_checkpoint
from .config import Config
from .evaluation import evaluate
from .model import ActorCritic, choose_device
from .ppo import Rollout, optimize
from .telemetry import Telemetry, atomic_json
from .vector import VectorMario


def set_reproducibility(seed: int, threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # CPU is the reproducible reference. GPU kernels can vary across hardware.
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Curriculum:
    def __init__(self, stages: tuple[str, ...], window: int, threshold: float, current: int = 0):
        self.stages = stages
        self.window = window
        self.threshold = threshold
        self.current = current
        self.successes: deque[bool] = deque(maxlen=window)

    def observe(self, episode: dict) -> bool:
        if len(self.stages) <= 1 or self.current >= len(self.stages) - 1:
            return False
        if episode.get("stage") != self.stages[self.current]:
            return False
        self.successes.append(bool(episode.get("flagGet", False)))
        if len(self.successes) == self.window and sum(self.successes) / self.window >= self.threshold:
            self.current += 1
            self.successes.clear()
            return True
        return False


def train(config: Config, resume: Path | None = None) -> dict:
    config.validate()
    output = Path(config.output_dir)
    if output.exists() and (output / "latest.pt").exists() and resume is None:
        raise FileExistsError(f"{output} already contains a run; use --resume or another output directory")
    set_reproducibility(config.seed, config.torch_threads)
    device = choose_device(config.device)
    resume_data = read_checkpoint(resume) if resume else None
    starting_step = int(resume_data["step"]) if resume_data else 0
    starting_update = int(resume_data["update"]) if resume_data else 0
    if starting_step >= config.total_steps:
        raise ValueError("total_steps must exceed the resumed checkpoint's step count")
    start_time = time.monotonic()
    telemetry = None
    model = None
    optimizer = None
    update = starting_update
    global_step = starting_step
    status = "completed"
    with VectorMario(config) as environments:
        model = ActorCritic(environments.observation_shape, environments.action_count).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, eps=1e-5)
        stage = int(resume_data["stage"]) if resume_data else 0
        if resume_data:
            saved_config = Config.from_dict(resume_data["config"])
            protected = ("env_id", "stages", "frame_stack", "frame_skip", "image_size", "action_set", "reward_scale")
            if any(getattr(config, key) != getattr(saved_config, key) for key in protected):
                raise ValueError("Resume requires the checkpoint's environment, preprocessing, action and reward configuration")
            model.load_state_dict(resume_data["model"], strict=True)
            optimizer.load_state_dict(resume_data["optimizer"])
            restore_rng(resume_data)
        observations = environments.set_stage(stage) if stage else environments.reset()
        curriculum = Curriculum(config.stages, config.curriculum_window,
                                config.curriculum_success_threshold, stage)
        telemetry = Telemetry(output, config.to_dict(), model.architecture(), resume=bool(resume))
        telemetry.data["run"]["device"] = str(device)
        atomic_json(output / "config.json", config.to_dict())
        telemetry.event("startup", global_step, {
            "parameters": sum(p.numel() for p in model.parameters()),
            "numEnvs": config.num_envs,
            "resume": bool(resume),
            "note": "Resume restores model, optimizer and RNG; emulator episodes restart.",
        })
        try:
            while global_step < config.total_steps:
                update += 1
                remaining = config.total_steps - global_step
                steps = min(config.rollout_steps, math.ceil(remaining / config.num_envs))
                rollout = Rollout.allocate(steps, config.num_envs, environments.observation_shape, device)
                advance_stage = False
                if config.anneal_learning_rate:
                    rate = config.learning_rate * max(0.0, 1 - global_step / config.total_steps)
                    for group in optimizer.param_groups:
                        group["lr"] = rate
                for step in range(steps):
                    tensor = torch.as_tensor(observations, device=device)
                    rollout.observations[step].copy_(tensor)
                    with torch.no_grad():
                        actions, log_probs, _, values = model.act(tensor)
                    next_observations, rewards, terminated, truncated, infos = environments.step(actions.cpu().numpy())
                    # Auto-reset returns reset observations, but truncation values
                    # must use the actual final observations from the old episode.
                    bootstrap_observations = next_observations.copy()
                    for index, info in enumerate(infos):
                        if "final_observation" in info:
                            bootstrap_observations[index] = info["final_observation"]
                        if "episode" in info:
                            telemetry.event("episode", global_step + config.num_envs, {"worker": index, **info["episode"]})
                            if curriculum.observe(info["episode"]):
                                advance_stage = True
                    with torch.no_grad():
                        next_values = model.value(torch.as_tensor(bootstrap_observations, device=device))
                    rollout.actions[step].copy_(actions)
                    rollout.log_probs[step].copy_(log_probs)
                    rollout.values[step].copy_(values)
                    rollout.next_values[step].copy_(next_values)
                    rollout.rewards[step].copy_(torch.as_tensor(rewards, device=device))
                    rollout.terminated[step].copy_(torch.as_tensor(terminated, device=device))
                    rollout.ended[step].copy_(torch.as_tensor(terminated | truncated, device=device))
                    observations = next_observations
                    global_step += config.num_envs
                measurements = optimize(model, optimizer, rollout, config)
                elapsed = time.monotonic() - start_time
                measurements.update({"update": update, "learningRate": optimizer.param_groups[0]["lr"],
                                     "fps": (global_step - starting_step) / max(elapsed, 1e-6),
                                     "elapsedSeconds": elapsed,
                                     "meanScaledReward": float(rollout.rewards.mean().item()),
                                     "stageIndex": curriculum.current})
                telemetry.event("metrics", global_step, measurements)
                print(f"update={update} transitions={global_step} loss={measurements['loss']:.4f} "
                      f"entropy={measurements['entropy']:.3f} fps={measurements['fps']:.1f}", flush=True)
                if advance_stage:
                    observations = environments.set_stage(curriculum.current)
                    telemetry.event("curriculum", global_step, {"stageIndex": curriculum.current,
                                    "environment": config.stages[curriculum.current]})
                if config.evaluate_interval and update % config.evaluate_interval == 0:
                    active_env = config.stages[curriculum.current] if config.stages else config.env_id
                    evaluation_config = replace(config, env_id=active_env)
                    report = evaluate(model, evaluation_config, config.evaluation_episodes,
                                      config.seed + 100_000, output=output / f"evaluation-{update:06d}")
                    telemetry.event("evaluation", global_step, report)
                if update % config.checkpoint_interval == 0:
                    save_checkpoint(output / f"checkpoint-{global_step:012d}.pt", model, optimizer,
                                    config, global_step, update, curriculum.current)
                    save_checkpoint(output / "latest.pt", model, optimizer, config,
                                    global_step, update, curriculum.current)
                telemetry.flush()
        except KeyboardInterrupt:
            status = "interrupted"
            print("Training interrupted; saving a resumable checkpoint.", flush=True)
        except BaseException as error:
            status = "failed"
            telemetry.event("error", global_step, {"type": type(error).__name__, "message": str(error)})
            raise
        finally:
            save_checkpoint(output / "latest.pt", model, optimizer, config, global_step, update, curriculum.current)
            telemetry.data["run"]["totalTransitions"] = global_step
            telemetry.finish(status)
    return {"output": str(output), "status": status, "step": global_step,
            "checkpoint": str(output / "latest.pt")}
