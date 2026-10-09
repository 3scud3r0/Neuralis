"""Measured policy rollouts, success intervals, PNG frames and optional video."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import math
import numpy as np
import torch
from PIL import Image

from .config import Config
from .environment import MarioEnv
from .model import ActorCritic
from .telemetry import atomic_json


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("Wilson interval requires 0 <= successes <= positive trials")
    rate = successes / trials
    denominator = 1 + z * z / trials
    center = (rate + z * z / (2 * trials)) / denominator
    half = z * math.sqrt(rate * (1 - rate) / trials + z * z / (4 * trials * trials)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def evaluate(model: ActorCritic, config: Config, episodes: int = 5, seed: int = 100_042,
             deterministic: bool = True, output: Path | None = None,
             record_video: bool = False, capture_every: int = 120,
             max_captures: int = 24, human: bool = False) -> dict:
    if episodes < 1 or capture_every < 1 or max_captures < 0:
        raise ValueError("Invalid evaluation episode or capture count")
    evaluation_config = replace(config, stages=())
    device = next(model.parameters()).device
    was_training = model.training
    torch_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(seed)
    env = MarioEnv(evaluation_config, seed)
    results = []
    capture_index = 0
    captures = []
    inspection = None
    writer = None
    if output:
        output.mkdir(parents=True, exist_ok=True)
    if record_video and not output:
        raise ValueError("Video recording requires an output directory")
    if record_video:
        import imageio.v2 as imageio
        writer = imageio.get_writer(output / "evaluation.mp4", fps=60 / config.frame_skip,
                                    codec="libx264", quality=7, macro_block_size=16)
    model.eval()
    try:
        for episode in range(episodes):
            observation = env.reset(seed=seed + episode * 1009)
            for step in range(config.max_episode_steps):
                tensor = torch.as_tensor(observation[None], device=device)
                if inspection is None:
                    inspection = model.inspect(tensor)
                with torch.no_grad():
                    action = int(model.act(tensor, deterministic=deterministic)[0].item())
                result = env.step(action)
                observation = result.observation
                if human:
                    env.render_human()
                if writer is not None and episode == 0:
                    writer.append_data(env.render_rgb())
                if output and capture_index < max_captures and (step % capture_every == 0 or result.done):
                    capture_index += 1
                    name = f"{capture_index:03d}-episode-{episode + 1:02d}-step-{step + 1:05d}.png"
                    Image.fromarray(env.render_rgb()).save(output / name)
                    captures.append({"file": name, "episode": episode + 1, "step": step + 1,
                                     "action": action, "xPos": int(result.info.get("x_pos", 0))})
                if result.done:
                    results.append({"episode": episode + 1, **result.info["episode"]})
                    break
        successes = sum(bool(result["flagGet"]) for result in results)
        returns = [result["reward"] for result in results]
        report = {
            "environment": config.env_id,
            "episodes": len(results),
            "seed": seed,
            "episodeSeeds": [seed + episode * 1009 for episode in range(episodes)],
            "initialNoopMax": config.noop_max,
            "deterministic": deterministic,
            "successes": successes,
            "successRate": successes / len(results),
            "successRate95CI": list(wilson_interval(successes, len(results))),
            "returns": returns,
            "meanReturn": float(np.mean(returns)),
            "stdReturn": float(np.std(returns)),
            "maxX": max(result["xPos"] for result in results),
            "meanEpisodeLength": float(np.mean([result["length"] for result in results])),
            "results": results,
            "captures": captures,
            "note": "Finite empirical evaluation across seeded initial no-ops. Wilson interval assumes independent episodes; deterministic NES starts can remain correlated. A small sample does not establish perfect play.",
        }
        if output:
            atomic_json(output / "evaluation.json", report)
            atomic_json(output / "inspection.json", inspection)
        return report
    finally:
        if writer is not None:
            writer.close()
        env.close()
        model.train(was_training)
        torch.set_rng_state(torch_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
