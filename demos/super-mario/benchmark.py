#!/usr/bin/env python3
"""Compare real PPO/Double-DQN agents and controls on identical seeded starts."""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import time
import torch

from mario_rl.config import Config
from mario_rl.environment import MarioEnv
from mario_rl.evaluation import evaluate
from mario_rl.policies import BaselinePolicy, load_any_policy
from mario_rl.telemetry import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="*", type=Path)
    parser.add_argument("--env-id", default="SuperMarioBros-1-1-v0")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=100042)
    parser.add_argument("--max-steps", type=int, default=4500)
    parser.add_argument("--baselines", action="store_true", help="Include random, right and right+jump controls")
    parser.add_argument("--output", type=Path, default=Path("runs/benchmark"))
    args = parser.parse_args()
    if not args.checkpoints and not args.baselines:
        parser.error("Provide at least one checkpoint or --baselines")
    torch.set_num_threads(2)
    results = []
    reference_config = Config(env_id=args.env_id, max_episode_steps=args.max_steps)
    agents = []
    for checkpoint in args.checkpoints:
        model, config, data = load_any_policy(checkpoint)
        config = replace(config, env_id=args.env_id, stages=(), max_episode_steps=args.max_steps)
        reference_keys = ("frame_skip", "frame_stack", "image_size", "action_set", "noop_max")
        if agents and any(getattr(config, key) != getattr(reference_config, key) for key in reference_keys):
            raise ValueError("Fair comparison requires matching observation, action, repeat and initial-noop settings")
        reference_config = config
        agents.append((checkpoint.parent.name, model, config, data["step"]))
    if args.baselines:
        env = MarioEnv(reference_config, args.seed)
        try:
            action_count = env.action_count
        finally:
            env.close()
        # SIMPLE_MOVEMENT and RIGHT_ONLY both use right at 1 and right+A at 2.
        agents.extend([
            ("random-control", BaselinePolicy(action_count, "random"), reference_config, 0),
            ("right-control", BaselinePolicy(action_count, "fixed", 1), reference_config, 0),
            ("right-jump-control", BaselinePolicy(action_count, "fixed", 2), reference_config, 0),
        ])
    for index, (name, model, config, transitions) in enumerate(agents):
        started = time.monotonic()
        report = evaluate(model, config, args.episodes, args.seed,
                          output=args.output / f"{index + 1:02d}-{name}", max_captures=0)
        row = {"name": name, "checkpointTransitions": transitions, "successRate": report["successRate"],
               "successRate95CI": report["successRate95CI"], "meanReturn": report["meanReturn"],
               "maxX": report["maxX"], "elapsedSeconds": time.monotonic() - started,
               "report": report}
        results.append(row)
        print(f"{name}: success={report['successes']}/{report['episodes']} "
              f"return={report['meanReturn']:.1f} maxX={report['maxX']}", flush=True)
    atomic_json(args.output / "benchmark.json", {
        "schema": "neuralis.mario.benchmark/v1", "environment": args.env_id,
        "episodesPerPolicy": args.episodes, "seed": args.seed, "results": results,
        "note": "Identical seeded initial no-ops and action limits. Reported intervals assume independent episodes; deterministic starts may remain correlated.",
    })


if __name__ == "__main__":
    main()
