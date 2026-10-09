#!/usr/bin/env python3
"""Train a real CNN policy in the NES emulator, with resumable PPO updates."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from mario_rl.checkpoint import read_checkpoint
from mario_rl.config import Config
from mario_rl.training import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON experiment configuration")
    parser.add_argument("--resume", type=Path, help="Neuralis PPO checkpoint; resumes into the configured output")
    parser.add_argument("--steps", type=int, help="Total target transitions, including resumed transitions")
    parser.add_argument("--envs", type=int, help="Number of rollout workers")
    parser.add_argument("--output", type=str, help="Run output directory")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--sync", action="store_true", help="Run emulators in this process for debugging")
    args = parser.parse_args()
    if args.config:
        config = Config.load(args.config)
    elif args.resume:
        config = Config.from_dict(read_checkpoint(args.resume)["config"])
    else:
        config = Config()
    overrides = {key: value for key, value in {
        "total_steps": args.steps, "num_envs": args.envs, "output_dir": args.output,
        "device": args.device, "seed": args.seed,
    }.items() if value is not None}
    if args.sync:
        overrides["vector_mode"] = "sync"
    config = replace(config, **overrides).validate()
    print(json.dumps(train(config, args.resume), indent=2))


if __name__ == "__main__":
    main()
