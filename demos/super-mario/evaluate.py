#!/usr/bin/env python3
"""Evaluate held-out seeds and capture actual emulator frames from a checkpoint."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import torch

from mario_rl.policies import load_any_policy
from mario_rl.evaluation import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=100042)
    parser.add_argument("--env-id", help="Registered Mario stage; default uses trained stage")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="cpu")
    parser.add_argument("--output", type=Path, default=Path("runs/evaluation"))
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--stochastic", action="store_true", help="Sample actions instead of using argmax")
    parser.add_argument("--max-steps", type=int, help="Optional per-episode action limit")
    parser.add_argument("--capture-every", type=int, default=120)
    parser.add_argument("--max-captures", type=int, default=24)
    args = parser.parse_args()
    model, config, checkpoint = load_any_policy(args.checkpoint, args.device)
    torch.set_num_threads(config.torch_threads)
    if args.env_id:
        config = replace(config, env_id=args.env_id)
    elif config.stages:
        config = replace(config, env_id=config.stages[checkpoint.get("stage", 0)])
    if args.max_steps is not None:
        config = replace(config, max_episode_steps=args.max_steps).validate()
    report = evaluate(model, config, args.episodes, args.seed, not args.stochastic,
                      args.output, args.video, args.capture_every, args.max_captures)
    report["checkpointTransitions"] = checkpoint["step"]
    from mario_rl.telemetry import atomic_json
    atomic_json(args.output / "evaluation.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
