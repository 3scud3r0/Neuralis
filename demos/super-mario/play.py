#!/usr/bin/env python3
"""Play a policy in a native NES window; use evaluate.py on headless servers."""
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
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=100042)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="cpu")
    args = parser.parse_args()
    model, config, checkpoint = load_any_policy(args.checkpoint, args.device)
    torch.set_num_threads(config.torch_threads)
    if config.stages:
        config = replace(config, env_id=config.stages[checkpoint.get("stage", 0)])
    print(json.dumps(evaluate(model, config, args.episodes, args.seed, not args.stochastic,
                              human=True), indent=2))


if __name__ == "__main__":
    main()
