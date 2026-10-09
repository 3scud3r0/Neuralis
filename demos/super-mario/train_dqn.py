#!/usr/bin/env python3
"""Train the alternative Dueling Double-DQN policy with resumable replay."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

from mario_rl.config import Config
from mario_rl.dqn import DQNConfig, read_dqn, train_dqn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--output")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"))
    parser.add_argument("--sync", action="store_true")
    args = parser.parse_args()
    if args.config:
        data = json.loads(args.config.read_text())
        if set(data) != {"environment", "algorithm"}:
            raise ValueError("Double-DQN JSON requires environment and algorithm objects")
        config = Config.from_dict(data["environment"])
        algorithm = DQNConfig.from_dict(data["algorithm"])
    elif args.resume:
        data = read_dqn(args.resume)
        config = Config.from_dict(data["config"])
        algorithm = DQNConfig.from_dict(data["dqn_config"])
    else:
        config = Config(output_dir="runs/mario-double-dqn")
        algorithm = DQNConfig()
    overrides = {key: value for key, value in {
        "total_steps": args.steps, "output_dir": args.output, "device": args.device,
    }.items() if value is not None}
    if args.sync:
        overrides["vector_mode"] = "sync"
    config = replace(config, **overrides).validate()
    print(json.dumps(train_dqn(config, algorithm, args.resume), indent=2))


if __name__ == "__main__":
    main()
