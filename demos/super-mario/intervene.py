#!/usr/bin/env python3
"""Measure activation ablation effects on actual seeded emulator evaluations."""
import argparse
from pathlib import Path
import torch
from mario_rl.evaluation import evaluate
from mario_rl.interpretability import activation_intervention
from mario_rl.policies import load_any_policy
from mario_rl.telemetry import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--layer", default="trunk.1")
    parser.add_argument("--channels", default="", help="Comma-separated channel indices; empty means all channels")
    parser.add_argument("--scale", type=float, default=0, help="0 ablates, 1 is identity")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=100042)
    parser.add_argument("--output", type=Path, default=Path("runs/intervention"))
    args = parser.parse_args()
    channels = tuple(int(channel.strip()) for channel in args.channels.split(',') if channel.strip())
    model, config, data = load_any_policy(args.checkpoint)
    torch.set_num_threads(config.torch_threads)
    original = evaluate(model, config, args.episodes, args.seed, output=args.output / "original", max_captures=6)
    with activation_intervention(model, args.layer, channels, args.scale):
        changed = evaluate(model, config, args.episodes, args.seed, output=args.output / "intervened", max_captures=6)
    report = {
        "schema": "neuralis.mario.intervention/v1", "layer": args.layer, "channels": list(channels),
        "scale": args.scale, "checkpointTransitions": data["step"],
        "original": original, "intervened": changed,
        "meanReturnDelta": changed["meanReturn"] - original["meanReturn"],
        "successRateDelta": changed["successRate"] - original["successRate"],
        "note": "Same seeded starts and per-episode limits. Effects are observed in this finite evaluation; they do not prove general causal semantics.",
    }
    atomic_json(args.output / "intervention.json", report)
    print(f"Mean native return: {original['meanReturn']:.1f} → {changed['meanReturn']:.1f}")


if __name__ == "__main__":
    main()
