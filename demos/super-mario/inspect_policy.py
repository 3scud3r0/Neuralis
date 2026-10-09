#!/usr/bin/env python3
"""Compute true input attributions for one real emulator observation."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch

from mario_rl.environment import MarioEnv
from mario_rl.interpretability import grad_cam, gradient_attribution, heatmap_rgb, integrated_gradients
from mario_rl.policies import load_any_policy
from mario_rl.telemetry import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--method", choices=("gradient", "integrated-gradients", "grad-cam"), default="integrated-gradients")
    parser.add_argument("--steps", type=int, default=32, help="Integrated-gradient quadrature steps")
    parser.add_argument("--advance", type=int, default=10, help="Policy action steps before capturing observation")
    parser.add_argument("--action", type=int, help="Output action to explain, default policy argmax")
    parser.add_argument("--seed", type=int, default=100042)
    parser.add_argument("--output", type=Path, default=Path("runs/attribution"))
    args = parser.parse_args()
    if args.advance < 0:
        parser.error("advance cannot be negative")
    model, config, checkpoint = load_any_policy(args.checkpoint)
    torch.set_num_threads(config.torch_threads)
    env = MarioEnv(config, args.seed)
    try:
        observation = env.reset()
        for _ in range(args.advance):
            with torch.no_grad():
                action = int(model.act(torch.as_tensor(observation[None]), deterministic=True)[0].item())
            result = env.step(action)
            observation = result.observation
            if result.done:
                break
        with torch.no_grad():
            action = int(model.act(torch.as_tensor(observation[None]), deterministic=True)[0].item()) if args.action is None else args.action
        if args.method == "gradient":
            result = gradient_attribution(model, observation, action)
        elif args.method == "grad-cam":
            result = grad_cam(model, observation, action)
        else:
            result = integrated_gradients(model, observation, action, args.steps)
        args.output.mkdir(parents=True, exist_ok=True)
        Image.fromarray(env.render_rgb()).save(args.output / "001-emulator-observation.png")
        Image.fromarray(observation[-1]).save(args.output / "002-network-input.png")
        Image.fromarray(heatmap_rgb(result.attribution)).resize((336, 336), Image.Resampling.NEAREST).save(args.output / "003-attribution-map.png")
        np.save(args.output / "attribution.npy", result.attribution, allow_pickle=False)
        atomic_json(args.output / "attribution.json", {
            "method": result.method, "action": result.action, "score": result.score,
            "baselineScore": result.baseline_score, "completenessResidual": result.completeness_residual,
            "checkpointTransitions": checkpoint["step"], "observationShape": list(observation.shape),
            "note": "Attribution of the selected output score. Integrated gradients uses an all-black observation baseline; map colors are per-map normalized.",
        })
        print(f"Saved {result.method} attribution for action {action} to {args.output}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
