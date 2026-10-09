#!/usr/bin/env python3
"""Check legacy package compatibility and the actual emulator, without training."""
import argparse
import json
from pathlib import Path
import platform
import shutil

from PIL import Image
from mario_rl.config import Config
from mario_rl.environment import MarioEnv
from mario_rl.telemetry import package_versions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="SuperMarioBros-1-1-v0")
    parser.add_argument("--capture", type=Path, help="Optional actual emulator PNG")
    args = parser.parse_args()
    versions = package_versions()
    print(json.dumps({"versions": versions, "compiler": shutil.which("g++"),
                      "videoEncoder": shutil.which("ffmpeg"), "system": platform.system()}, indent=2))
    env = MarioEnv(Config(env_id=args.env_id), 42)
    try:
        observation = env.reset()
        total = 0.0
        for _ in range(64):
            transition = env.step(1)
            total += transition.info["raw_reward"]
            if transition.done:
                break
        print(json.dumps({"emulator": "ready", "observationShape": list(observation.shape),
                          "actions": env.action_count, "rawReward": total,
                          "xPosition": env.last_info.get("x_pos")}, indent=2))
        if args.capture:
            args.capture.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(env.render_rgb()).save(args.capture)
    finally:
        env.close()


if __name__ == "__main__":
    main()
