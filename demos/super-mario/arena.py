#!/usr/bin/env python3
"""Execute a saved Mario policy and record real action frames in Neuralis RL Arena."""
from __future__ import annotations

import argparse
import base64
from dataclasses import replace
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import struct
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
import uuid
import zlib

import numpy as np
import torch

from mario_rl.environment import MarioEnv
from mario_rl.policies import load_any_policy


class ArenaError(RuntimeError):
    """A recording failed; no successful trajectory claim should be made."""


def encode_rgb_png(frame: np.ndarray) -> bytes:
    """Encode bounded RGB uint8 frames using standard noninterlaced PNG filter 0."""
    image = np.asarray(frame)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("Arena capture must be a uint8 RGB frame")
    height, width, _ = image.shape
    if not 1 <= height <= 480 or not 1 <= width <= 640:
        raise ValueError("Arena capture dimensions must be at most 640 by 480")

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))

    rows = b"".join(b"\x00" + row.tobytes() for row in image)
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows, level=6)) + chunk(b"IEND", b""))


class NoRedirect(HTTPRedirectHandler):
    """Do not forward API credentials to a redirected origin."""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class ArenaRecorder:
    def __init__(self, url: str, *, token: str | None = None, timeout: float = 10.0):
        parsed = urlsplit(url)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.path not in {"", "/"}):
            raise ValueError("--url must be the API HTTP(S) origin without credentials or a path")
        try:
            loopback = ipaddress.ip_address(parsed.hostname).is_loopback
        except ValueError:
            loopback = parsed.hostname == "localhost"
        if token and parsed.scheme != "https" and not loopback:
            raise ValueError("A token requires HTTPS except for a loopback API")
        if not math.isfinite(timeout) or not 0 < timeout <= 60:
            raise ValueError("HTTP timeout must be in (0, 60] seconds")
        if token and ("\r" in token or "\n" in token):
            raise ValueError("Invalid API token header")
        self._url, self._token, self.timeout = url.rstrip("/"), token, timeout
        self._opener = build_opener(NoRedirect())

    def record(self, episode_id: str, payload: dict) -> dict:
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", episode_id):
            raise ValueError("Episode ID must contain 1–128 letters, digits, '.', '_', or '-'")
        data = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._token:
            headers["Authorization"] = "Bearer " + self._token
        request = Request(self._url + "/api/v1/extensions/arena/episodes/"
                          + quote(episode_id, safe="") + "/frames",
                          data=data, headers=headers, method="POST")
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                raw = response.read(65537)
                if len(raw) > 65536:
                    raise ArenaError("Arena response exceeded the 64 KiB limit")
                try:
                    result = json.loads(raw) if raw else {}
                except (ValueError, UnicodeError) as error:
                    raise ArenaError("Arena returned an invalid JSON acknowledgment") from error
                if not isinstance(result, dict):
                    raise ArenaError("Arena acknowledgment must be a JSON object")
                if result.get("stored") is not True or result.get("step") != payload["step"]:
                    raise ArenaError("Arena did not confirm storage of the requested frame step")
                return result
        except HTTPError as error:
            # Server bodies and request headers are deliberately never logged.
            raise ArenaError(f"Arena rejected frame step {payload['step']} with HTTP {error.code}") from None
        except (URLError, TimeoutError, OSError):
            raise ArenaError(f"Arena transport failed at step {payload['step']}; delivery is unconfirmed") from None


@torch.no_grad()
def policy_decision(model, observation: np.ndarray, algorithm: str, stochastic: bool) -> dict:
    """Return actual action-head outputs and the action chosen from this observation."""
    tensor = torch.as_tensor(observation[None], device=next(model.parameters()).device)
    scores, value = model(tensor)
    if scores.ndim != 2 or scores.shape[0] != 1 or not torch.isfinite(scores).all():
        raise ValueError("Policy returned invalid or nonfinite action scores")
    scores = scores[0]
    if scores.numel() > 4096 or scores.abs().max().item() > 1e6:
        raise ValueError("Policy action scores exceed Arena limits")
    if algorithm == "ppo":
        distribution = torch.distributions.Categorical(logits=scores)
        action = distribution.sample() if stochastic else scores.argmax()
        outputs = distribution.probs
        diagnostics = {"policy_kind": "action_probabilities",
                       "logits": scores.cpu().tolist(),
                       "entropy": float(distribution.entropy().item())}
    elif algorithm == "double-dqn":
        if stochastic:
            raise ValueError("Double-DQN evaluation uses greedy actions; --stochastic applies to PPO")
        action, outputs = scores.argmax(), scores
        diagnostics = {"policy_kind": "q_values", "selection": "greedy"}
    else:
        raise ValueError("Unsupported policy algorithm")
    estimate = float(value[0].item())
    if not math.isfinite(estimate):
        raise ValueError("Policy returned a nonfinite value estimate")
    return {"policy": outputs.cpu().tolist(), "action": int(action.item()),
            "diagnostics": {**diagnostics, "value_estimate": estimate}}


def stream_episode(model, config, recorder: ArenaRecorder, episode_id: str, *,
                   algorithm: str, seed: int, max_steps: int = 300, fps: float = 15,
                   stochastic: bool = False, checkpoint_metadata: dict | None = None) -> dict:
    """Record each transition in order; stop immediately on an unacknowledged frame."""
    if not 1 <= max_steps <= 10000 or not math.isfinite(fps) or not 0 < fps <= 60:
        raise ValueError("Use 1–10000 steps and an upload rate in (0, 60] FPS")
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", episode_id):
        raise ValueError("Invalid episode ID")
    config = replace(config, stages=(), max_episode_steps=min(config.max_episode_steps, max_steps)).validate()
    env = MarioEnv(config, seed)
    uploaded, total_raw_reward = 0, 0.0
    storage_backend = None
    model.eval()
    try:
        if env.action_count != model.actions:
            raise ValueError("Checkpoint action count differs from the configured Mario environment")
        observation = env.reset(seed=seed)
        for step in range(config.max_episode_steps):
            began = time.monotonic()
            decision = policy_decision(model, observation, algorithm, stochastic)
            transition = env.step(decision["action"])
            raw_reward = float(transition.info.get("raw_reward", transition.reward))
            if not math.isfinite(raw_reward) or abs(raw_reward) > 1e6:
                raise ValueError("Emulator reward exceeds Arena limits")
            metadata = {
                **(checkpoint_metadata or {}), **decision["diagnostics"],
                "source": "mario_checkpoint_emulator", "algorithm": algorithm,
                "environment": config.env_id, "seed": seed,
                "environment_step": env.episode_steps, "emulator_frames": env.episode_frames,
                "frame_skip": config.frame_skip, "reward_kind": "raw_frame_skip_sum",
                "scaled_reward": float(transition.reward), "truncated": transition.truncated,
                "policy_observation": "pre_action_frame_stack", "frame_observation": "post_action_rgb",
                "x_pos": int(transition.info.get("x_pos", 0)),
                "flag_get": bool(transition.info.get("flag_get", False)),
                "snapshot_supported": False,
                "diagnostic_kind": "measured_policy_outputs",
            }
            acknowledgment = recorder.record(episode_id, {
                "step": step, "frame_png": base64.b64encode(encode_rgb_png(env.render_rgb())).decode("ascii"),
                "policy": decision["policy"], "action": decision["action"],
                "reward": raw_reward, "terminated": transition.terminated, "metadata": metadata,
            })
            storage_backend = acknowledgment.get("storage_backend", storage_backend)
            uploaded += 1
            total_raw_reward += raw_reward
            observation = transition.observation
            if transition.done:
                return {"episode_id": episode_id, "frames_recorded": uploaded,
                        "raw_reward": total_raw_reward, "terminated": transition.terminated,
                        "truncated": transition.truncated,
                        "storage_backend": storage_backend,
                        "flag_get": bool(transition.info.get("flag_get", False)),
                        "max_x": env.episode_max_x}
            remaining = 1 / fps - (time.monotonic() - began)
            if remaining > 0:
                time.sleep(remaining)
        raise ArenaError("Mario environment failed to report the action limit as truncation")
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Neuralis API origin")
    parser.add_argument("--episode-id", default=None, help="New unique trajectory ID (default: generated)")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--fps", type=float, default=15)
    parser.add_argument("--timeout", type=float, default=10)
    parser.add_argument("--seed", type=int, default=100042)
    parser.add_argument("--env-id", help="Override the saved Mario stage")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="cpu")
    parser.add_argument("--stochastic", action="store_true", help="Sample PPO actions; DDQN stays greedy")
    args = parser.parse_args()
    if not 1 <= args.episodes <= 100 or not 0 <= args.seed < 2**32:
        parser.error("Use 1–100 episodes and a seed in [0, 2^32)")
    try:
        recorder = ArenaRecorder(args.url, token=os.environ.get("NEURALIS_TOKEN"), timeout=args.timeout)
        model, config, checkpoint = load_any_policy(args.checkpoint, args.device)
        torch.set_num_threads(config.torch_threads)
        algorithm = "ppo" if checkpoint["format"] == "neuralis.mario.ppo/v1" else "double-dqn"
        if args.stochastic and algorithm != "ppo":
            raise ValueError("--stochastic is available only for PPO checkpoints")
        stage = args.env_id or (config.stages[checkpoint.get("stage", 0)] if config.stages else config.env_id)
        config = replace(config, env_id=stage)
        checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
        identifier = args.episode_id or "mario-" + uuid.uuid4().hex
        results = []
        for episode in range(args.episodes):
            episode_id = identifier if args.episodes == 1 else f"{identifier}-{episode + 1:03d}"
            seed = (args.seed + episode * 1009) % 2**32
            torch.manual_seed(seed)
            result = stream_episode(model, config, recorder, episode_id, algorithm=algorithm,
                                    seed=seed, max_steps=args.max_steps, fps=args.fps,
                                    stochastic=args.stochastic,
                                    checkpoint_metadata={"checkpoint_sha256": checkpoint_hash,
                                                         "checkpoint_transitions": int(checkpoint["step"])})
            results.append(result)
        print(json.dumps({"algorithm": algorithm, "episodes": results,
                          "note": "Measured emulator rollouts; finite runs do not establish perfect play."}, indent=2))
    except (ArenaError, ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"Mario Arena recording failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
