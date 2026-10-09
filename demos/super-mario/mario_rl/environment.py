"""NES environment adapter: real rewards, frame repeats and uint8 frame stacks.

The wrapper deliberately avoids depending on Gym's changing Wrapper API. It
accepts both four-item and five-item step results. Time-limit transitions are
kept distinct from true terminal transitions so PPO can bootstrap correctly.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any
import numpy as np
from PIL import Image

from .config import Config


@dataclass
class Step:
    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


def unpack_reset(result: Any) -> tuple[np.ndarray, dict[str, Any]]:
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
        return np.asarray(result[0]), result[1]
    return np.asarray(result), {}


def unpack_step(result: tuple) -> Step:
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
    elif len(result) == 4:
        obs, reward, done, info = result
        truncated = bool(info.get("TimeLimit.truncated", False))
        terminated = bool(done) and not truncated
    else:
        raise RuntimeError(f"Unsupported step result with {len(result)} fields")
    return Step(np.asarray(obs), float(reward), bool(terminated), bool(truncated), dict(info))


def preprocess(frame: np.ndarray, size: int) -> np.ndarray:
    """Use Pillow's RGB-to-luminance transform, area-like BOX downsampling."""
    if frame.ndim != 3 or frame.shape[-1] not in {3, 4}:
        raise ValueError(f"Expected RGB/RGBA frame; received {frame.shape}")
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("L")
    return np.asarray(image.resize((size, size), Image.Resampling.BOX), dtype=np.uint8)


class MarioEnv:
    def __init__(self, config: Config, seed: int, raw_env: Any | None = None):
        self.config = config
        self.seed_value = seed
        self.rng = np.random.default_rng(seed)
        self.frames: deque[np.ndarray] = deque(maxlen=config.frame_stack)
        self.stage_index = 0
        self.stages = config.stages or (config.env_id,)
        self.env = raw_env if raw_env is not None else self._create(self.stages[0])
        self._external_env = raw_env is not None
        self.action_count = int(self.env.action_space.n)
        self.observation_shape = (config.frame_stack, config.image_size, config.image_size)
        self.episode_steps = 0
        self.episode_frames = 0
        self.episode_reward = 0.0
        self.episode_max_x = 0
        self.episode_number = 0
        self.latest_rgb: np.ndarray | None = None
        self.last_info: dict[str, Any] = {}
        self._seed_pending = True

    def _create(self, env_id: str):
        import gym_super_mario_bros
        from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY, COMPLEX_MOVEMENT
        from nes_py.wrappers import JoypadSpace
        actions = {"simple": SIMPLE_MOVEMENT, "right": RIGHT_ONLY, "complex": COMPLEX_MOVEMENT}
        # Gym 0.23 is pinned: no passive checker requiring a newer Gym API.
        raw = gym_super_mario_bros.make(env_id)
        return JoypadSpace(raw, actions[self.config.action_set])

    @property
    def stage(self) -> str:
        return self.stages[self.stage_index]

    def set_stage(self, index: int) -> None:
        if self._external_env:
            raise RuntimeError("Cannot replace an externally supplied environment")
        if not 0 <= index < len(self.stages):
            raise ValueError("Stage index is outside the configured curriculum")
        if index == self.stage_index:
            return
        self.env.close()
        self.stage_index = index
        self.env = self._create(self.stage)
        if int(self.env.action_space.n) != self.action_count:
            raise RuntimeError("Curriculum action spaces must have the same size")
        self._seed_pending = True

    def _raw_reset(self):
        if self._seed_pending:
            try:
                result = self.env.reset(seed=self.seed_value)
            except TypeError:
                if hasattr(self.env, "seed"):
                    self.env.seed(self.seed_value)
                result = self.env.reset()
            self._seed_pending = False
        else:
            result = self.env.reset()
        return unpack_reset(result)

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.seed_value = int(seed)
            self.rng = np.random.default_rng(seed)
            self._seed_pending = True
        frame, info = self._raw_reset()
        # Random initial no-ops diversify training; evaluation sets noop_max=0.
        for _ in range(int(self.rng.integers(0, self.config.noop_max + 1))):
            transition = unpack_step(self.env.step(0))
            frame, info = transition.observation, transition.info
            if transition.done:
                frame, info = self._raw_reset()
        self.latest_rgb = frame.copy()
        processed = preprocess(frame, self.config.image_size)
        self.frames.clear()
        for _ in range(self.config.frame_stack):
            self.frames.append(processed.copy())
        self.episode_steps = 0
        self.episode_frames = 0
        self.episode_reward = 0.0
        self.episode_max_x = int(info.get("x_pos", 0))
        self.last_info = dict(info)
        return self.observation()

    def observation(self) -> np.ndarray:
        return np.stack(tuple(self.frames), axis=0)

    def step(self, action: int) -> Step:
        if not 0 <= int(action) < self.action_count:
            raise ValueError(f"Action {action} is outside [0, {self.action_count})")
        if not self.frames:
            raise RuntimeError("reset() must precede step()")
        recent: deque[np.ndarray] = deque(maxlen=2)
        reward = 0.0
        transition: Step | None = None
        for _ in range(self.config.frame_skip):
            transition = unpack_step(self.env.step(int(action)))
            reward += transition.reward
            recent.append(transition.observation)
            self.episode_frames += 1
            if transition.done:
                break
        assert transition is not None
        self.latest_rgb = transition.observation.copy()
        # Max-pool the final two frames to reduce NES sprite flicker.
        pooled = np.maximum(recent[0], recent[-1])
        self.frames.append(preprocess(pooled, self.config.image_size))
        self.episode_steps += 1
        self.episode_reward += reward
        self.episode_max_x = max(self.episode_max_x, int(transition.info.get("x_pos", 0)))
        terminated = transition.terminated
        truncated = transition.truncated or (self.episode_steps >= self.config.max_episode_steps and not terminated)
        info = dict(transition.info)
        info["raw_reward"] = reward
        info["stage"] = self.stage
        if terminated or truncated:
            self.episode_number += 1
            info["episode"] = {
                "reward": self.episode_reward,
                "length": self.episode_steps,
                "frames": self.episode_frames,
                "flagGet": bool(info.get("flag_get", False)),
                "xPos": self.episode_max_x,
                "stage": self.stage,
                "terminated": terminated,
                "truncated": truncated,
            }
        self.last_info = info
        return Step(self.observation(), reward * self.config.reward_scale, terminated, truncated, info)

    def render_rgb(self) -> np.ndarray:
        if self.latest_rgb is None:
            raise RuntimeError("No rendered frame is available before reset()")
        return self.latest_rgb.copy()

    def render_human(self) -> None:
        self.env.render()

    def close(self) -> None:
        self.env.close()
