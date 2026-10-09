"""Numerical and lifecycle regression tests, independent of emulator asset files."""
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import pytest
import torch

from mario_rl.checkpoint import load_policy, read_checkpoint, save_checkpoint
from mario_rl.config import Config
from mario_rl.environment import MarioEnv, preprocess, unpack_reset, unpack_step
from mario_rl.evaluation import wilson_interval
from mario_rl.model import ActorCritic
from mario_rl.ppo import Rollout, generalized_advantages, optimize
from mario_rl.telemetry import Telemetry, atomic_json
from mario_rl.training import Curriculum
from mario_rl.vector import auto_step


@pytest.fixture(autouse=True)
def bounded_torch_threads():
    torch.set_num_threads(2)


class Space:
    n = 3


class FakeEnv:
    action_space = Space()
    def __init__(self, new_api=False):
        self.new_api = new_api
        self.steps = 0
        self.closed = False
        self.seeds = []
    def reset(self, seed=None):
        self.seeds.append(seed)
        self.steps = 0
        frame = np.zeros((240, 256, 3), dtype=np.uint8)
        return (frame, {"x_pos": 0}) if self.new_api else frame
    def step(self, action):
        self.steps += 1
        frame = np.full((240, 256, 3), self.steps * 5, dtype=np.uint8)
        info = {"x_pos": self.steps * 3, "flag_get": False}
        done = self.steps == 5
        if self.new_api:
            return frame, 2.0, done, False, info
        return frame, 2.0, done, info
    def close(self):
        self.closed = True


def test_config_rejects_unknown_and_invalid():
    with pytest.raises(ValueError, match="Unknown"):
        Config.from_dict({"rollout_step": 16})
    with pytest.raises(ValueError, match="positive"):
        Config.from_dict({"num_envs": 0})
    with pytest.raises(ValueError, match="image_size"):
        Config.from_dict({"image_size": 8})
    assert Config.from_dict({"stages": ["SuperMarioBros-1-1-v0"]}).stages == ("SuperMarioBros-1-1-v0",)


@pytest.mark.parametrize("new_api", [False, True])
def test_environment_repeat_stack_and_terminal(new_api):
    raw = FakeEnv(new_api)
    config = Config(frame_skip=4, noop_max=0, reward_scale=0.1)
    env = MarioEnv(config, 13, raw)
    initial = env.reset()
    assert initial.shape == (4, 84, 84)
    assert initial.dtype == np.uint8
    first = env.step(1)
    assert first.reward == pytest.approx(0.8)
    assert not first.done
    assert first.observation[-1].mean() == 20
    assert first.observation[0].mean() == 0
    final = env.step(1)
    assert final.terminated and not final.truncated
    assert final.info["episode"]["reward"] == 10
    assert final.info["episode"]["frames"] == 5
    assert final.info["episode"]["xPos"] == 15
    env.close()
    assert raw.closed


def test_local_time_limit_autoreset_preserves_final_frame():
    env = MarioEnv(Config(frame_skip=1, max_episode_steps=2, noop_max=0), 9, FakeEnv())
    env.reset()
    assert not auto_step(env, 1).done
    final = auto_step(env, 1)
    assert final.truncated and not final.terminated
    assert final.info["final_observation"][-1].mean() == 10
    assert final.observation.mean() == 0
    assert final.info["episode"]["length"] == 2


def test_api_timeout_mapping_and_reset():
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    result = unpack_step((frame, 1, True, {"TimeLimit.truncated": True}))
    assert result.truncated and not result.terminated
    assert unpack_reset((frame, {"seed": 1}))[1] == {"seed": 1}
    with pytest.raises(RuntimeError):
        unpack_step((frame, 1, True))


def test_preprocessing_validation():
    assert preprocess(np.zeros((240, 256, 3), dtype=np.uint8), 84).shape == (84, 84)
    with pytest.raises(ValueError, match="RGB"):
        preprocess(np.zeros((2, 2)), 84)


def test_gae_terminal_and_timeout_distinction():
    rewards = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    values = torch.tensor([[0.5, 0.5], [0.6, 0.6]])
    next_values = torch.tensor([[10.0, 10.0], [3.0, 3.0]])
    terminated = torch.tensor([[True, False], [False, False]])
    ended = torch.tensor([[True, True], [False, False]])
    advantages, returns = generalized_advantages(rewards, values, next_values, terminated, ended, 0.9, 0.95)
    # Death ignores terminal value; timeout bootstraps without next-episode trace.
    assert advantages[0, 0].item() == pytest.approx(0.5)
    assert advantages[0, 1].item() == pytest.approx(9.5)
    assert returns[0, 1].item() == pytest.approx(10)
    assert advantages[1, 0].item() == pytest.approx(4.1)


def test_gae_multi_step_trace():
    zeros = torch.zeros((3, 1))
    rewards = torch.ones((3, 1))
    flags = torch.zeros((3, 1), dtype=torch.bool)
    advantages, _ = generalized_advantages(rewards, zeros, zeros, flags, flags, 1, 1)
    assert advantages.flatten().tolist() == [3, 2, 1]


def test_real_ppo_update_changes_policy_and_is_finite():
    torch.manual_seed(7)
    model = ActorCritic((4, 84, 84), 7)
    config = Config(num_envs=2, rollout_steps=4, minibatch_size=4, update_epochs=2)
    rollout = Rollout.allocate(4, 2, (4, 84, 84), torch.device("cpu"))
    rollout.observations.random_(0, 255)
    with torch.no_grad():
        actions, probs, _, values = model.act(rollout.observations.flatten(0, 1))
    rollout.actions.copy_(actions.reshape(4, 2))
    rollout.log_probs.copy_(probs.reshape(4, 2))
    rollout.values.copy_(values.reshape(4, 2))
    rollout.next_values.zero_()
    rollout.rewards.copy_(torch.arange(8).reshape(4, 2).float())
    rollout.terminated.fill_(True)
    rollout.ended.fill_(True)
    previous = model.actor.weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    result = optimize(model, optimizer, rollout, config)
    assert all(np.isfinite(result[key]) for key in ("loss", "approxKL", "entropy", "gradientNorm"))
    assert not torch.equal(previous, model.actor.weight)
    assert 0 <= result["clipFraction"] <= 1
    assert result["minibatches"] >= 2


def test_checkpoint_safe_roundtrip(tmp_path):
    torch.manual_seed(3)
    config = Config()
    model = ActorCritic((4, 84, 84), 7)
    optimizer = torch.optim.Adam(model.parameters())
    path = tmp_path / "policy.pt"
    save_checkpoint(path, model, optimizer, config, 512, 4, 0)
    loaded, restored, data = load_policy(path)
    assert data["step"] == 512
    assert restored.env_id == config.env_id
    assert all(torch.equal(value, loaded.state_dict()[name]) for name, value in model.state_dict().items())
    assert len(loaded.architecture()["layers"]) == 6
    inspection = loaded.inspect(torch.zeros((1, 4, 84, 84), dtype=torch.uint8))
    assert len(inspection["logits"][0]) == 7
    assert len(inspection["layers"][0]["sample"]) <= 128


def test_curriculum_requires_rolling_success_and_correct_stage():
    stages = ("SuperMarioBros-1-1-v0", "SuperMarioBros-1-2-v0")
    curriculum = Curriculum(stages, 3, 2 / 3)
    assert not curriculum.observe({"stage": stages[1], "flagGet": True})
    assert not curriculum.observe({"stage": stages[0], "flagGet": True})
    assert not curriculum.observe({"stage": stages[0], "flagGet": False})
    assert curriculum.observe({"stage": stages[0], "flagGet": True})
    assert curriculum.current == 1
    assert not curriculum.observe({"stage": stages[1], "flagGet": True})


def test_wilson_interval_small_sample_honesty():
    low, high = wilson_interval(0, 3)
    assert low == pytest.approx(0)
    assert 0.5 < high < 0.6
    low, high = wilson_interval(3, 3)
    assert 0.4 < low < 0.5
    assert high == pytest.approx(1)


def test_atomic_telemetry_and_nonfinite_rejection(tmp_path):
    telemetry = Telemetry(tmp_path, Config().to_dict(), {"layers": []})
    telemetry.event("metrics", 128, {"loss": 0.5})
    telemetry.event("episode", 128, {"reward": 10, "flagGet": False})
    telemetry.finish("completed")
    data = json.loads((tmp_path / "neuralis-mario-run.json").read_text())
    assert data["run"]["totalTransitions"] == 128
    assert data["metrics"][0]["loss"] == 0.5
    assert len((tmp_path / "events.jsonl").read_text().splitlines()) == 2
    with pytest.raises(ValueError):
        atomic_json(tmp_path / "invalid.json", {"value": float("nan")})
