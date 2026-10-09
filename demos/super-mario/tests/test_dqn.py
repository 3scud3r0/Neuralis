"""Replay, Double-DQN targets and checkpoint integrity numerical regressions."""
import numpy as np
import pytest
import torch

from mario_rl.config import Config
from mario_rl.diagnostics import summarize
from mario_rl.dqn import DQNConfig, DuelingQNetwork, double_dqn_targets, linear_schedule, update_q, update_target
from mario_rl.replay import NStepAccumulator, PriorityTree, PrioritizedReplay, Transition


def frame(value):
    return np.full((1, 2, 2), value, dtype=np.uint8)


def test_nstep_returns_and_terminal_flush():
    nstep = NStepAccumulator(3, 0.5)
    assert nstep.append(frame(0), 0, 1, frame(1), False, False) == []
    assert nstep.append(frame(1), 1, 2, frame(2), False, False) == []
    ready = nstep.append(frame(2), 2, 3, frame(3), True, False)
    assert [transition.reward for transition in ready] == [2.75, 3.5, 3]
    assert [transition.discount for transition in ready] == [0.125, 0.25, 0.5]
    assert all(transition.terminated for transition in ready)
    assert all(np.array_equal(transition.next_observation, frame(3)) for transition in ready)
    assert len(nstep.queue) == 0


def test_nstep_timeout_preserves_bootstrap():
    nstep = NStepAccumulator(3, 0.9)
    nstep.append(frame(0), 0, 1, frame(1), False, False)
    ready = nstep.append(frame(1), 1, 2, frame(2), False, True)
    assert len(ready) == 2
    assert ready[0].reward == pytest.approx(2.8)
    assert ready[0].discount == pytest.approx(0.81)
    assert not ready[0].terminated


def test_priority_tree_exact_prefix_search():
    tree = PriorityTree(3)
    for index, priority in enumerate([1, 2, 4]):
        tree.update(index, priority)
    assert tree.sum[1] == 7
    assert tree.minimum[1] == 1
    assert [tree.find(mass) for mass in [0, 0.9, 1, 2.99, 3, 6.99]] == [0, 0, 1, 1, 2, 2]
    with pytest.raises(ValueError):
        tree.find(7)


def test_prioritized_sampling_and_importance_weights():
    replay = PrioritizedReplay(4, (1, 2, 2), 42, alpha=1)
    for index in range(4):
        replay.add(Transition(frame(index), index, 1, frame(index + 1), False, 0.9))
    replay.update_priorities(np.arange(4), np.array([1, 1, 1, 50]))
    counts = np.zeros(4)
    for _ in range(100):
        batch = replay.sample(4, 1)
        assert (batch["weights"] > 0).all() and (batch["weights"] <= 1).all()
        for index in batch["indices"]:
            counts[index] += 1
    assert counts[3] > sum(counts[:3]) * 5
    assert replay.sample(4, 0)["weights"].tolist() == [1, 1, 1, 1]


def test_replay_ring_and_checksum_snapshot(tmp_path):
    replay = PrioritizedReplay(3, (1, 2, 2), 8)
    for index in range(5):
        replay.add(Transition(frame(index), 0, index, frame(index + 1), False, 0.9))
    assert len(replay) == 3 and replay.position == 2
    path = tmp_path / "buffer.npz"
    checksum = replay.save(path)
    restored = PrioritizedReplay.load(path, checksum)
    assert restored.position == replay.position
    assert np.array_equal(restored.observations, replay.observations)
    # RNG restoration reproduces stratified replay sampling exactly.
    assert np.array_equal(restored.sample(8, 0.4)["indices"], replay.sample(8, 0.4)["indices"])
    with pytest.raises(ValueError, match="checksum"):
        PrioritizedReplay.load(path, "f" * 64)


def test_replay_memory_budget():
    with pytest.raises(MemoryError, match="budget"):
        PrioritizedReplay(100000, (4, 84, 84), 0, max_memory_mb=10)


def test_double_dqn_uses_online_selection_target_estimation():
    online = torch.tensor([[1., 5., 3.], [2., 9., 1.]])
    target = torch.tensor([[100., 7., 20.], [50., 8., 60.]])
    returns = double_dqn_targets(online, target, torch.tensor([2., 3.]),
                                torch.tensor([0.5, 0.9]), torch.tensor([False, True]))
    assert returns.tolist() == [5.5, 3.0]


def test_schedule_bounds_and_target_interpolation():
    assert linear_schedule(1, 0.1, 10, -1) == 1
    assert linear_schedule(1, 0.1, 10, 5) == pytest.approx(0.55)
    assert linear_schedule(1, 0.1, 10, 20) == pytest.approx(0.1)
    torch.set_num_threads(2)
    online = DuelingQNetwork((4, 84, 84), 7)
    target = DuelingQNetwork((4, 84, 84), 7)
    with torch.no_grad():
        online.actor.bias.fill_(2)
        target.actor.bias.zero_()
    update_target(online, target, 0.25)
    assert torch.all(target.actor.bias == 0.5)


def test_double_dqn_update_is_finite_and_changes_weights():
    torch.set_num_threads(2)
    torch.manual_seed(8)
    online = DuelingQNetwork((4, 84, 84), 7)
    target = DuelingQNetwork((4, 84, 84), 7)
    target.load_state_dict(online.state_dict())
    replay = PrioritizedReplay(8, (4, 84, 84), 8)
    rng = np.random.default_rng(8)
    for index in range(8):
        obs = rng.integers(0, 255, size=(4, 84, 84), dtype=np.uint8)
        replay.add(Transition(obs, index % 7, float(index), obs, index % 2 == 0, 0.99))
    before = online.actor.weight.detach().clone()
    optimizer = torch.optim.Adam(online.parameters(), lr=0.001)
    result = update_q(online, target, optimizer, replay, DQNConfig(batch_size=4), 0.4, torch.device('cpu'))
    assert all(np.isfinite(value) for value in result.values())
    assert not torch.equal(before, online.actor.weight)


def test_diagnostics_reports_evidence_instead_of_perfection():
    data = {"episodes": [{"flagGet": False, "truncated": True, "xPos": 300, "stage": "1-1"}] * 15,
            "metrics": [{"entropy": 0.01, "approxKL": 0.2, "clipFraction": 0.8}]}
    report = summarize(data)
    codes = {finding["code"] for finding in report["findings"]}
    assert {"no-flags", "timeout-heavy", "repeated-failure-position", "policy-entropy-low", "policy-update-large", "ppo-clipping-heavy"} <= codes
    assert report["successes"] == 0
