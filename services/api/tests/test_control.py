import time

import pytest
import torch

from neuralis.core.schemas import CommandRequest, RunConfig
from neuralis.runtime.supervisor import StateConflict


def command(supervisor, run_id, name, identifier=""):
    return supervisor.command(run_id, CommandRequest(command=name, command_id=identifier))


def wait_terminal(supervisor, run_id, timeout=30):
    live = supervisor.get_live(run_id)
    live.thread.join(timeout)
    assert not live.thread.is_alive(), "training did not terminate within the test budget"
    assert live.state in {"completed", "stopped"}, supervisor.repository.get_run(run_id)
    return live


def test_single_step_pause_and_idempotent_commands(supervisor):
    row = supervisor.create(RunConfig(steps=100, step_delay_ms=2))
    result = command(supervisor, row["id"], "step", "one")
    assert result["step"] == 1
    time.sleep(0.03)
    assert supervisor.get_live(row["id"]).step == 1
    assert command(supervisor, row["id"], "step", "one") == result
    assert supervisor.get_live(row["id"]).step == 1
    command(supervisor, row["id"], "resume")
    time.sleep(0.02)
    paused = command(supervisor, row["id"], "pause")
    step = paused["step"]
    assert not supervisor.get_live(row["id"]).in_flight
    time.sleep(0.03)
    assert supervisor.get_live(row["id"]).step == step
    result = command(supervisor, row["id"], "step")
    assert result["step"] == step + 1
    command(supervisor, row["id"], "stop")
    wait_terminal(supervisor, row["id"])


def test_xor_learns_real_truth_table(supervisor):
    row = supervisor.create(RunConfig(steps=250, learning_rate=0.03, capture_every=100))
    command(supervisor, row["id"], "start")
    live = wait_terminal(supervisor, row["id"])
    metrics = supervisor.repository.events(row["id"], 0, 1000, "metrics")["events"]
    assert len(metrics) == 250
    assert metrics[-1]["payload"]["validation_loss"] < metrics[0]["payload"]["validation_loss"] * 0.2
    assert metrics[-1]["payload"]["validation_accuracy"] == 1
    assert live.latest_tensors["tensors"]
    assert supervisor.repository.checkpoints(row["id"])


@pytest.mark.parametrize("model_id,steps,learning_rate", [("cnn-shapes", 120, 0.005), ("tiny-transformer", 60, 0.005)])
def test_other_models_improve_validation_loss(supervisor, model_id, steps, learning_rate):
    row = supervisor.create(RunConfig(model_id=model_id, steps=steps, learning_rate=learning_rate,
                                     batch_size=32, capture_every=100, validation_every=10))
    command(supervisor, row["id"], "start")
    wait_terminal(supervisor, row["id"], timeout=60)
    metrics = supervisor.repository.events(row["id"], 0, 1000, "metrics")["events"]
    assert metrics[-1]["payload"]["validation_loss"] < metrics[0]["payload"]["validation_loss"]


def test_checkpoint_preserves_optimizer_rng_next_step(supervisor):
    first = supervisor.create(RunConfig(steps=20, learning_rate=0.01, capture_every=20))
    command(supervisor, first["id"], "step")
    manifest = command(supervisor, first["id"], "checkpoint")
    second = supervisor.restore(first["id"], manifest["id"])
    command(supervisor, first["id"], "step")
    command(supervisor, second["id"], "step")
    first_live, second_live = supervisor.get_live(first["id"]), supervisor.get_live(second["id"])
    for first_parameter, second_parameter in zip(first_live.model.parameters(), second_live.model.parameters()):
        assert torch.equal(first_parameter, second_parameter)
    command(supervisor, first["id"], "stop")
    command(supervisor, second["id"], "stop")


def test_invalid_transition_is_rejected(supervisor):
    row = supervisor.create(RunConfig(steps=1))
    with pytest.raises(StateConflict):
        command(supervisor, row["id"], "pause")
    command(supervisor, row["id"], "stop")
    with pytest.raises(StateConflict):
        command(supervisor, row["id"], "resume")


def test_snapshot_copied_training_hooks_do_not_capture_inference(supervisor):
    row = supervisor.create(RunConfig(steps=20, capture_every=1))
    command(supervisor, row["id"], "step")
    live = supervisor.get_live(row["id"])
    before = live.latest_tensors
    copied, data, config, step = supervisor.snapshot_model(row["id"])
    assert all(not module._forward_hooks for module in copied.modules())
    copied(data.validation.inputs)
    assert live.latest_tensors is before
    command(supervisor, row["id"], "stop")
