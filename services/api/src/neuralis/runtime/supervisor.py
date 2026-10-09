import copy
from dataclasses import dataclass, field
import threading
import time
from typing import Any
import uuid

import torch
from torch import nn

from neuralis.core.schemas import CommandRequest, RunConfig, RunState, TERMINAL_STATES
from neuralis.datasets.synthetic import dataset_for
from neuralis.datasets.imports import DatasetStore
from neuralis.models.registry import build_model
from neuralis.runtime.capture import CaptureSession
from neuralis.runtime.devices import module_device, optimizer_for, resolve_device, scheduled_learning_rate
from neuralis.storage.checkpoints import CheckpointStore
from neuralis.storage.repository import Repository


class StateConflict(ValueError):
    pass


def loss_for(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


def build_training_model(config: RunConfig) -> nn.Module:
    model = build_model(config.model_id, config.seed, config.architecture)
    if config.adapter.enabled:
        from neuralis.runtime.training.qlora_engine import LoRAConfig, inject_lora
        inject_lora(model, LoRAConfig(rank=config.adapter.rank, alpha=config.adapter.alpha,
                                     target_modules=tuple(config.adapter.target_modules)),
                    quantize_base=config.adapter.quantize_base, seed=config.seed + 2000)
    return model.to(resolve_device(config.device))


def evaluate(model, split):
    was_training = model.training
    model.eval()
    total_loss, correct, count = 0., 0, 0
    try:
        with torch.no_grad():
            for start in range(0, len(split), 128):
                device = module_device(model)
                x, y = split.inputs[start:start + 128].to(device), split.targets[start:start + 128].to(device)
                logits = model(x)
                elements = y.numel()
                total_loss += float(loss_for(logits, y)) * elements
                correct += int((logits.argmax(-1) == y).sum())
                count += elements
    finally:
        model.train(was_training)
    return {"validation_loss": total_loss / count, "validation_accuracy": correct / count,
            "validation_examples": len(split), "validation_predictions": count}


@dataclass
class LiveRun:
    id: str
    config: RunConfig
    model: nn.Module
    optimizer: Any
    generator: torch.Generator
    dataset: Any
    state: RunState = RunState.CREATED
    step: int = 0
    condition: threading.Condition = field(default_factory=lambda: threading.Condition(threading.RLock()))
    model_lock: threading.RLock = field(default_factory=threading.RLock)
    permits: int = 0
    stop_requested: bool = False
    in_flight: bool = False
    thread: threading.Thread | None = None
    latest_tensors: dict = field(default_factory=dict)
    validation: dict = field(default_factory=dict)
    commands: dict = field(default_factory=dict)


class Supervisor:
    """CPU jobs are bounded threads; controls commit at optimizer boundaries.

    Threads avoid unsafe post-fork PyTorch state and share a small local runtime.
    They are cooperative: unsupported kernels that hang require restarting the API.
    Remote execution uses a separate authenticated worker lease queue.
    """

    def __init__(self, repository: Repository, max_parallel: int = 4):
        self.repository = repository
        self.checkpoints = CheckpointStore(repository.directory)
        self.datasets = DatasetStore(repository)
        self.max_parallel = max_parallel
        self.runs: dict[str, LiveRun] = {}
        self._lock = threading.RLock()
        torch.set_num_threads(1)

    def create(self, config: RunConfig):
        with self._lock:
            if len(self.runs) >= 256:
                removable = [key for key, value in self.runs.items() if value.state in TERMINAL_STATES]
                if not removable:
                    raise StateConflict("Local runtime has reached its loaded-model limit")
                del self.runs[removable[0]]
            model = build_training_model(config)
            dataset = self.dataset_for(config)
            optimizer = optimizer_for(model, config)
            generator = torch.Generator().manual_seed(config.seed + 1000)
            row = self.repository.create_run(config.model_dump())
            live = LiveRun(row["id"], config, model, optimizer, generator, dataset)
            self.runs[live.id] = live
            self.emit(live, "state", {"state": live.state, "reason": "created"})
            return self.repository.get_run(live.id)

    def get_live(self, run_id: str) -> LiveRun:
        with self._lock:
            if run_id not in self.runs:
                row = self.repository.get_run(run_id)
                checkpoints = self.repository.checkpoints(run_id)
                if not checkpoints:
                    raise StateConflict("Model is no longer loaded and no checkpoint is available")
                config = RunConfig.model_validate(row["config"])
                model = build_training_model(config)
                optimizer = optimizer_for(model, config)
                generator = torch.Generator().manual_seed(config.seed + 1000)
                manifest = self.checkpoints.load(checkpoints[0]["id"], model, optimizer, generator)
                self.runs[run_id] = LiveRun(run_id, config, model, optimizer, generator,
                                            self.dataset_for(config),
                                            RunState(row["state"]), manifest["step"])
            return self.runs[run_id]

    def dataset_for(self, config):
        if config.dataset_id:
            metadata = self.datasets.metadata(config.dataset_id)
            if metadata["project_id"] != config.project_id:
                raise ValueError("Imported dataset and run must belong to the same project")
            return self.datasets.load(config.dataset_id, config.model_id)
        return dataset_for(config.model_id, config.seed, config.architecture.context_length)

    def emit(self, live, kind, payload):
        return self.repository.append_event(live.id, live.step, kind, payload)

    def _state(self, live, state, reason):
        live.state = RunState(state)
        self.repository.update_run(live.id, state=state, step=live.step)
        self.emit(live, "state", {"state": state, "reason": reason})

    def _ensure_capacity(self, live):
        # Lock ordering never acquires another live run's condition.
        active = sum(run.thread is not None and run.thread.is_alive() for run in self.runs.values())
        if live.thread is None and active >= self.max_parallel:
            raise StateConflict(f"Local quota reached: {self.max_parallel} concurrent jobs")

    def command(self, run_id: str, request: CommandRequest):
        live = self.get_live(run_id)
        with live.condition:
            if request.command_id and request.command_id in live.commands:
                return live.commands[request.command_id]
            command = request.command
            if command == "checkpoint":
                result = self.save_checkpoint(live)
            elif command == "start":
                if live.state != RunState.CREATED:
                    raise StateConflict(f"Cannot start a {live.state} run")
                self._ensure_capacity(live)
                self._state(live, RunState.RUNNING, "start")
                self._launch(live)
                result = self._ack(live, command)
            elif command == "pause":
                if live.state != RunState.RUNNING:
                    raise StateConflict(f"Cannot pause a {live.state} run")
                self._state(live, RunState.PAUSED, "pause-request")
                deadline = time.monotonic() + 10
                while live.in_flight:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise StateConflict("Pause is pending; kernel has not reached an optimizer boundary")
                    live.condition.wait(remaining)
                result = self._ack(live, command)
            elif command == "step":
                if live.state not in {RunState.CREATED, RunState.PAUSED}:
                    raise StateConflict(f"Cannot step a {live.state} run")
                self._ensure_capacity(live)
                if live.state == RunState.CREATED:
                    self._state(live, RunState.PAUSED, "single-step")
                target = live.step + 1
                live.permits += 1
                self._launch(live)
                live.condition.notify_all()
                deadline = time.monotonic() + 30
                while live.step < target and live.state not in TERMINAL_STATES:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise StateConflict("Single step is pending; check run state before retrying")
                    live.condition.wait(remaining)
                result = self._ack(live, command)
            elif command == "resume":
                if live.state != RunState.PAUSED:
                    raise StateConflict(f"Cannot resume a {live.state} run")
                self._state(live, RunState.RUNNING, "resume")
                live.condition.notify_all()
                result = self._ack(live, command)
            else:
                if live.state in TERMINAL_STATES:
                    raise StateConflict(f"Cannot stop a {live.state} run")
                live.stop_requested = True
                live.condition.notify_all()
                if live.thread is None:
                    self._state(live, RunState.STOPPED, "stop-before-start")
                deadline = time.monotonic() + 10
                while live.thread and live.state not in TERMINAL_STATES:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise StateConflict("Stop is pending; kernel has not reached an optimizer boundary")
                    live.condition.wait(remaining)
                result = self._ack(live, command)
            if request.command_id:
                live.commands[request.command_id] = result
                if len(live.commands) > 512:
                    del live.commands[next(iter(live.commands))]
            return result

    @staticmethod
    def _ack(live, command):
        return {"run_id": live.id, "command": command, "state": live.state, "step": live.step,
                "acknowledged_at_boundary": not live.in_flight}

    def _launch(self, live):
        if live.thread is None:
            live.thread = threading.Thread(target=self._train, args=(live,), name=f"neuralis-{live.id[:8]}", daemon=True)
            live.thread.start()

    def _train(self, live):
        capture = None
        try:
            capture = CaptureSession(live.model, live.config.capture)
            with live.model_lock:
                live.validation = evaluate(live.model, live.dataset.validation)
                self.emit(live, "validation", {**live.validation, "baseline": True})
            while live.step < live.config.steps:
                with live.condition:
                    while live.state == RunState.PAUSED and live.permits == 0 and not live.stop_requested:
                        live.condition.wait()
                    if live.stop_requested:
                        break
                    if live.permits:
                        live.permits -= 1
                    live.in_flight = True
                start = time.perf_counter()
                with live.model_lock:
                    live.model.train()
                    indices = torch.randint(len(live.dataset.train), (live.config.batch_size,), generator=live.generator)
                    x, targets = live.dataset.train.batch(indices)
                    device = module_device(live.model)
                    x, targets = x.to(device), targets.to(device)
                    for group in live.optimizer.param_groups:
                        group["lr"] = scheduled_learning_rate(live.config, live.step)
                    capture.begin(live.step + 1)
                    live.optimizer.zero_grad(set_to_none=True)
                    logits = live.model(x)
                    loss = loss_for(logits, targets)
                    if not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite training loss; reduce learning rate or inspect inputs")
                    loss.backward()
                    gradient_norm = torch.nn.utils.clip_grad_norm_(live.model.parameters(), live.config.gradient_clip or float("inf"))
                    if not torch.isfinite(gradient_norm):
                        raise RuntimeError("Nonfinite gradient norm; optimizer update was not applied")
                    live.optimizer.step()
                    live.step += 1
                    tensors = capture.finish()
                    if tensors["tensors"]:
                        live.latest_tensors = tensors
                        self.emit(live, "tensors", tensors)
                    if live.step % live.config.validation_every == 0 or live.step == live.config.steps:
                        live.validation = evaluate(live.model, live.dataset.validation)
                    metrics = {"loss": float(loss.detach()), "accuracy": float((logits.argmax(-1) == targets).float().mean()),
                               "gradient_norm": float(gradient_norm), "learning_rate": live.optimizer.param_groups[0]["lr"],
                               "step_ms": (time.perf_counter() - start) * 1000,
                               "capture_ms": tensors["capture_ms"], **live.validation}
                    self.emit(live, "metrics", metrics)
                    self.repository.update_run(live.id, step=live.step)
                    if live.config.checkpoint_every and live.step % live.config.checkpoint_every == 0:
                        self.save_checkpoint(live)
                with live.condition:
                    live.in_flight = False
                    live.condition.notify_all()
                    if live.config.step_delay_ms:
                        live.condition.wait(live.config.step_delay_ms / 1000)
            with live.condition:
                self._state(live, RunState.STOPPED if live.stop_requested else RunState.COMPLETED,
                            "cancelled" if live.stop_requested else "step-budget-complete")
                live.in_flight = False
                live.condition.notify_all()
            # Save final actual weights; failures are surfaced instead of claiming durability.
            self.save_checkpoint(live)
        except BaseException as error:
            with live.condition:
                live.in_flight = False
                self.repository.update_run(live.id, error=f"{type(error).__name__}: {error}")
                self._state(live, RunState.FAILED, str(error))
                live.condition.notify_all()
        finally:
            if capture:
                capture.close()

    def save_checkpoint(self, live):
        with live.model_lock:
            manifest = self.checkpoints.save(live.id, live.config, live.model, live.optimizer, live.generator, live.step)
            self.repository.add_checkpoint(live.id, manifest["id"], live.step, manifest)
            self.emit(live, "checkpoint", {"id": manifest["id"], "step": live.step})
            return manifest

    def restore(self, run_id, checkpoint_id):
        source = self.repository.get_run(run_id)
        candidates = self.repository.checkpoints(run_id)
        if checkpoint_id not in {item["id"] for item in candidates}:
            raise KeyError(checkpoint_id)
        config = RunConfig.model_validate(source["config"])
        config.name += " (resumed)"
        row = self.create(config)
        live = self.get_live(row["id"])
        manifest = self.checkpoints.load(checkpoint_id, live.model, live.optimizer, live.generator)
        live.step = manifest["step"]
        self.repository.update_run(live.id, step=live.step)
        self.emit(live, "restored", {"source_run": run_id, "checkpoint_id": checkpoint_id, "step": live.step})
        return self.repository.get_run(live.id)

    def snapshot_model(self, run_id):
        live = self.get_live(run_id)
        with live.model_lock:
            model = copy.deepcopy(live.model)
            # Training capture hooks close over the live CaptureSession; copied hooks must be cleared.
            for module in model.modules():
                module._forward_hooks.clear()
                module._forward_pre_hooks.clear()
                module._backward_hooks.clear()
            model.eval()
            return model, live.dataset, live.config, live.step

    def shutdown(self):
        for live in list(self.runs.values()):
            with live.condition:
                live.stop_requested = True
                live.condition.notify_all()
        for live in list(self.runs.values()):
            if live.thread:
                live.thread.join(timeout=10)
