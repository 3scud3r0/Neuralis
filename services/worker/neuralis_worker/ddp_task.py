from __future__ import annotations

import json
import os
import sys
import time
from datetime import timedelta

from .tasks import TaskConfig
from .task_runner import emit


def main() -> int:
    try:
        import torch
        import torch.distributed as distributed
        from torch.nn.parallel import DistributedDataParallel
    except ImportError:
        emit({"type": "error", "message": "Install the distributed extra to execute DDP tasks"})
        return 1
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    config = TaskConfig.parse(json.loads(os.environ["NEURALIS_TASK_CONFIG"]))
    if config.task != "ddp-xor":
        raise ValueError("DDP launcher accepts only ddp-xor tasks")
    started = time.monotonic()
    distributed.init_process_group("gloo", timeout=timedelta(seconds=60))
    try:
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
        torch.manual_seed(config.seed)
        model = torch.nn.Sequential(torch.nn.Linear(2, config.hidden_size), torch.nn.Tanh(), torch.nn.Linear(config.hidden_size, 1))
        model = DistributedDataParallel(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
        inputs = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
        targets = torch.tensor([[0.], [1.], [1.], [0.]])
        # Every rank has exactly the same batch size. Splitting a repeated dataset
        # distributes observations without uneven-rank gradient averaging.
        repetitions = max(16, world_size)
        inputs = inputs.repeat(repetitions, 1)
        targets = targets.repeat(repetitions, 1)
        index = torch.arange(inputs.shape[0])[rank::world_size]
        local_inputs, local_targets = inputs[index], targets[index]
        criterion = torch.nn.BCEWithLogitsLoss()
        history = []
        for step in range(1, config.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            logits = model(local_inputs)
            loss = criterion(logits, local_targets)
            loss.backward()
            optimizer.step()
            if step == 1 or step == config.steps or step % config.capture_every == 0:
                statistics = torch.tensor([loss.item(), ((logits >= 0) == (local_targets >= .5)).float().mean().item()])
                rank_statistics = [torch.zeros_like(statistics) for _ in range(world_size)]
                distributed.all_gather(rank_statistics, statistics)
                distributed.all_reduce(statistics)
                statistics /= world_size
                if rank == 0:
                    event = {"type": "metric", "step": step, "loss": statistics[0].item(), "accuracy": statistics[1].item(), "world_size": world_size, "ranks": [{"rank": i, "loss": item[0].item(), "accuracy": item[1].item()} for i, item in enumerate(rank_statistics)], "elapsed_seconds": time.monotonic() - started}
                    emit(event)
                    history.append(event)
                    if len(history) > 1000:
                        history.pop(0)
        if rank == 0:
            with torch.no_grad():
                final_logits = model.module(torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]]))
                values = final_logits.sigmoid().flatten().tolist()
                final_loss = criterion(final_logits, torch.tensor([[0.], [1.], [1.], [0.]])).item()
            emit({"type": "result", "result": {"task": "ddp-xor", "seed": config.seed, "steps": config.steps,
                "world_size": world_size, "backend": "gloo", "loss": final_loss, "metrics": history, "predictions": values,
                "accuracy": sum((value >= .5) == bool(target) for value, target in zip(values, [0, 1, 1, 0])) / 4,
                "checkpoint": {name: value.tolist() for name, value in model.module.state_dict().items()}}})
        distributed.barrier()
        return 0
    except Exception as exc:
        if rank == 0:
            emit({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        return 1
    finally:
        distributed.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())
