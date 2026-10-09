import itertools
import json
import threading
import time
import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from neuralis.core.schemas import CommandRequest, RunConfig


SUPPORTED_PARAMETERS = {"learning_rate", "weight_decay", "batch_size", "gradient_clip", "optimizer.name",
                        "optimizer.momentum", "schedule.name", "architecture.hidden_widths", "architecture.activation"}


class SweepRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(default="Hyperparameter sweep", min_length=1, max_length=120)
    base: RunConfig = Field(default_factory=RunConfig)
    grid: dict[str, list[Any]]
    seeds: list[int] = Field(default_factory=lambda: [42], min_length=1, max_length=8)
    objective: str = Field(default="validation_loss", pattern=r"^(validation_loss|validation_accuracy)$")

    @model_validator(mode="after")
    def bounded_grid(self):
        if not self.grid or not set(self.grid).issubset(SUPPORTED_PARAMETERS):
            raise ValueError(f"Grid parameters must come from {sorted(SUPPORTED_PARAMETERS)}")
        size = len(self.seeds)
        for values in self.grid.values():
            if not values or len(values) > 16:
                raise ValueError("Every sweep parameter requires 1 to 16 values")
            size *= len(values)
        if size > 64 or self.base.steps > 5000:
            raise ValueError("Sweep budget is 64 trials and 5,000 optimizer steps per trial")
        if any(seed < 0 or seed > 2147483647 for seed in self.seeds):
            raise ValueError("Sweep seed out of range")
        # Validate every combination before launching any work.
        list(expand_grid(self))
        return self


def expand_grid(request):
    keys = list(request.grid)
    for values in itertools.product(*(request.grid[key] for key in keys)):
        assignments = dict(zip(keys, values))
        for seed in request.seeds:
            configuration = request.base.model_dump()
            for path, value in assignments.items():
                parts = path.split(".")
                destination = configuration
                for part in parts[:-1]:
                    destination = destination[part]
                destination[parts[-1]] = value
            configuration["seed"] = seed
            configuration["name"] = f"{request.name}: seed {seed}, trial {assignments}"[:120]
            yield RunConfig.model_validate(configuration), assignments


class SweepManager:
    def __init__(self, supervisor):
        self.supervisor = supervisor
        self.repository = supervisor.repository
        self.threads = {}
        self.cancelled = set()
        self._lock = threading.RLock()
        with self.repository.transaction() as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS sweeps(id TEXT PRIMARY KEY,project_id TEXT NOT NULL,state TEXT NOT NULL,created_at REAL NOT NULL,request TEXT NOT NULL,trials TEXT NOT NULL,error TEXT)")
            connection.execute("UPDATE sweeps SET state='interrupted',error='Backend restarted; trials are not silently replayed' WHERE state='running'")

    def create(self, request):
        with self._lock:
            active = sum(thread.is_alive() for thread in self.threads.values())
            if active >= 2:
                raise ValueError("Two sweeps are already running; wait or cancel one")
            sweep_id = str(uuid.uuid4())
            with self.repository.transaction() as connection:
                connection.execute("INSERT INTO sweeps VALUES(?,?,?,?,?,?,?)", (sweep_id, request.base.project_id, "running", time.time(), json.dumps(request.model_dump()), "[]", None))
            thread = threading.Thread(target=self._execute, args=(sweep_id, request), daemon=True, name=f"sweep-{sweep_id[:8]}")
            self.threads[sweep_id] = thread
            thread.start()
        return self.get(sweep_id)

    def _save(self, sweep_id, state, trials, error=None):
        with self.repository.transaction() as connection:
            connection.execute("UPDATE sweeps SET state=?,trials=?,error=? WHERE id=?", (state, json.dumps(trials, allow_nan=False), error, sweep_id))

    def _execute(self, sweep_id, request):
        trials = []
        try:
            for config, assignment in expand_grid(request):
                if sweep_id in self.cancelled:
                    self._save(sweep_id, "cancelled", trials)
                    return
                row = self.supervisor.create(config)
                self.supervisor.command(row["id"], CommandRequest(command="start", command_id=f"sweep-{sweep_id}"))
                live = self.supervisor.get_live(row["id"])
                while live.thread.is_alive():
                    if sweep_id in self.cancelled and live.state not in {"completed", "stopped", "failed"}:
                        self.supervisor.command(live.id, CommandRequest(command="stop", command_id=f"cancel-{sweep_id}"))
                    live.thread.join(timeout=0.25)
                run = self.repository.get_run(live.id)
                trial = {"run_id": live.id, "seed": config.seed, "parameters": assignment,
                         "state": run["state"], "step": live.step, "metrics": live.validation, "error": run["error"]}
                trials.append(trial)
                self._save(sweep_id, "running", trials)
            self._save(sweep_id, "cancelled" if sweep_id in self.cancelled else "completed", trials)
        except Exception as error:
            self._save(sweep_id, "failed", trials, f"{type(error).__name__}: {error}")

    def get(self, sweep_id):
        with self.repository.transaction() as connection:
            row = connection.execute("SELECT * FROM sweeps WHERE id=?", (sweep_id,)).fetchone()
        if row is None:
            raise KeyError(sweep_id)
        value = dict(row)
        value["request"] = json.loads(value["request"])
        value["trials"] = json.loads(value["trials"])
        objective = value["request"]["objective"]
        valid = [trial for trial in value["trials"] if trial["state"] == "completed" and objective in trial["metrics"]]
        if valid:
            best = (min if objective == "validation_loss" else max)(valid, key=lambda trial: trial["metrics"][objective])
            value["best_run_id"] = best["run_id"]
            value["best_objective"] = best["metrics"][objective]
        else:
            value["best_run_id"] = value["best_objective"] = None
        value["interpretation"] = "Validation-selected configuration; evaluate on a separate untouched test set before reporting generalization."
        return value

    def list(self, project_id):
        with self.repository.transaction() as connection:
            rows = connection.execute("SELECT id FROM sweeps WHERE project_id=? ORDER BY created_at DESC LIMIT 100", (project_id,)).fetchall()
        return [self.get(row["id"]) for row in rows]

    def cancel(self, sweep_id):
        value = self.get(sweep_id)
        if value["state"] != "running":
            raise ValueError("Only a running sweep can be cancelled")
        with self._lock:
            self.cancelled.add(sweep_id)
        return {"id": sweep_id, "cancel_requested": True}

    def shutdown(self):
        with self._lock:
            self.cancelled.update(self.threads)
        for thread in list(self.threads.values()):
            thread.join(timeout=10)
