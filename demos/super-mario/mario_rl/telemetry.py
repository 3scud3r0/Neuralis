"""Append-only event logs and atomic, strictly finite JSON exports."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import tempfile
from typing import Any
import importlib.metadata


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def package_versions() -> dict:
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("torch", "numpy", "gym", "gym-super-mario-bros", "nes-py", "Pillow"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    return versions


class Telemetry:
    def __init__(self, output: Path, config: dict, architecture: dict, resume: bool = False):
        self.output = output
        self.path = output / "neuralis-mario-run.json"
        output.mkdir(parents=True, exist_ok=True)
        if resume and self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
            self.data["run"]["status"] = "running"
            self.data["run"]["resumedAt"] = utc_now()
            self.data.setdefault("resumeHistory", []).append({
                "at": self.data["run"]["resumedAt"], "previousConfig": self.data.get("config", {}),
                "newConfig": config,
            })
            self.data["config"] = config
            self.data["architecture"] = architecture
        else:
            self.data = {
                "schema": "neuralis.mario.run/v1",
                "run": {
                    "id": output.name,
                    "algorithm": "PPO (clipped actor-critic)",
                    "environment": config["env_id"],
                    "seed": config["seed"],
                    "device": config["device"],
                    "totalTransitions": 0,
                    "startedAt": utc_now(),
                    "status": "running",
                    "rewardDescription": "Native gym-super-mario-bros reward; PPO reward multiplied by reward_scale",
                },
                "config": config,
                "versions": package_versions(),
                "architecture": architecture,
                "metrics": [],
                "episodes": [],
                "events": [],
                "evaluations": [],
            }
        self.handle = (output / "events.jsonl").open("a", encoding="utf-8")

    def event(self, kind: str, step: int, data: dict) -> None:
        event = {"kind": kind, "step": int(step), "time": utc_now(), **data}
        self.handle.write(json.dumps(event, allow_nan=False) + "\n")
        self.handle.flush()
        if kind == "metrics":
            self.data["metrics"].append(event)
        elif kind == "episode":
            self.data["episodes"].append(event)
        elif kind == "evaluation":
            self.data["evaluations"].append(event)
            self.data["evaluation"] = data
        else:
            self.data["events"].append(event)
        self.data["run"]["totalTransitions"] = int(step)
        self.data["run"]["updatedAt"] = event["time"]

    def flush(self) -> None:
        atomic_json(self.path, self.data)

    def finish(self, status: str) -> None:
        self.data["run"]["status"] = status
        self.data["run"]["finishedAt"] = utc_now()
        self.flush()
        self.handle.close()
