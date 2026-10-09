from __future__ import annotations

from typing import Any
import json
import os
import sys

from .tasks import TaskConfig, train_xor


def emit(value: dict[str, Any]) -> None:
    print("NEURALIS_EVENT " + json.dumps(value, allow_nan=False, separators=(",", ":")), flush=True)


def main() -> int:
    try:
        raw = os.environ.pop("NEURALIS_TASK_CONFIG", "{}")
        if len(raw) > 8192:
            raise ValueError("Task configuration exceeds 8 KiB")
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("Task config must be an object")
        task = TaskConfig.parse(value)
        if task.task != "xor":
            raise ValueError("Distributed tasks require the dedicated DDP launcher")
        result = train_xor(task, emit)
        emit({"type": "result", "result": result})
        return 0
    except Exception as exc:
        emit({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        return 1


if __name__ == "__main__":
    sys.exit(main())
