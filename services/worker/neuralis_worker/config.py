from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit
import os
import re


@dataclass(frozen=True)
class WorkerConfig:
    endpoint: str
    token: str = field(repr=False)
    worker_id: str = "local-cpu"
    cpu_threads: int = 2
    memory_mb: int = 2048
    timeout_seconds: int = 300
    lease_seconds: int = 60
    poll_seconds: float = 2.0
    work_dir: Path = Path(".neuralis-worker")
    allow_distributed: bool = False
    world_size: int = 2

    def validate(self) -> "WorkerConfig":
        parsed = urlsplit(self.endpoint)
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Endpoint cannot contain credentials, query parameters or fragments")
        if parsed.scheme != "https" and not (
            parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"}
        ):
            raise ValueError("Remote coordinator requires HTTPS")
        if not parsed.hostname:
            raise ValueError("Coordinator endpoint requires a hostname")
        if not self.token or len(self.token) < 16 or any(ord(ch) < 32 for ch in self.token):
            raise ValueError("A valid coordinator token of at least 16 characters is required")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", self.worker_id):
            raise ValueError("Worker ID must contain only letters, digits, underscores and dashes")
        if not 1 <= self.cpu_threads <= 64:
            raise ValueError("CPU thread budget must be between 1 and 64")
        if not 128 <= self.memory_mb <= 131072:
            raise ValueError("Memory budget must be between 128 MiB and 128 GiB")
        if not 1 <= self.timeout_seconds <= 86400:
            raise ValueError("Job timeout must be between 1 second and 24 hours")
        if not 15 <= self.lease_seconds <= 300:
            raise ValueError("Lease duration must be between 15 and 300 seconds")
        if not 0.1 <= self.poll_seconds <= 60:
            raise ValueError("Poll interval must be between 0.1 and 60 seconds")
        if not 1 <= self.world_size <= 8 or (self.allow_distributed and self.world_size > self.cpu_threads):
            raise ValueError("World size must fit within the CPU thread budget and at most 8 ranks")
        return self

    @classmethod
    def from_environment(cls, **overrides: object) -> "WorkerConfig":
        values: dict[str, object] = {
            "endpoint": os.environ.get("NEURALIS_COORDINATOR", "http://127.0.0.1:8000"),
            "token": os.environ.get("NEURALIS_WORKER_TOKEN", os.environ.get("NEURALIS_TOKEN", "")),
            "worker_id": os.environ.get("NEURALIS_WORKER_ID", "local-cpu"),
            "cpu_threads": int(os.environ.get("NEURALIS_WORKER_CPU_THREADS", "2")),
            "memory_mb": int(os.environ.get("NEURALIS_WORKER_MEMORY_MB", "2048")),
            "timeout_seconds": int(os.environ.get("NEURALIS_WORKER_TIMEOUT", "300")),
            "work_dir": Path(os.environ.get("NEURALIS_WORKER_DIR", ".neuralis-worker")),
        }
        values.update({key: value for key, value in overrides.items() if value is not None})
        return cls(**values).validate()  # type: ignore[arg-type]
