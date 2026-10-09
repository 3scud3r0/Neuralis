from __future__ import annotations

from collections import deque
from dataclasses import asdict, replace
from pathlib import Path
from threading import Lock, Thread
from typing import Any
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time

from .config import WorkerConfig
from .tasks import TaskConfig


class ExecutionError(RuntimeError):
    pass


def capabilities(config: WorkerConfig) -> dict[str, Any]:
    torch = importlib.util.find_spec("torch") is not None
    distributed = torch and config.allow_distributed and os.name == "posix"
    return {
        "tasks": ["xor"] + (["ddp-xor"] if distributed else []),
        "cpu_count": config.cpu_threads,
        "memory_mb": config.memory_mb,
        "max_parallel": 1,
        "python": sys.version.split()[0],
        "torch_available": torch,
        "ddp_available": distributed,
        "hard_address_space_limit": os.name == "posix",
        "aggregate_rss_monitor": sys.platform.startswith("linux"),
    }


def _limits(config: WorkerConfig, distributed: bool):
    def apply() -> None:
        import resource
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_CPU, (config.timeout_seconds, config.timeout_seconds + 1))
        resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024 * 1024, 16 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
        if not distributed:
            # PyTorch reserves large virtual address ranges. DDP memory is monitored
            # using aggregate RSS on Linux instead of restricting virtual mappings.
            memory = config.memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    return apply


def _linux_tree_resources(pid: int) -> dict[str, float]:
    """Read non-secret process counters for the job and its direct descendants."""
    if not sys.platform.startswith("linux"):
        return {"max_rss_mb": 0.0, "cpu_seconds": 0.0}
    pending = [pid]
    visited: set[int] = set()
    rss = 0
    ticks = 0
    page_size = os.sysconf("SC_PAGE_SIZE")
    ticks_per_second = os.sysconf("SC_CLK_TCK")
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        try:
            stat = Path(f"/proc/{current}/stat").read_text()
            # The process name in parentheses can contain spaces.
            fields = stat[stat.rfind(")") + 2:].split()
            ticks += int(fields[11]) + int(fields[12])
            rss += int(fields[21]) * page_size
            children = Path(f"/proc/{current}/task/{current}/children").read_text()
            pending.extend(int(child) for child in children.split())
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError, IndexError):
            continue
    return {"max_rss_mb": rss / (1024 * 1024), "cpu_seconds": ticks / ticks_per_second}


class JobProcess:
    MAX_LINE = 1024 * 1024

    def __init__(self, task: TaskConfig, config: WorkerConfig):
        if task.cpu_threads is not None and task.cpu_threads > config.cpu_threads:
            raise ExecutionError("Job CPU request exceeds the worker budget")
        if task.memory_mb is not None and task.memory_mb > config.memory_mb:
            raise ExecutionError("Job memory request exceeds the worker budget")
        if task.world_size is not None and task.world_size > config.world_size:
            raise ExecutionError("Job rank request exceeds the worker budget")
        config = replace(config, cpu_threads=task.cpu_threads or config.cpu_threads,
            memory_mb=task.memory_mb or config.memory_mb, world_size=task.world_size or config.world_size,
            allow_distributed=config.allow_distributed and task.task == "ddp-xor").validate()
        self.task = task
        self.config = config
        self.started = time.monotonic()
        self.events: deque[dict[str, Any]] = deque(maxlen=200)
        self.errors: deque[str] = deque(maxlen=30)
        self.latest_metric: dict[str, Any] = {}
        self.result: dict[str, Any] | None = None
        self.reader_error: str | None = None
        self.lock = Lock()
        self._peak_rss = 0.0
        self._cpu_seconds = 0.0
        self.cancel_reason: str | None = None
        distributed = task.task == "ddp-xor"
        if distributed and "ddp-xor" not in capabilities(config)["tasks"]:
            raise ExecutionError("DDP task requires opt-in, POSIX and an installed PyTorch runtime")
        config.work_dir.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-m", "neuralis_worker.task_runner"]
        if distributed:
            command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
                f"--nproc-per-node={config.world_size}", "--module", "neuralis_worker.ddp_task"]
        environment = os.environ.copy()
        # Coordinator tokens are never inherited by model-training subprocesses.
        for name in ["NEURALIS_TOKEN", "NEURALIS_WORKER_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"]:
            environment.pop(name, None)
        environment.update({
            "OMP_NUM_THREADS": str(max(1, config.cpu_threads // (config.world_size if distributed else 1))),
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
            "NEURALIS_TASK_CONFIG": json.dumps(asdict(task), allow_nan=False),
        })
        # Preserve package importability when running from a source checkout.
        package_root = str(Path(__file__).resolve().parent.parent)
        environment["PYTHONPATH"] = package_root + os.pathsep + environment.get("PYTHONPATH", "")
        options: dict[str, Any] = {}
        if os.name == "posix":
            options.update(start_new_session=True, preexec_fn=_limits(config, distributed))
        else:
            options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        self.process = subprocess.Popen(command, cwd=config.work_dir.resolve(), env=environment,
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **options)
        self.stdout_reader = Thread(target=self._read_stdout, daemon=True)
        self.stderr_reader = Thread(target=self._read_stderr, daemon=True)
        self.stdout_reader.start()
        self.stderr_reader.start()

    def _read_stdout(self) -> None:
        assert self.process.stdout is not None
        while True:
            data = self.process.stdout.readline(self.MAX_LINE + 1)
            if not data:
                return
            if len(data) > self.MAX_LINE:
                with self.lock:
                    self.reader_error = "Training output exceeded the line budget"
                return
            prefix = b"NEURALIS_EVENT "
            if not data.startswith(prefix):
                with self.lock:
                    self.errors.append(data[:4096].decode("utf-8", errors="replace").strip())
                continue
            try:
                event = json.loads(data[len(prefix):])
                if not isinstance(event, dict):
                    raise ValueError("Event must be an object")
            except (ValueError, UnicodeDecodeError):
                with self.lock:
                    self.reader_error = "Training subprocess produced invalid JSON"
                return
            with self.lock:
                if event.get("type") == "metric":
                    self.latest_metric = {key: value for key, value in event.items() if key != "type"}
                    self.events.append(event)
                elif event.get("type") == "result" and isinstance(event.get("result"), dict):
                    self.result = event["result"]
                elif event.get("type") == "error":
                    self.errors.append(str(event.get("message", "Training failed"))[:1000])
                else:
                    self.reader_error = "Training subprocess produced an unsupported event"
                    return

    def _read_stderr(self) -> None:
        assert self.process.stderr is not None
        while True:
            data = self.process.stderr.readline(4097)
            if not data:
                return
            # Keep bounded diagnostics; avoid echoing process environment or arguments.
            with self.lock:
                self.errors.append(data[:4096].decode("utf-8", errors="replace").strip())

    def poll(self) -> int | None:
        resources = _linux_tree_resources(self.process.pid)
        self._peak_rss = max(self._peak_rss, resources["max_rss_mb"])
        self._cpu_seconds = max(self._cpu_seconds, resources["cpu_seconds"])
        if self.process.poll() is None:
            if time.monotonic() - self.started > self.config.timeout_seconds:
                self.cancel("Job exceeded its wall-clock budget")
            elif resources["max_rss_mb"] > self.config.memory_mb:
                self.cancel("Job exceeded its aggregate memory budget")
            elif self.reader_error:
                self.cancel(self.reader_error)
        return self.process.poll()

    def resources(self) -> dict[str, Any]:
        return {"max_rss_mb": self._peak_rss, "cpu_seconds": self._cpu_seconds,
            "elapsed_seconds": time.monotonic() - self.started, "cpu_threads": self.config.cpu_threads}

    def metric(self) -> dict[str, Any]:
        with self.lock:
            return dict(self.latest_metric)

    def cancel(self, reason: str) -> None:
        self.cancel_reason = reason
        if self.process.poll() is None:
            if os.name == "posix":
                try:
                    os.killpg(self.process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            else:
                self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    try:
                        os.killpg(self.process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    self.process.kill()
                self.process.wait(timeout=5)

    def finish(self) -> tuple[str, dict[str, Any], str | None]:
        code = self.process.wait()
        self.stdout_reader.join(timeout=2)
        self.stderr_reader.join(timeout=2)
        if self.cancel_reason:
            return "cancelled", self.resources(), self.cancel_reason
        if code != 0 or self.reader_error or self.result is None:
            error = self.reader_error or (self.errors[-1] if self.errors else f"Subprocess exited with code {code}")
            return "failed", self.resources(), error
        return "completed", {**self.result, "resources": self.resources()}, None

    def close(self) -> None:
        if self.process.poll() is None:
            self.cancel("Worker stopped")
        for stream in [self.process.stdout, self.process.stderr]:
            if stream:
                stream.close()
