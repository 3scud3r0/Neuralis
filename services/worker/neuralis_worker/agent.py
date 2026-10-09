from __future__ import annotations

from threading import Event
from typing import Any
import logging
import time

from .config import WorkerConfig
from .executor import JobProcess, capabilities
from .protocol import Coordinator, CoordinatorError, Lease
from .tasks import TaskConfig
from .ledger import Ledger

log = logging.getLogger("neuralis.worker")


class Worker:
    def __init__(self, config: WorkerConfig, coordinator: Coordinator | None = None):
        self.config = config.validate()
        self.coordinator = coordinator or Coordinator(config.endpoint, config.token, config.worker_id)
        self.stop_event = Event()
        self.active: JobProcess | None = None
        self.completed = 0
        self.failed = 0
        self.ledger = Ledger(config.work_dir)

    def stop(self) -> None:
        self.stop_event.set()
        if self.active:
            self.active.cancel("Worker shutdown requested")

    def execute(self, lease: Lease) -> tuple[str, dict[str, Any], str | None]:
        task = TaskConfig.parse(lease.config)
        self.active = JobProcess(task, self.config)
        heartbeat_period = max(1.0, lease.lease_seconds / 3)
        last_ack = time.monotonic()
        next_heartbeat = time.monotonic() + min(0.5, heartbeat_period)
        try:
            while self.active.poll() is None:
                if self.stop_event.is_set():
                    self.active.cancel("Worker shutdown requested")
                    break
                now = time.monotonic()
                if now >= next_heartbeat:
                    try:
                        cancelled = self.coordinator.heartbeat(lease, self.active.metric(), self.active.resources())
                        last_ack = time.monotonic()
                        if cancelled:
                            self.active.cancel("Coordinator cancelled this job")
                            break
                        next_heartbeat = time.monotonic() + heartbeat_period
                    except CoordinatorError as exc:
                        log.warning("Heartbeat failed: %s", exc)
                        if not exc.retryable or time.monotonic() - last_ack >= lease.lease_seconds:
                            self.active.cancel("Coordinator lease could not be renewed")
                            break
                        next_heartbeat = time.monotonic() + min(2.0, heartbeat_period)
                self.stop_event.wait(0.05)
            return self.active.finish()
        finally:
            self.active.close()
            self.active = None

    def flush_outbox(self) -> bool:
        for lease, state, result, error in self.ledger.pending():
            self.ledger.mark_delivery_attempt(lease.lease_id)
            try:
                self.coordinator.complete(lease, state, result, error)
                self.ledger.delivered(lease.lease_id)
            except CoordinatorError as exc:
                log.warning("Completion delivery pending for %s: %s", lease.job_id, exc)
                if not exc.retryable:
                    raise
                return False
        return True

    def run(self, *, once: bool = False, max_jobs: int | None = None) -> None:
        supported = capabilities(self.config)
        self.coordinator.register(supported)
        log.info("Registered %s; enabled tasks: %s", self.config.worker_id, ", ".join(supported["tasks"]))
        recovered = self.ledger.recover()
        if recovered:
            log.warning("Recovered %d interrupted leases as failed; execution will not repeat", recovered)
        acquired = 0
        while not self.stop_event.is_set():
            if not self.flush_outbox():
                if once:
                    raise CoordinatorError("Completion remains pending in the durable worker ledger", retryable=True)
                self.stop_event.wait(self.config.poll_seconds)
                continue
            self.ledger.check_quota()
            try:
                lease = self.coordinator.lease(supported["tasks"], {"cpu_threads": self.config.cpu_threads, "memory_mb": self.config.memory_mb}, self.config.lease_seconds)
            except CoordinatorError as exc:
                if not exc.retryable:
                    raise
                log.warning("Lease acquisition failed: %s", exc)
                if once:
                    raise
                self.stop_event.wait(self.config.poll_seconds)
                continue
            if lease is None:
                if once:
                    return
                self.stop_event.wait(self.config.poll_seconds)
                continue
            if not self.ledger.reserve(lease):
                log.warning("Duplicate lease %s was not executed again", lease.lease_id)
                if once:
                    return
                self.stop_event.wait(self.config.poll_seconds)
                continue
            acquired += 1
            try:
                state, result, error = self.execute(lease)
            except (ValueError, RuntimeError, OSError) as exc:
                state, result, error = "failed", {}, f"{type(exc).__name__}: {exc}"
            self.ledger.complete(lease, state, result, error)
            self.flush_outbox()
            self.completed += state == "completed"
            self.failed += state != "completed"
            log.info("Job %s finished with state %s", lease.job_id, state)
            if once or (max_jobs is not None and acquired >= max_jobs):
                return
