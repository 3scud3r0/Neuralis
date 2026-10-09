from __future__ import annotations

from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from tempfile import TemporaryDirectory
import json
import time
import unittest

from neuralis_worker.agent import Worker
from neuralis_worker.config import WorkerConfig
from neuralis_worker.executor import JobProcess, capabilities
from neuralis_worker.protocol import Coordinator, CoordinatorError, Lease
from neuralis_worker.tasks import TaskConfig, train_xor

TOKEN = "test-authentication-token-32-bytes"


class CoordinatorFixture:
    def __init__(self, *, cancel: bool = False, bad_config: bool = False):
        self.messages: list[tuple[str, dict]] = []
        self.cancel = cancel
        self.bad_config = bad_config
        self.issued = False
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                if self.headers.get("Authorization") != f"Bearer {TOKEN}":
                    self.send_response(401)
                    self.end_headers()
                    return
                size = int(self.headers.get("Content-Length", "0"))
                value = json.loads(self.rfile.read(size))
                owner.messages.append((self.path, value))
                result: object = {"ok": True}
                if self.path.endswith("/lease"):
                    if owner.issued:
                        result = None
                    else:
                        owner.issued = True
                        result = {"lease_id": "lease-1", "job_id": "job-1", "lease_seconds": 15,
                            "config": {"task": "xor", "steps": 100000 if owner.cancel else 500, "capture_every": 50}}
                        if owner.bad_config:
                            result["config"] = {"task": "shell", "command": "touch unsafe"}
                elif self.path.endswith("/heartbeat"):
                    result = {"cancel": owner.cancel}
                encoded = json.dumps(result).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.endpoint = f"http://127.0.0.1:{self.server.server_port}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


class TrainingTests(unittest.TestCase):
    def test_xor_converges_with_real_gradients(self):
        events = []
        result = train_xor(TaskConfig(steps=2500), events.append)
        self.assertEqual(result["accuracy"], 1)
        self.assertLess(result["loss"], .03)
        self.assertGreater(events[0]["loss"], events[-1]["loss"])
        self.assertEqual(len(result["predictions"]), 4)

    def test_seed_is_repeatable(self):
        first = train_xor(TaskConfig(steps=100, seed=123), lambda _: None)
        second = train_xor(TaskConfig(steps=100, seed=123), lambda _: None)
        self.assertEqual(first["checkpoint"], second["checkpoint"])
        self.assertEqual(first["predictions"], second["predictions"])

    def test_config_refuses_untrusted_executables_and_nan(self):
        for value in [{"command": "python"}, {"task": "shell"}, {"steps": True}, {"learning_rate": float("nan")}, {"hidden_size": 1000000}]:
            with self.assertRaises(ValueError):
                TaskConfig.parse(value)


class SecurityTests(unittest.TestCase):
    def test_configuration_validates_transport_and_budgets(self):
        with TemporaryDirectory() as directory:
            config = WorkerConfig("http://127.0.0.1:8000", TOKEN, work_dir=Path(directory))
            self.assertEqual(config.validate(), config)
            for change in [{"endpoint": "http://remote.example"}, {"endpoint": "https://user:pass@remote.example"}, {"token": "short"}, {"worker_id": "../other"}, {"cpu_threads": 0}, {"memory_mb": 1}]:
                with self.assertRaises(ValueError):
                    replace(config, **change).validate()
            self.assertNotIn(TOKEN, repr(config))

    def test_lease_parse_checks_identifiers_and_duration(self):
        self.assertIsNone(Lease.parse(None))
        with self.assertRaises(CoordinatorError):
            Lease.parse({"lease_id": "a", "job_id": "b", "config": {}, "lease_seconds": 1})
        with self.assertRaises(CoordinatorError):
            Lease.parse({"lease_id": "", "job_id": "b", "config": {}})

    def test_unauthenticated_worker_is_refused(self):
        fixture = CoordinatorFixture()
        try:
            coordinator = Coordinator(fixture.endpoint, "wrong-token", "worker")
            with self.assertRaises(CoordinatorError) as error:
                coordinator.register({})
            self.assertIn("401", str(error.exception))
            self.assertFalse(error.exception.retryable)
            self.assertEqual(fixture.messages, [])
        finally:
            fixture.close()


class LifecycleTests(unittest.TestCase):
    def test_subprocess_reports_a_verified_result(self):
        with TemporaryDirectory() as directory:
            config = WorkerConfig("http://127.0.0.1:8000", TOKEN, work_dir=Path(directory))
            job = JobProcess(TaskConfig(steps=2500), config)
            try:
                while job.poll() is None:
                    time.sleep(.02)
                state, result, error = job.finish()
                self.assertEqual(state, "completed", error)
                self.assertEqual(result["accuracy"], 1)
                self.assertIsNone(error)
                self.assertIn("max_rss_mb", result["resources"])
            finally:
                job.close()

    def test_complete_protocol_records_actual_metrics(self):
        fixture = CoordinatorFixture()
        try:
            with TemporaryDirectory() as directory:
                config = WorkerConfig(fixture.endpoint, TOKEN, worker_id="cpu-1", work_dir=Path(directory))
                worker = Worker(config)
                worker.run(once=True)
                self.assertEqual(worker.completed, 1)
                paths = [path for path, _ in fixture.messages]
                self.assertIn("/api/v1/workers/register", paths)
                completions = [body for path, body in fixture.messages if path.endswith("/complete")]
                self.assertEqual(len(completions), 1)
                self.assertEqual(completions[0]["state"], "completed")
                self.assertIn("checkpoint", completions[0]["result"])
                self.assertEqual(completions[0]["lease_id"], "lease-1")
        finally:
            fixture.close()

    def test_remote_cancellation_stops_long_job(self):
        fixture = CoordinatorFixture(cancel=True)
        try:
            with TemporaryDirectory() as directory:
                worker = Worker(WorkerConfig(fixture.endpoint, TOKEN, work_dir=Path(directory)))
                started = time.monotonic()
                worker.run(once=True)
                self.assertLess(time.monotonic() - started, 5)
                completions = [body for path, body in fixture.messages if path.endswith("/complete")]
                self.assertEqual(completions[0]["state"], "cancelled")
                self.assertIn("cancelled", completions[0]["error"])
        finally:
            fixture.close()

    def test_unregistered_remote_task_is_rejected_before_spawn(self):
        fixture = CoordinatorFixture(bad_config=True)
        try:
            with TemporaryDirectory() as directory:
                worker = Worker(WorkerConfig(fixture.endpoint, TOKEN, work_dir=Path(directory)))
                worker.run(once=True)
                completions = [body for path, body in fixture.messages if path.endswith("/complete")]
                self.assertEqual(completions[0]["state"], "failed")
                self.assertIn("Unsupported task", completions[0]["error"])
                self.assertFalse((Path(directory) / "unsafe").exists())
        finally:
            fixture.close()

    def test_local_timeout_terminates_job(self):
        with TemporaryDirectory() as directory:
            config = WorkerConfig("http://127.0.0.1:8000", TOKEN, timeout_seconds=1, work_dir=Path(directory))
            job = JobProcess(TaskConfig(steps=100000, hidden_size=256), config)
            try:
                while job.poll() is None:
                    time.sleep(.05)
                state, result, error = job.finish()
                self.assertIn(state, {"failed", "cancelled"})
                self.assertIsNotNone(error)
                self.assertLess(result["elapsed_seconds"], 5)
            finally:
                job.close()


if __name__ == "__main__":
    unittest.main()
