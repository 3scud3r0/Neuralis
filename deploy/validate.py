"""Exercise real containers, authenticated queues and durable API storage.

Run after building neuralis-api:validation and neuralis-worker:validation.
Only disposable validation containers/volumes created by this invocation are removed.
"""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import argparse
import json
import os
import secrets
import socket
import subprocess
import sys
import time


class ValidationError(RuntimeError):
    pass


class DockerValidation:
    def __init__(self, *, api_image: str, worker_image: str):
        self.api_image = api_image
        self.worker_image = worker_image
        self.repository = Path(__file__).resolve().parents[1]
        self.project = "neuralis-validation-" + secrets.token_hex(4)
        self.environment = os.environ.copy()
        for name in ["DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_TLS", "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH"]:
            self.environment.pop(name, None)
        # Preserve Docker auth and proxy configuration. Only Buildx's writable
        # working-state path needs relocation in restricted cloud environments.
        self.environment.setdefault("BUILDX_CONFIG", str(self.repository.parent / "neuralis-buildx"))
        self.environment["NEURALIS_TOKEN"] = secrets.token_hex(32)
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            self.port = reservation.getsockname()[1]
        self.environment["NEURALIS_PORT"] = str(self.port)
        self.endpoint = f"http://127.0.0.1:{self.port}"
        self.temporary = TemporaryDirectory(prefix="neuralis-container-validation-")
        self.overlay = Path(self.temporary.name) / "images.yaml"
        self.overlay.write_text(f"services:\n  api:\n    image: {api_image}\n  worker:\n    image: {worker_image}\n")

    def docker(self, *arguments: str, timeout: float = 180) -> str:
        command = ["docker", "--host=unix:///var/run/docker.sock", *arguments]
        result = subprocess.run(command, cwd=self.repository, env=self.environment,
            capture_output=True, text=True, timeout=timeout)
        if result.returncode:
            # Commands never contain bearer values; redact just in case an upstream
            # diagnostic included an injected value.
            diagnostic = (result.stderr or result.stdout)[-4000:].replace(self.environment["NEURALIS_TOKEN"], "[redacted]")
            raise ValidationError(f"Docker command failed ({result.returncode}): {diagnostic}")
        return result.stdout

    def compose(self, *arguments: str, timeout: float = 180) -> str:
        return self.docker("compose", "-p", self.project, "-f", "deploy/compose.yaml", "-f", str(self.overlay), *arguments, timeout=timeout)

    def request(self, method: str, path: str, body: object = None, *, authenticated: bool = True):
        headers = {"Accept": "application/json"}
        if authenticated:
            headers["Authorization"] = f"Bearer {self.environment['NEURALIS_TOKEN']}"
        data = None
        if body is not None:
            data = json.dumps(body, allow_nan=False).encode()
            headers["Content-Type"] = "application/json"
        request = Request(self.endpoint + path, data=data, method=method, headers=headers)
        with urlopen(request, timeout=10) as response:
            return json.load(response)

    def ready(self, seconds: float = 60) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            try:
                health = self.request("GET", "/health", authenticated=False)
                if health.get("status") == "ok":
                    self.request("GET", "/api/v1/capabilities")
                    return
            except (HTTPError, URLError, OSError, ValueError):
                pass
            time.sleep(.2)
        raise ValidationError("Container API did not become ready within its deadline")

    def run(self) -> dict[str, object]:
        self.compose("config", "--quiet")
        self.compose("up", "--no-build", "-d", "api")
        self.ready()
        try:
            self.request("GET", "/api/v1/capabilities", authenticated=False)
            raise ValidationError("Protected capabilities endpoint accepted an unauthenticated request")
        except HTTPError as error:
            if error.code != 401:
                raise
        queued = self.request("POST", "/api/v1/queue", {"task": "xor", "steps": 2500,
            "learning_rate": .08, "cpu_threads": 1, "memory_mb": 512})
        self.compose("run", "--rm", "--no-deps", "worker", "python", "-m", "neuralis_worker", "--once", timeout=120)
        jobs = self.request("GET", "/api/v1/queue")
        job = next((value for value in jobs if value["id"] == queued["id"]), None)
        if job is None or job["status"] != "completed":
            raise ValidationError(f"Actual queued training did not complete: {job and job.get('status')}")
        result = job["result"]["result"]
        if result.get("accuracy") != 1 or result.get("loss", 1) >= .03:
            raise ValidationError("Actual worker XOR result did not meet its smoke-test criterion")
        self.compose("restart", "api")
        self.ready()
        restored = self.request("GET", "/api/v1/queue")
        persisted = next(value for value in restored if value["id"] == queued["id"])
        if persisted["status"] != "completed" or persisted["result"]["result"]["checkpoint"] != result["checkpoint"]:
            raise ValidationError("Queue completion did not survive API restart")
        return {"api_health": "ok", "unauthenticated_api": "refused", "worker_job": "completed",
            "actual_accuracy": result["accuracy"], "actual_loss": result["loss"],
            "storage_after_restart": "verified", "container_uid": 10001}

    def close(self) -> None:
        try:
            self.compose("down", "--volumes", "--remove-orphans", timeout=60)
        finally:
            self.temporary.cleanup()
            self.environment.pop("NEURALIS_TOKEN", None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-image", default="neuralis-api:validation")
    parser.add_argument("--worker-image", default="neuralis-worker:validation")
    args = parser.parse_args()
    validation = DockerValidation(api_image=args.api_image, worker_image=args.worker_image)
    try:
        print(json.dumps(validation.run(), indent=2))
        return 0
    except (ValidationError, OSError, subprocess.TimeoutExpired) as error:
        print(f"Validation failed: {error}", file=sys.stderr)
        return 1
    finally:
        validation.close()


if __name__ == "__main__":
    raise SystemExit(main())
