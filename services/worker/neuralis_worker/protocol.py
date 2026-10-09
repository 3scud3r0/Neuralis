from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import quote
from urllib.request import Request, build_opener, HTTPRedirectHandler, HTTPSHandler
from urllib.error import HTTPError, URLError
import json
import ssl
import time


class CoordinatorError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward a bearer token to a redirect destination.
        return None


@dataclass(frozen=True)
class Lease:
    lease_id: str
    job_id: str
    config: dict[str, Any]
    lease_seconds: int = 60

    @classmethod
    def parse(cls, value: object) -> "Lease | None":
        if value is None or value == {}:
            return None
        if not isinstance(value, dict):
            raise CoordinatorError("Lease response must be an object or null")
        lease_id, job_id, config = value.get("lease_id"), value.get("job_id"), value.get("config")
        if not isinstance(lease_id, str) or not isinstance(job_id, str) or not lease_id or not job_id:
            raise CoordinatorError("Lease response is missing its identifiers")
        if len(lease_id) > 128 or len(job_id) > 128:
            raise CoordinatorError("Lease identifiers exceed the size budget")
        if not isinstance(config, dict):
            raise CoordinatorError("Lease config must be an object")
        duration = value.get("lease_seconds", 60)
        if not isinstance(duration, int) or not 15 <= duration <= 300:
            raise CoordinatorError("Invalid lease duration")
        return cls(lease_id, job_id, config, duration)


class Coordinator:
    MAX_RESPONSE = 1024 * 1024

    def __init__(self, endpoint: str, token: str, worker_id: str, timeout: float = 10.0):
        self.endpoint = endpoint.rstrip("/")
        self._token = token
        self.worker_id = worker_id
        self.timeout = timeout
        self.ssl_context = ssl.create_default_context()
        self.opener = build_opener(NoRedirect(), HTTPSHandler(context=self.ssl_context))
        # urllib's HTTPS handler verifies system trust by default; custom bundles can
        # be selected with SSL_CERT_FILE without disabling hostname verification.

    def _post(self, path: str, payload: dict[str, Any], retries: int = 0) -> Any:
        encoded = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode()
        if len(encoded) > self.MAX_RESPONSE:
            raise CoordinatorError("Request exceeds 1 MiB")
        request = Request(
            f"{self.endpoint}/api/v1{path}", data=encoded, method="POST",
            headers={"Authorization": f"Bearer {self._token}", "Content-Type": "application/json", "Accept": "application/json"},
        )
        for attempt in range(retries + 1):
            try:
                with self.opener.open(request, timeout=self.timeout) as response:
                    if response.status == 204:
                        return None
                    data = response.read(self.MAX_RESPONSE + 1)
                    if len(data) > self.MAX_RESPONSE:
                        raise CoordinatorError("Coordinator response exceeds 1 MiB")
                    if not data:
                        return None
                    return json.loads(data)
            except HTTPError as exc:
                retryable = exc.code in {429, 500, 502, 503, 504}
                error = CoordinatorError(f"Coordinator returned HTTP {exc.code}", retryable=retryable)
            except (URLError, TimeoutError, OSError) as exc:
                error = CoordinatorError(f"Coordinator unavailable ({type(exc).__name__})", retryable=True)
            except (ValueError, UnicodeDecodeError) as exc:
                raise CoordinatorError("Coordinator returned invalid JSON") from exc
            if not error.retryable or attempt == retries:
                raise error
            time.sleep(min(2 ** attempt * 0.2, 3))
        raise AssertionError("Unreachable retry state")

    @property
    def prefix(self) -> str:
        return f"/workers/{quote(self.worker_id, safe='')}"

    def register(self, capabilities: dict[str, Any]) -> Any:
        return self._post("/workers/register", {"worker_id": self.worker_id, "capabilities": capabilities}, retries=2)

    def lease(self, tasks: list[str], resources: dict[str, int], seconds: int) -> Lease | None:
        # A lease is a mutation: never silently retry acquisition after a timeout.
        return Lease.parse(self._post(f"{self.prefix}/lease", {"tasks": tasks, "resources": resources, "lease_seconds": seconds}))

    def heartbeat(self, lease: Lease, metrics: dict[str, Any], resources: dict[str, Any]) -> bool:
        response = self._post(f"{self.prefix}/heartbeat", {
            "lease_id": lease.lease_id, "job_id": lease.job_id, "state": "running",
            "metrics": metrics, "resources": resources,
        })
        if response is None:
            return False
        if not isinstance(response, dict) or not isinstance(response.get("cancel", False), bool):
            raise CoordinatorError("Heartbeat response must provide a boolean cancel flag")
        return response.get("cancel", False)

    def complete(self, lease: Lease, state: str, result: dict[str, Any], error: str | None = None) -> Any:
        payload = {"lease_id": lease.lease_id, "state": state, "result": result}
        if error:
            payload["error"] = error[:1000]
        # Completion is keyed by lease_id and expected to be idempotent.
        return self._post(f"{self.prefix}/jobs/{quote(lease.job_id, safe='')}/complete", payload, retries=2)
