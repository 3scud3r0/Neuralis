import json
import time
import uuid

from pydantic import BaseModel, ConfigDict, Field, model_validator


class WorkerRegistration(BaseModel):
    model_config = ConfigDict(extra="forbid")
    worker_id: str = Field(min_length=1, max_length=120, pattern=r"^[A-Za-z0-9_.-]+$")
    capabilities: dict


class LeaseRequest(BaseModel):
    tasks: list[str] = Field(default_factory=lambda: ["xor"])
    resources: dict = Field(default_factory=dict)
    lease_seconds: int = Field(default=60, ge=10, le=300)


class HeartbeatRequest(BaseModel):
    lease_id: str
    job_id: str
    state: str = "running"
    metrics: dict = Field(default_factory=dict)
    resources: dict = Field(default_factory=dict)


class CompletionRequest(BaseModel):
    lease_id: str
    state: str = Field(pattern=r"^(completed|failed|cancelled)$")
    result: dict = Field(default_factory=dict)
    error: str | None = Field(default=None, max_length=4000)


class QueueRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project_id: str = "local"
    task: str = Field(default="xor", pattern=r"^(xor|ddp-xor)$")
    seed: int = Field(default=42, ge=0, le=2147483647)
    steps: int = Field(default=100, ge=1, le=10000)
    learning_rate: float = Field(default=0.01, gt=0, le=1)
    cpu_threads: int = Field(default=1, ge=1, le=64)
    memory_mb: int = Field(default=512, ge=128, le=65536)
    world_size: int = Field(default=2, ge=2, le=8)

    @model_validator(mode="after")
    def distributed_resource_budget(self):
        if self.task == "ddp-xor" and self.cpu_threads < self.world_size:
            raise ValueError("DDP requires cpu_threads at least world_size; request the actual worker budget")
        return self


class LeaseConflict(ValueError):
    pass


class Coordinator:
    """Transactional leases; expired work is marked lost and never silently replayed."""

    def __init__(self, repository):
        self.repository = repository

    def register(self, request):
        encoded = json.dumps(request.capabilities, allow_nan=False)
        if len(encoded) > 32768:
            raise ValueError("Worker capability description is too large")
        with self.repository.transaction() as connection:
            connection.execute("INSERT INTO workers VALUES(?,?,?,?) ON CONFLICT(id) DO UPDATE SET capabilities=excluded.capabilities,heartbeat=excluded.heartbeat,status=excluded.status",
                               (request.worker_id, encoded, time.time(), "online"))
        return {"worker_id": request.worker_id, "registered": True, "protocol_version": "1.0"}

    def enqueue(self, request):
        job_id = str(uuid.uuid4())
        with self.repository.transaction() as connection:
            row = connection.execute("SELECT quota FROM projects WHERE id=?", (request.project_id,)).fetchone()
            if row is None:
                raise KeyError(request.project_id)
            pending = connection.execute("SELECT COUNT(*) FROM queue_jobs WHERE project_id=? AND status IN ('queued','running')", (request.project_id,)).fetchone()[0]
            if pending >= row["quota"]:
                raise LeaseConflict("Project queue quota reached")
            connection.execute("INSERT INTO queue_jobs(id,project_id,config,status,created_at) VALUES(?,?,?,?,?)",
                               (job_id, request.project_id, json.dumps(request.model_dump()), "queued", time.time()))
        return {"id": job_id, "status": "queued", "config": request.model_dump()}

    def expire(self, connection, now):
        connection.execute("UPDATE queue_jobs SET status='lost',result=? WHERE status='running' AND lease_expiry<?",
                           (json.dumps({"error": "Worker lease expired; explicitly enqueue a new attempt"}), now))
        connection.execute("UPDATE workers SET status='offline' WHERE heartbeat<?", (now - 120,))

    def lease(self, worker_id, request):
        now = time.time()
        with self.repository.transaction() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self.expire(connection, now)
            worker = connection.execute("SELECT * FROM workers WHERE id=?", (worker_id,)).fetchone()
            if worker is None:
                raise KeyError(worker_id)
            advertised = json.loads(worker["capabilities"])
            tasks = set(request.tasks).intersection(advertised.get("tasks", []))
            active = connection.execute("SELECT COUNT(*) FROM queue_jobs WHERE worker_id=? AND status='running'", (worker_id,)).fetchone()[0]
            if active >= max(1, min(64, int(advertised.get("max_parallel", 1)))):
                return None
            candidates = connection.execute("SELECT * FROM queue_jobs WHERE status='queued' ORDER BY created_at LIMIT 100").fetchall()
            selected = None
            for candidate in candidates:
                config = json.loads(candidate["config"])
                if config["task"] not in tasks:
                    continue
                if config["cpu_threads"] > request.resources.get("cpu_threads", advertised.get("cpu_count", 1)):
                    continue
                if config["memory_mb"] > request.resources.get("memory_mb", advertised.get("memory_mb", 0)):
                    continue
                selected = candidate
                break
            if selected is None:
                return None
            lease_id = str(uuid.uuid4())
            expiry = now + request.lease_seconds
            connection.execute("UPDATE queue_jobs SET status='running',worker_id=?,lease_id=?,lease_expiry=? WHERE id=? AND status='queued'",
                               (worker_id, lease_id, expiry, selected["id"]))
            connection.execute("UPDATE workers SET heartbeat=?,status='online' WHERE id=?", (now, worker_id))
        return {"lease_id": lease_id, "job_id": selected["id"], "expires_at": expiry,
                "lease_seconds": request.lease_seconds, "config": json.loads(selected["config"])}

    @staticmethod
    def _verify(connection, worker_id, job_id, lease_id):
        row = connection.execute("SELECT * FROM queue_jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        if row["worker_id"] != worker_id or row["lease_id"] != lease_id:
            raise LeaseConflict("Worker or lease does not own this job")
        if row["status"] not in {"running", "cancel-requested"} or row["lease_expiry"] < time.time():
            raise LeaseConflict("Lease is expired or job is no longer running")
        return row

    def heartbeat(self, worker_id, request):
        now = time.time()
        with self.repository.transaction() as connection:
            row = self._verify(connection, worker_id, request.job_id, request.lease_id)
            connection.execute("UPDATE workers SET heartbeat=?,status='online' WHERE id=?", (now, worker_id))
            connection.execute("UPDATE queue_jobs SET lease_expiry=?,result=? WHERE id=?",
                               (now + 60, json.dumps({"metrics": request.metrics, "resources": request.resources}, allow_nan=False), request.job_id))
        return {"cancel": row["status"] == "cancel-requested", "lease_seconds": 60, "expires_at": now + 60}

    def complete(self, worker_id, job_id, request):
        with self.repository.transaction() as connection:
            row = connection.execute("SELECT * FROM queue_jobs WHERE id=?", (job_id,)).fetchone()
            if row and row["worker_id"] == worker_id and row["lease_id"] == request.lease_id and row["status"] == request.state:
                return {"job_id": job_id, "state": row["status"], "idempotent": True}
            self._verify(connection, worker_id, job_id, request.lease_id)
            connection.execute("UPDATE queue_jobs SET status=?,result=? WHERE id=?",
                               (request.state, json.dumps({"result": request.result, "error": request.error}, allow_nan=False), job_id))
        return {"job_id": job_id, "state": request.state}

    def jobs(self, project_id=None):
        with self.repository.transaction() as connection:
            self.expire(connection, time.time())
            if project_id:
                rows = connection.execute("SELECT * FROM queue_jobs WHERE project_id=? ORDER BY created_at DESC LIMIT 100", (project_id,)).fetchall()
            else:
                rows = connection.execute("SELECT * FROM queue_jobs ORDER BY created_at DESC LIMIT 100").fetchall()
        return [{**dict(row), "config": json.loads(row["config"]),
                 "result": json.loads(row["result"]) if row["result"] else None} for row in rows]

    def cancel(self, job_id):
        with self.repository.transaction() as connection:
            row = connection.execute("SELECT status FROM queue_jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            if row["status"] not in {"queued", "running"}:
                raise LeaseConflict("Only queued or running jobs can be cancelled")
            status = "cancelled" if row["status"] == "queued" else "cancel-requested"
            connection.execute("UPDATE queue_jobs SET status=? WHERE id=?", (status, job_id))
        return {"id": job_id, "status": status}

    def workers(self):
        with self.repository.transaction() as connection:
            self.expire(connection, time.time())
            rows = connection.execute("SELECT * FROM workers ORDER BY id").fetchall()
        return [{**dict(row), "capabilities": json.loads(row["capabilities"])} for row in rows]
