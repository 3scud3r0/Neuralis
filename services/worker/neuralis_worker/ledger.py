"""Durable completion outbox: never rerun a leased job after a crash."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
import json
import sqlite3
import time

from .protocol import Lease


class LedgerError(RuntimeError):
    pass


class Ledger:
    MAX_RESULT_BYTES = 1024 * 1024

    def __init__(self, directory: Path, max_history: int = 1000, quota_mb: int = 128):
        if not 10 <= max_history <= 100000 or not 8 <= quota_mb <= 4096:
            raise ValueError("Invalid ledger history or quota budget")
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = directory / "worker.sqlite3"
        if self.path.is_symlink():
            raise LedgerError("Worker ledger cannot be a symbolic link")
        self.directory = directory.resolve()
        self.max_history = max_history
        self.quota_bytes = quota_mb * 1024 * 1024
        with self.connection() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS jobs (
                    lease_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    lease_seconds INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    result_json TEXT,
                    error TEXT,
                    acquired_at REAL NOT NULL,
                    completed_at REAL,
                    delivered_at REAL,
                    delivery_attempts INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS jobs_outbox ON jobs(state, delivered_at);
                CREATE INDEX IF NOT EXISTS jobs_history ON jobs(acquired_at);
                CREATE TABLE IF NOT EXISTS schema_info (version INTEGER NOT NULL);
                INSERT INTO schema_info SELECT 1 WHERE NOT EXISTS (SELECT 1 FROM schema_info);
            """)
            version = db.execute("SELECT version FROM schema_info").fetchone()[0]
            if version != 1:
                raise LedgerError("Unsupported worker ledger schema")

        self.path.chmod(0o600)

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=FULL")
        try:
            with db:
                yield db
        finally:
            db.close()

    def reserve(self, lease: Lease) -> bool:
        encoded = json.dumps(lease.config, allow_nan=False, separators=(",", ":"))
        if len(encoded) > 8192:
            raise LedgerError("Job config exceeds ledger budget")
        self.prune()
        self.check_quota()
        with self.connection() as db:
            previous = db.execute("SELECT job_id,config_json FROM jobs WHERE lease_id=?", (lease.lease_id,)).fetchone()
            if previous is not None:
                if previous["job_id"] != lease.job_id or previous["config_json"] != encoded:
                    raise LedgerError("Coordinator reused a lease identifier for a different job")
                return False
            db.execute("INSERT INTO jobs (lease_id,job_id,config_json,lease_seconds,state,acquired_at) VALUES (?,?,?,?,?,?)",
                (lease.lease_id, lease.job_id, encoded, lease.lease_seconds, "running", time.time()))
        return True

    def complete(self, lease: Lease, state: str, result: dict[str, Any], error: str | None) -> None:
        if state not in {"completed", "failed", "cancelled"}:
            raise LedgerError("Invalid completion state")
        encoded = json.dumps(result, allow_nan=False, separators=(",", ":"))
        if len(encoded.encode()) > self.MAX_RESULT_BYTES:
            state, encoded, error = "failed", "{}", "Training result exceeded the 1 MiB delivery budget"
        with self.connection() as db:
            row = db.execute("SELECT state FROM jobs WHERE lease_id=?", (lease.lease_id,)).fetchone()
            if row is None:
                raise LedgerError("Cannot complete a lease that was not reserved")
            if row["state"] != "running":
                return
            db.execute("UPDATE jobs SET state=?,result_json=?,error=?,completed_at=? WHERE lease_id=?",
                (state, encoded, error[:1000] if error else None, time.time(), lease.lease_id))

    def recover(self) -> int:
        # Interrupted training may have emitted side effects. Mark it failed rather
        # than repeating execution under an expired or ambiguous lease.
        with self.connection() as db:
            cursor = db.execute("UPDATE jobs SET state='failed',result_json='{}',error=?,completed_at=? WHERE state='running'",
                ("Worker restarted before training completed; execution was not repeated", time.time()))
            return cursor.rowcount

    def pending(self) -> list[tuple[Lease, str, dict[str, Any], str | None]]:
        with self.connection() as db:
            rows = db.execute("SELECT * FROM jobs WHERE state!='running' AND delivered_at IS NULL ORDER BY completed_at LIMIT 100").fetchall()
        return [(Lease(row["lease_id"], row["job_id"], json.loads(row["config_json"]), row["lease_seconds"]),
            row["state"], json.loads(row["result_json"] or "{}"), row["error"]) for row in rows]

    def mark_delivery_attempt(self, lease_id: str) -> None:
        with self.connection() as db:
            db.execute("UPDATE jobs SET delivery_attempts=delivery_attempts+1 WHERE lease_id=?", (lease_id,))

    def delivered(self, lease_id: str) -> None:
        with self.connection() as db:
            db.execute("UPDATE jobs SET delivered_at=? WHERE lease_id=? AND state!='running'", (time.time(), lease_id))

    def stats(self) -> dict[str, Any]:
        with self.connection() as db:
            rows = db.execute("SELECT state,count(*) AS count FROM jobs GROUP BY state").fetchall()
            pending = db.execute("SELECT count(*) FROM jobs WHERE delivered_at IS NULL AND state!='running'").fetchone()[0]
        return {"states": {row["state"]: row["count"] for row in rows}, "pending_deliveries": pending,
            "storage_bytes": self.usage(), "quota_bytes": self.quota_bytes}

    def usage(self) -> int:
        return sum(path.stat().st_size for path in self.directory.glob("worker.sqlite3*") if path.is_file())

    def check_quota(self) -> None:
        if self.usage() > self.quota_bytes - self.MAX_RESULT_BYTES:
            raise LedgerError("Worker completion ledger storage budget is exhausted; deliver pending results before acquiring work")

    def prune(self) -> int:
        with self.connection() as db:
            cursor = db.execute("DELETE FROM jobs WHERE delivered_at IS NOT NULL AND lease_id NOT IN (SELECT lease_id FROM jobs ORDER BY acquired_at DESC LIMIT ?)", (self.max_history,))
            count = cursor.rowcount
        if count:
            with self.connection() as db:
                db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return count
