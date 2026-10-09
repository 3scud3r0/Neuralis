"""Transactional metadata and bounded event history for the local profile."""

from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any
import uuid


class Repository:
    def __init__(self, directory: Path, max_events_per_run: int = 20000):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.database = directory / "metadata.sqlite3"
        self.max_events_per_run = max_events_per_run
        self._lock = threading.RLock()
        self._initialize()

    @contextmanager
    def transaction(self):
        with self._lock:
            connection = sqlite3.connect(self.database, timeout=10)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys=ON")
            try:
                yield connection
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
            finally:
                connection.close()

    def _initialize(self):
        with self.transaction() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS schema_version(version INTEGER NOT NULL);
                INSERT INTO schema_version SELECT 1 WHERE NOT EXISTS (SELECT 1 FROM schema_version);
                CREATE TABLE IF NOT EXISTS runs(
                    id TEXT PRIMARY KEY, project_id TEXT NOT NULL, config TEXT NOT NULL,
                    state TEXT NOT NULL, step INTEGER NOT NULL, created_at REAL NOT NULL,
                    updated_at REAL NOT NULL, error TEXT, sequence INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS events(
                    run_id TEXT NOT NULL REFERENCES runs(id), sequence INTEGER NOT NULL,
                    timestamp REAL NOT NULL, step INTEGER NOT NULL, type TEXT NOT NULL,
                    payload TEXT NOT NULL, PRIMARY KEY(run_id,sequence));
                CREATE INDEX IF NOT EXISTS event_type ON events(run_id,type,sequence);
                CREATE TABLE IF NOT EXISTS checkpoints(
                    id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id),
                    step INTEGER NOT NULL, created_at REAL NOT NULL, manifest TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS experiments(
                    id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id),
                    created_at REAL NOT NULL, request TEXT NOT NULL, result TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS projects(
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT NOT NULL,
                    created_at REAL NOT NULL, quota INTEGER NOT NULL DEFAULT 4);
                CREATE TABLE IF NOT EXISTS members(
                    project_id TEXT NOT NULL REFERENCES projects(id), subject TEXT NOT NULL,
                    role TEXT NOT NULL, PRIMARY KEY(project_id,subject));
                CREATE TABLE IF NOT EXISTS audit(
                    id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp REAL NOT NULL,
                    actor TEXT NOT NULL, project_id TEXT, action TEXT NOT NULL,
                    target TEXT NOT NULL, outcome TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS workers(
                    id TEXT PRIMARY KEY, capabilities TEXT NOT NULL, heartbeat REAL NOT NULL,
                    status TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS queue_jobs(
                    id TEXT PRIMARY KEY, project_id TEXT NOT NULL, config TEXT NOT NULL,
                    status TEXT NOT NULL, worker_id TEXT, lease_id TEXT, lease_expiry REAL,
                    created_at REAL NOT NULL, result TEXT);
            """)
            connection.execute("INSERT OR IGNORE INTO projects VALUES(?,?,?,?,?)",
                               ("local", "Local workspace", "Single-user CPU research", time.time(), 4))
            # A process restart interrupts active jobs; no duplicate work is silently resumed.
            connection.execute("UPDATE runs SET state='interrupted',error=? WHERE state IN ('running','paused')",
                               ("Backend restarted; resume explicitly from a checkpoint",))

    def create_run(self, config: dict) -> dict:
        run_id = str(uuid.uuid4())
        now = time.time()
        with self.transaction() as connection:
            connection.execute("INSERT INTO runs(id,project_id,config,state,step,created_at,updated_at) VALUES(?,?,?,?,?,?,?)",
                               (run_id, config["project_id"], json.dumps(config), "created", 0, now, now))
        return self.get_run(run_id)

    @staticmethod
    def _run(row) -> dict:
        value = dict(row)
        value["config"] = json.loads(value["config"])
        return value

    def get_run(self, run_id: str) -> dict:
        with self.transaction() as connection:
            row = connection.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(run_id)
        return self._run(row)

    def list_runs(self, project_id: str | None = None, limit: int = 100, offset: int = 0) -> list[dict]:
        with self.transaction() as connection:
            if project_id:
                rows = connection.execute("SELECT * FROM runs WHERE project_id=? ORDER BY created_at DESC LIMIT ? OFFSET ?",
                                          (project_id, limit, offset)).fetchall()
            else:
                rows = connection.execute("SELECT * FROM runs ORDER BY created_at DESC LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        return [self._run(row) for row in rows]

    def update_run(self, run_id: str, *, state: str | None = None, step: int | None = None, error: str | None = None):
        updates: dict[str, Any] = {"updated_at": time.time()}
        if state is not None:
            updates["state"] = str(state)
        if step is not None:
            updates["step"] = step
        if error is not None:
            updates["error"] = error
        sql = "UPDATE runs SET " + ",".join(f"{key}=?" for key in updates) + " WHERE id=?"
        with self.transaction() as connection:
            connection.execute(sql, (*updates.values(), run_id))

    def append_event(self, run_id: str, step: int, event_type: str, payload: dict) -> dict:
        with self.transaction() as connection:
            row = connection.execute("SELECT sequence FROM runs WHERE id=?", (run_id,)).fetchone()
            if row is None:
                raise KeyError(run_id)
            sequence = row["sequence"] + 1
            timestamp = time.time()
            encoded = json.dumps(payload, allow_nan=False, separators=(",", ":"))
            connection.execute("INSERT INTO events VALUES(?,?,?,?,?,?)", (run_id, sequence, timestamp, step, event_type, encoded))
            connection.execute("UPDATE runs SET sequence=? WHERE id=?", (sequence, run_id))
            cutoff = sequence - self.max_events_per_run
            if cutoff > 0:
                connection.execute("DELETE FROM events WHERE run_id=? AND sequence<=?", (run_id, cutoff))
        return {"protocol_version": "1.0", "run_id": run_id, "sequence": sequence, "timestamp": timestamp,
                "step": step, "type": event_type, "payload": payload, "rank": 0, "world_size": 1}

    def events(self, run_id: str, after: int = 0, limit: int = 500, event_type: str | None = None):
        sql = "SELECT * FROM events WHERE run_id=? AND sequence>?"
        parameters: list[Any] = [run_id, after]
        if event_type:
            sql += " AND type=?"
            parameters.append(event_type)
        sql += " ORDER BY sequence LIMIT ?"
        parameters.append(limit)
        with self.transaction() as connection:
            rows = connection.execute(sql, parameters).fetchall()
            oldest = connection.execute("SELECT MIN(sequence) FROM events WHERE run_id=?", (run_id,)).fetchone()[0]
        items = []
        for row in rows:
            value = dict(row)
            value["payload"] = json.loads(value["payload"])
            value.update(protocol_version="1.0", rank=0, world_size=1)
            items.append(value)
        return {"events": items, "oldest_sequence": oldest, "gap": oldest is not None and after < oldest - 1,
                "next_sequence": items[-1]["sequence"] if items else after}

    def add_checkpoint(self, run_id: str, checkpoint_id: str, step: int, manifest: dict):
        with self.transaction() as connection:
            connection.execute("INSERT INTO checkpoints VALUES(?,?,?,?,?)",
                               (checkpoint_id, run_id, step, time.time(), json.dumps(manifest)))

    def checkpoints(self, run_id: str):
        with self.transaction() as connection:
            rows = connection.execute("SELECT * FROM checkpoints WHERE run_id=? ORDER BY step DESC", (run_id,)).fetchall()
        return [{**dict(row), "manifest": json.loads(row["manifest"])} for row in rows]

    def save_experiment(self, run_id: str, request: dict, result: dict):
        experiment_id = str(uuid.uuid4())
        with self.transaction() as connection:
            connection.execute("INSERT INTO experiments VALUES(?,?,?,?,?)",
                               (experiment_id, run_id, time.time(), json.dumps(request), json.dumps(result, allow_nan=False)))
        return {"id": experiment_id, "run_id": run_id, "request": request, "result": result}

    def experiments(self, run_id: str | None = None):
        with self.transaction() as connection:
            if run_id:
                rows = connection.execute("SELECT * FROM experiments WHERE run_id=? ORDER BY created_at DESC", (run_id,)).fetchall()
            else:
                rows = connection.execute("SELECT * FROM experiments ORDER BY created_at DESC LIMIT 100").fetchall()
        return [{**dict(row), "request": json.loads(row["request"]), "result": json.loads(row["result"])} for row in rows]

    def audit(self, actor: str, project_id: str | None, action: str, target: str, outcome: str = "success"):
        with self.transaction() as connection:
            connection.execute("INSERT INTO audit(timestamp,actor,project_id,action,target,outcome) VALUES(?,?,?,?,?,?)",
                               (time.time(), actor, project_id, action, target, outcome))

    def audit_records(self, project_id: str | None = None, limit: int = 100):
        with self.transaction() as connection:
            if project_id:
                rows = connection.execute("SELECT * FROM audit WHERE project_id=? ORDER BY id DESC LIMIT ?", (project_id, limit)).fetchall()
            else:
                rows = connection.execute("SELECT * FROM audit ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(row) for row in rows]
