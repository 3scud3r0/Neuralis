"""Persist observations and policy outputs without serializing executable objects.

SQLite is the standard-library baseline. DuckDB is selected explicitly and never
silently substituted: installing it is optional. Both store actual captured RGB
frames, policy tensors, and optional opaque emulator snapshots.
"""

from dataclasses import dataclass
from io import BytesIO
import json
import math
from pathlib import Path
import re
import sqlite3
from threading import RLock
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TrajectoryRecord:
    episode_id: str
    step: int
    frame: np.ndarray
    policy: np.ndarray
    action: int
    reward: float
    terminated: bool
    snapshot: bytes | None
    metadata: dict[str, Any]

    def public(self, *, include_arrays: bool = False) -> dict[str, Any]:
        value = {
            "episode_id": self.episode_id, "step": self.step,
            "frame_shape": list(self.frame.shape), "policy_shape": list(self.policy.shape),
            "action": self.action, "reward": self.reward, "terminated": self.terminated,
            "has_snapshot": self.snapshot is not None, "metadata": self.metadata,
        }
        if include_arrays:
            value.update(frame=self.frame.tolist(), policy=self.policy.tolist())
        return value


def _encode(array: np.ndarray) -> bytes:
    buffer = BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def _decode(blob: bytes) -> np.ndarray:
    return np.load(BytesIO(bytes(blob)), allow_pickle=False)


class TrajectoryStore:
    """Append-only episode storage with explicit byte and record budgets.

    Steps must be contiguous starting at zero. A terminal record closes its
    episode. The database is local and callers must enforce project access.
    """

    def __init__(self, path: str | Path, *, backend: str = "sqlite",
                 max_records: int = 10000, max_record_bytes: int = 2 * 1024 * 1024,
                 max_total_bytes: int = 256 * 1024 * 1024):
        if backend not in {"sqlite", "duckdb"}:
            raise ValueError("Storage backend must be sqlite or duckdb")
        if not 1 <= max_records <= 1000000:
            raise ValueError("Record budget must be between one and one million")
        if not 1024 <= max_record_bytes <= 16 * 1024 * 1024:
            raise ValueError("Record byte budget must fit between one KiB and 16 MiB")
        if not max_record_bytes <= max_total_bytes <= 4 * 1024**3:
            raise ValueError("Total byte budget must fit between one record and four GiB")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.backend = backend
        self.max_records = max_records
        self.max_record_bytes = max_record_bytes
        self.max_total_bytes = max_total_bytes
        self._lock = RLock()
        if backend == "duckdb":
            try:
                import duckdb
            except ImportError as error:
                raise RuntimeError("DuckDB storage requires the optional duckdb package") from error
            self._connection = duckdb.connect(str(self.path))
        else:
            self._connection = sqlite3.connect(str(self.path), check_same_thread=False)
            self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("""CREATE TABLE IF NOT EXISTS trajectories (
            episode_id VARCHAR NOT NULL, step BIGINT NOT NULL, frame BLOB NOT NULL,
            policy BLOB NOT NULL, action BIGINT NOT NULL, reward DOUBLE NOT NULL,
            terminated BOOLEAN NOT NULL, snapshot BLOB, metadata VARCHAR NOT NULL,
            payload_bytes BIGINT NOT NULL, PRIMARY KEY (episode_id, step))""")
        self._connection.commit()

    @staticmethod
    def _episode(episode_id: str) -> None:
        if not isinstance(episode_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", episode_id):
            raise ValueError("Episode ID must contain one to 100 letters, digits, underscores or hyphens")

    def append(self, episode_id: str, step: int, frame: np.ndarray, policy: np.ndarray,
               action: int, reward: float, *, terminated: bool = False,
               snapshot: bytes | None = None, metadata: dict[str, Any] | None = None) -> TrajectoryRecord:
        self._episode(episode_id)
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise ValueError("Step must be a nonnegative integer")
        image = np.asarray(frame)
        tensor = np.asarray(policy)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Frame must be an H x W x 3 uint8 RGB image")
        if not 1 <= image.shape[0] <= 1080 or not 1 <= image.shape[1] <= 1920:
            raise ValueError("Frame dimensions exceed the capture budget")
        if tensor.dtype.kind not in "fiu" or not 1 <= tensor.ndim <= 4 or not 1 <= tensor.size <= 65536:
            raise ValueError("Policy must be a numeric tensor of at most 65536 elements")
        if not np.isfinite(tensor).all() or not math.isfinite(reward):
            raise ValueError("Policy and reward must contain finite values")
        if not isinstance(action, int) or isinstance(action, bool) or action < 0:
            raise ValueError("Action must be a nonnegative integer")
        if snapshot is not None and not isinstance(snapshot, bytes):
            raise ValueError("Emulator snapshot must be opaque bytes")
        details = metadata if metadata is not None else {}
        if not isinstance(details, dict):
            raise ValueError("Metadata must be a JSON object")
        encoded_metadata = json.dumps(details, allow_nan=False, sort_keys=True)
        if len(encoded_metadata.encode()) > 65536:
            raise ValueError("Metadata exceeds the 64 KiB budget")
        frame_bytes, policy_bytes = _encode(image), _encode(tensor)
        size = len(frame_bytes) + len(policy_bytes) + len(snapshot or b"") + len(encoded_metadata.encode())
        if size > self.max_record_bytes:
            raise ValueError("Captured record exceeds the byte budget")
        with self._lock:
            # Transaction serializes budget and contiguous-step checks with writes.
            self._connection.execute("BEGIN TRANSACTION")
            try:
                count, total = self._connection.execute(
                    "SELECT COUNT(*), COALESCE(SUM(payload_bytes), 0) FROM trajectories").fetchone()
                if count >= self.max_records or total + size > self.max_total_bytes:
                    raise ValueError("Trajectory storage budget exhausted")
                previous = self._connection.execute(
                    "SELECT step, terminated FROM trajectories WHERE episode_id=? ORDER BY step DESC LIMIT 1",
                    (episode_id,)).fetchone()
                expected = 0 if previous is None else previous[0] + 1
                if step != expected or (previous is not None and previous[1]):
                    raise ValueError("Append requires the next contiguous step of an open episode")
                self._connection.execute("INSERT INTO trajectories VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (episode_id, step, frame_bytes, policy_bytes, action, float(reward),
                     bool(terminated), snapshot, encoded_metadata, size))
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
        return TrajectoryRecord(episode_id, step, image.copy(), tensor.copy(), action, float(reward),
                                bool(terminated), snapshot, json.loads(encoded_metadata))

    @staticmethod
    def _record(row: tuple) -> TrajectoryRecord:
        return TrajectoryRecord(row[0], row[1], _decode(row[2]), _decode(row[3]), row[4], row[5],
                                bool(row[6]), None if row[7] is None else bytes(row[7]), json.loads(row[8]))

    def get(self, episode_id: str, step: int) -> TrajectoryRecord:
        self._episode(episode_id)
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM trajectories WHERE episode_id=? AND step=?", (episode_id, step)).fetchone()
        if row is None:
            raise KeyError((episode_id, step))
        return self._record(row)

    def steps(self, episode_id: str, *, start: int = 0, limit: int = 100) -> list[TrajectoryRecord]:
        self._episode(episode_id)
        if start < 0 or not 1 <= limit <= 1000:
            raise ValueError("Read requires nonnegative start and a limit between one and 1000")
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM trajectories WHERE episode_id=? AND step>=? ORDER BY step LIMIT ?",
                (episode_id, start, limit)).fetchall()
        return [self._record(row) for row in rows]

    def summary(self) -> dict[str, Any]:
        with self._lock:
            rows = self._connection.execute("""SELECT episode_id, COUNT(*), SUM(reward),
                MIN(step), MAX(step), MAX(CAST(terminated AS INTEGER)),
                SUM(CASE WHEN snapshot IS NOT NULL THEN 1 ELSE 0 END)
                FROM trajectories GROUP BY episode_id ORDER BY episode_id""").fetchall()
            count, size = self._connection.execute(
                "SELECT COUNT(*), COALESCE(SUM(payload_bytes), 0) FROM trajectories").fetchone()
        return {"backend": self.backend, "records": count, "payload_bytes": size,
                "max_records": self.max_records, "max_total_bytes": self.max_total_bytes,
                "episodes": [{"episode_id": r[0], "steps": r[1], "return": r[2],
                              "first_step": r[3], "last_step": r[4], "terminated": bool(r[5]),
                              "snapshots": r[6]} for r in rows]}

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def episodes(self) -> list[dict[str, Any]]:
        return self.summary()["episodes"]

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
