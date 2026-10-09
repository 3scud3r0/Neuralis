"""Instrument an existing trusted Python/PyTorch application without uploading code.

Example:
    with Inspector(model, JSONLRecorder(Path('telemetry.jsonl'))) as inspector:
        for step in range(100):
            with inspector.step(step + 1):
                loss = loss_function(model(inputs), targets)
                loss.backward()
                optimizer.step()
                inspector.metric('loss', float(loss.detach()))

The SDK owns hooks, not the user's optimizer, runtime, data, or remote execution.
"""

from contextlib import contextmanager
import json
from pathlib import Path
import queue
import threading
import time
from typing import Callable, Protocol
import uuid

import torch

from neuralis.core.schemas import CaptureConfig
from neuralis.runtime.capture import CaptureSession


class EventSink(Protocol):
    def write(self, event: dict, priority: bool = False) -> bool: ...
    def close(self) -> None: ...


class JSONLRecorder:
    """Bounded asynchronous JSONL capture with explicit backpressure accounting."""

    def __init__(self, path: Path, capacity: int = 128, maximum_record_bytes: int = 2 * 1024 * 1024):
        if capacity < 2 or capacity > 10000:
            raise ValueError("Recorder capacity must be between 2 and 10,000 events")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.maximum_record_bytes = maximum_record_bytes
        self.queue = queue.Queue(maxsize=capacity)
        self.dropped = 0
        self.written = 0
        self.bytes_written = 0
        self.error = None
        self._closed = False
        self._lock = threading.Lock()
        self._sentinel = object()
        self._thread = threading.Thread(target=self._writer, daemon=True, name="neuralis-jsonl-recorder")
        self._thread.start()

    def write(self, event: dict, priority=False):
        encoded = json.dumps(event, allow_nan=False, separators=(",", ":")) + "\n"
        with self._lock:
            if self._closed:
                raise RuntimeError("Recorder is closed")
            if self.error is not None:
                raise RuntimeError(f"Recorder failed: {self.error}")
            if len(encoded.encode("utf-8")) > self.maximum_record_bytes:
                if priority:
                    raise ValueError("Priority event exceeds the record byte budget")
                self.dropped += 1
                return False
            try:
                self.queue.put(encoded, block=priority, timeout=1 if priority else None)
                return True
            except queue.Full:
                if priority:
                    raise RuntimeError("Recorder cannot persist a priority event within the one-second deadline")
                self.dropped += 1
                return False

    def _writer(self):
        try:
            with self.path.open("a", encoding="utf-8") as stream:
                while True:
                    record = self.queue.get()
                    try:
                        if record is self._sentinel:
                            break
                        stream.write(record)
                        self.written += 1
                        self.bytes_written += len(record.encode("utf-8"))
                        if self.written % 16 == 0:
                            stream.flush()
                    finally:
                        self.queue.task_done()
                stream.flush()
        except BaseException as error:
            self.error = f"{type(error).__name__}: {error}"
            while True:
                try:
                    self.queue.get_nowait()
                    self.queue.task_done()
                except queue.Empty:
                    break

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
        if self._thread.is_alive():
            self.queue.put(self._sentinel, timeout=5)
            self._thread.join(timeout=10)
        if self._thread.is_alive():
            raise RuntimeError("Recorder writer did not finish within the shutdown deadline")
        if self.error:
            raise RuntimeError(f"Recorder failed: {self.error}")

    def status(self):
        return {"written": self.written, "dropped": self.dropped, "pending": self.queue.qsize(),
                "capacity": self.queue.maxsize, "bytes_written": self.bytes_written, "error": self.error}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


class CallbackSink:
    def __init__(self, callback: Callable[[dict], None]):
        self.callback = callback

    def write(self, event, priority=False):
        self.callback(event)
        return True

    def close(self):
        return None


class Inspector:
    def __init__(self, model, sink: EventSink, capture: CaptureConfig | None = None, run_id: str | None = None,
                 rank: int = 0, world_size: int = 1, shard: str | None = None, close_sink: bool = True):
        if rank < 0 or world_size < 1 or rank >= world_size:
            raise ValueError("Distributed rank must be inside the declared world size")
        self.model = model
        self.sink = sink
        self.config = capture or CaptureConfig()
        self.run_id = run_id or str(uuid.uuid4())
        self.rank = rank
        self.world_size = world_size
        self.shard = shard
        self.close_sink = close_sink
        self.sequence = 0
        self.current_step = 0
        self.metrics = {}
        self.session = None
        self.active_step = False
        self.dropped = 0

    def _emit(self, kind, payload, priority=False):
        self.sequence += 1
        event = {"protocol_version": "1.0", "run_id": self.run_id, "sequence": self.sequence,
                 "timestamp": time.time(), "step": self.current_step, "type": kind, "payload": payload,
                 "rank": self.rank, "world_size": self.world_size, "shard": self.shard}
        accepted = self.sink.write(event, priority)
        if not accepted:
            self.dropped += 1
        return accepted

    def metric(self, name, value):
        if not self.active_step:
            raise RuntimeError("Metrics must be recorded within an inspector step")
        if not isinstance(value, (float, int)) or not torch.isfinite(torch.tensor(float(value))):
            raise ValueError("Metrics must be finite scalar numbers")
        self.metrics[str(name)] = value

    @contextmanager
    def step(self, number):
        if self.session is None or self.active_step:
            raise RuntimeError("Enter the inspector context and avoid nested instrumentation steps")
        if number <= self.current_step:
            raise ValueError("Instrumented step numbers must increase strictly")
        self.current_step = int(number)
        self.metrics = {}
        self.active_step = True
        self.session.begin(number)
        started = time.perf_counter()
        try:
            yield self
            captured = self.session.finish()
            if captured["tensors"]:
                self._emit("tensors", captured)
            self._emit("metrics", {**self.metrics, "step_ms": (time.perf_counter() - started) * 1000,
                                   "capture_ms": captured["capture_ms"], "dropped_events": self.dropped})
        except BaseException as error:
            self.session.enabled = False
            self._emit("error", {"category": type(error).__name__, "message": str(error)}, priority=True)
            raise
        finally:
            self.active_step = False

    def __enter__(self):
        if self.session is not None:
            raise RuntimeError("Inspector context is already active")
        self.session = CaptureSession(self.model, self.config)
        self._emit("state", {"state": "attached", "parameters": sum(parameter.numel() for parameter in self.model.parameters()),
                             "capture": self.config.model_dump(), "coverage": "Explicitly selected modules on this rank only"}, priority=True)
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            if self.session:
                self.session.close()
            self.session = None
            self._emit("state", {"state": "detached", "dropped_events": self.dropped}, priority=True)
        finally:
            if self.close_sink:
                self.sink.close()


def replay_jsonl(path: Path, after_sequence=0, limit=10000):
    if limit < 1 or limit > 100000:
        raise ValueError("Replay limit exceeds the bounded parser budget")
    events, gaps, previous = [], [], after_sequence
    malformed = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                event = json.loads(line)
                sequence = event["sequence"]
                if event.get("protocol_version") != "1.0" or not isinstance(sequence, int):
                    raise ValueError("Invalid protocol envelope")
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                malformed.append(line_number)
                continue
            if sequence <= after_sequence:
                continue
            if sequence > previous + 1:
                gaps.append({"after": previous, "before": sequence})
            elif sequence <= previous:
                malformed.append(line_number)
                continue
            events.append(event)
            previous = sequence
            if len(events) >= limit:
                break
    return {"events": events, "gaps": gaps, "malformed_lines": malformed, "next_sequence": previous}
