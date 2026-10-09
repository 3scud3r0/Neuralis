"""Inspect captured time and replay only through an actual emulator adapter."""

from collections.abc import Callable
from typing import Any

from .trajectory_db import TrajectoryRecord, TrajectoryStore


class TemporalDebugger:
    def __init__(self, store: TrajectoryStore, *, restore: Callable[[bytes], None] | None = None,
                 advance: Callable[[int], Any] | None = None):
        self.store = store
        self.restore = restore
        self.advance = advance

    def capabilities(self) -> dict:
        return {"observation_playback": True, "state_replay": self.restore is not None and self.advance is not None,
                "replay_requires": "An emulator snapshot and caller-supplied restore/advance callbacks",
                "storage_backend": self.store.backend}

    def seek(self, episode_id: str, step: int) -> TrajectoryRecord:
        return self.store.get(episode_id, step)

    def replay(self, episode_id: str, step: int, *, actions: list[int] | None = None,
               end_step: int | None = None) -> dict:
        if self.restore is None or self.advance is None:
            raise RuntimeError("State replay requires an emulator restore and advance adapter")
        record = self.store.get(episode_id, step)
        if record.snapshot is None:
            raise ValueError("The selected record has no emulator snapshot")
        if actions is not None and end_step is not None:
            raise ValueError("Specify branch actions or a recorded end step, not both")
        if actions is None:
            if end_step is None or not step <= end_step <= step + 1000:
                raise ValueError("Recorded replay must end within 1000 steps")
            # A snapshot belongs to the observation before its stored action.
            records = self.store.steps(episode_id, start=step, limit=max(1, end_step - step))
            expected = list(range(step, end_step))
            if [r.step for r in records][:len(expected)] != expected:
                raise ValueError("Replay range is incomplete")
            selected = [r.action for r in records[:len(expected)]]
        else:
            if len(actions) > 1000 or any(not isinstance(a, int) or isinstance(a, bool) or a < 0 for a in actions):
                raise ValueError("Branch requires at most 1000 nonnegative integer actions")
            selected = actions
        self.restore(record.snapshot)
        outputs = [self.advance(action) for action in selected]
        return {"episode_id": episode_id, "snapshot_step": step, "actions": selected,
                "outputs": outputs, "mode": "emulator-replay"}

    def timeline(self, episode_id: str, *, start: int = 0, limit: int = 100) -> list[dict]:
        return [record.public() for record in self.store.steps(episode_id, start=start, limit=limit)]
