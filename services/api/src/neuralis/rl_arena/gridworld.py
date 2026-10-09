"""Small reproducible CPU arena; explicitly not a Mario emulator."""

import json
import uuid

import numpy as np

from .trajectory_db import TrajectoryStore


class GridWorld:
    actions = ((0, -1), (1, 0), (0, 1), (-1, 0))

    def __init__(self, size: int = 6):
        if not 3 <= size <= 20:
            raise ValueError("Grid size must be between three and 20")
        self.size = size
        self.position = (0, 0)
        self.goal = (size - 1, size - 1)
        self.steps = 0

    def snapshot(self) -> bytes:
        return json.dumps({"size": self.size, "position": self.position, "steps": self.steps}).encode()

    def restore(self, snapshot: bytes) -> None:
        value = json.loads(snapshot)
        position = tuple(value["position"])
        if value["size"] != self.size or len(position) != 2 or any(not isinstance(x, int) or not 0 <= x < self.size for x in position):
            raise ValueError("Invalid gridworld snapshot")
        if not isinstance(value["steps"], int) or value["steps"] < 0:
            raise ValueError("Invalid gridworld snapshot step")
        self.position, self.steps = position, value["steps"]

    def policy(self) -> np.ndarray:
        scores = []
        for dx, dy in self.actions:
            nx, ny = self.position[0] + dx, self.position[1] + dy
            valid = 0 <= nx < self.size and 0 <= ny < self.size
            scores.append(-abs(self.goal[0] - nx) - abs(self.goal[1] - ny) if valid else -100)
        return np.asarray(scores, dtype=np.float32)

    def frame(self, scale: int = 12) -> np.ndarray:
        image = np.full((self.size, self.size, 3), (25, 30, 45), dtype=np.uint8)
        image[self.goal[1], self.goal[0]] = (60, 190, 90)
        image[self.position[1], self.position[0]] = (70, 130, 240)
        return np.repeat(np.repeat(image, scale, axis=0), scale, axis=1)

    def advance(self, action: int) -> dict:
        if not 0 <= action < len(self.actions):
            raise ValueError("Unknown gridworld action")
        if self.position == self.goal:
            raise ValueError("Gridworld episode has already terminated")
        dx, dy = self.actions[action]
        nx = max(0, min(self.size - 1, self.position[0] + dx))
        ny = max(0, min(self.size - 1, self.position[1] + dy))
        self.position = (nx, ny)
        self.steps += 1
        done = self.position == self.goal
        return {"position": list(self.position), "reward": 1.0 if done else -0.01, "terminated": done}


def record_episode(store: TrajectoryStore, *, episode_id: str | None = None, size: int = 6,
                   max_steps: int = 1000) -> dict:
    if not 1 <= max_steps <= 1000:
        raise ValueError("Episode step budget must fit one to 1000")
    world = GridWorld(size)
    identifier = episode_id or "grid-" + uuid.uuid4().hex
    total_reward = 0.0
    for step in range(min(max_steps, 2 * (size - 1))):
        frame, policy, state = world.frame(), world.policy(), world.snapshot()
        action = int(np.argmax(policy))
        result = world.advance(action)
        total_reward += result["reward"]
        store.append(identifier, step, frame, policy, action, result["reward"],
                     terminated=result["terminated"], snapshot=state,
                     metadata={"environment": "deterministic-gridworld", "policy": "Manhattan-distance greedy",
                               "before_action": True, "next_position": result["position"]})
    return {"episode_id": identifier, "environment": "deterministic-gridworld", "steps": world.steps,
            "return": total_reward, "terminated": world.position == world.goal,
            "truncated": world.position != world.goal, "storage_backend": store.backend}


def record_gridworld_demo(store: TrajectoryStore, *, seed: int = 42, steps: int = 64,
                          episode_id: str | None = None, size: int = 6) -> dict:
    # The baseline policy is deterministic and has no stochastic parameter.
    result = record_episode(store, episode_id=episode_id, size=size, max_steps=steps)
    result.update(seed=seed, stochastic=False, requested_step_budget=steps)
    return result
