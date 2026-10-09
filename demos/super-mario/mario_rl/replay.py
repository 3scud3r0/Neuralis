"""Prioritized replay with sum/min segment trees, n-step returns and snapshots."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np


@dataclass
class Transition:
    observation: np.ndarray
    action: int
    reward: float
    next_observation: np.ndarray
    terminated: bool
    discount: float


class NStepAccumulator:
    """Flush every prefix at an episode boundary; bootstrap truncations only."""
    def __init__(self, n_steps: int, gamma: float):
        if n_steps < 1 or not 0 <= gamma <= 1:
            raise ValueError("n_steps must be positive and gamma must be in [0,1]")
        self.n_steps = n_steps
        self.gamma = gamma
        self.queue: deque[tuple] = deque()

    def _prefix(self) -> Transition:
        reward = 0.0
        discount = 1.0
        last = None
        for item in list(self.queue)[:self.n_steps]:
            reward += discount * item[2]
            discount *= self.gamma
            last = item
            if item[4] or item[5]:
                break
        assert last is not None
        first = self.queue[0]
        return Transition(first[0].copy(), int(first[1]), reward,
                          last[3].copy(), bool(last[4]), discount)

    def append(self, observation, action, reward, next_observation, terminated, truncated):
        self.queue.append((observation.copy(), int(action), float(reward),
                           next_observation.copy(), bool(terminated), bool(truncated)))
        ready = []
        if terminated or truncated:
            while self.queue:
                ready.append(self._prefix())
                self.queue.popleft()
        elif len(self.queue) >= self.n_steps:
            ready.append(self._prefix())
            self.queue.popleft()
        return ready

    def clear(self):
        self.queue.clear()


class PriorityTree:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.size = 1 << (capacity - 1).bit_length()
        self.sum = np.zeros(2 * self.size, dtype=np.float64)
        self.minimum = np.full(2 * self.size, np.inf, dtype=np.float64)

    def update(self, index: int, priority: float):
        if not 0 <= index < self.capacity or not np.isfinite(priority) or priority <= 0:
            raise ValueError("Priority index/value is invalid")
        position = self.size + index
        self.sum[position] = priority
        self.minimum[position] = priority
        position //= 2
        while position:
            self.sum[position] = self.sum[position * 2] + self.sum[position * 2 + 1]
            self.minimum[position] = min(self.minimum[position * 2], self.minimum[position * 2 + 1])
            position //= 2

    def find(self, mass: float) -> int:
        total = float(self.sum[1])
        if not 0 <= mass < total:
            raise ValueError("Sampling mass must be in [0, total priority)")
        position = 1
        while position < self.size:
            left = 2 * position
            if mass < self.sum[left]:
                position = left
            else:
                mass -= self.sum[left]
                position = left + 1
        index = position - self.size
        if index >= self.capacity:
            raise RuntimeError("Priority tree selected an unused leaf")
        return index


class PrioritizedReplay:
    def __init__(self, capacity: int, observation_shape: tuple[int, ...], seed: int,
                 alpha: float = 0.6, epsilon: float = 1e-5, max_memory_mb: int = 768):
        if capacity < 1 or not 0 <= alpha <= 1 or epsilon <= 0:
            raise ValueError("Replay capacity/alpha/epsilon is invalid")
        memory = capacity * (2 * np.prod(observation_shape) + 4 + 4 + 1 + 4 + 8)
        if memory > max_memory_mb * 1024 * 1024:
            raise MemoryError(f"Replay needs approximately {memory / 1024 ** 2:.1f} MiB; "
                              f"configured budget is {max_memory_mb} MiB. Reduce replay_capacity.")
        self.capacity = capacity
        self.observation_shape = tuple(observation_shape)
        self.alpha = alpha
        self.epsilon = epsilon
        self.max_memory_mb = max_memory_mb
        self.rng = np.random.default_rng(seed)
        shape = (capacity, *observation_shape)
        self.observations = np.empty(shape, dtype=np.uint8)
        self.next_observations = np.empty(shape, dtype=np.uint8)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.terminated = np.empty(capacity, dtype=bool)
        self.discounts = np.empty(capacity, dtype=np.float32)
        self.priorities = np.empty(capacity, dtype=np.float64)
        self.tree = PriorityTree(capacity)
        self.position = 0
        self.length = 0
        self.max_priority = 1.0

    def __len__(self):
        return self.length

    def add(self, transition: Transition):
        if transition.observation.shape != self.observation_shape or transition.next_observation.shape != self.observation_shape:
            raise ValueError("Replay observation shapes do not match")
        index = self.position
        self.observations[index] = transition.observation
        self.next_observations[index] = transition.next_observation
        self.actions[index] = transition.action
        self.rewards[index] = transition.reward
        self.terminated[index] = transition.terminated
        self.discounts[index] = transition.discount
        self.priorities[index] = self.max_priority
        self.tree.update(index, self.max_priority ** self.alpha)
        self.position = (index + 1) % self.capacity
        self.length = min(self.length + 1, self.capacity)

    def sample(self, batch_size: int, beta: float):
        if self.length == 0 or batch_size < 1 or not 0 <= beta <= 1:
            raise ValueError("Replay is empty or sample parameters are invalid")
        total = float(self.tree.sum[1])
        width = total / batch_size
        masses = (np.arange(batch_size) + self.rng.random(batch_size)) * width
        masses = np.minimum(masses, np.nextafter(total, 0))
        indices = np.asarray([self.tree.find(float(mass)) for mass in masses], dtype=np.int64)
        probabilities = self.tree.sum[self.tree.size + indices] / total
        minimum_probability = float(self.tree.minimum[1]) / total
        maximum_weight = (self.length * minimum_probability) ** (-beta)
        weights = (self.length * probabilities) ** (-beta) / maximum_weight
        return {
            "indices": indices,
            "observations": self.observations[indices],
            "next_observations": self.next_observations[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "terminated": self.terminated[indices],
            "discounts": self.discounts[indices],
            "weights": weights.astype(np.float32),
        }

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        if len(indices) != len(td_errors):
            raise ValueError("Priority index/error lengths differ")
        for index, error in zip(indices, td_errors):
            if not 0 <= index < self.length or not np.isfinite(error):
                raise ValueError("Replay priority index or TD error is invalid")
            priority = abs(float(error)) + self.epsilon
            self.priorities[index] = priority
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(int(index), priority ** self.alpha)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {"format": "neuralis.mario.replay/v1", "capacity": self.capacity,
                    "shape": self.observation_shape, "position": self.position,
                    "length": self.length, "alpha": self.alpha, "epsilon": self.epsilon,
                    "maxPriority": self.max_priority, "rng": self.rng.bit_generator.state}
        temporary = path.with_suffix(path.suffix + ".tmp")
        arrays = {name: getattr(self, name)[:self.length] for name in (
            "observations", "next_observations", "actions", "rewards", "terminated", "discounts", "priorities")}
        try:
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, metadata=np.asarray(json.dumps(metadata)), **arrays)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return file_sha256(path)

    @classmethod
    def load(cls, path: Path, expected_sha256: str, max_memory_mb: int = 768):
        if file_sha256(path) != expected_sha256:
            raise ValueError("Replay checksum mismatch; checkpoint and replay must be a matching pair")
        with zipfile.ZipFile(path) as archive:
            if sum(info.file_size for info in archive.infolist()) > max_memory_mb * 1024 * 1024 + 65536:
                raise MemoryError("Replay archive exceeds the configured uncompressed memory budget")
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
            if metadata["format"] != "neuralis.mario.replay/v1":
                raise ValueError("Unsupported replay format")
            replay = cls(metadata["capacity"], tuple(metadata["shape"]), 0,
                         metadata["alpha"], metadata["epsilon"], max_memory_mb)
            replay.length = metadata["length"]
            replay.position = metadata["position"]
            if not 0 <= replay.length <= replay.capacity or not 0 <= replay.position < replay.capacity:
                raise ValueError("Replay snapshot has invalid ring-buffer counters")
            for name in ("observations", "next_observations", "actions", "rewards", "terminated", "discounts", "priorities"):
                source = archive[name]
                destination = getattr(replay, name)
                if source.shape != destination[:replay.length].shape or source.dtype != destination.dtype:
                    raise ValueError(f"Replay snapshot {name} shape/dtype does not match")
                destination[:replay.length] = source
            replay.max_priority = float(metadata["maxPriority"])
            replay.rng.bit_generator.state = metadata["rng"]
            for index in range(replay.length):
                replay.tree.update(index, replay.priorities[index] ** replay.alpha)
        return replay


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()
