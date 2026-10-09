"""Spawn-based parallel rollout workers with explicit cleanup and error forwarding."""
from __future__ import annotations

import multiprocessing as mp
from multiprocessing.connection import Connection
import traceback
from typing import Any
import numpy as np

from .config import Config
from .environment import MarioEnv, Step


def auto_step(env: MarioEnv, action: int) -> Step:
    result = env.step(action)
    if result.done:
        result.info["final_observation"] = result.observation.copy()
        result.observation = env.reset()
    return result


def _worker(connection: Connection, config_data: dict, seed: int) -> None:
    env = None
    try:
        env = MarioEnv(Config.from_dict(config_data), seed)
        connection.send(("ok", (env.action_count, env.observation_shape)))
        while True:
            command, payload = connection.recv()
            if command == "close":
                break
            if command == "reset":
                value = env.reset()
            elif command == "step":
                value = auto_step(env, int(payload))
            elif command == "stage":
                env.set_stage(int(payload))
                value = env.reset()
            else:
                raise ValueError(f"Unknown worker command: {command}")
            connection.send(("ok", value))
    except EOFError:
        pass
    except BaseException:
        try:
            connection.send(("error", traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if env is not None:
            env.close()
        connection.close()


class VectorMario:
    """Synchronous control of independently seeded NES emulators.

    Process mode uses spawn on every platform; no forked Torch state, inherited
    GPU context or thread pool. Sync mode is useful for debugging and tiny runs.
    """
    def __init__(self, config: Config):
        self.config = config
        self.closed = False
        self.envs: list[MarioEnv] = []
        self.connections: list[Connection] = []
        self.processes: list[mp.Process] = []
        self.num_envs = config.num_envs
        try:
            if config.vector_mode == "sync":
                for index in range(config.num_envs):
                    self.envs.append(MarioEnv(config, config.seed + index * 1009))
                spaces = [(env.action_count, env.observation_shape) for env in self.envs]
            else:
                context = mp.get_context("spawn")
                for index in range(config.num_envs):
                    parent, child = context.Pipe()
                    process = context.Process(
                        target=_worker,
                        args=(child, config.to_dict(), config.seed + index * 1009),
                        daemon=True,
                        name=f"mario-rollout-{index}",
                    )
                    process.start()
                    child.close()
                    self.connections.append(parent)
                    self.processes.append(process)
                spaces = [self._receive(connection) for connection in self.connections]
            self.action_count, self.observation_shape = spaces[0]
            if any(space != spaces[0] for space in spaces):
                raise RuntimeError("Vector workers have inconsistent observation/action spaces")
        except BaseException:
            self.close()
            raise

    def _receive(self, connection: Connection) -> Any:
        if not connection.poll(self.config.worker_timeout):
            raise TimeoutError("Mario worker timed out; inspect environment startup or reduce load")
        try:
            status, result = connection.recv()
        except (EOFError, OSError) as error:
            raise RuntimeError("Mario worker exited unexpectedly") from error
        if status != "ok":
            raise RuntimeError(f"Mario worker failed:\n{result}")
        return result

    def reset(self) -> np.ndarray:
        if self.closed:
            raise RuntimeError("Vector environment is closed")
        if self.envs:
            observations = [env.reset() for env in self.envs]
        else:
            for connection in self.connections:
                connection.send(("reset", None))
            observations = [self._receive(connection) for connection in self.connections]
        return np.stack(observations)

    def set_stage(self, stage: int) -> np.ndarray:
        if self.closed:
            raise RuntimeError("Vector environment is closed")
        if self.envs:
            for env in self.envs:
                env.set_stage(stage)
            observations = [env.reset() for env in self.envs]
        else:
            for connection in self.connections:
                connection.send(("stage", stage))
            observations = [self._receive(connection) for connection in self.connections]
        return np.stack(observations)

    def step(self, actions: np.ndarray):
        if self.closed:
            raise RuntimeError("Vector environment is closed")
        if np.asarray(actions).shape != (self.num_envs,):
            raise ValueError(f"Expected {self.num_envs} actions")
        if self.envs:
            results = [auto_step(env, int(action)) for env, action in zip(self.envs, actions)]
        else:
            for connection, action in zip(self.connections, actions):
                connection.send(("step", int(action)))
            results = [self._receive(connection) for connection in self.connections]
        return (
            np.stack([result.observation for result in results]),
            np.asarray([result.reward for result in results], dtype=np.float32),
            np.asarray([result.terminated for result in results], dtype=bool),
            np.asarray([result.truncated for result in results], dtype=bool),
            [result.info for result in results],
        )

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for env in self.envs:
            env.close()
        for connection in self.connections:
            try:
                connection.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for process in self.processes:
            process.join(timeout=3)
            if process.is_alive():
                process.terminate()
                process.join(timeout=3)
        for connection in self.connections:
            connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
