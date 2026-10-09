"""Bounded trajectory capture and optional emulator/transport adapters."""

from .trajectory_db import TrajectoryRecord, TrajectoryStore
from .temporal_debugger import TemporalDebugger
from .frames import encode_frame_png
from .gridworld import GridWorld, record_gridworld_demo

__all__ = ["TrajectoryRecord", "TrajectoryStore", "TemporalDebugger", "encode_frame_png", "GridWorld", "record_gridworld_demo"]
