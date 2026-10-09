"""Verified synthetic examples using explicit teacher, student and judge adapters."""

from .foundry import Foundry, JudgeResult, run_math_demo, export_jsonl

__all__ = ["Foundry", "JudgeResult", "run_math_demo", "export_jsonl"]
