"""Public judge-provider interface; the arithmetic demo uses no language model.

Use Foundry with caller-provided model callables for an actual LLM judge. Its
public JudgeResult scores are recorded alongside the teacher and student answers.
"""

from .foundry import Foundry, JudgeResult, exact_math_judge, run_math_demo

__all__ = ["Foundry", "JudgeResult", "exact_math_judge", "run_math_demo"]
