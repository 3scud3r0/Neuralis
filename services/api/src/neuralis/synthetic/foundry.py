"""Bounded rejection sampling from caller-controlled, observable providers."""

from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import time
from typing import Any


@dataclass(frozen=True)
class JudgeResult:
    score: float
    accepted: bool
    reason: str


class Foundry:
    """Providers return final public answers; private reasoning is never requested.

    Providers are ordinary synchronous callables, making local inference, SDKs
    and deterministic fixtures interchangeable. Remote timeouts, credentials and
    model loading are explicitly the provider owner's responsibility.
    """

    def __init__(self, teacher: Callable[[str], str], student: Callable[[str], str],
                 judge: Callable[[str, str, str], JudgeResult], *, threshold: float = 0.9,
                 teacher_name: str = "caller-teacher", student_name: str = "caller-student"):
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Acceptance threshold must be between zero and one")
        self.teacher, self.student, self.judge = teacher, student, judge
        self.threshold = threshold
        self.teacher_name, self.student_name = teacher_name, student_name

    @staticmethod
    def _text(value: Any, label: str) -> str:
        if not isinstance(value, str) or not value.strip() or len(value) > 32000:
            raise ValueError(f"{label} must be nonempty public text of at most 32000 characters")
        return value

    def run(self, prompts: Iterable[str], *, count: int = 20, max_attempts: int = 200,
            keep_rejections: bool = False) -> dict:
        if not 1 <= count <= 1000 or not count <= max_attempts <= 10000:
            raise ValueError("Requested count must fit one to 1000; attempts must fit count to 10000")
        accepted, rejected = [], []
        attempts = 0
        scores = []
        started = time.perf_counter()
        iterator = iter(prompts)
        while attempts < max_attempts and len(accepted) < count:
            try:
                prompt = self._text(next(iterator), "Prompt")
            except StopIteration:
                break
            teacher = self._text(self.teacher(prompt), "Teacher answer")
            student = self._text(self.student(prompt), "Student answer")
            judgment = self.judge(prompt, teacher, student)
            if not isinstance(judgment, JudgeResult) or not math.isfinite(judgment.score) or not 0 <= judgment.score <= 1:
                raise ValueError("Judge must return a JudgeResult with a finite zero-to-one score")
            if not isinstance(judgment.accepted, bool) or not isinstance(judgment.reason, str) or len(judgment.reason) > 8000:
                raise ValueError("Judge decision and reason are invalid")
            attempts += 1
            scores.append(judgment.score)
            decision = judgment.accepted and judgment.score >= self.threshold
            record = {"id": f"sample-{attempts:06d}", "prompt": prompt,
                      "response": student, "teacher_response": teacher,
                      "teacher": self.teacher_name, "student": self.student_name,
                      "judge": asdict(judgment), "accepted": decision,
                      "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": student}]}
            if decision:
                accepted.append(record)
            elif keep_rejections:
                rejected.append(record)
        return {"records": accepted, "rejections": rejected,
                "statistics": {"attempted": attempts, "accepted": len(accepted), "rejected": attempts - len(accepted),
                               "acceptance_rate": len(accepted) / attempts if attempts else 0.0,
                               "mean_judge_score": sum(scores) / len(scores) if scores else None,
                               "requested": count, "complete": len(accepted) == count,
                               "threshold": self.threshold, "elapsed_seconds": time.perf_counter() - started}}


def _parse_math(prompt: str) -> tuple[int, str, int]:
    import re
    match = re.fullmatch(r"Compute (-?\d+) ([+*\-]) (-?\d+)\. Return only the integer answer\.", prompt)
    if match is None:
        raise ValueError("The deterministic math provider only handles its documented prompt format")
    left, operator, right = int(match[1]), match[2], int(match[3])
    if abs(left) > 100000 or abs(right) > 100000:
        raise ValueError("Arithmetic operands exceed the demo budget")
    return left, operator, right


def exact_math_teacher(prompt: str) -> str:
    left, operator, right = _parse_math(prompt)
    if operator == "+":
        answer = left + right
    elif operator == "-":
        answer = left - right
    else:
        answer = left * right
    return str(answer)


def exact_math_judge(prompt: str, teacher: str, student: str) -> JudgeResult:
    expected = exact_math_teacher(prompt)
    # Verify independently against the task, rather than trusting the teacher.
    teacher_valid = teacher.strip() == expected
    student_valid = student.strip() == expected
    correct = teacher_valid and student_valid
    return JudgeResult(1.0 if correct else 0.0, correct,
                       "Teacher and student match the independently computed integer" if correct
                       else "An answer differs from the independently computed integer")


def run_math_demo(count: int = 20, seed: int = 42, threshold: float = 0.9) -> dict:
    """Actual arithmetic generation with intentionally imperfect demo student.

    This demonstrates data filtering, not an LLM or a trained student model.
    Every fourth student answer is perturbed to exercise real rejection.
    """
    if not 1 <= count <= 1000:
        raise ValueError("Demo count must fit one to 1000")
    generator = random.Random(seed)
    calls = 0

    def student(prompt):
        nonlocal calls
        calls += 1
        answer = int(exact_math_teacher(prompt))
        return str(answer + (1 if calls % 4 == 0 else 0))

    def prompts():
        for _ in range(count * 2):
            left, right = generator.randint(-99, 99), generator.randint(-99, 99)
            operator = generator.choice(["+", "-", "*"])
            yield f"Compute {left} {operator} {right}. Return only the integer answer."

    result = Foundry(exact_math_teacher, student, exact_math_judge, threshold=threshold,
                     teacher_name="exact-integer-teacher", student_name="perturbed-arithmetic-demo").run(
                         prompts(), count=count, max_attempts=count * 2, keep_rejections=True)
    result["statistics"].update(seed=seed, provider_kind="deterministic-arithmetic-demo", trained_model=False)
    return result


def export_jsonl(records: Iterable[dict], path: str | Path | None = None) -> str:
    values = []
    for record in records:
        if len(values) >= 10000:
            raise ValueError("JSONL export exceeds the 10000-record budget")
        if not isinstance(record, dict) or record.get("accepted") is not True:
            raise ValueError("Training export accepts approved records only")
        text = json.dumps(record, ensure_ascii=False, allow_nan=False)
        if len(text.encode()) > 128 * 1024:
            raise ValueError("Export record exceeds the 128 KiB budget")
        values.append(text)
    encoded = "\n".join(values) + ("\n" if values else "")
    if path is not None:
        Path(path).write_text(encoded, encoding="utf-8")
    return encoded
