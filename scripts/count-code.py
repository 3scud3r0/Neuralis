#!/usr/bin/env python3
"""Report own production, test and tooling source without generated dependencies.

This is a conservative physical-line metric, not a statement of software quality.
It excludes blank lines and comment-only lines, reports multiline docstrings as
documentation, and separates test files from production. Run cloc independently
for a language-aware cross-check when it is available.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import tokenize
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTENSIONS = {".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".py", ".rs", ".css", ".scss", ".sh"}
EXCLUDED_PARTS = {
    "node_modules", "dist", "target", ".git", ".venv", ".venvs", "__pycache__",
    ".pytest_cache", ".ruff_cache", ".mypy_cache", "coverage", "htmlcov", "generated",
    "test-results", "playwright-report", "artifacts", "checkpoints", "recordings",
}


def python_documentation_lines(source: str) -> set[int]:
    """Identify standalone string blocks rather than counting docstrings as code."""
    documented: set[int] = set()
    previous_type = tokenize.NEWLINE
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type == tokenize.STRING and previous_type in {
                tokenize.INDENT, tokenize.NEWLINE, tokenize.DEDENT, tokenize.NL,
            }:
                documented.update(range(token.start[0], token.end[0] + 1))
            if token.type not in {tokenize.COMMENT, tokenize.ENCODING}:
                previous_type = token.type
    except (tokenize.TokenError, IndentationError):
        pass
    return documented


def source_lines(path: Path) -> dict[str, int]:
    text = path.read_text(encoding="utf-8")
    counts = {"physical": 0, "code": 0, "blank": 0, "comment": 0}
    documentation = python_documentation_lines(text) if path.suffix == ".py" else set()
    in_block = False
    for number, line in enumerate(text.splitlines(), start=1):
        counts["physical"] += 1
        stripped = line.strip()
        if not stripped:
            counts["blank"] += 1
            continue
        if number in documentation:
            counts["comment"] += 1
            continue
        if path.suffix in {".py", ".sh"}:
            is_comment = stripped.startswith("#")
        else:
            is_comment = stripped.startswith("//")
            if in_block:
                if "*/" in stripped:
                    after = stripped.split("*/", 1)[1].strip()
                    in_block = False
                    is_comment = not after
                else:
                    is_comment = True
            elif stripped.startswith("/*"):
                if "*/" in stripped:
                    is_comment = not stripped.split("*/", 1)[1].strip()
                else:
                    in_block = True
                    is_comment = True
        counts["comment" if is_comment else "code"] += 1
    return counts


def category(path: Path) -> str:
    relative = path.relative_to(ROOT)
    parts = relative.parts
    name = path.name
    if "tests" in parts or "__tests__" in parts or re.search(r"\.(test|spec)\.", name) or name.startswith("test_"):
        return "tests"
    if parts[0] in {"scripts", "benchmarks"}:
        return "tooling"
    if name.endswith(".config.ts") or name in {"build.rs"}:
        return "configuration_code"
    return "production"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--minimum", type=int, default=0, help="Optional required production-line threshold")
    args = parser.parse_args()
    totals: dict[str, int] = defaultdict(int)
    languages: dict[str, int] = defaultdict(int)
    files = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.suffix not in EXTENSIONS:
            continue
        relative = path.relative_to(ROOT)
        if EXCLUDED_PARTS.intersection(relative.parts):
            continue
        try:
            lines = source_lines(path)
        except (UnicodeError, OSError):
            continue
        kind = category(path)
        totals[kind] += lines["code"]
        languages[path.suffix] += lines["code"]
        files.append({"path": str(relative), "category": kind, **lines})
    result = {
        "metric": "physical nonblank non-comment source lines; tests/tooling/config separated",
        "production_target": 30000,
        "production_target_met": totals["production"] >= 30000,
        "totals": dict(totals),
        "languages": dict(languages),
        "file_count": len(files),
        "files": files,
    }
    if args.output:
        target = args.output if args.output.is_absolute() else ROOT / args.output
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "files"}, indent=2))
    if args.minimum and totals["production"] < args.minimum:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
