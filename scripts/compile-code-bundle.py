#!/usr/bin/env python3
"""Compile the repository's textual source and configuration into one reviewable TXT.

The bundle is intentionally deterministic: paths are sorted, every section has a
stable marker, and a manifest records byte counts and SHA-256 digests. Installed
dependencies, generated builds, caches, model checkpoints, media and the bundle
itself are excluded so the result remains useful as a source archive.
"""
from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "neuralis-all-code.txt"
TEXT_EXTENSIONS = {
    ".c", ".cc", ".cpp", ".css", ".d.ts", ".go", ".h", ".html", ".ini", ".java",
    ".js", ".json", ".jsx", ".lock", ".mjs", ".md", ".py", ".rs", ".scss", ".sh",
    ".sql", ".toml", ".ts", ".tsx", ".txt", ".xml", ".yaml", ".yml",
}
TEXT_FILENAMES = {".gitignore", ".env.example", "Dockerfile", "Makefile", "README"}
EXCLUDED_DIRECTORIES = {
    ".git", ".venv", ".venvs", ".cache", "node_modules", "dist", "target", "__pycache__",
    ".pytest_cache", ".ruff_cache", ".mypy_cache", "coverage", "htmlcov", "test-results",
    "playwright-report", "runs", "checkpoints", "recordings", "artifacts", "generated",
}
EXCLUDED_PREFIXES = ("screenshots/", "demos/super-mario/evidence/")
MAX_FILE_BYTES = 8 * 1024 * 1024


def is_source_file(path: Path, output: Path) -> bool:
    if path == output or not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
        return False
    relative = path.relative_to(ROOT).as_posix()
    if any(part in EXCLUDED_DIRECTORIES for part in path.relative_to(ROOT).parts):
        return False
    if any(relative.startswith(prefix) for prefix in EXCLUDED_PREFIXES):
        return False
    if path.name in {"neuralis-all-code.txt", "code-count.json"}:
        return False
    if path.name in TEXT_FILENAMES or path.name.startswith("Dockerfile") or path.name.startswith("README"):
        return True
    return path.suffix.lower() in TEXT_EXTENSIONS


def read_text(path: Path) -> str | None:
    data = path.read_bytes()
    if b"\x00" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compile_bundle(output: Path) -> tuple[int, int, str]:
    files: list[tuple[str, str]] = []
    skipped = 0
    for path in sorted(ROOT.rglob("*")):
        if not is_source_file(path, output):
            continue
        text = read_text(path)
        if text is None:
            skipped += 1
            continue
        if not text.endswith("\n"):
            text += "\n"
        files.append((path.relative_to(ROOT).as_posix(), text))

    generated = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    manifest_lines = [
        "NEURALIS STUDIO — COMPLETE TEXT SOURCE BUNDLE",
        "================================================",
        f"Generated (UTC): {generated}",
        f"Repository root: {ROOT}",
        "Encoding: UTF-8",
        "Contents: source code, tests, scripts, configuration and documentation",
        "Excluded: installed dependencies, builds, caches, binaries, media, checkpoints and runtime data",
        f"Included files: {len(files)}",
        f"Skipped binary/unreadable candidates: {skipped}",
        "",
        "FILE MANIFEST",
        "-------------",
    ]
    for relative, text in files:
        manifest_lines.append(f"{relative}\t{len(text.encode('utf-8'))} bytes\tsha256:{digest(text)}")
    manifest_lines.extend(["", "SOURCE FILES", "------------", ""])
    sections: list[str] = []
    for relative, text in files:
        sections.append(
            f"\n{'=' * 100}\n"
            f"BEGIN FILE: {relative}\n"
            f"{'=' * 100}\n"
            f"{text}"
            f"{'=' * 100}\n"
            f"END FILE: {relative}\n"
            f"{'=' * 100}\n"
        )
    content = "\n".join(manifest_lines) + "\n" + "".join(sections)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8", newline="\n")
    return len(files), len(content.encode("utf-8")), hashlib.sha256(content.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    files, size, checksum = compile_bundle(output)
    print(f"Wrote {output}")
    print(f"Included {files} files; {size:,} UTF-8 bytes")
    print(f"Bundle SHA-256: {checksum}")


if __name__ == "__main__":
    main()
