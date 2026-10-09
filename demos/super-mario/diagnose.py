#!/usr/bin/env python3
"""Analyze exported failure patterns, exploration and optimizer behavior."""
import argparse
import json
from pathlib import Path
from mario_rl.diagnostics import summarize
from mario_rl.telemetry import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    result = summarize(report, args.window)
    if args.output:
        atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
