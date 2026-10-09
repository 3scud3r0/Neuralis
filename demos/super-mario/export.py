#!/usr/bin/env python3
"""Produce portable Neuralis telemetry with a measured evaluation attached."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from mario_rl.telemetry import atomic_json


def read_object(path: Path) -> dict:
    if path.stat().st_size > 100 * 1024 * 1024:
        raise ValueError("JSON input exceeds the 100 MiB export budget")
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain an object")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Run directory containing neuralis-mario-run.json")
    parser.add_argument("--evaluation", type=Path, help="Actual evaluation.json report")
    parser.add_argument("--inspection", type=Path, help="Actual inspection.json report")
    parser.add_argument("--output", type=Path, default=Path("runs/mario-export.json"))
    parser.add_argument("--csv", type=Path, help="Optional flat PPO metrics CSV")
    args = parser.parse_args()
    data = read_object(args.run / "neuralis-mario-run.json")
    if data.get("schema") != "neuralis.mario.run/v1":
        raise ValueError("Unsupported telemetry schema")
    if args.evaluation:
        data["evaluation"] = read_object(args.evaluation)
    if args.inspection:
        data["inspection"] = read_object(args.inspection)
    atomic_json(args.output, data)
    if args.csv:
        metrics = data.get("metrics", [])
        fields = sorted({key for metric in metrics for key in metric})
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(metrics)
    print(f"Exported {len(data.get('metrics', []))} PPO updates to {args.output}")


if __name__ == "__main__":
    main()
