"""Evidence-backed training diagnostics; heuristics never imply solved play."""
from __future__ import annotations

from collections import Counter
import math
from typing import Any


def numeric(values):
    return [float(value) for value in values if isinstance(value, (float, int)) and math.isfinite(value)]


def summarize(data: dict[str, Any], window: int = 100) -> dict:
    if window < 1:
        raise ValueError("Diagnostic window must be positive")
    episodes = data.get("episodes", [])[-window:]
    metrics = data.get("metrics", [])[-window:]
    findings = []
    def finding(code, severity, message, evidence):
        findings.append({"code": code, "severity": severity, "message": message, "evidence": evidence})
    if not metrics:
        finding("no-updates", "warning", "No completed optimization updates were exported.", {"metricCount": 0})
    if not episodes:
        finding("no-complete-episodes", "info", "No full episodes were observed in this window.", {"window": window})
    successes = sum(bool(episode.get("flagGet")) for episode in episodes)
    if episodes and successes == 0:
        finding("no-flags", "warning", "No flag completion was recorded in the selected training window.",
                {"episodes": len(episodes), "successes": successes})
    timeouts = sum(bool(episode.get("truncated")) for episode in episodes)
    if episodes and timeouts / len(episodes) > 0.8:
        finding("timeout-heavy", "warning", "Most episodes reach the configured action limit; inspect stalling or the limit.",
                {"timeouts": timeouts, "episodes": len(episodes)})
    positions = numeric([episode.get("xPos") for episode in episodes])
    if len(positions) >= 10 and max(positions) - min(positions) < 32 and successes == 0:
        finding("repeated-failure-position", "warning", "Episodes repeatedly end in a narrow X-position range.",
                {"minX": min(positions), "maxX": max(positions), "episodes": len(positions)})
    entropies = numeric([metric.get("entropy") for metric in metrics])
    if entropies and entropies[-1] < 0.1:
        finding("policy-entropy-low", "warning", "Action entropy is low; inspect premature exploration collapse.",
                {"latestEntropy": entropies[-1]})
    kls = numeric([metric.get("approxKL") for metric in metrics])
    if kls and max(kls) > 0.1:
        finding("policy-update-large", "warning", "Some PPO updates have a large approximate KL.", {"maxKL": max(kls)})
    clips = numeric([metric.get("clipFraction") for metric in metrics])
    if clips and sum(clips) / len(clips) > 0.5:
        finding("ppo-clipping-heavy", "warning", "Most sampled policy ratios are clipped; inspect update size.",
                {"meanClipFraction": sum(clips) / len(clips)})
    errors = [event for event in data.get("events", []) if event.get("kind") == "error"]
    for event in errors[-5:]:
        finding("runtime-error", "error", str(event.get("message", "Runtime failure")),
                {"step": event.get("step"), "type": event.get("type")})
    stage_counts = Counter(str(episode.get("stage", "unknown")) for episode in episodes)
    return {"schema": "neuralis.mario.diagnostics/v1", "window": window,
            "episodes": len(episodes), "updates": len(metrics), "successes": successes,
            "observedSuccessRate": successes / len(episodes) if episodes else None,
            "stageCounts": dict(stage_counts), "findings": findings,
            "note": "Heuristic diagnostics based on exported measurements, not a policy quality guarantee."}
