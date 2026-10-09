"""Observable policy diagnostics, never a reconstruction of private reasoning.

The module name preserves the requested integration name. Its outputs are
measured action probabilities and explicitly caller-provided public rationale.
"""

import math
from typing import Sequence

import numpy as np


def policy_diagnostics(logits: Sequence[float], *, chosen_action: int | None = None,
                       action_labels: Sequence[str] | None = None,
                       public_rationale: str | None = None) -> dict:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 1 or not 1 <= values.size <= 4096 or not np.isfinite(values).all():
        raise ValueError("Diagnostics require one to 4096 finite action logits")
    if chosen_action is not None and (not isinstance(chosen_action, int) or not 0 <= chosen_action < values.size):
        raise ValueError("Chosen action is outside the policy action space")
    if action_labels is not None and (len(action_labels) != values.size or any(not isinstance(x, str) for x in action_labels)):
        raise ValueError("Action labels must match the logits")
    if public_rationale is not None and (not isinstance(public_rationale, str) or len(public_rationale) > 8000):
        raise ValueError("Public rationale must be a caller-provided string of at most 8000 characters")
    probabilities = np.exp(values - values.max())
    probabilities /= probabilities.sum()
    nonzero = probabilities[probabilities > 0]
    entropy = float(-(nonzero * np.log(nonzero)).sum())
    ranking = np.argsort(-probabilities, kind="stable")
    top = int(ranking[0])
    return {
        "kind": "observable-policy-diagnostics", "probabilities": probabilities.tolist(),
        "entropy_nats": entropy, "normalized_entropy": entropy / math.log(values.size) if values.size > 1 else 0.0,
        "top_action": top, "chosen_action": chosen_action,
        "chosen_probability": None if chosen_action is None else float(probabilities[chosen_action]),
        "margin": float(probabilities[ranking[0]] - probabilities[ranking[1]]) if values.size > 1 else 1.0,
        "action_labels": list(action_labels) if action_labels is not None else None,
        "public_rationale": public_rationale,
        "rationale_source": "caller-provided" if public_rationale is not None else None,
    }
