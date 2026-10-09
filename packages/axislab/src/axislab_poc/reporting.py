"""Portable local reports: no telemetry, no external JavaScript/CSS or assets."""
from __future__ import annotations
import html
import json
from pathlib import Path
from typing import Any, Sequence
from .lenses import LensReadout


def _payload(readouts: Sequence[LensReadout], *,
             stage_labels: Sequence[str] | None = None,
             metadata: dict[str, Any] | None = None) -> dict:
    if not readouts:
        raise ValueError("Need at least one readout")
    if stage_labels is None:
        stage_labels = [f"stage_{i}" for i in range(len(readouts))]
    if len(stage_labels) != len(readouts):
        raise ValueError("stage_labels/readouts mismatch")
    return {
        "schema": "axislab.reasoning_observables.v1",
        "disclaimer": "Token projections are probabilistic proxies, not hidden chain-of-thought transcripts.",
        "metadata": metadata or {},
        "stages": [{"label": label, **read.as_dict()} for label, read in zip(stage_labels, readouts)],
    }


def export_trace_json(filename: str | Path, readouts: Sequence[LensReadout], *,
                      stage_labels: Sequence[str] | None = None,
                      metadata: dict[str, Any] | None = None) -> Path:
    destination = Path(filename)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(_payload(readouts, stage_labels=stage_labels,
                                               metadata=metadata), indent=2,
                                               ensure_ascii=False), encoding="utf8")
    return destination


def export_trace_html(filename: str | Path, readouts: Sequence[LensReadout], *,
                      stage_labels: Sequence[str] | None = None,
                      metadata: dict[str, Any] | None = None) -> Path:
    """Build an offline HTML report with a per-stage top-token table and rank plot.

    This report uses only supplied projection data; it never generates fictitious
    spoken explanations from intermediate numerical states.
    """
    data = _payload(readouts, stage_labels=stage_labels, metadata=metadata)
    stages = data["stages"]
    # The illustration is a target-rank trajectory if scalar ranks exist.
    available = [(i, s["target_rank"][0] if isinstance(s["target_rank"], list) else s["target_rank"])
                 for i,s in enumerate(stages)
                 if isinstance(s["target_rank"], int) or
                    (isinstance(s["target_rank"], list) and len(s["target_rank"]) == 1 and isinstance(s["target_rank"][0], int))]
    lines = ""
    if len(available) > 1:
        maximum = max(rank for _,rank in available)
        points = []
        for i, rank in available:
            x = 35 + 715 * i / max(1, len(stages)-1)
            y = 205 - 170 * (1 - (rank-1)/max(1,maximum-1))
            points.append(f"{x:.1f},{y:.1f}")
        lines = ('<div class="plot"><h2>Rank do token-alvo (menor = melhor)</h2>'
                 '<svg viewBox="0 0 800 255" role="img" aria-label="Evolução do rank">'
                 '<line x1="35" y1="225" x2="760" y2="225" stroke="#889" />'
                 f'<polyline fill="none" stroke="#5588ff" stroke-width="3" points="{" ".join(points)}"/>'
                 '</svg></div>')
    rows=[]
    for stage in stages:
        pairs = list(zip(_flatten(stage["token_ids"]), _flatten(stage["probabilities"])))
        top = ", ".join(f"{int(i)} ({float(p)*100:.1f}%)" for i,p in pairs[:10])
        rank = stage["target_rank"]
        rows.append(f'<tr><td>{html.escape(str(stage["label"]))}</td>'
                    f'<td>{html.escape(top)}</td>'
                    f'<td>{html.escape(str(rank))}</td></tr>')
    caution=html.escape(data["disclaimer"])
    rendered = f'''<!doctype html><html lang="pt-BR"><meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>AxisLab | Auditoria de estados</title>
    <style>body{{font:16px system-ui;background:#111824;color:#eaf0f7;max-width:1020px;margin:2rem auto;padding:0 1rem}}
    h1{{font-size:1.8rem}}h2{{font-size:1.1rem}}p{{line-height:1.55}}.note{{border:1px solid #5d6c7a;padding:16px;border-radius:9px}}
    table{{width:100%;border-collapse:collapse;overflow-wrap:anywhere}}td,th{{padding:12px;text-align:left;border-bottom:1px solid #344050}}
    th{{background:#243146}}.plot{{margin:24px 0}}svg{{width:100%;height:auto;max-height:320px}}</style>
    <h1>AxisLab — Projeções e trajetória latente</h1><p class="note">{caution}</p>
    <p>Top tokens identificados por IDs: consulte seu tokenizer local para rótulos correspondentes.</p>
    {lines}<table><thead><tr><th>Etapa</th><th>Tokens projetados (probabilidades)</th><th>Rank-alvo</th></tr></thead>
    <tbody>{''.join(rows)}</tbody></table>
    </html>'''
    destination = Path(filename)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(rendered, encoding="utf8")
    return destination


def _flatten(val):
    if isinstance(val, list):
        if val and isinstance(val[0], list):
            return _flatten(val[0])
        return val
    return [val]
