"""Portable circuit visualizations and descriptive attention diagnostics.

No attention edge is assumed causal merely because it has high attention mass.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import html
import math
import json
import torch
from torch import Tensor


@dataclass(frozen=True)
class CircuitNode:
    id: str
    group: str
    description: str


@dataclass(frozen=True)
class CircuitEdge:
    source: str
    target: str
    weight: float
    description: str = ""
    evidence: str = "descriptive"


def recurrent_attention_summary(patterns: Tensor, *,
                                memory_positions: list[int],
                                query_position: int = -1) -> list[dict]:
    """Summarize attention mass to memory positions across steps and heads.

    `patterns`: [steps,batch,heads,queries,keys], already softmax-normalized.
    Aggregates across batch. High score does not prove an iteration circuit.
    """
    if patterns.ndim != 5 or not patterns.is_floating_point():
        raise ValueError("Expected floating attention [steps,batch,heads,Q,K]")
    if not memory_positions or not all(0 <= i < patterns.shape[-1] for i in memory_positions):
        raise ValueError("memory_positions must be valid key locations")
    if not -patterns.shape[-2] <= query_position < patterns.shape[-2]:
        raise ValueError("query_position invalid")
    if not torch.isfinite(patterns).all() or (patterns < 0).any():
        raise ValueError("Invalid attention weights")
    total = patterns.sum(-1)
    if not torch.allclose(total, torch.ones_like(total), atol=1e-3):
        raise ValueError("Attention patterns should sum to 1 over keys")
    selected = patterns[..., query_position, memory_positions].sum(-1).mean(1)
    return [{"step": int(step), "head": int(head),
             "memory_attention_mass": float(selected[step, head])}
            for step in range(selected.shape[0]) for head in range(selected.shape[1])]


def export_circuit_html(filename: str | Path,
                        nodes: list[CircuitNode], edges: list[CircuitEdge], *,
                        title: str = "AxisLab circuit map") -> Path:
    """Create an offline interactive SVG map with a per-edge strength slider.

    Provide evidence='causal_test' only when the edge underwent a documented
    intervention; otherwise label it descriptive/correlational.
    """
    ids = [x.id for x in nodes]
    if not nodes or len(ids) != len(set(ids)):
        raise ValueError("Circuit nodes must be non-empty and have unique IDs")
    if any(e.source not in ids or e.target not in ids or not math.isfinite(e.weight) or e.weight < 0 for e in edges):
        raise ValueError("Each edge needs known nodes and finite nonnegative weight")
    groups = list(dict.fromkeys(node.group for node in nodes))
    by_group = {g: [n for n in nodes if n.group == g] for g in groups}
    width, height = 1000, max(310, 130 + 85 * max(len(v) for v in by_group.values()))
    pos = {}
    for col, group in enumerate(groups):
        x = 80 + col * 840/max(1, len(groups)-1)
        for row, node in enumerate(by_group[group]):
            pos[node.id] = (x, 110 + row*85)
    svg = []
    for edge in edges:
        x1,y1=pos[edge.source]; x2,y2=pos[edge.target]
        evidence = html.escape(edge.evidence,quote=True)
        description = html.escape(edge.description, quote=True)
        note = html.escape(edge.source + ' → ' + edge.target,quote=True)
        tooltip = f"{note} | weight={edge.weight:.4f} | {evidence} | {description}"
        svg.append(f'<line class="edge" data-weight="{edge.weight:.8f}" x1="{x1}" y1="{y1}" '
                   f'x2="{x2}" y2="{y2}" stroke="#56a3da" stroke-opacity="0.5" '
                   f'stroke-width="{min(8,max(1.2,edge.weight*4)):.2f}"><title>{tooltip}</title></line>')
    for node in nodes:
        x,y=pos[node.id]
        label=html.escape(node.id)
        detail=html.escape(node.description)
        group=html.escape(node.group)
        svg.append(f'<g><rect x="{x-66:.1f}" y="{y-20:.1f}" width="132" height="40" '
                   'rx="8" fill="#27374c" stroke="#6c98ca"/>'
                   f'<text x="{x:.1f}" y="{y+5:.1f}" text-anchor="middle" fill="white" '
                   f'font-size="13">{label}</text><title>{group}: {detail}</title></g>')
    code = """const bar=document.getElementById('threshold');
    function update(){let t=Number(bar.value);document.getElementById('cutoff').textContent=t.toFixed(2);
      document.querySelectorAll('line.edge').forEach(x=>x.style.display=Number(x.dataset.weight)>=t?'':'none');}
    bar.addEventListener('input',update);update();"""
    result = f'''<!doctype html><html lang="pt-BR"><meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>
    <style>body{{font-family:system-ui;background:#111923;color:#e6ecf6;max-width:1050px;margin:2rem auto;padding:0 1rem}}
    svg{{background:#192435;border-radius:10px;width:100%;height:auto}}input{{width:50%}}</style>
    <h1>{html.escape(title)}</h1><p>Mapa exploratório de componentes. Arestas não são provas de causalidade sem intervenção e controles.</p>
    <label>Mostrar arestas com peso ≥ <span id="cutoff">0</span>
    <input id="threshold" type="range" min="0" max="1" step="0.01" value="0"></label>
    <svg viewBox="0 0 {width} {height}" role="img" aria-label="Mapa do circuito">{''.join(svg)}</svg>
    <script>{code}</script></html>'''
    destination = Path(filename)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(result,encoding='utf8')
    return destination
