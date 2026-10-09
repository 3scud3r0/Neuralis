"""CPU-only reproducible observational demo, not a hidden chain-of-thought extractor."""
from pathlib import Path
import torch
from torch import nn
from axislab_poc import (decode_states, trace_token_ranks, export_trace_json,
                         export_trace_html, LocalActivationTrace, CircuitNode, CircuitEdge, export_circuit_html)


class IterativeToy(nn.Module):
    def __init__(self):
        super().__init__()
        self.step = nn.Linear(3, 3, bias=False)
        with torch.no_grad():
            self.step.weight.copy_(torch.tensor([[.6, .4, 0], [0, .55, 0], [0, 0, 1.]]))

    def forward(self, x):
        for _ in range(5):
            x = self.step(x)
        return x


def main():
    model = IterativeToy().eval()
    x = torch.tensor([[[0., 2., -0.1]]])
    with LocalActivationTrace(model, ['step'], max_entries=5) as capture:
        with torch.no_grad():
            final = model(x)
    states = [r.activation for r in capture.entries]
    readouts = decode_states(states, unembed_weight=6 * torch.eye(3),
                             target_ids=torch.tensor([0]), top_k=3)
    output = Path('axislab_reasoning_output')
    output.mkdir(exist_ok=True)
    capture.export_npz(output / 'local_states.npz')
    export_trace_json(output / 'projections.json', readouts,
                      stage_labels=[f'recurrence_{i}' for i in range(5)],
                      metadata={'model':'toy-local', 'kind':'observables-not-cot'})
    export_trace_html(output / 'report.html', readouts,
                      stage_labels=[f'recurrence_{i}' for i in range(5)])
    export_circuit_html(output / 'circuit.html',
        [CircuitNode('state', 'input', 'current state'),
         CircuitNode('recurrent', 'transform', 'same block repeated 5 times'),
         CircuitNode('unembedding', 'readout', 'vocabulary projection')],
        [CircuitEdge('state','recurrent',1.,evidence='architectural'),
         CircuitEdge('recurrent','unembedding',1.,evidence='architectural')])
    print('top IDs:', [r.token_ids[0,0].item() for r in readouts])
    print('target ranks:', trace_token_ranks(readouts))
    print('recorded:', len(capture.entries), 'dropped:', capture.dropped)
    print('final state:', final.detach().tolist())
    print('saved:', output.resolve())

if __name__ == '__main__': main()
