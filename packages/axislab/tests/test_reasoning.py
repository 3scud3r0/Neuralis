import json
import pytest
import torch
from torch import nn
from types import SimpleNamespace
from axislab_poc import (
    logit_lens, decode_states, TunedLens, fit_tuned_lens, trace_token_ranks,
    LocalActivationTrace, capture_hf_hidden_states, causal_patch_effect,
    sae_feature_patch, component_logit_contributions, FeatureHypothesis,
    inspect_sae_features, vocabulary_partition, export_trace_html, export_trace_json,
    TopKSAE, fit_axis, inspect_axis,
)


def test_logit_lens_gets_ids_probs_and_entropy():
    h = torch.tensor([[2., -1., 0.]])
    w = torch.eye(3)
    ans = logit_lens(h, w, top_k=2, target_ids=torch.tensor([0]))
    assert ans.token_ids.shape == (1, 2)
    assert ans.token_ids[0, 0] == 0
    assert ans.target_rank.tolist() == [1]
    assert float(ans.probabilities[0, 0]) > 0.7
    assert torch.isfinite(ans.entropy).all()


def test_logit_lens_invalid_shape_and_rank():
    with pytest.raises(ValueError):
        logit_lens(torch.ones(2, 3), torch.ones(4, 4))
    with pytest.raises(ValueError):
        logit_lens(torch.ones(2, 3), torch.ones(4, 3), top_k=5)
    with pytest.raises(ValueError):
        logit_lens(torch.ones(2, 3), torch.ones(4, 3), target_ids=torch.tensor([9, 8]))


def test_entropy_rank_path_and_serialization(tmp_path):
    states = [torch.tensor([[0., 1., 0.]]), torch.tensor([[0.5, 0.5, 0.]]), torch.tensor([[1., 0., 0.]])]
    reads = decode_states(states, 8 * torch.eye(3), target_ids=torch.tensor([0]), top_k=3)
    assert trace_token_ranks(reads) == [[2], [1], [1]]
    path = export_trace_json(tmp_path / 'trajectory.json', reads, stage_labels=['a', 'b', 'c'])
    data = json.loads(path.read_text())
    assert data['schema'] == 'axislab.reasoning_observables.v1'
    assert data['stages'][0]['target_rank'] == [2]
    page = export_trace_html(tmp_path / 'trajectory.html', reads, stage_labels=['a', 'b', 'c'])
    assert 'not hidden chain-of-thought' in page.read_text()


def test_lens_html_escaping(tmp_path):
    ans = logit_lens(torch.tensor([[1., 0.]]), torch.eye(2), target_ids=torch.tensor([0]), top_k=2)
    file = export_trace_html(tmp_path / 'safe.html', [ans], stage_labels=['<img src=x onerror=alert(1)>'])
    assert '<img src=x' not in file.read_text()
    assert '&lt;img' in file.read_text()


def test_tuned_lens_train_reduces_kl():
    torch.manual_seed(33)
    x = torch.randn(50, 3)
    w = torch.eye(3) * 2
    target = torch.nn.functional.linear(x + torch.tensor([0.7, -0.3, 0.2]), w)
    before = torch.nn.functional.kl_div(torch.nn.functional.log_softmax(x @ w.T, -1),
                                          target.softmax(-1), reduction='batchmean').item()
    tuned, stats = fit_tuned_lens(x, target, w, steps=180, lr=0.06)
    assert isinstance(tuned, TunedLens)
    assert stats['training_KL'] < before * 0.1
    assert tuned.read(x[:2], w, top_k=3).token_ids.shape == (2, 3)


def test_recorder_recurrent_capture_and_cleanup(tmp_path):
    class Recurrent(nn.Module):
        def __init__(self):
            super().__init__(); self.block = nn.Linear(2, 2, bias=False)
            with torch.no_grad(): self.block.weight.copy_(torch.eye(2))
        def forward(self, x):
            for _ in range(3): x = self.block(x)
            return x
    model = Recurrent()
    with LocalActivationTrace(model, ['block'], max_entries=2, mode='all') as trace:
        model(torch.ones(1, 2))
        assert len(trace.entries) == 2
        assert trace.dropped == 1
    assert not model.block._forward_hooks
    path = trace.export_npz(tmp_path / 'captured.npz')
    assert path.exists() and path.with_suffix('.json').exists()


def test_recorder_budget_and_error_cleanup():
    model = nn.Sequential(nn.Identity())
    try:
        with LocalActivationTrace(model, ['0'], mode='last_token', max_bytes=2) as recorder:
            model(torch.ones(1, 3, 4))
            assert recorder.dropped == 1
            raise RuntimeError('expected')
    except RuntimeError:
        pass
    assert not model[0]._forward_hooks


def test_hf_hidden_state_capture_for_attention_masks():
    class MockHF(nn.Module):
        def forward(self, input_ids, output_hidden_states, return_dict, attention_mask=None):
            x = input_ids.float().unsqueeze(-1).expand(-1, -1, 2).clone()
            return SimpleNamespace(hidden_states=(x, x + 10), logits=x)
    inp = torch.tensor([[1, 2, 0], [0, 7, 8]])
    mask = torch.tensor([[1, 1, 0], [0, 1, 1]])
    _out, states = capture_hf_hidden_states(MockHF(), inp, attention_mask=mask)
    assert torch.allclose(states[0], torch.tensor([[2., 2.], [8., 8.]]))
    assert states[1].shape == (2, 2)


def test_causal_patching_clean_corrupt():
    model = nn.Sequential(nn.Identity(), nn.Linear(2,1,bias=False))
    with torch.no_grad(): model[1].weight.copy_(torch.tensor([[1., 0.]]))
    res = causal_patch_effect(model, '0', torch.tensor([[2., 0.]]),
                              torch.tensor([[-1., 0.]]), lambda y: y.mean())
    assert res.clean == pytest.approx(2)
    assert res.corrupted == pytest.approx(-1)
    assert res.patched == pytest.approx(2)
    assert res.recovered_fraction == pytest.approx(1)
    assert not model[0]._forward_hooks


def test_causal_patching_no_difference_fraction_none():
    model = nn.Sequential(nn.Identity())
    x = torch.tensor([[2.]])
    res = causal_patch_effect(model, '0', x, x, lambda y: y.mean())
    assert res.recovered_fraction is None


def test_sae_patch_shared_dictionary():
    class TrivialSAE:
        def encode(self, x): return x
        def decode(self, x): return x
    base = torch.tensor([[1., 0.]])
    clean = torch.tensor([[0., 4.]])
    patched = sae_feature_patch(base, clean, TrivialSAE(), [1])
    assert torch.allclose(patched, torch.tensor([[1., 4.]]))
    with pytest.raises(ValueError): sae_feature_patch(base, clean, TrivialSAE(), [1,1])


def test_component_logit_additivity():
    w = torch.tensor([[2., 3.], [0., 1.]])
    pieces = {'attn': torch.tensor([1., 0.]), 'mlp': torch.tensor([0., 2.])}
    contributions = component_logit_contributions(pieces, w, target_token_id=0)
    assert sum(float(v) for v in contributions.values()) == pytest.approx(8.)
    assert (sum(pieces.values()) @ w[0]).item() == pytest.approx(8.)


def test_feature_labels_and_partition():
    sae = TopKSAE(3, 5, k=2)
    x = torch.randn(10, 3)
    labeled = FeatureHypothesis(0, 'aritmetica', 'heuristic_guess')
    out = inspect_sae_features(sae, x, labels={0:labeled}, top_k=5)
    assert len(out) == 5
    assert next(row for row in out if row['feature_index'] == 0)['label'] == 'aritmetica'
    labels = vocabulary_partition(torch.tensor([2, 4, 5, 9]), visible_vocab_size=5,
                                   latent_vocab_size=3)
    assert [r['kind'] for r in labels] == ['visible','visible','latent','special_or_unknown']


def test_axis_hypothesis_not_transcript():
    axis = fit_axis(torch.tensor([[1.,0.],[2.,0.]]),
                    torch.tensor([[-1.,0.],[-2.,0.]]), denoise_variance=0)
    out = inspect_axis(axis, torch.tensor([[4.,0.]]), description='ativação teste')
    assert out['provenance'] == 'heuristic_guess'
    assert 'not a transcript' in out['warning']


def test_trace_output_list_invalid():
    class Bad(nn.Module):
        def forward(self, x): return {'value': x}
    model = nn.Sequential(Bad())
    with pytest.raises(TypeError):
        with LocalActivationTrace(model, ['0'], mode='all'):
            model(torch.ones(1, 2))
    assert not model[0]._forward_hooks


def test_coda_lens_with_user_supplied_suffix():
    from axislab_poc import coda_lens
    hidden = torch.tensor([[0., 2.]])
    def suffix(x): return x.flip(-1)
    readout = coda_lens(hidden, suffix, torch.eye(2), top_k=2, target_ids=torch.tensor([0]))
    assert readout.token_ids[0,0].item() == 0
    assert 'coda_like' in readout.source


def test_attention_summary_recurrent_shape_and_normalization():
    from axislab_poc import recurrent_attention_summary
    patt = torch.tensor([[[[[.8,.2],[.1,.9]],[[.6,.4],[.7,.3]]]]],dtype=torch.float32)
    out = recurrent_attention_summary(patt, memory_positions=[1], query_position=-1)
    assert len(out) == 2
    assert out[0]['memory_attention_mass'] == pytest.approx(.9)
    with pytest.raises(ValueError):
        recurrent_attention_summary(patt * 2, memory_positions=[1])


def test_svg_circuit_interaction_offline(tmp_path):
    from axislab_poc import CircuitNode, CircuitEdge, export_circuit_html
    nodes = [CircuitNode('inp', 'encoder','input'), CircuitNode('head1','decoder','atten head')]
    edges = [CircuitEdge('inp','head1', 0.7, evidence='descriptive')]
    saved = export_circuit_html(tmp_path / 'circuit.html', nodes, edges)
    assert saved.exists()
    page = saved.read_text()
    assert 'threshold' in page and '<svg' in page and 'head1' in page
    with pytest.raises(ValueError):
        export_circuit_html(tmp_path / 'bad.html', nodes, [CircuitEdge('missing','head1',.5)])
