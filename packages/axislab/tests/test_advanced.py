"""Offline unit tests: deliberate toy ground truth, not claims about frontier LLMs."""
import math
import pytest
import torch
from torch import nn
from axislab_poc import (
    Axis, AxisBank, ActivationMoments, TopKSAE,
    fit_paired_axis, fit_pca_axis, fit_ridge_probe, evaluate_axis, roc_auc,
    activation_patch, projection_ablation, multi_axis_steering,
    train_topk_sae, sae_delta_intervention, sae_features,
    rank_one_update, batched_ridge_update, weight_edit,
    preservation_drift, forward_moments,
    transformer_lens_capture, saelens_encode_decode, nnsight_capture,
    repe_registry, upstream_model_edit, transformer_lens_steer, nnsight_steer, pca_subspace,
)


def test_paired_axis_direction_and_eval():
    torch.manual_seed(1)
    a = torch.randn(30, 4)
    b = a.clone()
    a[:, 1] += 2.0
    axis = fit_paired_axis(a,b)
    assert torch.allclose(axis.direction, torch.tensor([0.,1.,0.,0.]), atol=1e-6)
    assert evaluate_axis(axis,a,b)['auc'] > .95


def test_pca_paired_axis_positive_orientation():
    x = torch.arange(1, 21, dtype=torch.float32).unsqueeze(1)
    pos = torch.cat([x, torch.zeros_like(x)],1)
    neg = torch.zeros_like(pos)
    axis = fit_pca_axis(pos, neg)
    assert axis.direction[0] == pytest.approx(1.0)


def test_ridge_probe():
    a=torch.tensor([[3.,0.], [4.,0.], [5.,0.]])
    b=torch.tensor([[-3.,0.],[-4.,0.],[-5.,0.]])
    axis=fit_ridge_probe(a,b)
    assert axis.direction[0] > 0.99
    assert evaluate_axis(axis,a,b)['auc'] == pytest.approx(1)


def test_auc_ties():
    assert roc_auc(torch.tensor([1.,1.,0.,0.]),torch.tensor([1,0,1,0])) == pytest.approx(.5)
    assert roc_auc(torch.tensor([0., 1., 3., 4.]),torch.tensor([0,0,1,1])) == 1.0


def test_bank_and_orthogonalized_composition():
    x=Axis(torch.tensor([1.,0.]),torch.zeros(2),0)
    y=Axis(torch.tensor([1.,1.]),torch.zeros(2),0)
    b=AxisBank({'x':x,'y':y})
    assert b.cosine_matrix().shape==(2,2)
    assert b.compose({'x':1,'y':1},orthogonalize=True).tolist()==pytest.approx([1.,1.])
    with pytest.raises(KeyError): b.compose({'wrong':1})


def test_patch_and_rollback():
    m=nn.Sequential(nn.Identity())
    x=torch.zeros(1,3)
    with activation_patch(m,'0',torch.tensor([[1.,2.,3.]])):
        assert m(x).tolist()==[[1.,2.,3.]]
    assert torch.equal(m(x),x)
    assert len(m[0]._forward_hooks)==0


def test_ablation_and_multiaxis():
    m=nn.Sequential(nn.Identity())
    x=torch.tensor([[3.,4.]])
    with projection_ablation(m,'0',torch.tensor([1.,0.])):
        assert torch.allclose(m(x),torch.tensor([[0.,4.]]))
    with multi_axis_steering(m,'0',torch.eye(2),torch.tensor([2.,-1.])):
        assert torch.allclose(m(x),torch.tensor([[5.,3.]]))
    assert torch.equal(m(x),x)


def test_sae_topk_shapes_and_features():
    torch.manual_seed(5)
    sae=TopKSAE(4,10,2)
    x=torch.randn(2,3,4)
    rec,feat=sae(x)
    assert rec.shape==x.shape and feat.shape==(2,3,10)
    assert (feat!=0).sum(-1).max()<=2
    rebuilt,original=sae_features(sae,x,keep=[0,1])
    assert rebuilt.shape==x.shape and original.shape==feat.shape
    new_x=sae_delta_intervention(sae,x,feature_index=0,coefficient=2.)
    assert new_x.shape==x.shape
    assert torch.isfinite(new_x).all()


def test_train_sae_improves_over_baseline():
    torch.manual_seed(14)
    x=torch.randn(200,3)
    sae=TopKSAE(3,12,6)
    before=(sae(x)[0]-x).square().mean().item()
    metric=train_topk_sae(sae,x,steps=110,batch_size=32,lr=.012)
    assert metric.final_mse < before * .6
    assert metric.feature_density <= .5 + 1e-5


def test_rank_one_exact_rewrite_and_rollback():
    layer=nn.Linear(3,2,bias=True)
    with torch.no_grad():
        layer.weight.zero_()
        layer.bias.copy_(torch.tensor([1.,-1.]))
    key=torch.tensor([2.,0.,0.])
    target=torch.tensor([7.,3.])
    delta=rank_one_update(layer,key,target)
    old=layer(key).clone()
    with weight_edit(layer,delta,max_relative_norm=1e14):
        assert torch.allclose(layer(key),target)
    assert torch.allclose(layer(key),old)


def test_batched_memory_rewrites():
    layer=nn.Linear(3,2,bias=False)
    with torch.no_grad():layer.weight.zero_()
    keys=torch.tensor([[1.,0.],[0.,1.],[0.,0.]])
    targets=torch.tensor([[2.,0.],[0.,3.]])
    delta=batched_ridge_update(layer,keys,targets,ridge=1e-9)
    test_x=torch.tensor([[0.,0.,1.]])
    baseline=layer(test_x).clone()
    with weight_edit(layer,delta,max_relative_norm=1e14):
        assert torch.allclose(layer(keys.T),targets.T,atol=1e-5)
        assert preservation_drift(layer,test_x,baseline)==pytest.approx(0.,abs=1e-6)
    assert torch.allclose(layer.weight,torch.zeros_like(layer.weight))


def test_weight_edit_rejects_large_change():
    m=nn.Linear(2,2,bias=False)
    with pytest.raises(ValueError):
        with weight_edit(m,torch.full_like(m.weight,1e3),max_relative_norm=.01): pass


def test_weight_edit_error_cleanup():
    m=nn.Linear(2,2,bias=False)
    before=m.weight.detach().clone()
    with pytest.raises(RuntimeError):
        with weight_edit(m,torch.ones_like(m.weight),max_relative_norm=100):
            raise RuntimeError('abort')
    assert torch.allclose(m.weight,before)


def test_online_moments_match_torch_statistics():
    torch.manual_seed(4)
    data=torch.randn(22,5)
    moments=ActivationMoments()
    for chunk in data.split(4): moments.update(chunk)
    assert moments.count==22
    assert torch.allclose(moments.mean,data.double().mean(0),atol=1e-10)
    assert torch.allclose(moments.variance,data.double().var(0),atol=1e-10)


def test_forward_moments_capture_hooks_clean():
    m=nn.Sequential(nn.Identity())
    x=[torch.randn(3,2,4),torch.randn(2,2,4)]
    moments=forward_moments(m,'0',x)
    expected=torch.cat([q[:,-1] for q in x])
    assert moments.count==5
    assert torch.allclose(moments.mean,expected.double().mean(0))
    assert not m[0]._forward_hooks


def test_transformerlens_adapter_fake():
    class Bridge:
        def run_with_cache(self,prompt,names_filter):
            assert names_filter == "some_hook"
            return 'logits',{'some_hook':torch.tensor([1.,2.])}
    x,y=transformer_lens_capture(Bridge(),'hello','some_hook')
    assert x=='logits' and y.tolist()==[1.,2.]


def test_nnsight_adapter_fake():
    class Save:
        value=torch.tensor([[3.]])
    class Proxy:
        def save(self):return Save()
    class Module:
        output=Proxy()
    class Model:
        hidden=[Module()]
        def trace(self,inputs,remote=False):
            assert inputs=='hello' and not remote
            from contextlib import nullcontext
            return nullcontext()
    assert nnsight_capture(Model(),'hello','hidden.0').item()==3.


def test_saelens_adapter_contract():
    sae=TopKSAE(3,6,2)
    feat,rec=saelens_encode_decode(sae,torch.zeros(2,3))
    assert feat.shape==(2,6) and rec.shape==(2,3)


def test_external_missing_dep_errors():
    with pytest.raises(ImportError):repe_registry()
    with pytest.raises(ImportError):
        upstream_model_edit(None,None,[],None,algorithm='ROME')
    with pytest.raises(ValueError):
        upstream_model_edit(None,None,[],None,algorithm='other')


def test_paired_bad_inputs():
    with pytest.raises(ValueError):fit_paired_axis(torch.ones(3,3),torch.ones(2,3))
    with pytest.raises(ValueError):fit_pca_axis(torch.ones(2,3),torch.ones(2,2))


def test_no_stale_hooks_after_transform_error():
    from axislab_poc import transform_activation
    model=nn.Sequential(nn.Identity())
    with pytest.raises(ValueError):
        with transform_activation(model,'0',lambda x:x[...,0]):
            model(torch.ones(3,4))
    assert not model[0]._forward_hooks


def test_pca_subspace_orthonormal():
    torch.manual_seed(3)
    data = torch.randn(25, 5)
    basis = pca_subspace(data, 3)
    assert basis.shape == (3, 5)
    assert torch.allclose(basis @ basis.T, torch.eye(3), atol=1e-5)


def test_bridge_steering_local_only():
    class Bridge:
        def run_with_hooks(self, prompt, fwd_hooks):
            _, hook = fwd_hooks[0]
            return hook(torch.zeros(2,3,4))
    direction = torch.tensor([1.,0.,0.,0.])
    result = transformer_lens_steer(Bridge(), "hello", "resid", direction, coefficient=3)
    assert result[:, -1, 0].tolist() == [3., 3.]
    assert result[:, 0, 0].tolist() == [0., 0.]
    class RemoteTransformerBridge: pass
    with pytest.raises(NotImplementedError):
        transformer_lens_steer(RemoteTransformerBridge(), "hello", "resid", direction)


def test_nnsight_steering_synthetic_envoy():
    class Saved:
        def __init__(self, value): self.value=value
    class Activation:
        def __init__(self, x):self.x=x
        def __add__(self, other):return Activation(self.x + other)
        def save(self):return Saved(self.x)
    class Mod:
        def __init__(self):self.output=Activation(torch.zeros(2,3))
    class Model:
        hidden=[Mod()]
        def trace(self, text, remote=False):
            from contextlib import nullcontext
            return nullcontext()
    result=nnsight_steer(Model(),'hello','hidden.0',torch.tensor([1.,0.,0.]),coefficient=2)
    assert result.tolist()==[[2.,0.,0.],[2.,0.,0.]]
