# AxisLab Research Toolkit 0.3.0

AxisLab provides independent PyTorch algorithms for representation engineering,
sparse autoencoders, causal interventions, linear memory edits, and observability
of local model states. The default experiments run on CPU and download no models.
Optional adapters call upstream libraries; this package does not contain copies
of TransformerLens, nnsight, SAELens, RepE, ROME, or MEMIT.

This is the English documentation for the supplied research toolkit. The complete
Portuguese README is preserved byte for byte in [README.original.md](README.original.md).
All 28 supplied text files, including source, tests, license, and reference
reports, retain their original bytes. [provenance.json](provenance.json) records
their SHA256 digests and byte lengths. The supplied bundle excluded `.npz` binary
data; run the reasoning demo to generate a fresh trace. Reference reports are
examples from the bundle, not evidence of a new run.

## Installation and local checks

Requires Python 3.10 or later and PyTorch 2.3 or later. In this package directory:

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

From Neuralis, reuse its prepared API environment without resolving or upgrading
existing dependencies:

```bash
cd /workspace/Neuralis
UV_CACHE_DIR=/workspace/.cache/uv uv pip install --python /workspace/.venvs/neuralis-api/bin/python \
  --no-deps --no-build-isolation -e packages/axislab
/workspace/.venvs/neuralis-api/bin/python -m pytest -q packages/axislab/tests
```

Run demonstrations from a separate working directory so the supplied reference
reports remain intact:

```bash
AXISLAB_ROOT=/workspace/Neuralis/packages/axislab
AXISLAB_PYTHON=/workspace/.venvs/neuralis-api/bin/python
AXISLAB_RUN=$(mktemp -d /tmp/axislab-run.XXXXXX)
cd "$AXISLAB_RUN"
"$AXISLAB_PYTHON" "$AXISLAB_ROOT/demo.py"
"$AXISLAB_PYTHON" "$AXISLAB_ROOT/demo_advanced.py"
"$AXISLAB_PYTHON" "$AXISLAB_ROOT/demo_reasoning.py"
```

`demo.py` fits an axis and verifies temporary steering; `demo_advanced.py`
evaluates held-out examples, trains a small SAE, edits and restores linear
weights, and aggregates streaming moments. `demo_reasoning.py` captures five
recurrences of a small local network and writes `local_states.npz`, metadata,
projected readouts, and offline HTML reports under the temporary run directory.
The first original demo prints Portuguese labels; its source is preserved.

## Local capabilities

| Module | Main APIs | Purpose |
| --- | --- | --- |
| `core` | `Axis`, `fit_axis`, `capture_activations`, `temporary_steering` | Mean-difference axes, control PCA suppression, and temporary activation hooks |
| `representations` | `fit_paired_axis`, `fit_pca_axis`, `pca_subspace`, `fit_ridge_probe` | Paired contrasts, principal directions/subspaces, and linear probes |
| `representations` | `roc_auc`, `evaluate_axis`, `AxisBank` | Held-out evaluation, cosine comparison, and axis composition |
| `interventions` | `activation_patch`, `multi_axis_steering`, `projection_ablation`, `transform_activation` | Replace activations, add directions, and remove projections |
| `sparse` | `TopKSAE`, `train_topk_sae`, `sae_features`, `sae_delta_intervention` | Train a sparse SAE and edit features while retaining reconstruction residuals |
| `memory` | `rank_one_update`, `batched_ridge_update`, `weight_edit`, `preservation_drift` | Linear edits, norm limits, transactional rollback, and collateral-change measurements |
| `streaming` | `ActivationMoments`, `forward_moments` | Online means and variances without retaining complete activation history |
| `lenses` | `logit_lens`, `coda_lens`, `TunedLens`, `fit_tuned_lens`, `decode_states`, `trace_token_ranks` | Vocabulary projections and learned intermediate readouts |
| `traces` | `LocalActivationTrace`, `capture_hf_hidden_states` | Bounded, ordered captures of local/recurrent activations |
| `causal_reasoning` | `causal_patch_effect`, `sae_feature_patch`, `component_logit_contributions` | Clean/corrupt/patched comparisons and linear component contributions |
| `interpretation` | `FeatureHypothesis`, `inspect_sae_features`, `inspect_axis`, `vocabulary_partition` | Label-associated statistics and explicit token/latent-vocabulary partitions |
| `reporting`, `circuits` | `export_trace_json`, `export_trace_html`, `recurrent_attention_summary`, `export_circuit_html` | Offline readout reports and circuit diagrams with evidence labels |

Dimensions are explicit. Steering, ablation, activation patching, and weight-edit
contexts remove hooks or restore weights when the `with` block exits, including
exceptions. `weight_edit(..., keep=True)` deliberately retains an edit.

## Causal intervention on a small network

```python
import torch
from torch import nn
from axislab_poc import fit_paired_axis, temporary_steering

baseline = torch.tensor([[-1., 0.], [0., 0.], [1., 0.]])
positive = baseline + torch.tensor([2., 0.])
axis = fit_paired_axis(positive, baseline)
network = nn.Sequential(nn.Identity(), nn.Linear(2, 1, bias=False))
with torch.no_grad():
    network[1].weight.copy_(torch.tensor([[1., 0.]]))
x = torch.zeros(1, 2)
print(network(x).item())  # 0.0
with temporary_steering(network, "0", axis, coefficient=2.0):
    print(network(x).item())  # 2.0
print(network(x).item())  # 0.0: hook removed
```

`baseline` supplies control examples with two features. `positive` shifts only
the first feature. The fit averages `positive - baseline` and normalizes the
direction. The identity layer exposes an intermediate activation, and the linear
layer reads the first feature. The context adds two units in the fitted direction;
the final evaluation checks that the original behavior returns.

## Linear memory edits and rollback

```python
import torch
from torch import nn
from axislab_poc import rank_one_update, weight_edit

layer = nn.Linear(2, 1, bias=False)
key, target = torch.tensor([1., 0.]), torch.tensor([3.])
delta = rank_one_update(layer, key, target)
with weight_edit(layer, delta, max_relative_norm=100):
    print(layer(key))  # Approximately 3.0
# Original weights are restored here.
```

For the affine layer `y = Wx + b`, the minimum Frobenius-norm rank-one update is

```text
ΔW = (y* - Wk - b) kᵀ / (kᵀk)
```

For multiple keys `K` and targets `Y`, both arranged as columns:

```text
ΔW = (Y - WK - b) (KᵀK + λI)⁻¹ Kᵀ
```

The code uses `torch.linalg.solve` instead of explicitly forming an inverse and
supports ridge regularization. These functions edit a single linear mapping.
They do not locate facts in a Transformer or reproduce full ROME/MEMIT. MEMIT
distributes edits across layers and estimates internal preservation statistics.
Measure unrelated-input drift with `preservation_drift`; edits are not isolated
automatically.

## Train a small Top-K SAE

```python
import torch
from axislab_poc import TopKSAE, train_topk_sae

x = torch.randn(200, 16)
sae = TopKSAE(d_model=16, d_sae=64, k=4)
metrics = train_topk_sae(sae, x, steps=100)
reconstruction, features = sae(x[:3])
print(reconstruction.shape, features.shape)  # [3, 16], [3, 64]
```

The encoder projects 16 inputs into 64 ReLU features. Top-K retains at most four
positive values per sample; the decoder reconstructs the input, and Adam
optimizes mean squared error. A learned feature needs labeling, controls,
evaluation, and replication before it can be assigned a semantic interpretation.
This small implementation does not replace production SAELens training.

## Local state observability in v0.3

`logit_lens` projects activations `[..., width]` through an unembedding matrix
`[vocabulary, width]` and returns top token IDs, probabilities, entropy, and target
token ranks. Supply the output normalization required by the architecture
(LayerNorm/RMSNorm) before the language head.

`coda_lens` accepts a caller-provided function that applies the model's own Coda
block/suffix before projection. The caller must provide the correct architecture;
this generic interface is not a validated reproduction of Huginn's implementation.
`TunedLens` learns an affine intermediate readout against the final-layer
distribution while leaving base weights intact. Validate it on held-out data.

`LocalActivationTrace` uses local PyTorch hooks, records repeated calls to recurrent
modules in order, enforces an entry/memory budget, and exports `.npz` data plus
JSON metadata. `capture_hf_hidden_states` requests `output_hidden_states=True` on
a locally running Hugging Face model and can capture whole layers or the final
nonpadding token. Capture only required locations on large models.

`causal_patch_effect` compares clean, corrupted, and patched executions and measures
recovery. `sae_feature_patch` transfers selected features using the same SAE for
both activations. Feature indices from separately trained SAEs need alignment
before comparison. `component_logit_contributions` projects additive components
linearly; it does not account exactly for subsequent nonlinear normalizations.

`FeatureHypothesis` keeps manual, heuristic, and validated label origins explicit.
`vocabulary_partition` distinguishes ordinary and latent vocabulary IDs in a dual
vocabulary model you control; latent IDs do not automatically have natural-language
translations. `recurrent_attention_summary` measures attention mass toward memory
positions, which is descriptive evidence rather than causal proof.

### Inspect a prediction trajectory

```python
import torch
from axislab_poc import decode_states, export_trace_html, trace_token_ranks

states = [torch.tensor([[0., 1., 0.]]),
          torch.tensor([[0.5, 0.5, 0.]]),
          torch.tensor([[1., 0., 0.]])]
unembed = 8 * torch.eye(3)
readouts = decode_states(states, unembed, target_ids=torch.tensor([0]), top_k=3)
print(trace_token_ranks(readouts))  # [[2], [1], [1]]; ties share a rank
export_trace_html("reasoning_report.html", readouts,
                  stage_labels=["t0", "t1", "t2"])
```

These states are constructed to illustrate projected distributions. Improving
target ranks do not demonstrate a verbal internal reasoning process. The toolkit
records statistical readouts of local model states; it does not reconstruct a
hidden chain-of-thought transcript.

The circuit HTML from `demo_reasoning.py` contains labeled nodes and edges plus
an interactive intensity filter. Its toy diagram describes architecture. Add
causal evidence only after interventions measure the corresponding effects.

## Optional upstream integrations

Install optional extras according to your model's requirements:

```bash
python -m pip install -e '.[nnsight,saelens,transformerlens,test]'
```

These dependencies are not installed by Neuralis's default AxisLab setup. The
original bundle tested independent algorithms and mocked adapter contracts;
external backends were not validated end to end. RepE, ROME, and MEMIT are not
extras because their version requirements may conflict with current PyTorch or
Transformers. Install their original projects in separate environments as needed.

| Upstream | AxisLab adapter | Requirements and limitations |
| --- | --- | --- |
| [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) | `transformer_lens_capture`, `transformer_lens_steer` | Local `run_with_hooks`; ordinary hooks cannot mutate remote backends |
| [nnsight / NDIF](https://github.com/ndif-team/nnsight) | `nnsight_capture`, `nnsight_steer` | Uses `trace()`; remote execution requires account access, a hosted model, and external validation |
| [SAELens](https://github.com/decoderesearch/SAELens) | `saelens_encode_decode`, `sae_delta_intervention` | SAE objects must expose `encode`/`decode`; no production SAE training |
| [RepE](https://github.com/andyzoujm/representation-engineering) | `repe_registry` | Registers official pipelines; local paired/PCA methods are independent |
| [ROME](https://github.com/kmeng01/rome) | `upstream_model_edit(algorithm="ROME")` | Upstream installation and model-specific requests/hyperparameters |
| [MEMIT](https://github.com/kmeng01/memit) | `upstream_model_edit(algorithm="MEMIT")` | Upstream installation and model-specific requests/hyperparameters |

### TransformerLens example

```python
from transformer_lens.model_bridge import TransformerBridge
from axislab_poc import transformer_lens_capture

bridge = TransformerBridge.boot_transformers("gpt2", device="cpu")
print(list(bridge.hook_dict)[:20])
# Choose an actual exposed hook name for this architecture/version:
# logits, activation = transformer_lens_capture(bridge, "Hello world", hook_name)
```

This call may download model weights. It is an optional upstream example and was
not executed during the default CPU validation. Hook names vary by architecture
and library version.

### nnsight example

```python
# Requires a separately installed nnsight and a model you can access:
# from nnsight import TransformersModel
# from axislab_poc import nnsight_capture
# model = TransformersModel("openai-community/gpt2", dispatch=True)
# activation = nnsight_capture(model, "Hello", "transformer.h.0")
```

`remote=True` needs NDIF authentication and the requested hosted model. No remote
result is claimed. `nnsight_steer` replaces a module's full output during a trace;
token selection is not supported uniformly across architectures.

### Hugging Face local example

`examples/hf_local_lens.py` inspects `openai-community/gpt2` layer by layer using
its output-head weights and final normalization. Install `transformers`
separately and run the example when downloading and executing that model is
intended. Default tests and demos do not download it; no GPT-2 benchmark is
included in the default validation.

## Research protocol and limits

1. Split data by topic/group to avoid leakage and evaluate unseen examples.
2. Compare against norm-matched random directions, unsupervised PCA, prompting,
   and LoRA baselines.
3. Test dose response, direction ablation, activation patching, and preservation
   of unrelated tasks.
4. Record model/tokenizer versions, data, hook names, seeds, hardware, and
   hyperparameters.
5. Test interventions before applying them to agents with external tools.

Reasoning interpretations require held-out predictions, causal interventions
with random controls, normalization checks, replication across architectures,
and a distinction between natural-language descriptions and actual computation.
Linear axes/probes are approximations and need not identify unique mechanisms.
There is no claim of new knowledge creation, universal reasoning improvements,
subjective consciousness/pain control, trillion-parameter benchmarks, or validated
distributed execution.

The capture APIs operate on local or explicitly controlled models. They do not
access private traces through hosted APIs, replay encrypted envelopes, automate
prompt injection/tool-call replay, or transmit private reasoning elsewhere.
Authorized API audits can use responses, metadata, provider-supplied summaries,
and authorized test outputs.

## Attribution and references

AxisLab is independently implemented from public formulas and interfaces and is
distributed under the original [MIT license](LICENSE). Optional upstream projects
retain their own authorship, licenses, usage restrictions, and citation needs.

- [Pain-axis](https://github.com/valen-research/Pain-axis)
- [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)
- [nnsight](https://github.com/ndif-team/nnsight)
- [SAELens](https://github.com/decoderesearch/SAELens)
- [Representation Engineering](https://github.com/andyzoujm/representation-engineering)
- [ROME](https://github.com/kmeng01/rome)
- [MEMIT](https://github.com/kmeng01/memit)
- [SAEs and patching on CoT/GSM8K](https://arxiv.org/abs/2507.22928)
- [Huginn, Logit Lens, and Coda Lens](https://arxiv.org/abs/2507.02199)
- [Latent CoT study/notebooks](https://github.com/jialiang19/latent-cot-thinking)
- [Recurrent trajectories and multiple lenses](https://github.com/wenquanlu/huginn-latent-cot)
- [Experimental dual-vocabulary architecture](https://github.com/LE-WH/LatentReasoning)
- [API CoT leakage-risk study](https://arxiv.org/abs/2608.20055), referenced in the
  supplied README; extraction functionality is not incorporated.
