# Local interpretability extension

The Python extension measures actual tensors and logits in a caller-supplied
PyTorch model. It performs local dictionary training and temporary interventions.
It does not download models, infer human-readable feature semantics, expose
hidden thoughts, or prove that a measured direction represents a concept.
The bundled `TinyTransformer` is an intentionally small character-level model;
its projections and generated text are demonstrations of its actual parameters.

Run its numerical and lifecycle tests from the repository root:

```bash
PYTHONPATH=services/api/src python -m pytest services/api/tests/test_interpretability_extension.py -q
```

Use the configured API virtual environment's Python when dependencies are not
installed in the system interpreter.

## Capture selected activations

```python
import torch
from neuralis.models.registry import TinyTransformer
from neuralis.interpretability.activation_cache import ActivationCache

model = TinyTransformer(width=16, depth=1, heads=2)
tokens = torch.tensor([[1, 2, 3]])
with ActivationCache(model, ["blocks.0"], positions=[-1], max_bytes=4096) as cache:
    logits = model(tokens)
residual = cache.latest("blocks.0")
```

Hooks attach on context entry or `start()` and always detach on context exit,
including exceptions. `close()` retains records for inspection; `clear()` releases
them. Snapshots are detached, independent CPU copies. Repeated invocations create
separate records. `set_modules(names)` changes selected hooks while the context is
active, retaining prior records. Invalid selection leaves existing hooks intact.

Capture never samples or truncates a retained tensor. If an entire selected
tensor exceeds the remaining byte/record budget, it is skipped and `dropped`
increases. Inspect `complete` before treating a capture as exhaustive. The byte
budget counts raw retained tensor storage; Python/JSON serialization has additional
overhead. Defaults are 8 MiB and 64 records, with absolute limits of 64 MiB and
1024 records. Module names are explicit; no whole-model default capture occurs.
Positions select the penultimate axis of `[batch,sequence,channel]` output;
negative indices address positions from the end. Select residual modules when
using this convention, rather than treating attention matrices as residuals.
Tensor and tuple-first-tensor module outputs are supported.

## Train a TopK dictionary

```python
from neuralis.interpretability.sae.dictionary_learning import fit_dictionary

# Collect multiple examples first: this requires at least two observations.
observations = torch.randn(256, 16)
dictionary, report = fit_dictionary(
    observations, features=32, k=4, steps=100, seed=42,
    user_labels={3: "Researcher's candidate label"},
)
direction = dictionary.direction(3)
```

The trainable encoder keeps at most `k` positive activations per observation;
the decoder reconstructs the original activation units around a training-set
mean. Decoder columns are unit normalized after optimizer updates. Training uses
a deterministic shuffled 80/20 split and reports initial and final held-out MSE,
variance explained, activation frequency, mean active-feature count, and dead
features. Metrics can reveal a poor dictionary; improvement is not guaranteed.
Constant held-out variance yields `variance_explained=None`.

Input data must be finite `[observations,channels]` tensors within explicit data,
parameter, step, batch, and CPU-work budgets. Training takes place on CPU. RNG
initialization and sampling preserve the caller's global torch RNG state.
Features are identified as `Feature 0`, `Feature 1`, and so on. Only the caller's
`user_labels` supply human-readable labels; `label_source="user"` preserves their
provenance. Such labels remain hypotheses requiring independent evaluation.

## Reversible steering

```python
from neuralis.interpretability.interventions.steering_vectors import SteeringSession

with SteeringSession(
    model, "blocks.0", vectors={3: direction}, strengths={3: 0.5}, positions=[-1],
):
    steered_logits = model(tokens)
baseline_logits = model(tokens)
```

Every selected position receives the sum of `strength * direction` for the
supplied features. Omit positions to affect all positions. Directions are fixed
independent copies; parameters and original output tensors are not modified.
`update_strengths()` replaces the active strength mapping, supporting interactive
controls without registering additional hooks. Strengths are bounded to
`[-100,100]` and direction widths must match the selected output. Exceptions
remove the intervention hook. Compare resulting probabilities and behavior with
a baseline instead of assuming the intervention expresses the user's label.

## Activation patching

```python
from neuralis.interpretability.circuit_discovery.activation_patching import run_patching_experiment

corrupt = torch.tensor([[3, 2, 1]])
report = run_patching_experiment(
    model, tokens, corrupt, "blocks.0",
    metric=lambda logits: logits[0, -1, 1] - logits[0, -1, 2],
    positions=[-1],
)
```

The experiment computes clean and corrupt scores, substitutes clean activations
at the specified corrupt positions, computes the patched score, removes the
hook, and computes a restored score. Its metric must return one finite scalar.
The recovery ratio is `(patched-corrupt)/(clean-corrupt)`; it is deliberately not
clamped and becomes `None` when the clean/corrupt difference is negligible. A
negative ratio or a ratio above one can be a real outcome. This is a causal
sensitivity measurement for one supplied site, not automatic circuit discovery.

`ActivationPatcher` also accepts `source_positions` to map clean positions to
different corrupt positions. Full replacement requires identical shapes;
position replacement requires matching batch and channel shapes. The experiment
helper requires one invocation of its selected site; use distinct wrapper sites
for reused modules. Evaluation mode and each submodule's original training flag
are restored, along with the hooks.

## Token inference, generation, and logit lens

```python
from neuralis.interpretability.lenses import infer_tokens, generate_tokens, logit_lens

measured = infer_tokens(model, tokens, modules=["blocks.0"])
with ActivationCache(model, ["blocks.0"]) as cache:
    model(tokens)
projection = logit_lens(model, cache.latest("blocks.0"), top_k=5)
generated = generate_tokens(
    model, [1, 2, 3], max_new_tokens=8,
    modules=["blocks.0"], temperature=0.8, seed=42,
)
```

Inference returns real CPU logits, softmax probabilities, and selected snapshots.
The lens applies the model's final normalization and output head to a residual;
for the final TinyTransformer block it exactly reproduces the model's readout.
At intermediate layers it is an exploratory projection, not a prediction made
by that layer. `normalized=False` permits a head-only projection. Models with
`get_output_embeddings()` can supply a head, but normalized projection requires
an explicit `model.norm`; arbitrary Hugging Face architectures need a deliberate
adapter, and no weights or packages are downloaded by this extension.

The optional `axislab_logit_lens()` bridge calls the extracted AxisLab public
projection implementation and returns its measured top probabilities, entropy,
and optional target-token ranks. Add `packages/axislab/src` to `PYTHONPATH` or
install that local package to use the bridge. The required API workflow works
without it; its test is explicitly skipped when AxisLab is unavailable.

Generation accepts token IDs and a supplied model, limits context and vocabulary,
and performs actual autoregressive forward passes. Each step records context
IDs, the generated token, the base model distribution, the sampling distribution,
entropy, and selected last-position activations from the context that produced
that token. Temperature zero uses greedy selection with sampling probability
one, while retaining the unmodified model probability separately. Positive
temperature supports TopK and nucleus filtering with a local seeded generator.
The capture budget is shared across all generated steps; subsequent activations
are honestly marked dropped when exhausted. The output vocabulary is bounded to
4096 entries, context to 512 tokens or the supplied model's smaller capacity, and
generation to 256 tokens. A supplied EOS ID can terminate generation earlier.

Inference and interventions operate on a mutable model. Callers must serialize
access to each model instance: hooks and training flags are process-local and
should not be shared between concurrent requests without a model lock. No
operation makes claims about natural-language reasoning or private internal
thoughts; all returned measurements concern the caller's local model tensors.
