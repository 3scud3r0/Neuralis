# AxisLab in Neuralis

The supplied AxisLab Research 0.3.0 toolkit is available as the editable Python
package `axislab-poc`, imported with `import axislab_poc`. It provides real CPU
implementations of representation axes, sparse features, activation interventions,
linear memory edits, and bounded local-state observability. Its detailed English
workflow is in [the package README](../packages/axislab/README.md).

## Source integrity

The user supplied `AxisLab_0.3.0_CODIGO_COMPLETO.txt`, a consolidated UTF-8 bundle
with 28 `BEGIN FILE`/`END FILE` entries. The bundle's SHA256 is
`c7b00aa7a0891a3db0ce1a8b5bc09380e78faa32a751b60cc9ba51a2584d8bef`.
Every entry was reconstructed from its declared byte length and validated against
its supplied SHA256 before writing any files.

All original source, tests, reports, and the MIT license remain byte-exact. The
Portuguese README is preserved as `packages/axislab/README.original.md`; the new
English README is separate. [The provenance manifest](../packages/axislab/provenance.json)
maps all original paths to their imported destinations and records original byte
lengths and hashes. No original code corrections were needed for the verified
workflow. New import-safety regression tests are additional files.

The original bundle explicitly excluded binary `.npz` data. Existing HTML/JSON
reports are supplied reference artifacts. The reasoning demo generates fresh
binary and report files in a temporary run directory; it does not overwrite those
references.

## Repeat the import safely

From the repository root:

```bash
python scripts/import-axislab.py /path/to/AxisLab_0.3.0_CODIGO_COMPLETO.txt --check
python scripts/import-axislab.py /path/to/AxisLab_0.3.0_CODIGO_COMPLETO.txt /tmp/axislab-clean-import
```

The importer rejects unsafe/absolute/traversing paths, duplicate paths,
file/directory collisions, malformed metadata and markers, truncated payloads,
incorrect checksums or lengths, wrong file counts, and unexpected trailing data.
It consumes each payload by byte count so embedded marker-like text is preserved.
It validates the complete input before creating staged output and refuses existing
destination directories or symlinks. Import into a new directory to compare a
bundle without replacing local files. The importer preserves the original README
under its archival name and writes `provenance.json`; the English documentation
is maintained separately in the repository.

## Installation and validation

The prepared Neuralis API environment uses Python 3.12.14, PyTorch 2.8.0 CPU,
and pytest 8.4.1. AxisLab was installed into that environment without resolving
or upgrading its other dependencies:

```bash
cd /workspace/Neuralis
UV_CACHE_DIR=/workspace/.cache/uv uv pip install \
  --python /workspace/.venvs/neuralis-api/bin/python \
  --no-deps --no-build-isolation -e packages/axislab
/workspace/.venvs/neuralis-api/bin/python -m pytest -q packages/axislab/tests
```

Validation on 2026-10-09 passed **69 tests**: all **48 supplied numerical and
adapter-contract tests**, plus **21 import-integrity/safety cases**. The supplied
source hashes were rechecked after installation and demonstration runs; all 28
still matched. The package exports 58 public names under `axislab_poc.__all__`.

All three original CPU demonstrations ran successfully from
`/tmp/axislab-run.TMgXCO`, using the installed package and no model downloads:

| Demonstration | Actual result |
| --- | --- |
| `demo.py` | One control PCA component removed; output `0.0 → 1.9999 → 0.0` across steering and rollback |
| `demo_advanced.py` | Held-out AUC `0.9919`; steering `0.0 → 2.0 → 0.0`; final SAE reconstruction MSE `0.1263`; feature intervention changed the activation |
| `demo_advanced.py` memory/streaming | Linear target reached; original weights restored; four batch observations accumulated |
| `demo_reasoning.py` | Five recurrences captured, zero dropped; target ranks `[2, 1, 1, 1, 1]`; top token IDs `[1, 0, 0, 0, 0]` |

The new reasoning run produced `local_states.npz`, `local_states.json`,
`projections.json`, `report.html`, and `circuit.html`. The `.npz` archive was opened
with `allow_pickle=False`; it contains five finite arrays of shape `[1, 3]`.
Its final captured state was approximately `[0.438905, 0.100657, -0.1]`.
These are synthetic local-network results, not language-model benchmark results.
Temporary run directories are disposable; use the package README commands to
generate a new report.

## Integration boundaries

Importing AxisLab does not activate optional upstream dependencies. The local
algorithms and mocked adapter contracts were tested; external TransformerLens,
nnsight/NDIF, SAELens, RepE, ROME/MEMIT, and the downloading GPT-2 example were
not tested end to end. Use separate model-compatible environments for optional
integrations and preserve their upstream licenses.

Activation hooks require a running Python process and a model it controls.
GitHub Pages can host documentation and browser demonstrations; it cannot execute
this Python package. An optional Neuralis Python backend can use the toolkit's
local functions. Token projections, attention summaries, and toy circuit diagrams
are statistical or architectural observables. Causal claims require controlled
interventions and held-out validation.
