# Research extensions

The English browser and desktop interface shares the same research routes.
GitHub Pages runs the browser computations; installed Python integrations require
a connected local API. Capability checks distinguish installed adapters from
validated model/hardware support.

## Install and start

From the repository root with Python 3.12 or 3.13 and uv:

```bash
uv sync --project services/api --frozen --extra dev --extra research --extra streaming --extra lan
uv run --project services/api python -m neuralis
npm ci
npm run dev
```

Open **Research workbench**, choose **Connect backend**, enter the API URL and
check capabilities. Tokens stay in the page session. Loopback development uses
the existing single-user authentication; network-facing APIs require configured
authentication and explicit CORS origins.

In the Tauri app, choose **Use desktop runtime** to route requests through the
Rust IPC bridge. The ephemeral runtime credential stays in Rust. Graph surgery
also supports the desktop bridge; the browser URL/token controls apply to a
direct browser connection.

## What each module executes

| Module | Browser | Local Python |
| --- | --- | --- |
| AxisLab | Paired contrast directions, held-out AUC and numerical steering | Original AxisLab 0.3.0, exact imported source and license; `/extensions/axis/evaluate` |
| Efficient training | Packed NF4 reconstruction and analytical memory planner | Frozen FP32/NF4 LoRA, actual loss/accuracy, checkpoint/resume and supervisor controls |
| Interpretability | Actual captured tensors and Transformer residual readout | Dynamic cache, TopK SAE, reversible feature steering, clean/corrupt patching and AxisLab lens bridge |
| RL Arena | Real tabular Q-learning, trajectory import/export and scrubbing | DuckDB/SQLite observations, actual NES recorder, WebRTC, stored frame inspection and gridworld state branching |
| Data Foundry | Seeded arithmetic generation, independent validation and JSONL split | Caller-controlled teacher/student/judge, rejection sampling, verified arithmetic demo |
| GPU observatory | Real WGSL compute and instanced points, CPU canvas fallback; PCA and measured attention splines | Captured vectors may be supplied by local instrumentation |
| Graph surgery | Dense-chain editing and tensor-shape validation | Real bounded DAG compilation with residual add, concat, nonlinearities and layer normalization |
| Export | Modelfile and command recipe export | Validated external llama.cpp conversion adapter and GGUF structure inspection |
| Local RAG | Chunking, hashed-token cosine retrieval and optional browser persistence | Fernet-encrypted documents, metadata and vector index with persistent local key |

## Train LoRA with ordinary run controls

Create a registered run using the API or Python client:

```python
from neuralis.core.schemas import RunConfig, CommandRequest
from neuralis.runtime.supervisor import Supervisor
from neuralis.storage.repository import Repository
from pathlib import Path

supervisor = Supervisor(Repository(Path("/tmp/neuralis-lora-example")))
try:
    run = supervisor.create(RunConfig(
        model_id="mlp-xor", steps=50,
        adapter={"enabled": True, "rank": 2, "alpha": 4, "quantize_base": True},
    ))
    supervisor.command(run["id"], CommandRequest(command="step"))
    checkpoint = supervisor.save_checkpoint(supervisor.get_live(run["id"]))
    resumed = supervisor.restore(run["id"], checkpoint["id"])
    print(resumed["step"])
finally:
    supervisor.shutdown()
```

Base tensors remain frozen. The optimizer contains trainable adapter parameters
only. The reference NF4 layer actually packs two codes per byte but reconstructs
floating weights for execution; it is not a fused low-memory accelerator kernel.
For large local compatible models, `load_hf_qlora` is an explicit optional
Transformers/PEFT/bitsandbytes adapter with local-only loading by default.

## Record and inspect the real Mario environment

Install the separate environment described in `demos/super-mario/README.md`, train
or supply a compatible PPO/DDQN checkpoint, and run:

```bash
cd demos/super-mario
python arena.py runs/smoke/latest.pt --url http://127.0.0.1:8000 --max-steps 100 --fps 15
```

Open **Research workbench → RL Arena → Temporal debugger**, load episodes and seek
a frame. Stored PNG pixels, action values, raw rewards and recorder metadata come
from actual execution. Live video negotiates a WebRTC session with the API. To
watch a continuing episode, start its recorder after connecting the video client;
the recorder can use an explicit unique `--episode-id`.

PPO records action probabilities; DDQN records raw Q values. Current NES records
contain post-action pixels and clearly labeled pre-action policy diagnostics.
They do not contain emulator state snapshots. The gridworld adapter captures
pre-action snapshots and supports real branching from them. Pixel-only Mario
records cannot support state restoration or counterfactual emulator replay.

## Compile residual graphs

Open **Graph surgery → Local backend graph compiler** and use the residual preset.
The declarative compiler accepts a maximum of 32 nodes and two million parameters;
nodes must contribute to the output and be reachable from the input. It rejects
cycles, unknown nodes, unsupported operations and dimension mismatches before
allocating model parameters.

`neuralis.models.graph_surgery.compile_graph` returns an actual autograd-capable
`TensorGraph`. Its `recompile` method preserves matching named weights explicitly
and returns a change report. Structural edits require a new optimizer. The HTTP
evaluate action creates fresh seeded weights and does not silently modify an
existing trained run.

## Encrypted retrieval and exports

The API creates a private `0600` key at `NEURALIS_DATA/extensions/rag.key` and
persists encrypted collections in that directory. Back up the key separately
with appropriate access controls: losing it makes the encrypted corpus unreadable.
This protects file contents, not a logged-in user or a compromised running
process. Browser persistence remains unencrypted and the interface identifies it.

Baseline retrieval uses deterministic hashed lexical embeddings. Optional local
neural embedders and PDF readers require explicit installation and model files;
the baseline does not claim semantic embedding quality. Retrieved passages remain
source-linked evidence, not automatically generated answers.

GGUF export requires an actual compatible Hugging Face model and a trusted
llama.cpp conversion tool. Registered toy models are not Ollama language models.
The browser exports inert recipes; the Python converter checks declared budgets,
process timeout, output structure and existing destination files. Converter
execution and Ollama inference were not tested with large third-party weights.

## Validation and limits

The suites exercise numerical references, held-out dictionary reconstruction,
hook restoration, adapter freezing and exact resume, graph gradients, encrypted
retrieval/reopen, real DuckDB persistence and local WebRTC decoded-frame receipt.
AxisLab's original tests and safe-import tests run independently.

The managed cloud Chromium has `WebRtcIPHandling=disable_non_proxied_udp`.
It gathers zero local ICE candidates, so it cannot receive host-only WebRTC
video here. The interface reports that condition immediately. Actual encoded
frames were decoded by local aiortc peers, and real Mario observations were
persisted and inspected in the browser; these are separate checks. A permitted
browser/network or explicitly configured TURN infrastructure is required for
browser video transport. Screenshot 026 records the restriction, not a decoded
video claim. The environment policy was preserved.

```bash
uv run --project services/api pytest services/api/tests packages/axislab/tests
npm run typecheck
npm test
npm run build
python3 scripts/count-code.py --output docs/code-count.json
python3 scripts/compile-code-bundle.py
```

The WebGPU renderer bounds points to 8,192 and attention curves to 512. It does
not claim billion-parameter point rendering or hardware frame-rate guarantees.
SAE features are numerical directions; user-authored labels are hypotheses.
Policy diagnostics and supplied public explanations do not expose hidden thoughts.
CUDA, ROCm, MPS, real multicast discovery and multi-machine execution require
validation on those systems. The LAN protocol authenticates data but requires a
trusted LAN or separately configured encrypted tunnel for confidentiality.

See [AxisLab](axislab.md), [efficient runtime](efficient-runtime.md),
[interpretability](interpretability-extension.md) and
[research services](research-services.md) for contracts and examples.
