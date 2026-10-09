# Research services

These CPU services capture real arrays, verify generated answers, and encrypt
local retrieval data. They can be used independently of the API. The optional
API exposes capabilities for integrations unavailable in a particular install.

## Trajectories and temporal debugging

`neuralis.rl_arena.trajectory_db.TrajectoryStore` persists RGB `uint8` frames,
numeric policy tensors, actions, rewards, terminal flags, JSON metadata and
optional opaque emulator snapshots. The caller chooses `backend="duckdb"` for
the optional DuckDB engine or `backend="sqlite"` for the standard-library
baseline. The API chooses DuckDB when installed and reports the actual engine.

Append-only steps start at zero and remain contiguous. Terminal episodes reject
further appends. Limits bound frame resolution, tensor elements, record bytes,
total bytes and total records. A database transaction makes validation and the
append atomic. Reads have explicit limits, and NumPy arrays load without pickle.

```python
from neuralis.rl_arena import GridWorld, TemporalDebugger, TrajectoryStore
from neuralis.rl_arena.gridworld import record_gridworld_demo

with TrajectoryStore("arena.duckdb", backend="duckdb") as store:
    outcome = record_gridworld_demo(store, episode_id="example", size=6)
    captured = store.get(outcome["episode_id"], 0)
    print(captured.public())
    emulator = GridWorld(6)
    debugger = TemporalDebugger(store, restore=emulator.restore,
                                advance=emulator.advance)
    branch = debugger.replay("example", 0, actions=[2, 2, 1])
    print(branch["outputs"])
```

The deterministic gridworld is a runnable environment with a Manhattan-distance
greedy policy; it is not a Mario emulator or trained RL policy. Its `seed` is
reported for API consistency, and `stochastic=False` makes clear that it does
not change this deterministic baseline. A step budget can truncate a capture.
Frames and snapshots describe the observation **before** the stored action;
the reward and terminal flag describe that action's actual result.

`TemporalDebugger.seek` and `timeline` play back captured observations. State
replay requires an actual snapshot plus caller-supplied `restore` and `advance`
callbacks. Without these it raises a capability error. `end_step` is exclusive:
replaying from zero to six executes the six actions stored at steps zero to
five. Branch actions do not mutate the original trajectory. Restoring emulator
state must include its random generator and wrapper state where relevant;
captured RGB frames alone cannot recreate an emulator.

`cot_engine.policy_diagnostics` calculates softmax probabilities, entropy,
chosen-action probability and margin from supplied observable logits. An
optional public rationale is stored exactly as caller-provided. The module
does not infer or expose private chain-of-thought.

## Actual optional WebRTC

`webrtc_streamer.FrameQueue` holds a bounded number of RGB frames and drops the
oldest when full. Use it from one asyncio event loop. `WebRTCStreamer.answer`
accepts a genuine SDP offer and returns an aiortc SDP answer. Its real video
track converts queued pixels to AV frames. A media relay serves a bounded peer
count without an unbounded per-viewer buffer. Close streams and peers on shutdown.

Install optional `aiortc` and `av` for transport. The local integration test
negotiates two real peers and verifies decoded frame dimensions and pixels.
This checks CPU transport, not GPU encoding, WAN latency, or NAT traversal.
The adapter accepts an explicit aiortc configuration for STUN/TURN; the caller
must authenticate signaling and configure networking. No latency guarantee is
reported. PNG capture export uses actual pixels with a standard-library encoder.

The default configuration explicitly uses an empty ICE server list, so local
streaming does not contact a public STUN service. Browser ICE gathering must
produce a candidate before signaling begins. A browser that gathers no
candidates receives a clear transport error instead of waiting for video. In
this cloud environment, managed Chromium's `disable_non_proxied_udp` policy
blocks its host-candidate gathering. Real aiortc-to-aiortc transport is tested;
browser video receipt remains blocked by that policy. The browser validation
script reports `status: "blocked"` and `browser_video_verified: false` in that
case. The managed policy is preserved.

## Teacher, student, judge and rejection sampling

`synthetic.Foundry` accepts three callables: `teacher(prompt) -> str`,
`student(prompt) -> str`, and `judge(prompt, teacher, student) -> JudgeResult`.
The judge returns a public score in `[0, 1]`, an acceptance decision and a public
reason. Samples are retained only when the decision is true and the threshold
passes. Attempt count, requested count, completion state, acceptance rate and
mean score derive from the actual sampled outcomes. Bounds limit attempts,
text size and exported records. Caller-owned remote providers must configure
their own timeouts, credentials and model resources.

```python
from neuralis.synthetic import run_math_demo, export_jsonl

result = run_math_demo(count=24, seed=42)
print(result["statistics"])
export_jsonl(result["records"], "approved-training.jsonl")
```

The demo creates integer addition, subtraction and multiplication tasks. Its
exact arithmetic teacher supplies reference answers, the demo student perturbs
every fourth answer, and the judge independently computes the expected answer.
Rejected examples are observable separately. This is a reproducible filtering
demo, not evidence of a trained LLM. The exported accepted JSONL includes
user/assistant messages, provider labels and judgments. No hidden reasoning is
requested. `llm_as_a_judge.py` and `rejection_sampling.py` expose the public
interfaces; custom LLM implementations plug into the same callables.

## Encrypted local retrieval

`rag.LocalRAG` requires an explicit Fernet key as bytes and the `cryptography`
package. It has no plaintext fallback. The API supplies a generated key stored
in a private local file with owner-only permissions. Library callers control
key persistence; losing it prevents decryption. Back up a key separately from
the collection using your existing secure storage process. The key is not
included in status responses or serialized into the index.

```python
from pathlib import Path
from neuralis.rag import LocalRAG

# Provision this private file securely before opening the collection.
key = Path("/private/location/rag.key").read_bytes()
with LocalRAG("collection.sqlite", key=key) as rag:
    indexed = rag.add_document("Jupiter has many observed moons.",
                               source="Astronomy notes")
    response = rag.search("Jupiter moons", limit=3)
    print(response["matches"])
```

Document content, source, metadata, character ranges and normalized vectors
are all authenticated and encrypted before being written, including SQLite WAL
data. The encrypted index descriptor verifies the key and embedding adapter on
reopen. IDs, ciphertext lengths and counts remain visible. Search decrypts the
bounded collection in memory, computes normalized cosine scores and returns
plaintext matches to the authorized caller. Encryption at rest does not protect
against a compromised running process, and document deletion does not promise
physical secure erasure of historical database pages.

The default embedding is deterministic signed BLAKE2 hashing of words and
adjacent word pairs. It is an honest local lexical baseline. The optional
`SentenceTransformerEmbedder` loads an existing local model on CPU by default;
an explicit `allow_download=True` enables normal library model acquisition.
Changing adapter signatures requires rebuilding a separate collection.

`chunk_text` creates overlapping character ranges with preferred word
boundaries. `read_pdf` optionally uses `pypdf`, limits input bytes, page count
and extracted text, and rejects password-protected or textless PDFs. OCR is not
configured. `retrieve_and_answer` accepts a caller-provided generator; search
does not manufacture a model-generated answer.

## GGUF and Ollama

`export.GGUFConverter` invokes an explicitly supplied trusted local llama.cpp
`convert_hf_to_gguf.py` script with a Python executable, argument list and
timeout. It does not use a shell, download weights, overwrite an existing output,
or claim Neuralis toy classifiers match a supported LLM architecture. Supply a
real Hugging Face directory containing a declared architecture and local
weights, and the converter must actually support that architecture.

```python
from neuralis.export import GGUFConverter, write_ollama_modelfile

converter = GGUFConverter("/trusted/llama.cpp/convert_hf_to_gguf.py",
                          timeout_seconds=300)
converted = converter.convert("/models/compatible-hf-model",
                              "/models/converted.gguf", outtype="f16")
recipe = write_ollama_modelfile(converted["path"], "/models/Modelfile")
```

Conversion stages output in a temporary directory, drains logs with only a
bounded retained tail, checks completion and validates the resulting container
before publishing the destination. The validator checks GGUF version, required
architecture metadata, alignment, tensor dimensions/types, nonoverlapping
offsets and actual data bounds. Unsupported newer GGML types fail explicitly.
This validates container structure; only the target runtime can establish model
architecture compatibility and inference quality.

`write_ollama_modelfile` references an existing validated `.gguf` file and emits
bounded temperature/context settings. It does not import or start Ollama.
Run `ollama create <name> -f <Modelfile>` in an installed local Ollama runtime
to check real compatibility. `gguf_quantizer.py` and `ollama_exporter.py` expose
these public adapters. A supplied converter's `q8_0` output is supported;
additional quantization requires the appropriate external tool.

## Validation

Run the service tests in the API environment:

```bash
PYTHONPATH=services/api/src python -m pytest services/api/tests/test_research_services.py -q
```

The tests exercise actual SQLite and DuckDB persistence/reopen, atomic budget
failures, gridworld snapshot restoration and branches, PNG pixels, numerical
policy diagnostics, verified arithmetic filtering, encrypted search/reopen and
wrong-key rejection, chunk coverage, GGUF parsing and external subprocess
contracts. The optional WebRTC test performs local signaling and actual decoded
video receipt. GGUF subprocess fixtures validate the adapter contract; they do
not claim an external llama.cpp model conversion or Ollama inference was tested.
