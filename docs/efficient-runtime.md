# Efficient runtime and LAN stages

Neuralis includes CPU-tested adapter training, packed NF4 storage, explicit
offloading, and authenticated TCP pipeline stages. These are callable Python
utilities under `neuralis.runtime` and `neuralis.cluster`. The run supervisor can
use the same LoRA layers through its adapter configuration.

## Hardware dispatch

```python
from neuralis.runtime.acceleration import backend_capabilities, resolve_backend

print(backend_capabilities())
backend = resolve_backend("cpu")  # auto, cuda:0, rocm:0, mps are explicit choices
model = model.to(backend.device)
with backend.autocast():
    outputs = model(inputs.to(backend.device))
backend.synchronize()
```

CUDA availability comes from PyTorch's installed runtime. ROCm is identified by
`torch.version.hip`; it uses PyTorch's `cuda` device namespace. A CPU installation
does not advertise either accelerator. MPS requires an available Apple Metal
runtime. Invalid device indices, unavailable backends and unsupported precision
choices raise errors rather than falling back silently. `auto` prefers CUDA,
ROCm, MPS, then CPU. Precision defaults to float32. Autocast support does not
guarantee that every model operation supports that precision on every backend.

| Component | Verified in this environment | Additional requirements |
| --- | --- | --- |
| CPU dispatch and numerical execution | Yes | Installed PyTorch |
| CUDA / MPS / ROCm dispatch paths | Availability detection only | Compatible hardware/runtime; hardware validation still needed |
| Portable NF4 and LoRA | Yes | PyTorch |
| Fused bitsandbytes linear | No | Optional bitsandbytes and compatible CUDA |
| HF/PEFT language-model loader | No | Optional transformers, peft, bitsandbytes and compatible CUDA |
| TCP stage execution | Two independent loopback TCP servers | Explicitly configured hosts/ports and shared secret for LAN use |
| mDNS advertisements/browser | Parsing and configuration controls only | Optional zeroconf; multicast-capable LAN |
| Accelerator memory offloading | CPU hook/autograd behavior only | Hardware validation still needed |

CPU allocator usage is reported as unavailable by the backend memory helper.
CUDA memory metrics use PyTorch allocator counters; MPS reports PyTorch/driver
allocated memory. No estimated GPU speedup or memory savings is presented as a
measurement.

## Packed NF4

```python
import torch
from neuralis.runtime.quantization import quantize_nf4, dequantize_nf4

weights = torch.randn(128, 64)
packed = quantize_nf4(weights, block_size=64)
restored = dequantize_nf4(packed)
print(packed.storage_bytes, packed.compression_ratio)
```

Each block stores its absolute maximum in float32. Every normalized weight maps
to the closest value in the 16-entry NF4 codebook. Two indices occupy one uint8
byte. Partial blocks and odd element counts are supported; the unused final
nibble is zero's code. Zero blocks reconstruct exactly. NaN/infinite inputs,
invalid block sizes and tensors above 100 million elements are rejected. Chunked
conversion avoids allocating an elements-by-16 distance matrix.

For a full 64-element block from float32 weights, storage is 32 packed bytes plus
4 scale bytes instead of 256 bytes, before object and module metadata. This is a
storage ratio, not measured peak training-memory reduction. Scales remain float32
when modules change floating compute dtype. Quantization is lossy and does not
implement double quantization.

`NF4Linear` stores packed buffers in its normal `state_dict`; its bias is frozen.
It reconstructs the floating weight on each forward and calls ordinary
`torch.nn.functional.linear`. That preserves gradients with respect to inputs
while keeping base weights frozen. It also materializes a full floating matrix
for execution and backward, so this portable implementation is a reference
path, not a fused four-bit GEMM. `bitsandbytes_linear` is an explicitly optional
CUDA adapter for the optimized library's `Linear4bit`.

## LoRA and checkpoints

```python
import torch
from torch import nn
from neuralis.runtime.training import LoRAConfig, QLoRAEngine

torch.manual_seed(17)
model = nn.Sequential(nn.Linear(4, 12), nn.Tanh(), nn.Linear(12, 2))
engine = QLoRAEngine(model, LoRAConfig(rank=4, alpha=8),
                     quantize_base=True, learning_rate=0.02)
inputs = torch.randn(32, 4)
targets = (inputs[:, 0] > 0).long()
for _ in range(30):
    metrics = engine.train_batch(inputs, targets)
print(engine.evaluate_batch(inputs, targets))
engine.save_checkpoint("/tmp/neuralis-adapters.pt")
```

The adaptation is `base(x) + (alpha / rank) * B(A(x))`. `B` initializes to zero,
so an unquantized adapter initially reproduces the base exactly. The quantized
variant initially reproduces the reconstructed NF4 base. Only `lora_A` and
`lora_B` train. The engine leaves base BatchNorm buffers and base dropout in
evaluation mode; adapter dropout remains configurable and defaults to zero.

`inject_lora(model, config, quantize_base=False, seed=17)` can be used without the
engine. It mutates targeted linear children in place and returns their paths.
Empty targets select every existing `nn.Linear`; names, suffixes and shell-style
glob patterns are accepted. Every supplied target must match. Overcomplete
ranks are supported, including rank four on a two-output classifier. Injection
rejects already-adapted models and tied/shared selected parameters. Wrap a root
`nn.Linear` in `nn.Sequential` before injection. Explicit seeded injection uses a
local CPU/CUDA generator and preserves global torch RNG state; perform injection
on CPU before moving to MPS.

Limits are 256 selected layers, 20 million total adapter parameters, 10 million
adapter parameters per layer, and 100 million portable quantized base elements.
Training input and target tensors each have a 16-million-element budget. These
limits do not promise that any particular large architecture fits available RAM.

`train_batch` reports computed loss, microbatch/optimizer counts and gradient
norm. Accumulation scales each microbatch loss and steps after the configured
number of batches. Save/load requires an optimizer boundary. Checkpoints are
atomically written local files containing adapter tensors, optimizer state,
configuration, counters and torch RNG state. They are loaded with
`weights_only=True` and a 512 MiB file budget. Use trusted local snapshots rather
than accepting arbitrary checkpoint files over a network.

To resume, rebuild the **same frozen base**, construct an engine with the same
adapter configuration and accumulation count, and call `load_checkpoint(path)`.
A SHA-256 fingerprint of frozen tensors prevents silently loading adapters onto
different base weights. The base is not copied into these adapter-only snapshots.
Torch RNG restoration is global; this engine is intended for synchronous,
single-owner execution. The API supervisor uses its own full-run checkpoint
storage and control locks for concurrent jobs.

`LoRALinear.merged_linear()` exports a floating layer with the adapter update
added to its frozen weight. Quantized bases remain subject to NF4 reconstruction
error. `load_hf_qlora` is an opt-in HF/PEFT integration, defaults to
`local_files_only=True`, and always uses `trust_remote_code=False`. It never
downloads model weights unless explicitly called with local-only loading disabled.
The portable engine and the PEFT loader have distinct checkpoint contracts.
The loader returns a PEFT model and tokenizer for the Hugging Face training
workflow; it is not passed into the portable `QLoRAEngine`.

## Offloading

```python
from neuralis.cluster.memory_offloading import selective_cpu_offload, activation_offload

# The rest of the inference model and its inputs should use the compute device.
with selective_cpu_offload(model, ["0", "2"], compute_device="cpu") as offload:
    with torch.inference_mode():
        output = model(inputs)
    print(offload.stats)

# For training, offload saved accelerator activations, not live parameters.
with activation_offload(min_bytes=4096, max_bytes=256 * 1024 * 1024) as stats:
    loss = model(inputs).sum()
loss.backward()
```

Selective weight offload loads nonoverlapping named modules before forward,
evicts their parameters/buffers to CPU after forward, and restores their original
devices when closed. It requires inference mode/no-grad because moving weights
needed by backward would invalidate autograd. Selected modules cannot share
tensors outside their subtrees. Calls to the same model must be serialized.

Activation offload uses PyTorch saved-tensor hooks and preserves autograd. CPU
tensors stay in place. Accelerator tensors move to CPU only within the context's
byte budget; overflow stays on its original device. This bounds additional host
offload rather than promising a hard bound on all model memory. Restores and
transferred bytes are measured by actual hook activity.

## LAN split training

```python
import secrets
from neuralis.cluster.distributed import LANPipeline, PipelineClient, PipelineServer, PipelineStage

secret = secrets.token_bytes(32)  # Provision the same secret securely on each host.
part0 = nn.Sequential(nn.Linear(4, 8), nn.Tanh())
part1 = nn.Sequential(nn.Linear(8, 2))
stage0 = PipelineStage(part0, torch.optim.SGD(part0.parameters(), lr=0.01))
stage1 = PipelineStage(part1, torch.optim.SGD(part1.parameters(), lr=0.01))
with PipelineServer(stage0, secret) as server0, PipelineServer(stage1, secret) as server1:
    with PipelineClient(*server0.address, secret) as client0, PipelineClient(*server1.address, secret) as client1:
        pipeline = LANPipeline([client0, client1])
        result = pipeline.train_batch(inputs, targets, nn.functional.cross_entropy)
        print(result)
```

On separate machines, each process constructs its assigned stage and starts a
server bound to its explicitly selected LAN interface. The coordinator opens
clients to those hosts/ports. Models are locally constructed; no remote code or
model pickle is accepted. Each stage accepts/returns one tensor. Forward sends
detached activations and retains a bounded local autograd graph behind a ticket.
Backward passes gradients in reverse stage order, accumulates local parameter
gradients and returns input gradients. Optimizers step only after all queued
graphs are consumed. Direct client calls expose forward, backward, step, abort
and status for custom microbatch schedules.

The default limits are 16 MiB per frame, eight connections, four pending graphs,
64 MiB of retained input/output activation tensors and a 60-second training lease.
Intermediate tensors inside a locally chosen stage can exceed the input/output
budget; choose stage architecture and batch size to fit its machine. Frame and
socket deadlines prevent unbounded reads. Disconnects, expired leases and failed
requests discard pending work. One training connection leases a stage at a time.
Parameter-free stages can propagate gradients without an optimizer. Unused
parameter gradients are supported; interrupted differentiable paths between
stages are errors.

Challenge-response HMAC authenticates both peers. Every frame is authenticated
and carries a monotonically increasing connection sequence, preventing replay
and reordering within or across sessions. Lengths, tensor shapes, dtypes and
finite values are validated before tensor construction. The protocol never
unpickles network data. Shared secrets require at least 32 bytes and are not sent
in plaintext. HMAC does not encrypt activations; use a trusted LAN or an
independently configured VPN/TLS tunnel when confidentiality is needed.

Stage optimizer updates are **not atomic across machines**. A failure during
stepping can leave only some stages updated. There is no automatic training
retry: restore coordinated stage checkpoints before retrying such a failed batch.
Distributed scheduling, multi-machine recovery and accelerator execution have
not been validated in this environment. The numerical test compares actual
forward losses and parameter updates across two loopback TCP stage servers to
the same unsplit model for three SGD updates.

## Optional discovery

`LANDiscovery` uses the optional `zeroconf` package and service type
`_neuralis._tcp.local.`. Advertisements contain the protocol version, port and
available backend names; they contain no secret. The worker list is bounded and
ages out stale entries. Discovery data is untrusted: clients still authenticate
the advertised worker with the separately provisioned shared secret. Explicit
host/port configuration works without zeroconf or multicast networking.

Run the CPU suite from the repository root:

```bash
PYTHONPATH=services/api/src /workspace/.venvs/neuralis-api/bin/python -m pytest services/api/tests/test_efficient_runtime.py -q
```
