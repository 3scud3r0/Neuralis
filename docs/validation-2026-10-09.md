# Research extension validation — 2026-10-09

All results below come from executed checks in the managed cloud workspace.

| Check | Result |
| --- | --- |
| Python API and AxisLab tests | 176 passed |
| Browser numerical, interaction and integration tests | 166 passed |
| Isolated Python 3.11 Mario suite | 27 passed |
| Rust desktop core tests | 19 passed |
| Worker suite | 16 passed |
| TypeScript checks and production browser build | Passed |
| Local API container | Actual HTTP AxisLab, frozen-base LoRA, encrypted RAG and DuckDB trajectory operations passed |
| Real NES recorder against API | PPO and DDQN: three persisted native RGB frames each; ordered policy/reward read-back passed |
| Optional WebRTC | Local aiortc peers decoded actual RGB video frames; managed Chromium blocks host ICE candidates |
| WebGPU | SwiftShader WGSL compute/render and explicit CPU fallback exercised with non-background rendered pixels |

The suites total **404 passing tests**. A Starlette/AnyIO deprecation warning
does not affect the results. The Rust check exercises the native core; it does
not establish GUI packaging on Windows, macOS or every Linux distribution.

The source count is reproducible with `python3 scripts/count-code.py --minimum
30000 --output docs/code-count.json`. Dependencies, generated bundles, blank
lines, comments, tests and documentation are excluded from production totals.

Screenshots are actual browser captures. Screenshot 025 shows archived NES
observations; screenshot 026 explicitly shows the managed browser's WebRTC
restriction. Neither the capture manifest nor the interface claims decoded
browser video under that policy. No browser policy was overridden.

CPU demonstrations and numerical adapters were validated. CUDA, ROCm, MPS,
multicast discovery on real networks, large pretrained models, external GGUF
conversion, Ollama execution and cross-machine pipeline throughput still require
the user's compatible hardware, model files and infrastructure. The provided
Mario policies do not solve a level; short measured rollouts cannot prove
perfect gameplay.
