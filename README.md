# Neuralis Studio

Neuralis Studio is an open neural research workbench for the browser and desktop. It combines a local, reproducible learning engine with an optional PyTorch runtime for experiments that need a real Python process. The interface is English throughout.

The browser build runs without a server and includes inspectable models, training controls, weights, activations, gradients, attention views, interventions, representation analysis, CSV preprocessing, checkpoints, replay and a library of runnable demonstrations. The optional API runs registered PyTorch MLP, CNN and causal Transformer examples on CPU and exposes bounded telemetry, checkpoints, inference, attribution and controlled interventions. The desktop shell is a Tauri host for the same UI and a local or remote runtime.

The project does not claim access to the private weights or internal tools of Anthropic, OpenAI or xAI. A 3D graph is a measured representation of captured model data, not a visualization of private model thoughts. Large-laboratory adapters report their actual availability; PostgreSQL, S3, OIDC, Slurm, Kubernetes, TransformerLens and Circuit Tracer are optional capabilities and are not silently presented as configured in this checkout.

## Try the browser workbench

```bash
npm ci
npm run dev
```

Open `http://localhost:5173`. Browser demos execute on the current device. `npm run build` creates the static site in `apps/web/dist`, packages the self-contained demos and Mario evidence, and is the build used by GitHub Pages.

The repository includes a Pages workflow at `.github/workflows/pages.yml`. After the `main` branch is pushed and Pages is enabled for the repository with **GitHub Actions** as its source, the site is served at:

`https://3scud3r0.github.io/Neuralis/`

The workflow runs TypeScript checks, browser tests and the production build before publishing. A static Pages site cannot run the Python API or train Mario; it provides the CPU browser experiments, source demos, viewer and measured evidence. Use a local API or desktop runtime for larger models.

## Run the Python runtime

The validated runtime targets Python 3.12–3.13 and CPU by default. `uv` keeps dependencies and the CPU PyTorch wheel pinned.

```bash
uv sync --project services/api --extra dev
uv run --project services/api python -m neuralis
```

The API listens on `http://127.0.0.1:8000`. In Neuralis Studio open **Infrastructure**, enter that URL and connect. The frontend calls the versioned `/api/v1` API, receives WebSocket events, and reports gaps instead of pretending that dropped telemetry was captured. The API advertises the validated device and integration limits at `/api/v1/capabilities`.

Useful checks:

```bash
uv run --project services/api pytest services/api/tests
npm run typecheck
npm test
npm run build
```

Runtime data is stored outside the checkout by default in `/tmp/neuralis-data`. Set `NEURALIS_DATA`, `NEURALIS_HOST`, `NEURALIS_PORT` and explicit `NEURALIS_CORS_ORIGINS` when running it elsewhere. Do not use wildcard CORS for a laboratory deployment. Authentication and project controls are configuration-sensitive; the default loopback process is not a production multi-user service.

## Mario research demo

The separate `demos/super-mario` project uses `gym-super-mario-bros`/`nes_py`, PyTorch and a reproducible PPO or Double-DQN pipeline. It includes action wrappers, frame stacking, bounded PPO, a dueling prioritized n-step Double-DQN, checkpoints, replay integrity, evaluation, activation capture, integrated gradients, Grad-CAM, ablation and a static report viewer.

```bash
cd demos/super-mario
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python doctor.py
python train.py --config configs/smoke.json
python evaluate.py --checkpoint runs/latest/checkpoint.pt --episodes 3
```

The emulator smoke test and a 2,048-transition PPO run completed in this environment. The held-out short evaluation completed **0/3 flags**; the maximum observed x-position was 314 and mean native return was 250. The Double-DQN smoke test completed 512 transitions and 61 gradient updates; it also did not solve a level. These are measured research baselines, not a claim of perfect Mario play. Longer training requires a user-owned machine, legal game assets and substantially more compute.

The numbered Mario evidence and report are in `demos/super-mario/evidence/`. The browser viewer is `demos/super-mario/index.html`. Large checkpoints and replay buffers are intentionally not included in the static Pages artifact.

## Desktop and laboratory profiles

The Tauri shell is under `apps/desktop`. Its Rust core manages a local Python sidecar with an ephemeral port and token, secure workspace paths, checkpoint dialogs, diagnostics and remote connection metadata. The desktop package needs the platform's native WebView/GTK toolchain; the browser and Python workflows do not.

`deploy/compose.yaml` is the validated local container profile for the API and worker. `deploy/compose.cloud.yaml`, `deploy/oidc.env.example` and `docs/laboratory.md` describe the explicit laboratory extensions. They do not imply that a PostgreSQL/S3/cluster is available in this environment.

## Repository map

```text
apps/web/                 React, TypeScript, Three.js and browser training engine
apps/desktop/             Tauri host, runtime lifecycle and secure file integration
services/api/             FastAPI, PyTorch models, capture, interventions and persistence
services/worker/          Worker leases, bounded execution and distributed diagnostics
demos/                    Browser examples and isolated Super Mario RL project
deploy/                   Local/container laboratory profiles
docs/                     Desktop, laboratory and source-count documentation
screenshots/              Numbered captures of the implemented application
```

Generated dependencies, local runs, checkpoints, recordings, `node_modules`, virtual environments and build output are ignored. The source count is produced with `python3 scripts/count-code.py`; it separates production, tests, configuration and tooling and never counts dependencies or generated bundles.

## Scope and evidence

The current checkout is CPU-first and bounded. The browser and API tests compare gradients, attention masks, checkpoints, interventions and training metrics against real numerical references. The Pages workflow is configured but publication is owned by GitHub repository settings and the Actions run. The final capture manifest distinguishes actual screenshots from design references; generated concept images in the proposal are not presented as product evidence.
