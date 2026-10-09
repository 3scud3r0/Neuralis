# Mario Policy Lab

A real, isolated reinforcement-learning experiment for **Neuralis Studio**. Train a visual neural policy inside `gym-super-mario-bros` / `nes_py`, evaluate held-out seeded starts, inspect its activations and input attributions, and compare causal ablations. The browser page displays exported measurements; the emulator and training run locally in Python.

Two complete learning algorithms are included:

- **PPO**: shared visual CNN, categorical actor, value critic, clipped policy/value objectives, entropy regularization, timeout-aware generalized advantage estimation, gradient clipping, KL early stopping, learning-rate annealing and automatic stage curriculum.
- **Dueling Double-DQN**: shared visual CNN, value/advantage decomposition, separate target network, proportional prioritized replay, stratified segment-tree sampling, importance-sampling correction, n-step returns, epsilon-greedy exploration and hard/soft target updates.

The included smoke policies **have not solved Mario**. Perfect play cannot be promised: success must be measured after sufficient training on the stages, seeds and conditions of interest.

## Tested environment

Python **3.11.16**, PyTorch **2.7.1+cpu**, NumPy **1.26.4**, Gym **0.23.1**, `gym-super-mario-bros` **7.4.0**, `nes_py` **8.2.1**, Linux x86-64. A C++ compiler is needed to build `nes_py`. Python 3.12 and NumPy 2 are deliberately avoided for this legacy package combination.

The Studio backend has its own environment. Installing these legacy dependencies in that backend environment is unnecessary.

## Install

On Linux/macOS, from this directory:

```bash
bash bootstrap.sh
source .venv/bin/activate
python doctor.py
```

`bootstrap.sh` uses `uv` when present, otherwise Python 3.11's built-in venv. It preserves existing environments, verifies their Python version, installs the CPU Torch build from the official index, then installs the exact tested package versions in `requirements.lock`. macOS uses the native Torch distribution. TLS verification remains enabled.

For a manual Linux installation:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install 'torch==2.7.1+cpu' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.lock
```

Windows PowerShell requires Python 3.11 and Microsoft C++ Build Tools:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install torch==2.7.1+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.lock
python doctor.py
```

Windows/macOS instructions are provided for local development; the executable validation performed for this delivery was Linux. Native rendering also needs your platform's display and OpenGL libraries. Headless training and PNG/video capture work without a desktop window.

For CUDA, install an official compatible PyTorch CUDA distribution in the isolated environment and use `--device cuda`. `requirements.lock` records the tested CPU distribution; avoid replacing an existing GPU Torch build by blindly synchronizing that CPU lock. See the official [PyTorch installation selector](https://pytorch.org/get-started/locally/).

## Verify the actual emulator and training

```bash
python doctor.py --capture runs/doctor/001-emulator.png
python -m pytest -q
python train.py --config configs/smoke.json
python train_dqn.py --config configs/dqn-smoke.json
```

The smoke configurations execute real NES rollouts, real loss/backpropagation steps and real checkpoints. They use short action limits and small replay capacities to keep validation bounded. They are not sufficient to learn a strong policy.

## Train PPO

```bash
python train.py --config configs/level-1-1.json
python train.py --resume runs/level-1-1/latest.pt --steps 20000000
```

`--steps` is the total target number of environment transitions, including already trained transitions. Every action advances up to four emulator frames. Vector rollout size can overshoot the target by at most `num_envs - 1` transitions. Existing run directories with checkpoints require `--resume` or a new `--output`.

```bash
python train.py --config configs/level-1-1.json --envs 2 --device cpu --output runs/cpu-trial
python train.py --config configs/smoke.json --sync --output runs/debug
python train.py --config configs/curriculum.json
```

The curriculum advances to the next listed stage after the configured rolling training-episode success threshold. It does not imply that the completed stages are permanently mastered. PPO checkpoints preserve the model, optimizer, counters, stage and Python/NumPy/Torch RNG states. Resume restarts emulator episodes; exact mid-episode continuation is not claimed.

## Train Double-DQN

```bash
python train_dqn.py --config configs/dqn-level-1-1.json
python train_dqn.py --resume runs/dqn-level-1-1/latest-dqn.pt --steps 20000000
```

DQN checkpoints include a separate compressed replay snapshot referenced by filename and SHA-256. Keep `latest-dqn.pt` and the referenced `replay-*.npz` together when moving a run. Resume verifies the replay checksum, array types/shapes and memory budget before loading. The target network, optimizer, transition count, update count, replay priorities and replay/random generator states are restored. Emulator episodes and pending incomplete n-step prefixes restart.

The default 10,000-entry replay uses approximately **539 MiB** for two full uint8 observation stacks per entry, plus metadata. `replay_memory_mb` bounds allocation and archive loading. Increase capacity deliberately; this implementation favors transparent, independently inspectable replay over an implicit shared-frame encoding. Snapshot compression can be expensive for large replay buffers; tune `checkpoint_every`. Set `save_replay: false` to save weights/optimizer only, then resume will warm a new buffer.

## Evaluate and capture evidence

Both PPO and DQN checkpoints work with the same evaluation command:

```bash
python evaluate.py runs/level-1-1/latest.pt \
  --episodes 30 --seed 100042 --video \
  --output runs/level-1-1/evaluation
python evaluate.py runs/dqn-level-1-1/latest-dqn.pt \
  --episodes 30 --seed 100042 \
  --output runs/dqn-level-1-1/evaluation
python play.py runs/level-1-1/latest.pt --episodes 3
```

Use `play.py` on a computer with a display. `evaluate.py` saves actual numbered RGB PNGs, optional MP4 video, a bounded layer inspection and a machine-readable report. `--capture-every` and `--max-captures` bound the image budget. Video records the first episode at action-step frequency. Use `--max-steps` for a bounded inspection run; use the trained episode limit for a meaningful full-stage evaluation.

Completion means the emulator reports `flag_get`. Training returns preserve the native game's reward; PPO/DQN optimization uses native reward multiplied by `reward_scale` (default 0.1). Reward alone never proves completion.

Evaluation uses independent episode seed offsets and random initial no-ops. Action selection defaults to greedy; `--stochastic` samples PPO's categorical policy. DQN always uses greedy evaluation actions. Since NES is deterministic, seeded initial no-ops can still produce correlated starts. The exported Wilson 95% success interval assumes independent trials and explicitly reports that limitation. Use a wider stage/start distribution and multiple training seeds before making research claims.

## Fair policy comparison and diagnostics

```bash
python benchmark.py \
  runs/level-1-1/latest.pt runs/dqn-level-1-1/latest-dqn.pt \
  --baselines --episodes 30 --max-steps 4500 --output runs/comparison
python diagnose.py runs/level-1-1/neuralis-mario-run.json \
  --window 100 --output runs/level-1-1/diagnostics.json
```

The benchmark reuses the same seed sequence, initial-noop configuration and episode limits for all agents. It rejects mismatched preprocessing/action spaces. Optional controls are random actions, constant right, and constant right+jump. These controls are explicitly labeled, not trained policies. Diagnostics inspect measured flag failures, repeated failure locations, timeouts, low entropy, large KL, heavy clipping and reported runtime errors.

## Inspect and intervene

```bash
python inspect_policy.py runs/level-1-1/latest.pt \
  --method integrated-gradients --steps 32 --advance 10 \
  --output runs/level-1-1/attribution
python inspect_policy.py runs/level-1-1/latest.pt \
  --method grad-cam --output runs/level-1-1/grad-cam
python intervene.py runs/level-1-1/latest.pt \
  --layer trunk.1 --channels 0,1,2,3 --scale 0 --episodes 10 \
  --output runs/level-1-1/ablation
```

Attribution explains a selected actor logit or DQN Q-value. Integrated gradients uses an all-black input baseline and reports its numerical completeness residual. Grad-CAM uses the final convolution. Maps are bounded and normalized per observation; they are not semantic labels. Ablations scale selected channels (empty `--channels` means all), run original/intervened policies on the same seeded starts and export measured return/success differences. Hooks are removed even if evaluation fails. Model checkpoint weights are never changed by inspection or intervention.

## Export to the browser

```bash
python export.py runs/level-1-1 \
  --evaluation runs/level-1-1/evaluation/evaluation.json \
  --inspection runs/level-1-1/evaluation/inspection.json \
  --output runs/mario-export.json --csv runs/mario-metrics.csv
```

Open `index.html` through an HTTP server or the Neuralis GitHub Pages deployment. The bundled report loads automatically. Report import is bounded to 25 MiB, 100,000 optimization samples and 128 displayed architecture layers. Import your exported JSON to inspect measured optimization curves, model architecture, completed evaluation episodes and activation snapshots. Imported frame files stay in their local output directory; the viewer does not fabricate or substitute frames for them.

```bash
python -m http.server 8080
# Open http://127.0.0.1:8080/
```

GitHub Pages cannot execute Python, native NES emulation or CUDA. This page is a real report viewer, not a claim of browser-based Mario training. The broader Neuralis demo collection supplies browser-native numerical experiments.

## Evidence from this delivery

The `evidence/` directory contains actual exported measurements and numbered PNGs:

- PPO: **2,048 transitions, 16 updates**, including checkpoint resume; **19 completed training episodes**.
- Double-DQN: **512 transitions, 61 gradient updates**, including checksum-verified replay resume.
- PPO evaluation: three bounded seeded starts, **0/3 flag completions**, mean native return **249.0**, maximum X **314**.
- Double-DQN benchmark: **0/3 flag completions**, mean native return **239.7**, maximum X **306**.
- Random/right/right+jump controls: mean native returns **215.3 / 231.7 / 231.7**, all **0/3** flag completions.
- `001`–`012`: real emulator frames captured from PPO evaluation.
- `013`: a real integrated-gradients attribution map (16-step numerical quadrature; residual reported in `attribution.json`).
- Full output-head ablation: mean native return **249.0 → −25.3** on the same seeded starts; completion remains **0/3**. This short policy is not evidence of a meaningful learned Mario strategy.
- **27 numerical/lifecycle/integrity tests passed**, including PPO/DQN optimization, timeout handling, replay resume, attribution completeness and intervention hook cleanup.
- `020`–`022`: the actual report viewer in Chromium at desktop/inspection/mobile viewports.

These results validate installation, emulator execution, parallel rollouts, two learning pipelines, resume, evaluation and export. They do not establish a solved game or benchmark superiority. Short-sample returns have no demonstrated statistical significance.

## Project map

```text
super-mario/
├── bootstrap.sh           # Isolated installation
├── requirements.lock      # Exact tested versions
├── doctor.py              # Actual emulator health check
├── train.py               # PPO training CLI
├── train_dqn.py           # Double-DQN training CLI
├── evaluate.py            # Seeded evaluation + numbered captures
├── play.py                # Native rendering
├── benchmark.py           # Fair policy/control comparison
├── inspect_policy.py      # Input gradients / IG / Grad-CAM
├── intervene.py           # Activation ablation evaluation
├── diagnose.py            # Evidence-backed failure diagnostics
├── export.py              # Portable Neuralis JSON / CSV
├── configs/               # Bounded and research configurations
├── mario_rl/
│   ├── config.py          # Typed validated environment/training settings
│   ├── environment.py     # NES, Gym API adaptation, preprocessing
│   ├── vector.py          # Spawn workers, terminal obs, failure cleanup
│   ├── model.py           # CNN actor-critic and bounded layer inspection
│   ├── ppo.py             # GAE and clipped numerical objectives
│   ├── training.py        # PPO rollout, checkpoint and curriculum loop
│   ├── checkpoint.py      # Restricted-load atomic PPO checkpoints
│   ├── replay.py          # Prioritized trees, n-step returns, SHA-256 snapshots
│   ├── dqn.py             # Dueling Double-DQN optimizer and loop
│   ├── policies.py        # Policy loaders and explicit control baselines
│   ├── evaluation.py      # Actual performance and artifact capture
│   ├── interpretability.py# Gradient attribution and causal hooks
│   ├── diagnostics.py     # Failure pattern analysis
│   └── telemetry.py       # Finite JSON, JSONL and run provenance
├── tests/                 # Numerical, lifecycle, integrity regressions
├── evidence/              # Actual smoke outputs and numbered PNGs
├── index.html             # Static report viewer
├── viewer.js              # Curves, import, layers and frame navigation
└── styles.css             # Responsive English interface
```

Mario game assets remain the property of their respective rights holders. No ROM is added to this repository. The emulator package resolves the environment assets; use only assets you are authorized to run. Neuralis policy/training source follows the root repository license.
