# Actual execution evidence

These artifacts were produced by executed emulator rollouts, PyTorch optimization and Chromium page captures. They are not design mockups.

| Files | Source |
| --- | --- |
| `001`–`012` PNGs | PPO held-out evaluation; 3 independently seeded initial-noop starts, greedy actor, bounded action limit |
| `013-integrated-gradients.png` | Numerical attribution of the actual actor score at an emulator observation |
| `020-policy-lab-desktop.png` | Actual static viewer, Chromium, 1440 × 1100 viewport, full page |
| `021-policy-lab-inspection.png` | Actual viewer after selecting action entropy, a different activation layer and next captured frame |
| `022-policy-lab-mobile.png` | Actual responsive viewer, Chromium, 390 × 844 viewport, full page |
| `sample-run.json` | 2,048 real PPO transitions, 16 optimization updates; checkpoint resumed after 256 transitions |
| `ppo-metrics.csv` | The same measured PPO updates in tabular form |
| `ppo-evaluation.json` | Native episode returns, flag completion, actual X positions and confidence interval limitation |
| `ppo-inspection.json` | Real layer summaries and bounded activation samples |
| `dqn-run.json` | 512 real Double-DQN transitions, 61 gradient updates; checksum-verified prioritized replay resume |
| `benchmark.json` | Measured PPO / Double-DQN / random / right / right+jump comparison on identical starts |
| `attribution.json` | Score, baseline score and integrated-gradients completeness residual |
| `full-ablation.json` | Full shared-trunk activation ablation; the short smoke policy's selected actions remained unchanged |
| `output-ablation.json` | Output-head ablation, mean native return changed from 249.0 to −25.3 |
| `ppo-diagnostics.json` | Failure diagnostics derived from measured training episodes |

Both short trained policies completed **0/3** tested stages. The data demonstrates a runnable training/research pipeline, not perfect play or statistically established superiority. Native reward differs from the 0.1-scaled reward used for learning. The deterministic NES may retain correlated starts despite independent initial-noop seeds; the exported Wilson interval explicitly states its independence assumption.

The optional checkpoint/replay ZIP lives under the ignored local `runs/delivery/` directory and is excluded from static deployment. No ROM is distributed.
