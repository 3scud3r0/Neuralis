import { createEngine } from '../../apps/web/src/engine';
import type { RunConfig, RunSnapshot } from '../../apps/web/src/types';

export const configuration: RunConfig = {
  "demoId": "custom-data",
  "learningRate": 0.01,
  "maxSteps": 1200,
  "batchSize": 32,
  "seed": 42,
  "captureEvery": 10,
  "optimizer": "adam",
  "schedule": "constant",
  "gradientClip": 5,
  "weightDecay": 0,
  "dropout": 0,
  "customDataset": {
    "name": "Linear toy example",
    "task": "regression",
    "inputs": [
      [
        -1
      ],
      [
        -0.5
      ],
      [
        0
      ],
      [
        0.5
      ],
      [
        1
      ]
    ],
    "targets": [
      [
        -1.7
      ],
      [
        -0.7
      ],
      [
        0.3
      ],
      [
        1.3
      ],
      [
        2.3
      ]
    ],
    "validationInputs": [
      [
        -0.75
      ],
      [
        0.25
      ],
      [
        0.75
      ]
    ],
    "validationTargets": [
      [
        -1.2
      ],
      [
        0.8
      ],
      [
        1.8
      ]
    ],
    "featureNames": [
      "x"
    ]
  }
};

/** Run genuine deterministic numerical updates without starting a timer. */
export function runCustomData(steps = 100): { snapshot: RunSnapshot; checkpoint: string } {
  if (!Number.isInteger(steps) || steps < 1 || steps > configuration.maxSteps) {
    throw new Error('Requested steps must lie within the configured training budget.');
  }
  const engine = createEngine(configuration, () => undefined);
  try {
    for (let step = 0; step < steps; step++) engine.step();
    const snapshot = engine.getSnapshot();
    if (snapshot.state === 'failed') throw new Error(snapshot.message ?? 'Numerical training failed.');
    return { snapshot, checkpoint: engine.exportCheckpoint() };
  } finally {
    engine.dispose();
  }
}

/** Import the full optimizer/random state and perform one reproducible update. */
export function resumeCheckpoint(serialized: string): RunSnapshot {
  const engine = createEngine(configuration, () => undefined);
  try {
    engine.importCheckpoint(serialized);
    if (engine.getSnapshot().state !== 'completed') engine.step();
    return engine.getSnapshot();
  } finally {
    engine.dispose();
  }
}
