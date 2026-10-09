# Tiny digit recognition

Ten hand-designed 5x5 glyphs with independent pixel noise.

## Run in Neuralis Studio

From the repository root, run `npm ci`, then `npm run dev`. Open the URL printed by Vite, choose **Tiny digit recognition** in the demo catalog, and start training. GitHub Pages runs this same numerical browser engine without a Python server.

`config.json` contains a reproducible CPU-first configuration. `example.ts` is an importable source example that runs synchronous single-step updates, reports real measurements, exports a resumable checkpoint, and disposes of the engine.

## Algorithm and measurements

Categorical cross-entropy; held-out synthetic digit accuracy.

The glyphs are not MNIST and scores do not establish real handwriting performance.

Implementation: [`dense-model.ts`](../../apps/web/src/engine/dense-model.ts). Shared numerical operations live in [engine](../../apps/web/src/engine). Dataset generation uses a seeded stream, with independent training/validation draws. Finite-case toy datasets can naturally repeat contexts across splits.

## Controls and reproducibility

Pause stops scheduled computation; single-step performs one optimizer or solver update. Resume continues the same parameters and random stream. Stop preserves the current model for inspection. Checkpoints retain parameters, optimizer moments, stochastic state, metrics, configuration, and progress. Imports validate into a fresh model before replacing the current run.

Five optimizer families are available for neural demos: SGD, momentum, Adam, AdamW, and RMSProp. Warmup/cosine schedules, global gradient clipping, weight decay and seeded inverted dropout are available where supported. Classical solvers expose their actual algorithm rather than pretending to use those neural options.

Captured tensors and metrics are measured values. Training and visualization budgets are bounded, and all demos use browser JavaScript CPU computation. Results from these synthetic problems do not establish performance on production datasets.

## Validation

Run `npm test -- --run src/engine` from the web workspace, or `npm run test --workspace=@neuralis/web -- src/engine` from the repository root. The suite checks exact numerical derivatives, real learning, state transitions, immutable captures, and checkpoint resumption.
