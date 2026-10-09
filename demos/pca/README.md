# Principal-component explorer

Correlated three-dimensional observations with a small third-direction noise component.

## Run in Neuralis Studio

From the repository root, run `npm ci`, then `npm run dev`. Open the URL printed by Vite, choose **Principal-component explorer** in the demo catalog, and start training. GitHub Pages runs this same numerical browser engine without a Python server.

`config.json` contains a reproducible CPU-first configuration. `example.ts` is an importable source example that runs synchronous single-step updates, reports real measurements, exports a resumable checkpoint, and disposes of the engine.

## Algorithm and measurements

Reconstruction MSE; explained variance after projecting onto two axes.

Orthogonal power iteration updates axes. Learning rate/minibatch settings are ignored; node gradients are zero because this is not backpropagation.

Implementation: [`pca.ts`](../../apps/web/src/engine/pca.ts). Shared numerical operations live in [engine](../../apps/web/src/engine). Dataset generation uses a seeded stream, with independent training/validation draws. Finite-case toy datasets can naturally repeat contexts across splits.

## Controls and reproducibility

Pause stops scheduled computation; single-step performs one optimizer or solver update. Resume continues the same parameters and random stream. Stop preserves the current model for inspection. Checkpoints retain parameters, optimizer moments, stochastic state, metrics, configuration, and progress. Imports validate into a fresh model before replacing the current run.

Five optimizer families are available for neural demos: SGD, momentum, Adam, AdamW, and RMSProp. Warmup/cosine schedules, global gradient clipping, weight decay and seeded inverted dropout are available where supported. Classical solvers expose their actual algorithm rather than pretending to use those neural options.

Captured tensors and metrics are measured values. Training and visualization budgets are bounded, and all demos use browser JavaScript CPU computation. Results from these synthetic problems do not establish performance on production datasets.

## Validation

Run `npm test -- --run src/engine` from the web workspace, or `npm run test --workspace=@neuralis/web -- src/engine` from the repository root. The suite checks exact numerical derivatives, real learning, state transitions, immutable captures, and checkpoint resumption.
