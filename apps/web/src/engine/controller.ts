import type {
  EngineController, InterventionResult, InterventionSpec, MetricPoint,
  RunConfig, RunSnapshot, RunState,
} from '../types';
import { BanditModel } from './bandit';
import { getDemo } from './catalog';
import { DenseModel } from './dense-model';
import { CNNModel } from './cnn';
import { GridworldModel } from './gridworld';
import { KMeansModel } from './kmeans';
import { klDivergence } from './math';
import type { ResearchModel } from './model';
import { PCAModel } from './pca';
import { scheduledLearningRate } from './optimizers';
import { TransformerModel } from './transformer';
import { validateCustomDataset } from './datasets';

const MAX_METRICS = 512;
const CHECKPOINT_SCHEMA = 'neuralis-browser-checkpoint';
const CHECKPOINT_VERSION = 1;
const MAX_CHECKPOINT_BYTES = 64_000_000;

interface EngineCheckpoint {
  schema: typeof CHECKPOINT_SCHEMA;
  version: number;
  config: RunConfig;
  runId: string;
  step: number;
  elapsedMs: number;
  metrics: MetricPoint[];
  model: unknown;
}

export function validateConfig(value: RunConfig): RunConfig {
  if (!value || typeof value !== 'object') throw new Error('Run configuration is required.');
  getDemo(value.demoId);
  if (!Number.isFinite(value.learningRate) || value.learningRate <= 0 || value.learningRate > 1) {
    throw new Error('Learning rate must be finite and lie in (0, 1].');
  }
  if (!Number.isInteger(value.maxSteps) || value.maxSteps < 1 || value.maxSteps > 100_000) {
    throw new Error('Training steps must be an integer between 1 and 100,000.');
  }
  if (!Number.isInteger(value.batchSize) || value.batchSize < 1 || value.batchSize > 256) {
    throw new Error('Batch size must be an integer between 1 and 256.');
  }
  if (!Number.isInteger(value.seed) || value.seed < 0 || value.seed > 0xffffffff) {
    throw new Error('Seed must be a 32-bit unsigned integer.');
  }
  if (!Number.isInteger(value.captureEvery) || value.captureEvery < 1 || value.captureEvery > 10_000) {
    throw new Error('Capture interval must be an integer between 1 and 10,000.');
  }
  if (value.hiddenLayers && (!Array.isArray(value.hiddenLayers) || value.hiddenLayers.length > 8 ||
    value.hiddenLayers.some(size => !Number.isInteger(size) || size < 1 || size > 128))) {
    throw new Error('Custom hidden layers allow up to eight layers with 1–128 units each.');
  }
  if (value.optimizer !== undefined && !['adam', 'adamw', 'sgd', 'momentum', 'rmsprop'].includes(value.optimizer)) throw new Error('Unsupported optimizer.');
  if (value.schedule !== undefined && !['constant', 'cosine', 'warmup-cosine'].includes(value.schedule)) throw new Error('Unsupported learning-rate schedule.');
  if (!Number.isFinite(value.weightDecay ?? 0) || (value.weightDecay ?? 0) < 0 || (value.weightDecay ?? 0) > 1) throw new Error('Weight decay must lie in [0, 1].');
  if (!Number.isFinite(value.dropout ?? 0) || (value.dropout ?? 0) < 0 || (value.dropout ?? 0) >= 0.9) throw new Error('Dropout must lie in [0, 0.9).');
  if (!Number.isFinite(value.gradientClip ?? 5) || (value.gradientClip ?? 5) <= 0 || (value.gradientClip ?? 5) > 1000) throw new Error('Gradient clip must lie in (0, 1000].');
  if (value.warmupSteps !== undefined && (!Number.isInteger(value.warmupSteps) || value.warmupSteps < 0 || value.warmupSteps > value.maxSteps)) throw new Error('Warmup steps must lie between zero and maxSteps.');
  if (value.customDataset && value.demoId !== 'custom-data') throw new Error('Uploaded datasets require the custom-data demo.');
  if (value.demoId === 'custom-data' && !value.customDataset) throw new Error('Upload and prepare a dataset in the Data workbench before starting the custom-data demo.');
  return { demoId: value.demoId, learningRate: value.learningRate, maxSteps: value.maxSteps,
    batchSize: value.batchSize, seed: value.seed, captureEvery: value.captureEvery,
    hiddenLayers: value.hiddenLayers ? [...value.hiddenLayers] : undefined,
    optimizer: value.optimizer, weightDecay: value.weightDecay, dropout: value.dropout, gradientClip: value.gradientClip,
    schedule: value.schedule, warmupSteps: value.warmupSteps,
    customDataset: value.customDataset ? validateCustomDataset(value.customDataset) : undefined };
}

function buildModel(config: RunConfig): ResearchModel {
  const demo = getDemo(config.demoId);
  if (demo.id === 'pca') return new PCAModel(demo, config);
  if (demo.id === 'k-means') return new KMeansModel(demo, config);
  if (demo.id === 'gridworld') return new GridworldModel(config);
  if (demo.id === 'bandit') return new BanditModel(config);
  if (demo.id === 'tiny-cnn') return new CNNModel(demo, config);
  if (demo.id === 'tiny-transformer') return new TransformerModel(config);
  if (demo.id === 'super-mario') throw new Error('Super Mario runs in the separate local Python project: demos/super-mario.');
  return new DenseModel(demo, config);
}

/**
 * All work occurs in bounded event-loop slices. A studio step represents one
 * optimizer/minibatch update (or one PCA/Lloyd update), not one example.
 * Snapshots are immutable copies and expose actual validation measurements.
 */
export class BrowserEngine implements EngineController {
  private config: RunConfig;
  private model: ResearchModel;
  private state: RunState = 'idle';
  private currentStep = 0;
  private metrics: MetricPoint[] = [];
  private timer: ReturnType<typeof setTimeout> | null = null;
  private disposed = false;
  private runId: string;
  private elapsedMs = 0;
  private activeSince: number | null = null;
  private message: string | undefined;
  private snapshot: RunSnapshot;
  private readonly onSnapshot: (snapshot: RunSnapshot) => void;

  constructor(config: RunConfig, onSnapshot: (snapshot: RunSnapshot) => void) {
    this.config = validateConfig(config);
    this.model = buildModel(this.config);
    this.onSnapshot = onSnapshot;
    this.runId = `run-${this.config.demoId}-${Date.now().toString(36)}-${this.config.seed.toString(36)}`;
    this.snapshot = this.capture();
  }

  private assertLive(): void {
    if (this.disposed) throw new Error('This engine has been disposed.');
  }

  private time(): number {
    return this.elapsedMs + (this.activeSince === null ? 0 : Math.max(0, performance.now() - this.activeSince));
  }

  private cancelTimer(): void {
    if (this.timer !== null) clearTimeout(this.timer);
    this.timer = null;
  }

  private endActiveTime(): void {
    this.elapsedMs = this.time();
    this.activeSince = null;
  }

  private capture(): RunSnapshot {
    const result = this.model.evaluate();
    if (![result.loss, result.accuracy, result.gradientNorm].every(Number.isFinite)) {
      throw new Error('The model returned a nonfinite validation measurement.');
    }
    const point: MetricPoint = {
      step: this.currentStep, loss: result.loss, accuracy: result.accuracy,
      gradientNorm: result.gradientNorm, timestamp: Date.now(),
    };
    if (this.metrics.at(-1)?.step === this.currentStep) this.metrics[this.metrics.length - 1] = point;
    else this.metrics.push(point);
    if (this.metrics.length > MAX_METRICS) this.metrics.splice(0, this.metrics.length - MAX_METRICS);
    return {
      runId: this.runId, demoId: this.config.demoId, state: this.state, step: this.currentStep,
      epoch: this.currentStep * this.config.batchSize / this.model.trainingSize,
      loss: result.loss, accuracy: result.accuracy, learningRate: scheduledLearningRate(this.config, Math.max(0, this.currentStep - 1)),
      elapsedMs: this.time(), metrics: this.metrics.map(metric => ({ ...metric })),
      layers: this.model.capture(), samples: this.model.samples(),
      parameters: this.model.parameters, device: 'Browser CPU · JavaScript', message: this.message,
      seed: this.config.seed,
      extras: { ...this.model.extras(), gradientNorm: result.gradientNorm,
        optimizer: this.config.optimizer ?? 'adam', schedule: this.config.schedule ?? 'constant', baseLearningRate: this.config.learningRate,
        captureInterval: this.config.captureEvery, metricCapacity: MAX_METRICS,
        validationNote: 'Metrics are computed from the current model. No simulated training traces are used.',
        epochNote: ['pca', 'k-means'].includes(this.config.demoId) ?
          'Each studio step is a full-data solver iteration; epoch display is only a generic progress estimate.' :
          ['gridworld', 'bandit'].includes(this.config.demoId) ? 'Epoch is a generic transition-equivalent progress estimate, not a dataset epoch.' :
          'Epoch is sampled examples divided by training-set size; minibatches are sampled with replacement.',
      },
    };
  }

  private publish(recapture = true): void {
    if (recapture) this.snapshot = this.capture();
    else this.snapshot = { ...this.snapshot, state: this.state, message: this.message, elapsedMs: this.time() };
    this.onSnapshot(this.getSnapshot());
  }

  private fail(error: unknown): void {
    this.cancelTimer();
    this.endActiveTime();
    this.state = 'failed';
    this.message = error instanceof Error ? error.message : 'An unknown numerical error occurred.';
    this.publish(false);
  }

  private update(): void {
    this.model.train(scheduledLearningRate(this.config, this.currentStep), this.config.batchSize);
    this.currentStep++;
    if (this.currentStep >= this.config.maxSteps) {
      this.state = 'completed';
      this.endActiveTime();
      this.message = 'Configured training budget completed. Results reflect measured performance.';
    }
  }

  private schedule(): void {
    if (this.timer !== null || this.state !== 'running' || this.disposed) return;
    this.timer = setTimeout(() => {
      this.timer = null;
      if (this.state !== 'running' || this.disposed) return;
      try {
        const deadline = performance.now() + 8;
        let updates = 0;
        let needsCapture = false;
        do {
          this.update();
          updates++;
          if (this.currentStep % this.config.captureEvery === 0) needsCapture = true;
        } while (this.state === 'running' && updates < 8 && performance.now() < deadline);
        if (needsCapture || this.currentStep >= this.config.maxSteps) this.publish();
        this.schedule();
      } catch (error) {
        this.fail(error);
      }
    }, 0);
  }

  start(): void {
    this.assertLive();
    if (this.state === 'running') return;
    if (this.state === 'paused') { this.resume(); return; }
    if (this.state !== 'idle') throw new Error('Start a new engine to restart a completed, stopped, or failed run.');
    this.state = 'running';
    this.activeSince = performance.now();
    this.message = undefined;
    this.publish(false);
    this.schedule();
  }

  pause(): void {
    this.assertLive();
    if (this.state !== 'running') return;
    this.cancelTimer();
    this.endActiveTime();
    this.state = 'paused';
    this.publish();
  }

  resume(): void {
    this.assertLive();
    if (this.state === 'running') return;
    if (this.state !== 'paused') throw new Error('Only a paused run can be resumed.');
    if (this.currentStep >= this.config.maxSteps) throw new Error('The configured step budget has already been reached.');
    this.state = 'running';
    this.activeSince = performance.now();
    this.message = undefined;
    this.publish(false);
    this.schedule();
  }

  step(): void {
    this.assertLive();
    if (!['idle', 'paused'].includes(this.state)) throw new Error('Single-step requires an idle or paused run.');
    if (this.currentStep >= this.config.maxSteps) throw new Error('The configured step budget has already been reached.');
    this.state = 'paused';
    const before = performance.now();
    try {
      this.update();
      this.elapsedMs += Math.max(0, performance.now() - before);
      this.publish();
    } catch (error) {
      this.fail(error);
    }
  }

  stop(): void {
    this.assertLive();
    if (['completed', 'failed', 'stopped'].includes(this.state)) return;
    this.cancelTimer();
    this.endActiveTime();
    this.state = 'stopped';
    this.message = 'Run stopped. The current parameters remain available for inspection and export.';
    this.publish();
  }

  getSnapshot(): RunSnapshot {
    // structuredClone preserves optional fields and protects model state from consumers.
    return structuredClone(this.snapshot);
  }

  predict(input: number[], intervention?: InterventionSpec): number[] {
    this.assertLive();
    return this.model.predict(input, intervention);
  }

  intervene(input: number[], spec: InterventionSpec): InterventionResult {
    this.assertLive();
    const before = this.predict(input);
    const after = this.predict(input, spec);
    return { before, after, delta: after.map((value, i) => value - before[i]),
      input: [...input], spec: { ...spec }, klDivergence: klDivergence(before, after) };
  }

  exportCheckpoint(): string {
    this.assertLive();
    const checkpoint: EngineCheckpoint = {
      schema: CHECKPOINT_SCHEMA, version: CHECKPOINT_VERSION, config: { ...this.config }, runId: this.runId,
      step: this.currentStep, elapsedMs: this.time(), metrics: this.metrics.map(metric => ({ ...metric })),
      model: this.model.checkpoint(),
    };
    return JSON.stringify(checkpoint, null, 2);
  }

  importCheckpoint(serialized: string): void {
    this.assertLive();
    if (typeof serialized !== 'string' || serialized.length > MAX_CHECKPOINT_BYTES) {
      throw new Error('Checkpoint exceeds the 64 MB browser import limit.');
    }
    let saved: EngineCheckpoint;
    try { saved = JSON.parse(serialized) as EngineCheckpoint; }
    catch { throw new Error('Checkpoint is not valid JSON.'); }
    if (!saved || saved.schema !== CHECKPOINT_SCHEMA || saved.version !== CHECKPOINT_VERSION) {
      throw new Error('Unsupported checkpoint schema or version.');
    }
    const nextConfig = validateConfig(saved.config);
    if (!Number.isInteger(saved.step) || saved.step < 0 || saved.step > nextConfig.maxSteps ||
      !Number.isFinite(saved.elapsedMs) || saved.elapsedMs < 0 ||
      typeof saved.runId !== 'string' || saved.runId.length > 200 ||
      !Array.isArray(saved.metrics) || saved.metrics.length > MAX_METRICS ||
      saved.metrics.some((point, i) => !point || !Number.isInteger(point.step) || point.step < 0 || point.step > saved.step ||
        (i > 0 && point.step <= saved.metrics[i - 1].step) ||
        !Number.isFinite(point.loss) || point.loss < 0 || !Number.isFinite(point.accuracy) || point.accuracy < 0 || point.accuracy > 1 ||
        !Number.isFinite(point.timestamp) ||
        (point.gradientNorm !== undefined && (!Number.isFinite(point.gradientNorm) || point.gradientNorm < 0)))) {
      throw new Error('Checkpoint contains invalid run progress or metrics.');
    }
    // Validate and restore a fresh candidate before mutating this running engine.
    const candidate = buildModel(nextConfig);
    candidate.restore(saved.model);
    const result = candidate.evaluate();
    if (![result.loss, result.accuracy, result.gradientNorm].every(Number.isFinite)) {
      throw new Error('Checkpoint restored nonfinite model state.');
    }
    this.cancelTimer();
    this.config = nextConfig;
    this.model = candidate;
    this.currentStep = saved.step;
    this.elapsedMs = saved.elapsedMs;
    this.activeSince = null;
    this.metrics = saved.metrics.map(point => ({ ...point }));
    this.runId = saved.runId;
    this.state = saved.step >= nextConfig.maxSteps ? 'completed' : 'paused';
    this.message = 'Checkpoint loaded with optimizer state and deterministic random-stream position.';
    this.publish();
  }

  dispose(): void {
    if (this.disposed) return;
    this.cancelTimer();
    this.endActiveTime();
    this.disposed = true;
  }
}

export function createEngine(config: RunConfig, onSnapshot: (snapshot: RunSnapshot) => void): EngineController {
  return new BrowserEngine(config, onSnapshot);
}
