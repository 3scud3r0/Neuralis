/** Versioned REST and telemetry contract for the optional Python backend. */
export const BACKEND_PROTOCOL_VERSION = '1.0' as const;

export type BackendRunState =
  | 'created'
  | 'running'
  | 'paused'
  | 'completed'
  | 'stopped'
  | 'failed'
  | 'interrupted';

export type BackendModelId =
  | 'mlp-xor'
  | 'mlp-spiral'
  | 'cnn-shapes'
  | 'tiny-transformer';

export interface BackendCaptureConfig {
  modules: string[];
  every: number;
  max_elements: number;
  max_tensors: number;
  histogram_bins: number;
  activations: boolean;
  gradients: boolean;
  weights: boolean;
}

export interface BackendRunConfig {
  model_id: BackendModelId;
  name: string;
  project_id: string;
  seed: number;
  steps: number;
  batch_size: number;
  learning_rate: number;
  weight_decay: number;
  gradient_clip: number;
  capture: BackendCaptureConfig;
  checkpoint_every: number;
  validation_every: number;
  step_delay_ms: number;
  device: 'cpu';
}

export interface BackendRun {
  id: string;
  project_id: string;
  config: BackendRunConfig;
  state: BackendRunState;
  step: number;
  sequence: number;
  created_at: number;
  updated_at: number;
  error: string | null;
}

export type BackendCommand = 'start' | 'pause' | 'step' | 'resume' | 'stop' | 'checkpoint';

export interface BackendCommandRequest {
  command: BackendCommand;
  command_id?: string;
}

export interface BackendCommandResult {
  run_id: string;
  command: BackendCommand;
  state: BackendRunState;
  step: number;
  acknowledged_at_boundary: boolean;
}

export interface BackendMetrics {
  loss: number;
  accuracy: number;
  gradient_norm: number;
  learning_rate: number;
  validation_loss: number;
  validation_accuracy: number;
  step_ms: number;
  capture_ms: number;
}

export interface BackendTensorSummary {
  module: string;
  kind: 'activation' | 'activation-gradient' | 'weight' | 'parameter-gradient';
  step: number;
  shape: number[];
  dtype: string;
  elements: number;
  captured_elements: number;
  sampled: boolean;
  sampling: { method: 'uniform-stride'; stride: number };
  values: number[];
  nonfinite: number;
  bytes: number;
  mean: number | null;
  std: number | null;
  minimum: number | null;
  maximum: number | null;
  norm: number | null;
  sparsity: number | null;
  histogram: { counts: number[]; minimum: number; maximum: number };
  quantiles: number[];
}

export interface BackendCaptureEvent {
  tensors: BackendTensorSummary[];
  dropped: number;
  capture_ms: number;
  complete: boolean;
  budget: BackendCaptureConfig;
}

export interface BackendEvent<T = Record<string, unknown>> {
  protocol_version: typeof BACKEND_PROTOCOL_VERSION;
  run_id: string;
  sequence: number;
  timestamp: number;
  step: number;
  type: string;
  payload: T;
  rank: number;
  world_size: number;
}

export interface BackendEventPage {
  events: BackendEvent[];
  oldest_sequence: number | null;
  gap: boolean;
  next_sequence: number;
}

export interface BackendInferenceRequest {
  inputs?: unknown[];
  sample_indices?: number[];
  split?: 'train' | 'validation';
}

export interface BackendIntervention {
  module: string;
  operation: 'ablate' | 'scale' | 'steer' | 'patch';
  strength?: number;
  indices?: number[];
  vector?: number[];
}

export interface BackendExperimentRequest {
  run_id: string;
  input?: BackendInferenceRequest;
  interventions: BackendIntervention[];
  clean_input?: BackendInferenceRequest;
  target_class?: number;
}

export interface BackendCapabilityStatus {
  available: boolean;
  reason?: string;
}

export interface BackendCapabilities {
  version: string;
  protocol_version: string;
  mode: string;
  models: BackendModelId[];
  training: boolean;
  inference: boolean;
  interventions: BackendIntervention['operation'][];
  attribution: ('gradient-input' | 'integrated-gradients')[];
  integrations: Record<string, BackendCapabilityStatus>;
  limitations: string[];
}

export interface BackendProblem {
  category: string;
  message: string;
  action: string;
}

export interface WorkerLease {
  lease_id: string;
  job_id: string;
  expires_at: number;
  lease_seconds: number;
  config: {
    task: 'xor' | 'ddp-xor';
    seed: number;
    steps: number;
    learning_rate: number;
    cpu_threads: number;
    memory_mb: number;
    world_size: number;
  };
}

export function isBackendEvent(value: unknown): value is BackendEvent {
  if (typeof value !== 'object' || value === null) return false;
  const event = value as Record<string, unknown>;
  return event.protocol_version === BACKEND_PROTOCOL_VERSION
    && typeof event.run_id === 'string'
    && typeof event.sequence === 'number'
    && Number.isSafeInteger(event.sequence)
    && event.sequence > 0
    && typeof event.timestamp === 'number'
    && Number.isFinite(event.timestamp)
    && typeof event.step === 'number'
    && Number.isSafeInteger(event.step)
    && event.step >= 0
    && typeof event.type === 'string'
    && typeof event.payload === 'object'
    && event.payload !== null;
}

export function describeCapture(summary: BackendTensorSummary): string {
  const coverage = summary.sampled
    ? `${summary.captured_elements} of ${summary.elements} values, sampled every ${summary.sampling.stride}`
    : `${summary.elements} complete values`;
  return `${summary.module}: ${summary.kind}, ${summary.shape.join(' × ')}, ${coverage}`;
}

export function websocketProtocols(token?: string): string[] | undefined {
  return token ? [`neuralis-token.${token}`] : undefined;
}
