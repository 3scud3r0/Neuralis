export type RunState = 'idle' | 'running' | 'paused' | 'completed' | 'stopped' | 'failed';
export type DemoCategory = 'Classification' | 'Regression' | 'Representation' | 'Reinforcement' | 'Language' | 'Vision';
export interface DemoDefinition {
  id: string; title: string; description: string; category: DemoCategory;
  architecture: number[]; tags: string[]; difficulty: 'Introductory' | 'Intermediate' | 'Advanced';
  accent: string; task: string; inputLabels?: string[]; outputLabels?: string[];
}
export interface RunConfig {
  demoId: string; learningRate: number; maxSteps: number; batchSize: number;
  seed: number; captureEvery: number; hiddenLayers?: number[];
  optimizer?: 'adam' | 'adamw' | 'sgd' | 'momentum' | 'rmsprop';
  weightDecay?: number; dropout?: number; gradientClip?: number;
  schedule?: 'constant' | 'cosine' | 'warmup-cosine'; warmupSteps?: number;
  customDataset?: {
    name?: string; inputs: number[][]; targets: number[][];
    validationInputs?: number[][]; validationTargets?: number[][];
    task: 'classification' | 'regression'; labels?: string[]; featureNames?: string[];
  };
}
export interface MetricPoint { step: number; loss: number; accuracy: number; gradientNorm?: number; timestamp: number; }
export interface LayerData {
  id: string; name: string; size: number; shape: number[];
  activations: number[]; weights: number[][]; gradients: number[]; biases: number[];
}
export interface PredictionSample {
  input: number[]; target: number[]; prediction: number[]; label?: string;
}
export interface RunSnapshot {
  runId: string; demoId: string; state: RunState; step: number; epoch: number;
  loss: number; accuracy: number; learningRate: number; elapsedMs: number;
  metrics: MetricPoint[]; layers: LayerData[]; samples: PredictionSample[];
  parameters: number; device: string; message?: string; seed: number;
  extras?: Record<string, unknown>;
}
export interface InterventionSpec {
  layer: number; neuron: number; kind: 'ablate' | 'scale' | 'steer'; value: number;
}
export interface InterventionResult {
  before: number[]; after: number[]; delta: number[]; input: number[];
  spec: InterventionSpec; klDivergence?: number;
}
export interface EngineController {
  start(): void; pause(): void; resume(): void; step(): void; stop(): void;
  getSnapshot(): RunSnapshot;
  predict(input: number[], intervention?: InterventionSpec): number[];
  intervene(input: number[], spec: InterventionSpec): InterventionResult;
  exportCheckpoint(): string; importCheckpoint(checkpoint: string): void;
  dispose(): void;
}
export interface SceneProps {
  layers: LayerData[]; mode?: 'activations' | 'weights' | 'gradients';
  onSelect?: (layer: number, neuron: number) => void;
  selected?: { layer: number; neuron: number } | null;
  paused?: boolean; height?: number | string; className?: string;
}
