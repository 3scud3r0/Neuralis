import type { InterventionSpec, LayerData, PredictionSample } from '../types';
import type { BatchResult } from './network';

export interface ResearchModel {
  readonly parameters: number;
  readonly trainingSize: number;
  train(learningRate: number, batchSize: number): BatchResult;
  evaluate(): BatchResult;
  predict(input: number[], intervention?: InterventionSpec): number[];
  capture(): LayerData[];
  samples(): PredictionSample[];
  extras(): Record<string, unknown>;
  checkpoint(): unknown;
  restore(value: unknown): void;
}
