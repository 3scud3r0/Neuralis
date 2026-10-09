import type { InterventionSpec, LayerData, PredictionSample, RunConfig } from '../types';
import { argmax, dot, finiteVector, mean } from './math';
import type { ResearchModel } from './model';
import { DenseNetwork, type BatchResult, type NetworkCheckpoint } from './network';
import { Random } from './random';

const REWARD_COEFFICIENTS = [[0.8, -0.3, 0.2], [-0.2, 0.9, -0.1], [0.1, -0.4, 0.85]];
const expectedRewards = (context: number[]) => REWARD_COEFFICIENTS.map(coefficients => dot(coefficients, context));
const drawContext = (random: Random) => Array.from({ length: 3 }, () => random.uniform() * 2 - 1);

interface BanditCheckpoint {
  network: NetworkCheckpoint;
  random: ReturnType<Random['exportState']>;
  updates: number;
  rewards: number[];
  actions: number[];
}

/** Neural contextual value model with chosen-action squared-error updates. */
export class BanditModel implements ResearchModel {
  readonly parameters: number;
  readonly trainingSize = 128;
  private readonly network: DenseNetwork;
  private readonly random: Random;
  private readonly validation: number[][];
  private updates = 0;
  private rewards: number[] = [];
  private actions = [0, 0, 0];
  private readonly config: RunConfig;

  constructor(config: RunConfig) {
    this.config = { ...config };
    this.random = new Random(config.seed);
    this.network = new DenseNetwork([3, ...(config.hiddenLayers ?? [12]), 3], 'mse', this.random);
    this.parameters = this.network.parameters;
    const validationRandom = new Random((config.seed ^ 0x78198d) >>> 0);
    this.validation = Array.from({ length: 128 }, () => drawContext(validationRandom));
  }

  private epsilon(): number {
    return Math.max(0.08, 0.8 * Math.exp(-this.updates / 220));
  }

  train(learningRate: number, batchSize: number): BatchResult {
    const inputs: number[][] = [];
    const targets: number[][] = [];
    for (let i = 0; i < batchSize; i++) {
      const context = drawContext(this.random);
      const prediction = this.network.predict(context);
      const action = this.random.uniform() < this.epsilon() ? this.random.integer(3) : argmax(prediction);
      const reward = expectedRewards(context)[action] + this.random.normal(0, 0.08);
      const target = [...prediction];
      target[action] = reward;
      inputs.push(context);
      targets.push(target);
      this.actions[action]++;
      this.rewards.push(reward);
      if (this.rewards.length > 256) this.rewards.shift();
    }
    this.updates++;
    // Bandit targets use the inference readout; input/hidden dropout would change
    // unchosen-action targets and introduce a spurious loss, so it is disabled.
    return this.network.trainBatch(inputs, targets, learningRate, this.config.gradientClip ?? 5,
      { kind: this.config.optimizer, weightDecay: this.config.weightDecay });
  }

  evaluate(): BatchResult {
    const result = this.network.evaluate(this.validation, this.validation.map(expectedRewards));
    const correct = this.validation.filter(context => argmax(this.predict(context)) === argmax(expectedRewards(context))).length;
    return { ...result, accuracy: correct / this.validation.length };
  }

  predict(input: number[], intervention?: InterventionSpec): number[] {
    return this.network.predict(input, intervention);
  }

  capture(): LayerData[] {
    return this.network.capture(this.validation[0]);
  }

  samples(): PredictionSample[] {
    return this.validation.slice(0, 8).map(input => ({ input: [...input], target: expectedRewards(input), prediction: this.predict(input), label: 'Expected action reward' }));
  }

  extras(): Record<string, unknown> {
    const regret = this.validation.map(context => {
      const rewards = expectedRewards(context);
      return Math.max(...rewards) - rewards[argmax(this.predict(context))];
    });
    return { algorithm: 'Neural contextual bandit with chosen-action MSE updates', scoreLabel: 'Optimal greedy action fraction',
      lossLabel: 'Validation mean squared expected-reward prediction error', epsilon: this.epsilon(),
      actionCounts: [...this.actions], meanObservedReward: mean(this.rewards), recentRewards: [...this.rewards],
      meanExpectedRegret: mean(regret), rewardNoise: 0.08,
      weightGradients: this.network.layers.map(layer => layer.weightGradient.map(row => [...row])),
      architectureNote: 'A dense neural action-value function trained from sampled one-step rewards. This is not PPO or a full MDP.',
      dropoutNote: 'Bandit dropout is disabled to preserve chosen-action-only targets.',
    };
  }

  checkpoint(): BanditCheckpoint {
    return { network: this.network.exportCheckpoint(), random: this.random.exportState(), updates: this.updates, rewards: [...this.rewards], actions: [...this.actions] };
  }

  restore(value: unknown): void {
    const saved = value as BanditCheckpoint;
    if (!saved || !Number.isInteger(saved.updates) || saved.updates < 0 || !finiteVector(saved.rewards) || saved.rewards.length > 256 ||
      !finiteVector(saved.actions, 3) || saved.actions.some(value => !Number.isInteger(value) || value < 0)) throw new Error('Invalid bandit checkpoint.');
    const random = new Random(0);
    random.importState(saved.random);
    this.network.importCheckpoint(saved.network);
    this.random.importState(saved.random);
    this.updates = saved.updates;
    this.rewards = [...saved.rewards];
    this.actions = [...saved.actions];
  }
}
