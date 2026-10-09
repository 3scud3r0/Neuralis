import type { InterventionSpec, LayerData, PredictionSample, RunConfig } from '../types';
import { argmax, finiteMatrix, finiteVector, matrix, mean, zeros } from './math';
import type { ResearchModel } from './model';
import type { BatchResult } from './network';
import { Random } from './random';

export interface GridTransition { next: number; reward: number; terminal: boolean; }
export const GRID = {
  width: 6, height: 6, goal: 35, hazard: 20,
  walls: [8, 9, 15, 21, 25, 26, 27],
  actions: ['Up', 'Right', 'Down', 'Left'],
} as const;
const WALLS = new Set<number>(GRID.walls);
const OPEN = Array.from({ length: 36 }, (_, i) => i).filter(i => !WALLS.has(i) && i !== GRID.goal && i !== GRID.hazard);
const DISCOUNT = 0.95;

export function gridTransition(state: number, action: number): GridTransition {
  if (!Number.isInteger(state) || state < 0 || state >= 36 || !Number.isInteger(action) || action < 0 || action > 3) {
    throw new Error('Invalid grid state or action.');
  }
  if (state === GRID.goal || state === GRID.hazard) return { next: state, reward: 0, terminal: true };
  const row = Math.floor(state / 6);
  const column = state % 6;
  const dr = [-1, 0, 1, 0][action];
  const dc = [0, 1, 0, -1][action];
  const nextRow = row + dr;
  const nextColumn = column + dc;
  const proposed = nextRow * 6 + nextColumn;
  if (nextRow < 0 || nextRow >= 6 || nextColumn < 0 || nextColumn >= 6 || WALLS.has(proposed)) {
    return { next: state, reward: -0.05, terminal: false };
  }
  return { next: proposed, reward: proposed === GRID.goal ? 1 : proposed === GRID.hazard ? -1 : -0.025,
    terminal: proposed === GRID.goal || proposed === GRID.hazard };
}

function optimalTable(): number[][] {
  let values = zeros(36);
  for (let iteration = 0; iteration < 300; iteration++) {
    const next = values.map((value, state) => OPEN.includes(state) ?
      Math.max(...[0, 1, 2, 3].map(action => {
        const transition = gridTransition(state, action);
        return transition.reward + (transition.terminal ? 0 : DISCOUNT * values[transition.next]);
      })) : value);
    const change = Math.max(...next.map((value, i) => Math.abs(value - values[i])));
    values = next;
    if (change < 1e-12) break;
  }
  return Array.from({ length: 36 }, (_, state) => [0, 1, 2, 3].map(action => {
    if (!OPEN.includes(state)) return 0;
    const transition = gridTransition(state, action);
    return transition.reward + (transition.terminal ? 0 : DISCOUNT * values[transition.next]);
  }));
}

interface GridCheckpoint {
  table: number[][];
  lastTD: number[];
  random: ReturnType<Random['exportState']>;
  current: number;
  episodeSteps: number;
  episodeReward: number;
  episodes: number;
  transitions: number;
  returns: number[];
}

export class GridworldModel implements ResearchModel {
  readonly parameters = 144;
  readonly trainingSize = OPEN.length;
  private readonly random: Random;
  private table = matrix(36, 4);
  private lastTD = zeros(4);
  private current = 0;
  private episodeSteps = 0;
  private episodeReward = 0;
  private episodes = 0;
  private transitions = 0;
  private returns: number[] = [];
  private readonly optimum = optimalTable();

  constructor(config: RunConfig) {
    this.random = new Random(config.seed);
  }

  private epsilon(): number {
    return Math.max(0.05, 0.9 * Math.exp(-this.transitions / 6000));
  }

  train(learningRate: number, batchSize: number): BatchResult {
    let squaredTD = 0;
    this.lastTD.fill(0);
    for (let i = 0; i < batchSize; i++) {
      const action = this.random.uniform() < this.epsilon() ? this.random.integer(4) : argmax(this.table[this.current]);
      const transition = gridTransition(this.current, action);
      const target = transition.reward + (transition.terminal ? 0 : DISCOUNT * Math.max(...this.table[transition.next]));
      const error = target - this.table[this.current][action];
      this.table[this.current][action] += learningRate * error;
      this.lastTD[action] = -error;
      squaredTD += error * error;
      this.current = transition.next;
      this.episodeSteps++;
      this.episodeReward += transition.reward;
      this.transitions++;
      if (transition.terminal || this.episodeSteps >= 100) {
        this.returns.push(this.episodeReward);
        if (this.returns.length > 128) this.returns.shift();
        this.episodes++;
        this.current = this.random.choice(OPEN);
        this.episodeSteps = 0;
        this.episodeReward = 0;
      }
    }
    return { loss: squaredTD / batchSize, accuracy: this.evaluate().accuracy,
      gradientNorm: Math.sqrt(this.lastTD.reduce((total, value) => total + value * value, 0)) };
  }

  private rollout(start: number): { success: boolean; path: number[]; reward: number } {
    let current = start;
    let reward = 0;
    const path = [current];
    const seen = new Set<number>();
    for (let step = 0; step < 72; step++) {
      if (seen.has(current)) break;
      seen.add(current);
      const transition = gridTransition(current, argmax(this.table[current]));
      reward += transition.reward;
      current = transition.next;
      path.push(current);
      if (transition.terminal) return { success: current === GRID.goal, path, reward };
    }
    return { success: false, path, reward };
  }

  evaluate(): BatchResult {
    let residual = 0;
    for (const state of OPEN) for (let action = 0; action < 4; action++) {
      const transition = gridTransition(state, action);
      const target = transition.reward + (transition.terminal ? 0 : DISCOUNT * Math.max(...this.table[transition.next]));
      residual += (target - this.table[state][action]) ** 2;
    }
    return { loss: residual / (OPEN.length * 4), accuracy: OPEN.filter(state => this.rollout(state).success).length / OPEN.length,
      gradientNorm: Math.sqrt(this.lastTD.reduce((total, value) => total + value * value, 0)) };
  }

  predict(input: number[], intervention?: InterventionSpec): number[] {
    if (!finiteVector(input, 36)) throw new Error('Gridworld expects a 36-dimensional one-hot state vector.');
    const result = [...this.table[argmax(input)]];
    if (intervention) {
      if (intervention.layer !== 1 || !Number.isInteger(intervention.neuron) || intervention.neuron < 0 || intervention.neuron > 3 ||
        !['ablate', 'scale', 'steer'].includes(intervention.kind) || !Number.isFinite(intervention.value)) {
        throw new Error('Gridworld interventions target an action value in layer 1.');
      }
      const i = intervention.neuron;
      result[i] = intervention.kind === 'ablate' ? 0 : intervention.kind === 'scale' ? result[i] * intervention.value : result[i] + intervention.value;
    }
    return result;
  }

  capture(): LayerData[] {
    const state = zeros(36);
    state[this.current] = 1;
    return [
      { id: 'grid-state', name: 'One-hot grid state', size: 36, shape: [6, 6], activations: state, weights: [], gradients: zeros(36), biases: [] },
      { id: 'grid-values', name: 'Tabular action values', size: 4, shape: [4, 36], activations: [...this.table[this.current]],
        weights: [0, 1, 2, 3].map(action => this.table.map(row => row[action])), gradients: [...this.lastTD], biases: zeros(4) },
    ];
  }

  samples(): PredictionSample[] {
    return OPEN.slice(0, 8).map(state => {
      const input = zeros(36);
      input[state] = 1;
      return { input, target: [...this.optimum[state]], prediction: this.predict(input), label: `Grid cell ${state}: optimal versus learned Q values` };
    });
  }

  extras(): Record<string, unknown> {
    return { algorithm: 'Tabular Q-learning', scoreLabel: 'Greedy goal reach rate over every nonterminal open start',
      lossLabel: 'Mean squared Bellman residual over all open-state action pairs',
      grid: { ...GRID, current: this.current, table: this.table.map(row => [...row]), path: this.rollout(0).path },
      epsilon: this.epsilon(), episodes: this.episodes, environmentTransitions: this.transitions,
      recentReturns: [...this.returns], meanEpisodeReturn: mean(this.returns), discount: DISCOUNT,
      architectureNote: 'Weights visualize a tabular Q function; no neural network is trained. Gradients are temporal-difference residuals.',
      stepNote: 'One studio step performs batchSize environment transitions.',
    };
  }

  checkpoint(): GridCheckpoint {
    return { table: this.table.map(row => [...row]), lastTD: [...this.lastTD], random: this.random.exportState(), current: this.current,
      episodeSteps: this.episodeSteps, episodeReward: this.episodeReward, episodes: this.episodes, transitions: this.transitions, returns: [...this.returns] };
  }

  restore(value: unknown): void {
    const saved = value as GridCheckpoint;
    if (!saved || !finiteMatrix(saved.table, 36, 4) || !finiteVector(saved.lastTD, 4) ||
      !Number.isInteger(saved.current) || saved.current < 0 || saved.current >= 36 || WALLS.has(saved.current) ||
      !Number.isInteger(saved.episodeSteps) || saved.episodeSteps < 0 || saved.episodeSteps > 100 ||
      !Number.isInteger(saved.episodes) || saved.episodes < 0 || !Number.isInteger(saved.transitions) || saved.transitions < 0 ||
      !Number.isFinite(saved.episodeReward) || !finiteVector(saved.returns) || saved.returns.length > 128) {
      throw new Error('Invalid gridworld checkpoint.');
    }
    const random = new Random(0);
    random.importState(saved.random);
    this.table = saved.table.map(row => [...row]);
    this.lastTD = [...saved.lastTD];
    this.random.importState(saved.random);
    this.current = saved.current;
    this.episodeSteps = saved.episodeSteps;
    this.episodeReward = saved.episodeReward;
    this.episodes = saved.episodes;
    this.transitions = saved.transitions;
    this.returns = [...saved.returns];
  }
}
