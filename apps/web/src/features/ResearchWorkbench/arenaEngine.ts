import { GRID, GridworldModel, gridTransition } from '../../engine/gridworld';

export interface ArenaFrame {
  step: number; state: number; action: number; reward: number; done: boolean;
  activations: number[]; image?: string;
}
export interface ArenaEpisode {
  schema: 'neuralis.arena/v1'; environment: string; seed: number; frames: ArenaFrame[];
  training?: { transitions: number; episodes: number; successRate: number; bellmanResidual: number; table: number[][] };
}
export function trainArena(seed: number, transitions: number): ArenaEpisode {
  if (!Number.isSafeInteger(seed) || !Number.isInteger(transitions) || transitions < 0 || transitions > 100000) throw new Error('Use an integer seed and 0–100,000 training transitions.');
  const model = new GridworldModel({ demoId: 'gridworld', learningRate: 0.25, batchSize: 64, seed, maxSteps: 1000, captureEvery: 10 });
  for (let index = 0; index < transitions; index += 64) model.train(0.25, Math.min(64, transitions - index));
  const extras = model.extras();
  const table = (extras.grid as { table: number[][] }).table;
  const frames: ArenaFrame[] = [];
  let state = 0;
  for (let step = 0; step < 72; step++) {
    const activations = [...table[state]];
    const action = activations.indexOf(Math.max(...activations));
    const result = gridTransition(state, action);
    frames.push({ step, state, action, reward: result.reward, done: result.terminal, activations });
    state = result.next;
    if (result.terminal) {
      frames.push({ step: step + 1, state, action: -1, reward: 0, done: true, activations: [...table[state]] });
      break;
    }
  }
  const evaluation = model.evaluate();
  return { schema: 'neuralis.arena/v1', environment: 'Gridworld / tabular Q-learning', seed, frames,
    training: { transitions, episodes: Number(extras.episodes), successRate: evaluation.accuracy, bellmanResidual: evaluation.loss, table } };
}
export function decodeEpisode(text: string): ArenaEpisode {
  if (text.length > 8_000_000) throw new Error('Trajectory import exceeds 8 MB.');
  const input = JSON.parse(text) as ArenaEpisode;
  if (input?.schema !== 'neuralis.arena/v1' || typeof input.environment !== 'string' || input.environment.length > 200 || !Number.isSafeInteger(input.seed) ||
    !Array.isArray(input.frames) || !input.frames.length || input.frames.length > 2000) throw new Error('Expected a neuralis.arena/v1 trajectory with 1–2,000 frames.');
  const frames = input.frames.map((frame, index) => {
    if (!frame || frame.step !== index || !Number.isInteger(frame.state) || frame.state < 0 || frame.state >= GRID.width * GRID.height || !Number.isInteger(frame.action) || frame.action < -1 || frame.action > 3 || !Number.isFinite(frame.reward) || typeof frame.done !== 'boolean' ||
      !Array.isArray(frame.activations) || frame.activations.length !== 4 || frame.activations.some(value => !Number.isFinite(value))) throw new Error(`Invalid trajectory frame ${index}. Expected a bounded grid state and four action values.`);
    if (frame.image !== undefined && (typeof frame.image !== 'string' || frame.image.length > 500000 || !/^data:image\/(png|jpeg|webp);base64,[A-Za-z0-9+/=]+$/.test(frame.image))) throw new Error('Imported frame images must be bounded PNG/JPEG/WebP data URLs.');
    return { step: frame.step, state: frame.state, action: frame.action, reward: frame.reward, done: frame.done, activations: [...frame.activations], ...(frame.image ? { image: frame.image } : {}) };
  });
  return { schema: input.schema, environment: input.environment, seed: input.seed, frames };
}
export function episodeReturn(episode: ArenaEpisode, frame = episode.frames.length - 1): number { return episode.frames.slice(0, frame + 1).reduce((sum, item) => sum + item.reward, 0); }
export { GRID };
