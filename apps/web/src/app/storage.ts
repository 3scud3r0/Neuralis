import type { RunConfig, RunSnapshot } from '../types';

export interface RunRecord {
  id: string;
  demoId: string;
  step: number;
  loss: number;
  accuracy: number;
  state: RunSnapshot['state'];
  seed: number;
  createdAt: string;
  metrics: RunSnapshot['metrics'];
  config: RunConfig;
}
export interface SavedCheckpoint {
  id: string;
  name: string;
  demoId: string;
  step: number;
  createdAt: string;
  data: string;
}
export interface WorkspaceSettings {
  reducedMotion: boolean;
  dense: boolean;
  captureLimit: number;
  backendUrl: string;
}
const prefix = 'neuralis.v1.';

export function readStored<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(prefix + key);
    return raw ? JSON.parse(raw) as T : fallback;
  } catch {
    return fallback;
  }
}

export function writeStored(key: string, value: unknown): boolean {
  try {
    localStorage.setItem(prefix + key, JSON.stringify(value));
    return true;
  } catch {
    return false;
  }
}

export function removeStored(key: string): void {
  try { localStorage.removeItem(prefix + key); } catch { /* Storage may be blocked. */ }
}

export function downloadText(name: string, content: string, mime = 'application/json'): void {
  const url = URL.createObjectURL(new Blob([content], { type: mime }));
  const link = document.createElement('a');
  link.href = url;
  link.download = name;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function snapshotRecord(snapshot: RunSnapshot, config: RunConfig): RunRecord {
  return {
    id: snapshot.runId,
    demoId: snapshot.demoId,
    step: snapshot.step,
    loss: snapshot.loss,
    accuracy: snapshot.accuracy,
    state: snapshot.state,
    seed: snapshot.seed,
    createdAt: new Date().toISOString(),
    metrics: snapshot.metrics.slice(-1000),
    config: { ...config },
  };
}
