import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from 'react';
import { createEngine, DEMOS, validateConfig } from '../engine';
import type { EngineController, RunConfig, RunSnapshot } from '../types';
import {
  readStored, writeStored, snapshotRecord, downloadText,
  type RunRecord, type SavedCheckpoint, type WorkspaceSettings,
} from './storage';

export interface Notification {
  id: number;
  message: string;
  tone: 'success' | 'info' | 'error';
}
interface StudioValue {
  config: RunConfig;
  snapshot: RunSnapshot;
  engine: EngineController;
  history: RunRecord[];
  checkpoints: SavedCheckpoint[];
  settings: WorkspaceSettings;
  notifications: Notification[];
  setConfig: (changes: Partial<RunConfig>) => void;
  selectDemo: (id: string) => void;
  resetEngine: () => void;
  start: () => void;
  pause: () => void;
  step: () => void;
  stop: () => void;
  saveCheckpoint: (name?: string) => void;
  loadCheckpoint: (checkpoint: SavedCheckpoint) => void;
  importCheckpoint: (data: string) => void;
  deleteCheckpoint: (id: string) => void;
  exportRun: () => void;
  clearHistory: () => void;
  updateSettings: (updates: Partial<WorkspaceSettings>) => void;
  notify: (message: string, tone?: Notification['tone']) => void;
  dismiss: (id: number) => void;
}
const StudioContext = createContext<StudioValue | null>(null);
const defaultSettings: WorkspaceSettings = {
  reducedMotion: false,
  dense: false,
  captureLimit: 100,
  backendUrl: 'http://127.0.0.1:8000',
};
const initialConfig = (): RunConfig => {
  const defaults: RunConfig = {
    demoId: DEMOS[0].id,
    learningRate: 0.03,
    maxSteps: 1500,
    batchSize: 32,
    seed: 42,
    captureEvery: 10,
  };
  const stored = readStored<RunConfig>('config', defaults);
  try {
    return stored.demoId !== 'super-mario' ? validateConfig({ ...defaults, ...stored }) : defaults;
  } catch { return defaults; }
};

export function StudioProvider({ children }: { children: ReactNode }) {
  const [config, updateConfig] = useState<RunConfig>(initialConfig);
  const [settings, setSettings] = useState(() => readStored('settings', defaultSettings));
  const [history, setHistory] = useState<RunRecord[]>(() => readStored('runs', []));
  const [checkpoints, setCheckpoints] = useState<SavedCheckpoint[]>(() => readStored('checkpoints', []));
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [revision, setRevision] = useState(0);
  const [snapshot, setSnapshot] = useState<RunSnapshot | null>(null);
  const [controller, setController] = useState<EngineController | null>(null);
  const controllerRef = useRef<EngineController | null>(null);
  const configRef = useRef(config);
  const notificationId = useRef(0);
  const archiveRef = useRef<Set<string>>(new Set());
  const runConfigs = useRef<Map<string, RunConfig>>(new Map());
  const notify = useCallback((message: string, tone: Notification['tone'] = 'info') => {
    const id = ++notificationId.current;
    setNotifications(current => [...current.slice(-3), { id, message, tone }]);
    setTimeout(() => setNotifications(current => current.filter(item => item.id !== id)), 6500);
  }, []);

  const archive = useCallback((value: RunSnapshot) => {
    if (value.step === 0 || archiveRef.current.has(value.runId)) return;
    archiveRef.current.add(value.runId);
    const record = snapshotRecord(value, runConfigs.current.get(value.runId) ?? configRef.current);
    setHistory(current => {
      const next = [record, ...current.filter(run => run.id !== record.id)].slice(0, 50);
      writeStored('runs', next);
      return next;
    });
  }, []);

  useEffect(() => {
    const next = createEngine(configRef.current, value => {
      setSnapshot(value);
      if (value.state === 'completed' || value.state === 'stopped' || value.state === 'failed') archive(value);
    });
    controllerRef.current = next;
    runConfigs.current.set(next.getSnapshot().runId, { ...configRef.current });
    setController(next);
    setSnapshot(next.getSnapshot());
    return () => {
      const active = controllerRef.current ?? next;
      archive(active.getSnapshot());
      active.dispose();
      next.dispose();
      controllerRef.current = null;
    };
  }, [revision, archive]);

  const setConfig = useCallback((changes: Partial<RunConfig>) => {
    try {
      const next = validateConfig({ ...configRef.current, ...changes });
      if (next.demoId === 'super-mario') throw new Error('Super Mario runs in the local Python project.');
      configRef.current = next;
      writeStored('config', next);
      updateConfig(next);
      setRevision(value => value + 1);
    } catch (error) {
      notify(error instanceof Error ? error.message : 'Invalid configuration.', 'error');
    }
  }, [notify]);
  const selectDemo = useCallback((id: string) => {
    const demo = DEMOS.find(item => item.id === id);
    if (!demo || id === 'super-mario') return;
    setConfig({ demoId: id, hiddenLayers: undefined });
    notify(`${demo.title} initialized. Training runs locally in your browser.`);
  }, [setConfig, notify]);

  const resetEngine = useCallback(() => setRevision(value => value + 1), []);
  const start = useCallback(() => {
    const engine = controllerRef.current;
    if (!engine) return;
    const state = engine.getSnapshot().state;
    if (state === 'paused') engine.resume();
    else if (state === 'completed' || state === 'stopped' || state === 'failed') {
      const next = createEngine(configRef.current, value => {
        setSnapshot(value);
        if (['completed', 'stopped', 'failed'].includes(value.state)) archive(value);
      });
      engine.dispose();
      controllerRef.current = next;
      runConfigs.current.set(next.getSnapshot().runId, { ...configRef.current });
      setController(next);
      setSnapshot(next.getSnapshot());
      next.start();
    } else engine.start();
  }, [archive]);
  const pause = useCallback(() => controllerRef.current?.pause(), []);
  const step = useCallback(() => controllerRef.current?.step(), []);
  const stop = useCallback(() => controllerRef.current?.stop(), []);

  const saveCheckpoint = useCallback((name?: string) => {
    const engine = controllerRef.current;
    if (!engine) return;
    const current = engine.getSnapshot();
    const entry: SavedCheckpoint = {
      id: crypto.randomUUID(),
      name: name?.trim() || `Step ${current.step.toLocaleString()} · ${new Date().toLocaleTimeString()}`,
      demoId: current.demoId,
      step: current.step,
      createdAt: new Date().toISOString(),
      data: engine.exportCheckpoint(),
    };
    setCheckpoints(previous => {
      const next = [entry, ...previous].slice(0, 15);
      if (!writeStored('checkpoints', next)) notify('Browser storage is full. Export your checkpoint to a file.', 'error');
      return next;
    });
    notify('Checkpoint saved with model parameters and optimizer state.', 'success');
  }, [notify]);

  const importCheckpoint = useCallback((data: string) => {
    try {
      const metadata = JSON.parse(data) as { config?: RunConfig };
      if (!metadata.config) throw new Error('Checkpoint configuration is missing.');
      const importedConfig = validateConfig(metadata.config);
      controllerRef.current?.importCheckpoint(data);
      configRef.current = importedConfig;
      updateConfig(importedConfig);
      writeStored('config', importedConfig);
      const current = controllerRef.current?.getSnapshot();
      if (current) {
        runConfigs.current.set(current.runId, { ...importedConfig });
        archiveRef.current.delete(current.runId);
        setSnapshot(current);
      }
      notify('Checkpoint restored.', 'success');
    } catch (error) {
      notify(error instanceof Error ? error.message : 'Invalid checkpoint.', 'error');
    }
  }, [notify]);
  const loadCheckpoint = useCallback((checkpoint: SavedCheckpoint) => {
    importCheckpoint(checkpoint.data);
  }, [importCheckpoint, notify]);
  const deleteCheckpoint = useCallback((id: string) => {
    setCheckpoints(current => {
      const next = current.filter(entry => entry.id !== id);
      writeStored('checkpoints', next);
      return next;
    });
  }, []);
  const exportRun = useCallback(() => {
    const current = controllerRef.current?.getSnapshot();
    if (!current) return;
    downloadText(`neuralis-${current.demoId}-${current.step}.json`, JSON.stringify({
      schemaVersion: 1,
      config: configRef.current,
      snapshot: current,
    }, null, 2));
    notify('Run exported as JSON.', 'success');
  }, [notify]);
  const clearHistory = useCallback(() => {
    setHistory([]);
    writeStored('runs', []);
    notify('Local run history cleared. Checkpoints are retained.');
  }, [notify]);
  const updateSettings = useCallback((updates: Partial<WorkspaceSettings>) => {
    setSettings(current => {
      const next = { ...current, ...updates };
      writeStored('settings', next);
      return next;
    });
  }, []);

  useEffect(() => {
    document.documentElement.dataset.motion = settings.reducedMotion ? 'reduced' : 'full';
    document.documentElement.dataset.density = settings.dense ? 'dense' : 'comfortable';
  }, [settings]);

  if (!controller || !snapshot) return <div className="boot-screen"><span className="boot-mark">N</span><p>Initializing the local research engine…</p></div>;
  return <StudioContext.Provider value={{
    config, snapshot, engine: controller, history, checkpoints, settings, notifications,
    setConfig, selectDemo, resetEngine, start, pause, step, stop,
    saveCheckpoint, loadCheckpoint, importCheckpoint, deleteCheckpoint,
    exportRun, clearHistory, updateSettings, notify,
    dismiss: id => setNotifications(current => current.filter(item => item.id !== id)),
  }}>{children}</StudioContext.Provider>;
}

export function useStudio(): StudioValue {
  const context = useContext(StudioContext);
  if (!context) throw new Error('useStudio requires StudioProvider');
  return context;
}
