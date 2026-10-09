export type BackendConnectionState = 'disconnected' | 'connecting' | 'connected' | 'reconnecting' | 'failed';
export type BackendRunState = 'created' | 'running' | 'paused' | 'completed' | 'stopped' | 'failed' | 'interrupted';
export type BackendCommand = 'start' | 'pause' | 'step' | 'resume' | 'stop' | 'checkpoint';

export interface BackendModel {
  id: string;
  name?: string;
  title?: string;
  description?: string;
  parameters?: number;
  task?: string;
  [key: string]: unknown;
}
export interface BackendRunConfig {
  model_id: string;
  name: string;
  seed: number;
  steps: number;
  batch_size: number;
  learning_rate: number;
  capture_every: number;
  device: 'cpu';
  step_delay_ms?: number;
}
export interface BackendRun {
  id?: string;
  run_id?: string;
  state?: BackendRunState;
  status?: BackendRunState;
  step?: number;
  config?: BackendRunConfig;
  metrics?: unknown;
  [key: string]: unknown;
}
export interface BackendEvent {
  protocol_version: string;
  run_id: string;
  sequence: number;
  timestamp: number;
  step: number;
  type: string;
  payload: Record<string, unknown>;
  rank?: number;
  world_size?: number;
}
export interface BackendCapabilities {
  protocol_version?: string;
  version?: string;
  models?: unknown;
  device?: string;
  devices?: unknown;
  capabilities?: Record<string, unknown> | string[];
  [key: string]: unknown;
}
export interface BackendConnection {
  state: BackendConnectionState;
  baseUrl: string;
  capabilities: BackendCapabilities | null;
  lastError: string | null;
  latencyMs: number | null;
  connectedAt: number | null;
}
export interface StreamOptions {
  signal?: AbortSignal;
  afterSequence?: number;
  maxReconnects?: number;
  onState?: (state: BackendConnectionState) => void;
  onGap?: (previous: number, next: number) => void;
  onError?: (error: BackendError) => void;
}

export class BackendError extends Error {
  readonly status: number;
  readonly category: string;
  readonly action: string;
  constructor(message: string, status = 0, category = 'connection', action = '') {
    super(message);
    this.name = 'BackendError';
    this.status = status;
    this.category = category;
    this.action = action;
  }
}

const MAX_RESPONSE_BYTES = 32 * 1024 * 1024;
const DEFAULT_TIMEOUT = 15000;

export function validateBackendUrl(input: string): string {
  let url: URL;
  try {
    url = new URL(input.trim());
  } catch {
    throw new BackendError('Enter a complete HTTP or HTTPS backend URL.', 0, 'configuration');
  }
  if (!['http:', 'https:'].includes(url.protocol)) {
    throw new BackendError('The backend must use HTTP or HTTPS.', 0, 'configuration');
  }
  if (url.username || url.password || url.search || url.hash) {
    throw new BackendError('Credentials, queries and fragments do not belong in the backend URL.', 0, 'configuration');
  }
  if (typeof location !== 'undefined' && location.protocol === 'https:' && url.protocol === 'http:' &&
      !['localhost', '127.0.0.1', '[::1]'].includes(url.hostname)) {
    throw new BackendError('An HTTPS workbench requires HTTPS for a remote backend.', 0, 'configuration');
  }
  return url.toString().replace(/\/+$/, '').replace(/\/api\/v1$/, '');
}

function isObject(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

export function decodeEvent(value: unknown): BackendEvent {
  if (!isObject(value)) throw new BackendError('The server sent a malformed event.', 0, 'protocol');
  const sequence = Number(value.sequence);
  const step = Number(value.step);
  const timestamp = Number(value.timestamp);
  if (!Number.isSafeInteger(sequence) || sequence < 0 || !Number.isSafeInteger(step) || step < 0 ||
      !Number.isFinite(timestamp) || typeof value.run_id !== 'string' || typeof value.type !== 'string' ||
      !isObject(value.payload)) {
    throw new BackendError('The server event violates the telemetry contract.', 0, 'protocol');
  }
  const protocol = typeof value.protocol_version === 'string' ? value.protocol_version : '1.0';
  if (protocol.split('.')[0] !== '1') {
    throw new BackendError(`Unsupported telemetry protocol ${protocol}.`, 0, 'protocol', 'Update the client and backend together.');
  }
  return {
    protocol_version: protocol,
    run_id: value.run_id,
    sequence,
    timestamp,
    step,
    type: value.type,
    payload: value.payload,
    rank: Number.isSafeInteger(value.rank) ? Number(value.rank) : 0,
    world_size: Number.isSafeInteger(value.world_size) ? Number(value.world_size) : 1,
  };
}

async function parseResponse(response: Response): Promise<unknown> {
  const length = Number(response.headers.get('content-length'));
  if (Number.isFinite(length) && length > MAX_RESPONSE_BYTES) {
    throw new BackendError('The backend response exceeds the capture budget.', response.status, 'budget');
  }
  const text = await response.text();
  if (text.length > MAX_RESPONSE_BYTES) {
    throw new BackendError('The backend response exceeds the capture budget.', response.status, 'budget');
  }
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch {
    throw new BackendError('The backend returned an invalid JSON response.', response.status, 'protocol');
  }
}

function errorFromPayload(payload: unknown, status: number): BackendError {
  const object = isObject(payload) ? payload : {};
  const detail = isObject(object.detail) ? object.detail : object;
  const validation = Array.isArray(object.detail) ? object.detail : null;
  const message = typeof detail.message === 'string' ? detail.message :
    typeof object.detail === 'string' ? object.detail :
    validation ? `Invalid request: ${validation.map(item => isObject(item) ? String(item.msg || 'invalid field') : 'invalid field').join('; ')}` :
    `Backend request failed (${status}).`;
  return new BackendError(message, status,
    typeof detail.category === 'string' ? detail.category : 'request',
    typeof detail.action === 'string' ? detail.action : '');
}

export class BackendClient {
  readonly baseUrl: string;
  private token: string | null = null;
  private listeners = new Set<(connection: BackendConnection) => void>();
  private connection: BackendConnection;
  private controllers = new Set<AbortController>();
  private streams = new Set<() => void>();
  private disposed = false;

  constructor(baseUrl: string, token?: string) {
    this.baseUrl = validateBackendUrl(baseUrl);
    this.token = token || null;
    this.connection = {
      state: 'disconnected', baseUrl: this.baseUrl, capabilities: null,
      lastError: null, latencyMs: null, connectedAt: null,
    };
  }

  setSessionToken(token: string | null): void {
    this.token = token || null;
  }

  getConnection(): BackendConnection {
    return { ...this.connection };
  }

  subscribe(listener: (connection: BackendConnection) => void): () => void {
    this.listeners.add(listener);
    listener(this.getConnection());
    return () => { this.listeners.delete(listener); };
  }

  private publish(changes: Partial<BackendConnection>): void {
    this.connection = { ...this.connection, ...changes };
    this.listeners.forEach(listener => listener(this.getConnection()));
  }

  async connect(signal?: AbortSignal): Promise<BackendCapabilities> {
    this.publish({ state: 'connecting', lastError: null });
    const start = performance.now();
    try {
      const capabilities = await this.request<BackendCapabilities>('GET', '/capabilities', undefined, signal);
      const protocol = capabilities.protocol_version;
      if (protocol && protocol.split('.')[0] !== '1') {
        throw new BackendError(`Backend protocol ${protocol} is not supported.`, 0, 'protocol');
      }
      this.publish({
        state: 'connected', capabilities, connectedAt: Date.now(),
        latencyMs: Math.round(performance.now() - start), lastError: null,
      });
      return capabilities;
    } catch (error) {
      const normalized = error instanceof BackendError ? error : new BackendError(String(error));
      this.publish({ state: 'failed', lastError: normalized.message, capabilities: null });
      throw normalized;
    }
  }

  async request<T>(method: string, path: string, body?: unknown, signal?: AbortSignal, timeout = DEFAULT_TIMEOUT): Promise<T> {
    if (this.disposed) throw new BackendError('The backend client has been disposed.', 0, 'lifecycle');
    if (!path.startsWith('/') || path.startsWith('//')) throw new BackendError('Invalid API path.', 0, 'configuration');
    const controller = new AbortController();
    this.controllers.add(controller);
    const timer = setTimeout(() => controller.abort('timeout'), timeout);
    const abort = () => controller.abort(signal?.reason);
    signal?.addEventListener('abort', abort, { once: true });
    if (signal?.aborted) abort();
    try {
      const response = await fetch(`${this.baseUrl}/api/v1${path}`, {
        method,
        headers: {
          Accept: 'application/json',
          ...(body === undefined ? {} : { 'Content-Type': 'application/json' }),
          ...(this.token ? { Authorization: `Bearer ${this.token}` } : {}),
        },
        body: body === undefined ? undefined : JSON.stringify(body),
        signal: controller.signal,
        credentials: 'omit',
        cache: 'no-store',
      });
      const payload = await parseResponse(response);
      if (!response.ok) throw errorFromPayload(payload, response.status);
      return payload as T;
    } catch (error) {
      if (error instanceof BackendError) throw error;
      if (controller.signal.aborted) {
        throw new BackendError(signal?.aborted ? 'Request cancelled.' : 'The backend did not respond before the timeout.', 0, 'timeout');
      }
      throw new BackendError('Cannot reach the backend. Check its URL, CORS settings and whether it is running.', 0, 'connection');
    } finally {
      clearTimeout(timer);
      signal?.removeEventListener('abort', abort);
      this.controllers.delete(controller);
    }
  }

  async models(signal?: AbortSignal): Promise<BackendModel[]> {
    const result = await this.request<BackendModel[] | { models: BackendModel[] }>('GET', '/models', undefined, signal);
    return Array.isArray(result) ? result : result.models;
  }

  async runs(signal?: AbortSignal): Promise<BackendRun[]> {
    const result = await this.request<BackendRun[] | { runs: BackendRun[] }>('GET', '/runs', undefined, signal);
    return Array.isArray(result) ? result : result.runs;
  }

  createRun(config: BackendRunConfig, signal?: AbortSignal): Promise<BackendRun> {
    return this.request('POST', '/runs', config, signal);
  }

  run(id: string, signal?: AbortSignal): Promise<BackendRun> {
    return this.request('GET', `/runs/${encodeURIComponent(id)}`, undefined, signal);
  }

  command(id: string, command: BackendCommand, signal?: AbortSignal): Promise<unknown> {
    return this.request('POST', `/runs/${encodeURIComponent(id)}/commands`, {
      command, command_id: crypto.randomUUID(),
    }, signal);
  }

  inference(id: string, sampleIndices: number[] = [0, 1, 2, 3], signal?: AbortSignal): Promise<unknown> {
    return this.request('POST', `/runs/${encodeURIComponent(id)}/inference`, {
      sample_indices: sampleIndices, split: 'validation',
    }, signal);
  }

  tensors(id: string, signal?: AbortSignal): Promise<unknown> {
    return this.request('GET', `/runs/${encodeURIComponent(id)}/tensors`, undefined, signal);
  }

  metrics(id: string, signal?: AbortSignal): Promise<unknown> {
    return this.request('GET', `/runs/${encodeURIComponent(id)}/metrics`, undefined, signal);
  }

  checkpoints(id: string, signal?: AbortSignal): Promise<unknown> {
    return this.request('GET', `/runs/${encodeURIComponent(id)}/checkpoints`, undefined, signal);
  }

  experiment(body: {
    run_id: string;
    input: { sample_indices: number[]; split: string };
    interventions: { module: string; operation: 'ablate' | 'scale' | 'steer' | 'patch'; strength: number; indices?: number[] }[];
  }, signal?: AbortSignal): Promise<unknown> {
    return this.request('POST', '/experiments', body, signal);
  }

  attribution(id: string, method: string, integrationSteps = 32, signal?: AbortSignal): Promise<unknown> {
    return this.request('POST', `/runs/${encodeURIComponent(id)}/attribution`, {
      input: { sample_indices: [0], split: 'validation' },
      method, integration_steps: integrationSteps,
    }, signal, 60000);
  }

  async exportRun(id: string, signal?: AbortSignal): Promise<string> {
    const value = await this.request('GET', `/runs/${encodeURIComponent(id)}/export`, undefined, signal);
    return JSON.stringify(value, null, 2);
  }

  stream(id: string, onEvent: (event: BackendEvent) => void, options: StreamOptions = {}): () => void {
    let socket: WebSocket | null = null;
    let timer: ReturnType<typeof setTimeout> | null = null;
    let closed = false;
    let reconnects = 0;
    let lastSequence = options.afterSequence ?? -1;
    const maxReconnects = options.maxReconnects ?? 6;
    const connect = () => {
      if (closed || this.disposed || options.signal?.aborted) return;
      options.onState?.(reconnects ? 'reconnecting' : 'connecting');
      const url = new URL(`${this.baseUrl}/api/v1/ws/runs/${encodeURIComponent(id)}`);
      url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
      if (lastSequence >= 0) url.searchParams.set('after', String(lastSequence));
      // The backend accepts an ephemeral bearer credential as a negotiated
      // subprotocol. It is never copied into URLs, persistence or diagnostics.
      socket = this.token ? new WebSocket(url, [`neuralis-token.${this.token}`]) : new WebSocket(url);
      socket.onopen = () => {
        options.onState?.('connected');
      };
      socket.onmessage = message => {
        if (typeof message.data !== 'string' || message.data.length > MAX_RESPONSE_BYTES) {
          options.onError?.(new BackendError('Unsupported or oversized telemetry frame.', 0, 'protocol'));
          return;
        }
        try {
          const event = decodeEvent(JSON.parse(message.data));
          if (event.run_id !== id) throw new BackendError('A telemetry event belongs to another run.', 0, 'protocol');
          if (event.sequence <= lastSequence) return;
          if (lastSequence >= 0 && event.sequence > lastSequence + 1) options.onGap?.(lastSequence, event.sequence);
          lastSequence = event.sequence;
          reconnects = 0;
          onEvent(event);
        } catch (error) {
          options.onError?.(error instanceof BackendError ? error : new BackendError('Invalid telemetry JSON.', 0, 'protocol'));
        }
      };
      socket.onclose = event => {
        socket = null;
        if (closed || this.disposed) return;
        if (event.code === 1008 || event.code === 4001 || event.code === 4003) {
          options.onState?.('failed');
          options.onError?.(new BackendError('The server rejected the telemetry session.', 403, 'authentication'));
          return;
        }
        if (++reconnects > maxReconnects) {
          options.onState?.('failed');
          options.onError?.(new BackendError('Telemetry reconnection attempts exhausted. Refresh the run state.', 0, 'connection'));
          return;
        }
        options.onState?.('reconnecting');
        timer = setTimeout(connect, Math.min(15000, 500 * 2 ** (reconnects - 1)));
      };
      socket.onerror = () => {
        options.onError?.(new BackendError('Telemetry connection failed; checking the HTTP run state is still available.', 0, 'connection'));
      };
    };
    const close = () => {
      if (closed) return;
      closed = true;
      if (timer) clearTimeout(timer);
      if (socket) {
        socket.onclose = null;
        socket.close(1000, 'Client closed the stream');
      }
      this.streams.delete(close);
      options.signal?.removeEventListener('abort', close);
      options.onState?.('disconnected');
    };
    this.streams.add(close);
    options.signal?.addEventListener('abort', close, { once: true });
    connect();
    return close;
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.streams.forEach(close => close());
    this.controllers.forEach(controller => controller.abort('disposed'));
    this.controllers.clear();
    this.token = null;
    this.publish({ state: 'disconnected', capabilities: null });
    this.listeners.clear();
  }
}
