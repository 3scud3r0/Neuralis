import { BackendError, validateBackendUrl } from '../../integrations/backendClient';
import { desktop } from '../../../../desktop/bridge';
export interface ExtensionCapability { available: boolean; reason?: string; version?: string; encrypted?: boolean; }
export interface ExtensionCapabilities {
  axislab: ExtensionCapability; nf4: ExtensionCapability; lora: ExtensionCapability;
  rag: ExtensionCapability; webrtc: ExtensionCapability; gguf: ExtensionCapability;
  backends: Record<string, unknown> | unknown[];
}
export interface AxisResponse {
  direction: number[]; scores: { positive: number[]; negative: number[] };
  auc: number; training_samples: number; evaluation_samples: number; method: string;
}
export interface TrainingResponse {
  losses: number[]; base_parameters: number; trainable_parameters: number; max_base_change: number; method: string;
  quantization?: { original_bytes: number; packed_bytes: number; mse: number };
}
export interface RagBackendResult { results: { text: string; source: string; score: number }[]; encrypted: boolean; }
export interface ExportRecipe { modelfile: string; commands: string[]; requirements: string[]; }
export class ExtensionClient {
  readonly baseUrl: string;
  private readonly token: string;
  constructor(baseUrl: string, token = '', private readonly fetcher: typeof fetch = fetch, readonly useNative = false) {
    this.baseUrl = validateBackendUrl(baseUrl);
    this.token = useNative ? '' : token;
  }
  async request<T>(path: string, payload?: unknown): Promise<T> {
    const route = path.split('?')[0];
    const segments = route.split('/');
    if (!route || path.length > 2029 || path.includes('//') || /[\\#%\s\x00-\x1f]/.test(path) || segments.some(segment => !/^[A-Za-z0-9_.-]+$/.test(segment) || segment === '.' || segment === '..')) {
      throw new BackendError('Extension paths must be relative API routes without traversal or encoded segments.', 0, 'configuration');
    }
    if (this.useNative) {
      try {
        return await desktop.request<T>(payload === undefined ? 'GET' : 'POST', `/api/v1/extensions/${path}`, payload);
      } catch (failure) {
        const message = failure instanceof Error ? failure.message : typeof failure === 'string' ? failure : 'The desktop runtime request failed.';
        const status = Number(message.match(/HTTP (\d{3})\b/)?.[1] ?? 0);
        throw new BackendError(message, status, 'native', status === 403 ? 'An instance-admin runtime session is required for this operation.' : 'Check the desktop runtime status and backend diagnostics.');
      }
    }
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 60000);
    try {
      const response = await this.fetcher.call(globalThis, `${this.baseUrl}/api/v1/extensions/${path}`, {
        method: payload === undefined ? 'GET' : 'POST',
        headers: { Accept: 'application/json', ...(payload === undefined ? {} : { 'Content-Type': 'application/json' }), ...(this.token ? { Authorization: `Bearer ${this.token}` } : {}) },
        ...(payload === undefined ? {} : { body: JSON.stringify(payload) }), signal: controller.signal,
      });
      const length = Number(response.headers.get('content-length'));
      if (length > 8_000_000) throw new BackendError('Extension response exceeds the 8 MB budget.', response.status, 'budget');
      const text = await response.text();
      if (text.length > 8_000_000) throw new BackendError('Extension response exceeds the 8 MB budget.', response.status, 'budget');
      let data: unknown;
      try { data = JSON.parse(text); } catch { throw new BackendError('The extension returned invalid JSON.', response.status, 'protocol'); }
      if (!response.ok) {
        const value = data as { detail?: unknown; message?: string };
        const detail = typeof value.detail === 'object' && value.detail ? value.detail as { message?: string; action?: string } : {};
        const message = typeof value.detail === 'string' ? value.detail : detail.message ?? value.message ?? `Extension request failed (${response.status}).`;
        throw new BackendError(message, response.status, 'extension', detail.action ?? (response.status === 403 ? 'An instance-admin session is required for this operation.' : ''));
      }
      return data as T;
    } catch (failure) {
      if (failure instanceof BackendError) throw failure;
      if (controller.signal.aborted) throw new BackendError('The extension did not respond within 60 seconds.', 0, 'timeout');
      throw new BackendError(failure instanceof Error ? failure.message : 'Cannot reach the extension backend.', 0, 'connection', 'Check the backend URL, CORS policy and session token.');
    } finally { clearTimeout(timeout); }
  }
  capabilities(): Promise<ExtensionCapabilities> { return this.request('capabilities'); }
  axis(payload: { positive: number[][]; negative: number[][]; eval_positive?: number[][]; eval_negative?: number[][] }): Promise<AxisResponse> { return this.request('axis/evaluate', payload); }
  training(payload: { steps: number; rank: number; quantize_base: boolean; seed: number }): Promise<TrainingResponse> { return this.request('training/demo', payload); }
  index(payload: { text: string; source: string; collection: string }): Promise<{ collection: string; result: Record<string, unknown>; encrypted: boolean }> { return this.request('rag/index', payload); }
  search(payload: { query: string; collection: string; limit: number }): Promise<RagBackendResult> { return this.request('rag/search', payload); }
  recipe(payload: { model_name: string; gguf_path: string }): Promise<ExportRecipe> { return this.request('export/recipe', payload); }
}
