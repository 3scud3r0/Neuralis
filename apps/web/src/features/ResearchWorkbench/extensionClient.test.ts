import { afterEach, describe, expect, it, vi } from 'vitest';
import { ExtensionClient } from './extensionClient';
afterEach(() => vi.unstubAllGlobals());
describe('extension backend integration', () => {
  it('invokes the browser fetch function with its global receiver', async () => {
    const fetcher = function(this: unknown) {
      if (this !== globalThis) throw new TypeError('Illegal invocation');
      return Promise.resolve(new Response('{}'));
    } as typeof fetch;
    await expect(new ExtensionClient('http://127.0.0.1:8000', '', fetcher).capabilities()).resolves.toEqual({});
  });
  it('normalizes the base URL and passes the session token only in headers', async () => {
    const fetcher = vi.fn(async () => new Response(JSON.stringify({ losses: [2, 1] }), { status: 200, headers: { 'content-type': 'application/json' } }));
    const client = new ExtensionClient('http://127.0.0.1:8000/api/v1', 'session-secret', fetcher);
    await client.training({ steps: 2, rank: 1, seed: 42, quantize_base: true });
    const [url, options] = fetcher.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe('http://127.0.0.1:8000/api/v1/extensions/training/demo');
    expect(options.headers).toMatchObject({ Authorization: 'Bearer session-secret' });
    expect(JSON.parse(options.body as string)).toEqual({ steps: 2, rank: 1, seed: 42, quantize_base: true });
  });
  it('preserves server errors and the instance-admin action', async () => {
    const client = new ExtensionClient('http://127.0.0.1:8000', '', async () => new Response(JSON.stringify({ detail: 'Instance admin required' }), { status: 403 }));
    await expect(client.index({ text: 'test', source: 'source', collection: 'local' })).rejects.toMatchObject({ status: 403, message: 'Instance admin required', action: expect.stringContaining('instance-admin') });
  });
  it('rejects invalid JSON and responses above the capture budget', async () => {
    const malformed = new ExtensionClient('http://127.0.0.1:8000', '', async () => new Response('<html>error</html>'));
    await expect(malformed.capabilities()).rejects.toMatchObject({ category: 'protocol' });
    const oversized = new ExtensionClient('http://127.0.0.1:8000', '', async () => new Response('{}', { headers: { 'content-length': '9000000' } }));
    await expect(oversized.capabilities()).rejects.toMatchObject({ category: 'budget' });
  });
  it('routes desktop requests through IPC without sending JavaScript token headers', async () => {
    const invoke = vi.fn(async () => ({ losses: [0.4, 0.2] }));
    vi.stubGlobal('__TAURI__', { core: { invoke } });
    const fetcher = vi.fn(async () => { throw new Error('Browser fetch must not run for native requests'); });
    const client = new ExtensionClient('http://127.0.0.1:8000', 'unused-javascript-token', fetcher, true);
    const payload = { steps: 2, rank: 1, seed: 42, quantize_base: true };
    await expect(client.training(payload)).resolves.toEqual({ losses: [0.4, 0.2] });
    expect(client.useNative).toBe(true);
    expect(invoke).toHaveBeenCalledWith('backend_request', { method: 'POST', path: '/api/v1/extensions/training/demo', body: payload });
    expect(JSON.stringify(invoke.mock.calls)).not.toContain('unused-javascript-token');
    expect(fetcher).not.toHaveBeenCalled();
    await client.request('arena/episodes/episode-42/records/0?limit=10');
    expect(invoke).toHaveBeenLastCalledWith('backend_request', { method: 'GET', path: '/api/v1/extensions/arena/episodes/episode-42/records/0?limit=10', body: null });
  });
  it('reports native HTTP errors and never falls back to browser fetch', async () => {
    const invoke = vi.fn(async () => { throw 'Runtime error: Backend returned HTTP 403'; });
    vi.stubGlobal('__TAURI__', { core: { invoke } });
    const fetcher = vi.fn(async () => new Response('{}'));
    const client = new ExtensionClient('http://127.0.0.1:8000', '', fetcher, true);
    await expect(client.search({ query: 'local', collection: 'local', limit: 5 })).rejects.toMatchObject({ status: 403, category: 'native', action: expect.stringContaining('instance-admin') });
    expect(fetcher).not.toHaveBeenCalled();
  });
  it('rejects traversal and authority-like paths before either transport runs', async () => {
    const invoke = vi.fn(async () => ({}));
    vi.stubGlobal('__TAURI__', { core: { invoke } });
    const fetcher = vi.fn(async () => new Response('{}'));
    for (const native of [true, false]) {
      const client = new ExtensionClient('http://127.0.0.1:8000', '', fetcher, native);
      for (const path of ['../capabilities', 'axis/./evaluate', '/capabilities', '//evil.example', 'arena//episodes', 'arena/%2e%2e/secrets', 'arena\\episodes', 'capabilities#other', 'https://other.example']) {
        await expect(client.request(path)).rejects.toMatchObject({ category: 'configuration' });
      }
    }
    expect(invoke).not.toHaveBeenCalled();
    expect(fetcher).not.toHaveBeenCalled();
  });
});
