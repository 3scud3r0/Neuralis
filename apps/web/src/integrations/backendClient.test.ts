import { describe, expect, it, vi, afterEach } from 'vitest';
import { BackendClient, BackendError, decodeEvent, validateBackendUrl } from './backendClient';

afterEach(() => vi.unstubAllGlobals());

describe('remote backend contracts', () => {
  it('normalizes API roots without accepting embedded credentials', () => {
    expect(validateBackendUrl('http://127.0.0.1:8000/api/v1/')).toBe('http://127.0.0.1:8000');
    expect(() => validateBackendUrl('https://user:secret@example.com')).toThrow(BackendError);
    expect(() => validateBackendUrl('file:///tmp/backend')).toThrow(BackendError);
    expect(() => validateBackendUrl('https://example.com?token=x')).toThrow(BackendError);
  });
  it('rejects malformed, negative and future-protocol telemetry', () => {
    const event = { protocol_version: '1.0', run_id: 'run-1', sequence: 1, timestamp: 1, step: 2, type: 'metrics', payload: { loss: 0.3 } };
    expect(decodeEvent(event).payload.loss).toBe(0.3);
    expect(() => decodeEvent({ ...event, sequence: -1 })).toThrow();
    expect(() => decodeEvent({ ...event, protocol_version: '2.0' })).toThrow();
    expect(() => decodeEvent({ ...event, payload: [] })).toThrow();
  });
  it('reports real capabilities and latency after a successful handshake', async () => {
    const fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify({ protocol_version: '1.0', device: 'cpu' }), { status: 200 }));
    vi.stubGlobal('fetch', fetch);
    const client = new BackendClient('http://localhost:8000');
    const capabilities = await client.connect();
    expect(capabilities.device).toBe('cpu');
    expect(client.getConnection().state).toBe('connected');
    expect(fetch.mock.calls[0][0]).toBe('http://localhost:8000/api/v1/capabilities');
    client.dispose();
    expect(client.getConnection().state).toBe('disconnected');
  });
  it('does not advertise a successful connection when the server refuses access', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: { category: 'authorization', message: 'No project access', action: 'Request project access' } }), { status: 403 })));
    const client = new BackendClient('https://lab.example');
    await expect(client.connect()).rejects.toMatchObject({ status: 403, category: 'authorization' });
    expect(client.getConnection().state).toBe('failed');
    expect(client.getConnection().capabilities).toBeNull();
  });
  it('uses command identifiers and keeps credentials out of request paths', async () => {
    const fetch = vi.fn().mockResolvedValue(new Response('{}', { status: 200 }));
    vi.stubGlobal('fetch', fetch);
    const client = new BackendClient('https://lab.example', 'session-token');
    await client.command('a/b', 'pause');
    const [url, options] = fetch.mock.calls[0];
    expect(url).toBe('https://lab.example/api/v1/runs/a%2Fb/commands');
    const body = JSON.parse(options.body);
    expect(body.command).toBe('pause');
    expect(body.command_id.length).toBeGreaterThan(10);
    expect(options.headers.Authorization).toBe('Bearer session-token');
    expect(url).not.toContain('session-token');
  });
});
