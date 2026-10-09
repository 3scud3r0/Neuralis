// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { FoundryPanel } from './FoundryPanel';
import { RagPanel } from './RagPanel';
import { ResearchWorkbench } from './ResearchWorkbench';

beforeEach(() => localStorage.clear());
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });
describe('research workbench controls', () => {
  it('offers desktop IPC only in the desktop app and hides the browser token after selection', async () => {
    const browser = render(<ResearchWorkbench initialTab="foundry"/>);
    fireEvent.click(screen.getByRole('button', { name: 'Connect backend' }));
    expect(screen.queryByRole('button', { name: 'Use desktop runtime' })).toBeNull();
    browser.unmount();
    const flags = Object.fromEntries(['axislab', 'nf4', 'lora', 'rag', 'webrtc', 'gguf'].map(key => [key, { available: false }]));
    const invoke = vi.fn(async (command: string) => command === 'backend_status' ? { phase: 'ready', endpoint: 'http://127.0.0.1:8010' } : { ...flags, backends: {} });
    vi.stubGlobal('__TAURI__', { core: { invoke } });
    const fetcher = vi.fn(async () => { throw new Error('Native connection must not use browser fetch'); });
    vi.stubGlobal('fetch', fetcher);
    render(<ResearchWorkbench initialTab="foundry"/>);
    fireEvent.click(screen.getByRole('button', { name: 'Connect backend' }));
    fireEvent.change(screen.getByLabelText('Extension session token'), { target: { value: 'browser-only-test-token' } });
    fireEvent.click(screen.getByRole('button', { name: 'Use desktop runtime' }));
    await waitFor(() => expect(screen.getByRole('button', { name: 'Desktop runtime connected' })).toBeTruthy());
    expect(screen.queryByLabelText('Extension session token')).toBeNull();
    expect((screen.getByLabelText('Extension backend URL') as HTMLInputElement).value).toBe('http://127.0.0.1:8010');
    expect(invoke).toHaveBeenCalledWith('backend_request', { method: 'GET', path: '/api/v1/extensions/capabilities', body: null });
    expect(JSON.stringify(invoke.mock.calls)).not.toContain('browser-only-test-token');
    expect(fetcher).not.toHaveBeenCalled();
  });
  it('rejects a corrupted answer in the JSONL editor and reports its line', () => {
    render(<FoundryPanel/>);
    const editor = screen.getByLabelText('Arithmetic JSONL editor') as HTMLTextAreaElement;
    const lines = editor.value.trim().split('\n');
    const first = JSON.parse(lines[0]);
    first.messages[1].content = '999999';
    lines[0] = JSON.stringify(first);
    fireEvent.change(editor, { target: { value: lines.join('\n') } });
    expect(screen.getByText('Changes not validated')).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Revalidate edited source' }));
    expect(screen.getByText('Prompt or answer does not agree with independently recomputed metadata.')).toBeTruthy();
    expect(screen.getByText('Review required')).toBeTruthy();
    expect(screen.getByText('23', { selector: 'strong' })).toBeTruthy();
  });
  it('adds user text, searches it and explicitly persists an unencrypted collection', () => {
    const rendered = render(<RagPanel client={null} capabilities={null}/>);
    fireEvent.change(screen.getByLabelText('Source / filename'), { target: { value: 'Lab notebook' } });
    fireEvent.change(screen.getByLabelText('RAG document text'), { target: { value: 'The spectrograph measures wavelength using a calibrated prism and optical detector.' } });
    fireEvent.change(screen.getByLabelText('Retrieval query'), { target: { value: 'spectrograph wavelength' } });
    fireEvent.click(screen.getByRole('button', { name: 'Add to browser index' }));
    expect(screen.getByText('The spectrograph measures wavelength using a calibrated prism and optical detector.')).toBeTruthy();
    expect(localStorage.getItem('neuralis.v1.browser-rag')).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: 'Save in browser' }));
    expect(JSON.parse(localStorage.getItem('neuralis.v1.browser-rag')!).documents).toHaveLength(4);
    rendered.unmount();
    render(<RagPanel client={null} capabilities={null}/>);
    expect(screen.getByText('Lab notebook')).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Clear saved collection' }));
    expect(localStorage.getItem('neuralis.v1.browser-rag')).toBeNull();
    expect(screen.getByText('Lab notebook')).toBeTruthy();
  });
});
