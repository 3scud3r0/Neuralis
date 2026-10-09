export interface ArenaClient {
  request<T>(path: string, payload?: unknown): Promise<T>;
}
export interface EpisodeSummary {
  episode_id: string; steps: number; return: number;
  first_step: number; last_step: number; terminated: boolean; snapshots: number;
}
export interface StoredFrame {
  episode_id: string; step: number; action: number; reward: number;
  terminated: boolean; has_snapshot: boolean; frame_png: string;
  policy: number[]; metadata: Record<string, unknown>;
}
export function episodePath(id: string): string {
  if (!/^[A-Za-z0-9_-]{1,100}$/.test(id)) throw new Error('Choose a valid recorded episode identifier.');
  return `arena/episodes/${encodeURIComponent(id)}`;
}
export function decodeStoredFrame(value: unknown): StoredFrame {
  const row = value as StoredFrame;
  if (!row || typeof row.episode_id !== 'string' || !Number.isInteger(row.step) || row.step < 0 ||
      !Array.isArray(row.policy) || !row.policy.length || row.policy.length > 4096 || row.policy.some(v => !Number.isFinite(v)) ||
      !Number.isInteger(row.action) || row.action < 0 || row.action >= row.policy.length || !Number.isFinite(row.reward) ||
      typeof row.frame_png !== 'string' || row.frame_png.length > 1400000 || !/^[A-Za-z0-9+/]*={0,2}$/.test(row.frame_png)) {
    throw new Error('The backend returned an invalid or oversized trajectory record.');
  }
  return row;
}
export function policyBars(values: number[]): {value: number; fraction: number}[] {
  if (!values.length || values.length > 4096 || values.some(value => !Number.isFinite(value))) throw new Error('Invalid policy values.');
  const scale = Math.max(1e-12, ...values.map(Math.abs));
  return values.map(value => ({value, fraction: Math.abs(value) / scale}));
}
export async function waitForIce(peer: RTCPeerConnection, signal: AbortSignal, milliseconds = 8000): Promise<void> {
  if (signal.aborted) throw new Error('Connection cancelled.');
  const candidateError = () => /(?:^|\r?\n)a=candidate:/.test(peer.localDescription?.sdp ?? '') ? undefined : new Error(
    'This browser gathered no ICE candidates. Its network or managed browser policy may block WebRTC UDP. Use a permitted browser/network or configure a TURN relay.',
  );
  if (peer.iceGatheringState === 'complete') {
    const error = candidateError();
    if (error) throw error;
    return;
  }
  return new Promise((resolve, reject) => {
    const finish = (error?: Error) => {
      clearTimeout(timer);
      signal.removeEventListener('abort', abort);
      peer.removeEventListener('icegatheringstatechange', changed);
      error ? reject(error) : resolve();
    };
    const changed = () => { if (peer.iceGatheringState === 'complete') finish(candidateError()); };
    const abort = () => finish(new Error('Connection cancelled.'));
    const timer = setTimeout(() => finish(new Error('Local ICE gathering timed out. Check your network adapter.')), milliseconds);
    peer.addEventListener('icegatheringstatechange', changed);
    signal.addEventListener('abort', abort, {once: true});
    changed();
  });
}
