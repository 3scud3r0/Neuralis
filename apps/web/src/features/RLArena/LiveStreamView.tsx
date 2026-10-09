import { useEffect, useRef, useState } from 'react';
import { Radio, RefreshCw, Square, Video } from 'lucide-react';
import { episodePath, waitForIce, type ArenaClient, type EpisodeSummary } from './client';
import './arena.css';

export function LiveStreamView({client}: {client: ArenaClient | null}) {
  const [episodes, setEpisodes] = useState<EpisodeSummary[]>([]);
  const [episode, setEpisode] = useState('');
  const [status, setStatus] = useState('Disconnected');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const video = useRef<HTMLVideoElement>(null);
  const connection = useRef<RTCPeerConnection | null>(null);
  const pending = useRef<AbortController | null>(null);
  const activeEpisode = useRef<string | null>(null);
  const activeClient = useRef<ArenaClient | null>(null);
  const mounted = useRef(true);
  const stop = () => {
    pending.current?.abort();
    pending.current = null;
    connection.current?.close();
    connection.current = null;
    if (video.current) video.current.srcObject = null;
    const identifier = activeEpisode.current;
    const backend = activeClient.current;
    activeEpisode.current = null;
    activeClient.current = null;
    if (identifier && backend) void backend.request(`${episodePath(identifier)}/webrtc/close`, {}).catch(() => {});
    if (mounted.current) {setStatus('Disconnected');setBusy(false);}
  };
  useEffect(() => {
    mounted.current = true;
    return () => {mounted.current = false;stop();};
  }, []); // The live session holds its original backend until explicitly stopped.
  useEffect(() => {stop();setEpisodes([]);setEpisode('');}, [client]);
  const refresh = async () => {
    if (!client) return;
    setBusy(true);setError('');
    try {
      const values = await client.request<EpisodeSummary[]>('arena/episodes');
      if (!Array.isArray(values)) throw new Error('Invalid episode list.');
      setEpisodes(values);setEpisode(current => values.some(item => item.episode_id === current) ? current : values[0]?.episode_id ?? '');
    } catch (failure) {setError(failure instanceof Error ? failure.message : String(failure));}
    finally {setBusy(false);}
  };
  const connect = async () => {
    if (!client || !episode) return;
    stop();setBusy(true);setError('');setStatus('Negotiating');
    const controller = new AbortController();pending.current = controller;
    let peer: RTCPeerConnection | null = null;
    try {
      if (typeof RTCPeerConnection === 'undefined') throw new Error('This browser does not support WebRTC.');
      peer = new RTCPeerConnection({iceServers: []});
      connection.current = peer;
      activeEpisode.current = episode;activeClient.current = client;
      peer.ontrack = event => {
        if (video.current) {
          video.current.srcObject = new MediaStream([event.track]);
          void video.current.play().catch(() => { if (mounted.current) setStatus('Video received; click Play in the video player'); });
        }
      };
      peer.onconnectionstatechange = () => {if (mounted.current && peer === connection.current) setStatus(peer?.connectionState ?? 'Disconnected');};
      peer.addTransceiver('video', {direction: 'recvonly'});
      await peer.setLocalDescription(await peer.createOffer());
      await waitForIce(peer, controller.signal);
      const answer = await client.request<RTCSessionDescriptionInit>(`${episodePath(episode)}/webrtc`, {
        type: 'offer', sdp: peer.localDescription?.sdp,
      });
      if (controller.signal.aborted || peer !== connection.current) return;
      await peer.setRemoteDescription(answer);
      setStatus('Waiting for captured frames');
    } catch (failure) {
      if (!controller.signal.aborted && mounted.current) setError(failure instanceof Error ? failure.message : String(failure));
      if (peer === connection.current) stop();
    } finally {if (mounted.current) setBusy(false);}
  };
  return <section className="arena-runtime-card">
    <div className="arena-runtime-heading"><div><span className="eyebrow">LOCAL EMULATOR TRANSPORT</span><h3><Video size={18}/>Live captured video</h3></div><span className="arena-runtime-badge"><Radio size={12}/>{status}</span></div>
    <p>Connect the local API, select an episode, then start the Mario recorder with the same episode ID. WebRTC carries captured pixels; action evidence is persisted separately.</p>
    {!client && <p className="arena-runtime-empty">Connect a research backend above to enable live video.</p>}
    <div className="arena-runtime-controls"><button disabled={!client || busy} onClick={() => void refresh()}><RefreshCw size={14}/>Refresh episodes</button><select aria-label="Live episode" disabled={!!connection.current} value={episode} onChange={event => setEpisode(event.target.value)}><option value="">Select an episode</option>{episodes.map(item => <option key={item.episode_id} value={item.episode_id}>{item.episode_id} · {item.steps} records</option>)}</select><button disabled={!client || !episode || busy || !!connection.current} onClick={() => void connect()}>Connect video</button><button disabled={!connection.current} onClick={stop}><Square size={13}/>Stop</button></div>
    <video className="arena-live-video" ref={video} controls autoPlay muted playsInline aria-label="Captured emulator WebRTC video"/>
    {error && <p role="alert" className="arena-runtime-error">{error}</p>}
    <p className="arena-runtime-note">Requires the API streaming extra. Host ICE candidates support reachable peers; routed networks may need a configured TURN service. Encoding, throughput and latency depend on hardware and network conditions.</p>
  </section>;
}
