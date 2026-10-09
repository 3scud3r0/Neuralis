import {describe, expect, it} from 'vitest';
import {decodeStoredFrame, episodePath, policyBars, waitForIce} from './client';

describe('persisted arena transport', () => {
  it('keeps raw negative Q values and checks selected action against its vector', () => {
    const frame = {episode_id:'mario-1',step:12,policy:[-1.5,2,1],action:1,reward:3,frame_png:'AQID',metadata:{policy_kind:'q-values'},terminated:false,has_snapshot:false};
    expect(decodeStoredFrame(frame).policy).toEqual([-1.5,2,1]);
    expect(policyBars(frame.policy).map(item => item.fraction)).toEqual([.75,1,.5]);
    expect(() => decodeStoredFrame({...frame,action:3})).toThrow();
    expect(() => decodeStoredFrame({...frame,policy:[Infinity]})).toThrow();
    expect(() => decodeStoredFrame({...frame,frame_png:'javascript:bad'})).toThrow();
  });
  it('refuses unknown resource paths and handles an all-zero policy without NaN', () => {
    expect(episodePath('mario-record_1')).toBe('arena/episodes/mario-record_1');
    expect(() => episodePath('../private')).toThrow();
    expect(policyBars([0,0]).map(item => item.fraction)).toEqual([0,0]);
  });
  it('ICE waits resolve on completion and reject cancellation while releasing listeners', async () => {
    class Peer extends EventTarget {
      iceGatheringState = 'gathering';
      localDescription = {sdp:'v=0\r\na=candidate:1 1 udp 1 192.0.2.1 5000 typ host\r\n'};
    }
    const peer = new Peer();
    const controller = new AbortController();
    const pending = waitForIce(peer as unknown as RTCPeerConnection,controller.signal,100);
    peer.iceGatheringState = 'complete';peer.dispatchEvent(new Event('icegatheringstatechange'));
    await expect(pending).resolves.toBeUndefined();
    peer.iceGatheringState = 'gathering';
    const cancelled = waitForIce(peer as unknown as RTCPeerConnection,controller.signal,100);
    controller.abort();
    await expect(cancelled).rejects.toThrow('cancelled');
  });
  it('rejects already-complete gathering without candidates instead of waiting indefinitely for video', async () => {
    const peer = {iceGatheringState:'complete',localDescription:{sdp:'v=0\r\nm=video 9 UDP/TLS/RTP/SAVPF 96\r\na=recvonly\r\n'}};
    await expect(waitForIce(peer as unknown as RTCPeerConnection,new AbortController().signal)).rejects.toThrow('no ICE candidates');
  });
  it('rejects candidate-free completion and removes the gathering listener', async () => {
    class Peer extends EventTarget {
      iceGatheringState = 'gathering';
      localDescription = {sdp:'v=0\r\n'};
      listeners = 0;
      override addEventListener(...args: Parameters<EventTarget['addEventListener']>) {this.listeners++;super.addEventListener(...args);}
      override removeEventListener(...args: Parameters<EventTarget['removeEventListener']>) {this.listeners--;super.removeEventListener(...args);}
    }
    const peer = new Peer();
    const pending = waitForIce(peer as unknown as RTCPeerConnection,new AbortController().signal,100);
    peer.iceGatheringState = 'complete';peer.dispatchEvent(new Event('icegatheringstatechange'));
    await expect(pending).rejects.toThrow('managed browser policy');
    expect(peer.listeners).toBe(0);
  });
  it('accepts completed relay candidate gathering', async () => {
    const peer = {iceGatheringState:'complete',localDescription:{sdp:'v=0\r\na=candidate:1 1 udp 1 192.0.2.2 5000 typ relay\r\n'}};
    await expect(waitForIce(peer as unknown as RTCPeerConnection,new AbortController().signal)).resolves.toBeUndefined();
  });
});
