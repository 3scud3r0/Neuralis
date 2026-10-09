"""Optional real WebRTC video signaling; bounded RGB queue works without aiortc."""

import asyncio
from fractions import Fraction
import importlib.util
import time

import numpy as np


def capabilities() -> dict:
    available = importlib.util.find_spec("aiortc") is not None and importlib.util.find_spec("av") is not None
    return {"webrtc_available": available, "transport": "aiortc" if available else None,
            "requires": [] if available else ["aiortc", "av"], "latency_guarantee": None,
            "queue_policy": "drop oldest frames when full"}


class FrameQueue:
    """Use from one asyncio event loop; producer pushes actual captured frames."""

    def __init__(self, capacity: int = 3):
        if not 1 <= capacity <= 30:
            raise ValueError("Frame capacity must be between one and 30")
        self.queue: asyncio.Queue = asyncio.Queue(capacity)
        self.dropped = 0
        self.accepted = 0
        self.closed = False

    def push(self, frame: np.ndarray) -> None:
        if self.closed:
            raise RuntimeError("Frame queue is closed")
        frame = np.asarray(frame)
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Streaming requires an RGB uint8 image")
        if not 1 <= frame.shape[0] <= 1080 or not 1 <= frame.shape[1] <= 1920:
            raise ValueError("Frame exceeds the streaming resolution budget")
        if self.queue.full():
            self.queue.get_nowait()
            self.dropped += 1
        self.queue.put_nowait((frame.copy(), time.monotonic()))
        self.accepted += 1

    async def next(self) -> tuple[np.ndarray, float]:
        if self.closed and self.queue.empty():
            raise EOFError("Frame queue closed")
        result = await self.queue.get()
        if result is None:
            raise EOFError("Frame queue closed")
        return result

    def close(self) -> None:
        self.closed = True
        while not self.queue.empty():
            self.queue.get_nowait()
        self.queue.put_nowait(None)

    def statistics(self) -> dict:
        return {"accepted": self.accepted, "dropped": self.dropped, "queued": self.queue.qsize(),
                "capacity": self.queue.maxsize, "closed": self.closed}


def create_video_track(frames: FrameQueue, *, fps: int = 30):
    if not capabilities()["webrtc_available"]:
        raise RuntimeError("WebRTC requires optional aiortc and av packages")
    if not 1 <= fps <= 60:
        raise ValueError("Frame rate must be between one and 60")
    from aiortc import VideoStreamTrack
    from av import VideoFrame

    class CapturedVideoTrack(VideoStreamTrack):
        def __init__(self):
            super().__init__()
            self.started_at = None
            self.last_pts = -1

        async def recv(self):
            try:
                rgb, captured_at = await frames.next()
            except EOFError:
                self.stop()
                from aiortc.mediastreams import MediaStreamError
                raise MediaStreamError
            if self.started_at is None:
                self.started_at = captured_at
            pts = max(self.last_pts + 1, round((captured_at - self.started_at) * 90000))
            if self.last_pts >= 0:
                pts = max(pts, self.last_pts + round(90000 / fps))
            self.last_pts = pts
            frame = VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.pts = pts
            frame.time_base = Fraction(1, 90000)
            return frame

    return CapturedVideoTrack()


class WebRTCStreamer:
    """Peer ownership and HTTP authentication belong to the integration caller.

    Uses host ICE candidates unless the caller passes an aiortc configuration.
    NAT traversal may require user-configured STUN/TURN infrastructure.
    """

    def __init__(self, frames: FrameQueue, *, configuration=None, max_peers: int = 4):
        if not 1 <= max_peers <= 16:
            raise ValueError("Peer budget must be between one and 16")
        self.frames = frames
        self.configuration = configuration
        self.max_peers = max_peers
        self.peers = set()

    async def answer(self, offer: dict) -> dict:
        if not capabilities()["webrtc_available"]:
            raise RuntimeError("WebRTC requires optional aiortc and av packages")
        if len(self.peers) >= self.max_peers:
            raise RuntimeError("WebRTC peer budget exhausted")
        if offer.get("type") != "offer" or not isinstance(offer.get("sdp"), str) or len(offer["sdp"]) > 65536:
            raise ValueError("A bounded SDP offer is required")
        from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription
        from aiortc.contrib.media import MediaRelay
        if not hasattr(self, "_relay"):
            self._relay = MediaRelay()
            self._track = create_video_track(self.frames)
        configuration = self.configuration if self.configuration is not None else RTCConfiguration(iceServers=[])
        peer = RTCPeerConnection(configuration=configuration)
        self.peers.add(peer)

        @peer.on("connectionstatechange")
        async def state_change():
            if peer.connectionState in {"failed", "closed"}:
                self.peers.discard(peer)
                await peer.close()

        try:
            peer.addTrack(self._relay.subscribe(self._track, buffered=False))
            await asyncio.wait_for(peer.setRemoteDescription(RTCSessionDescription(**offer)), timeout=15)
            answer = await peer.createAnswer()
            await asyncio.wait_for(peer.setLocalDescription(answer), timeout=15)
            return {"type": peer.localDescription.type, "sdp": peer.localDescription.sdp}
        except Exception:
            self.peers.discard(peer)
            await peer.close()
            raise

    async def close(self) -> None:
        await asyncio.gather(*(peer.close() for peer in list(self.peers)))
        self.peers.clear()
        if hasattr(self, "_track"):
            self._track.stop()
