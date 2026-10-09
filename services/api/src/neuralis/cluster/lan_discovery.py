"""Optional mDNS discovery. Advertisements never establish authentication."""

from dataclasses import asdict, dataclass
import ipaddress
import re
import socket
import threading
import time

from neuralis.runtime.acceleration.backends import backend_capabilities


SERVICE_TYPE = "_neuralis._tcp.local."


def _zeroconf():
    try:
        import zeroconf
    except ImportError as error:
        raise RuntimeError("mDNS discovery is optional; install zeroconf or configure worker host/port explicitly") from error
    return zeroconf


@dataclass(frozen=True)
class DiscoveredWorker:
    name: str
    addresses: tuple[str, ...]
    port: int
    protocol: str
    backends: tuple[str, ...]
    last_seen: float


class LANDiscovery:
    def __init__(self, *, max_workers: int = 128):
        if not 1 <= max_workers <= 1024:
            raise ValueError("max_workers must be between 1 and 1024")
        zc = _zeroconf()
        self._zc_module = zc
        self._zc = zc.Zeroconf()
        self.max_workers = max_workers
        self._workers: dict[str, DiscoveredWorker] = {}
        self._lock = threading.RLock()
        self._advertisement = None
        self._closed = False
        self._browser = zc.ServiceBrowser(self._zc, SERVICE_TYPE, self)

    def advertise(self, name: str, port: int, addresses: list[str]) -> None:
        if self._closed:
            raise RuntimeError("Discovery is closed")
        if self._advertisement is not None:
            raise ValueError("This discovery instance is already advertising")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,48}", name) or not 1 <= port <= 65535:
            raise ValueError("Worker name or port is invalid")
        if not addresses or len(addresses) > 8:
            raise ValueError("Provide between 1 and 8 LAN addresses")
        packed = [ipaddress.ip_address(address).packed for address in addresses]
        capabilities = backend_capabilities()["backends"]
        backends = ",".join(key for key, value in capabilities.items() if value["available"])
        info = self._zc_module.ServiceInfo(SERVICE_TYPE, name + "." + SERVICE_TYPE,
                                          addresses=packed, port=port,
                                          properties={"protocol": "neuralis-lan-v1", "backends": backends},
                                          server=name + ".local.")
        self._zc.register_service(info)
        self._advertisement = info

    def add_service(self, zeroconf, service_type, name):
        self.update_service(zeroconf, service_type, name)

    def update_service(self, zeroconf, service_type, name):
        if self._closed or len(name) > 256:
            return
        info = zeroconf.get_service_info(service_type, name, timeout=500)
        if info is None or not 1 <= info.port <= 65535:
            return
        protocol = info.properties.get(b"protocol", b"").decode("utf-8", errors="replace")
        if protocol != "neuralis-lan-v1":
            return
        advertised = info.properties.get(b"backends", b"").decode("utf-8", errors="replace")
        backends = tuple(value for value in advertised.split(",") if value in {"cpu", "cuda", "rocm", "mps"})
        addresses = tuple(info.parsed_addresses()[:8])
        with self._lock:
            if name not in self._workers and len(self._workers) >= self.max_workers:
                oldest = min(self._workers, key=lambda key: self._workers[key].last_seen)
                del self._workers[oldest]
            self._workers[name] = DiscoveredWorker(name, addresses, info.port, protocol, backends, time.monotonic())

    def remove_service(self, zeroconf, service_type, name):
        with self._lock:
            self._workers.pop(name, None)

    def workers(self, *, max_age: float = 120) -> list[dict]:
        if max_age <= 0:
            raise ValueError("max_age must be positive")
        cutoff = time.monotonic() - max_age
        with self._lock:
            return [asdict(worker) for worker in self._workers.values() if worker.last_seen >= cutoff]

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._browser.cancel()
        if self._advertisement is not None:
            self._zc.unregister_service(self._advertisement)
        self._zc.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def local_lan_addresses() -> list[str]:
    """Return hostname-resolved addresses; users still choose interface bindings."""
    return sorted({entry[4][0] for entry in socket.getaddrinfo(socket.gethostname(), None)
                   if not ipaddress.ip_address(entry[4][0]).is_loopback})[:8]
