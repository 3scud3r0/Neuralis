"""Bounded, authenticated split forward/backward over persistent TCP connections.

The wire format is length-prefixed JSON metadata + tensor bytes + HMAC. It never
unpickles network data. HMAC authenticates peers and frames, but does not encrypt
activations: use a trusted LAN or a separately configured VPN/TLS tunnel.
"""

from dataclasses import dataclass
import hmac
import json
import math
import secrets
import socket
import socketserver
import struct
import threading
import time
from typing import Callable
import uuid

import torch
from torch import nn


MAGIC = b"NLP1"
MAX_HEADER = 65536
DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32,
          "float64": torch.float64, "int64": torch.int64, "int32": torch.int32,
          "uint8": torch.uint8, "bool": torch.bool}


class ProtocolError(ValueError):
    pass


class AuthenticationError(ProtocolError):
    pass


def _secret(value: str | bytes) -> bytes:
    value = value.encode("utf-8") if isinstance(value, str) else value
    if not isinstance(value, bytes) or not 32 <= len(value) <= 4096:
        raise ValueError("LAN authentication secret must contain 32 to 4096 bytes")
    return value


def _recv_exact(connection: socket.socket, size: int, deadline: float) -> bytes:
    data = bytearray()
    while len(data) < size:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("LAN receive deadline exceeded")
        connection.settimeout(remaining)
        part = connection.recv(size - len(data))
        if not part:
            raise ConnectionError("LAN peer disconnected")
        data.extend(part)
    return bytes(data)


def _encode(metadata: dict, tensors: dict[str, torch.Tensor], sequence: int, max_bytes: int) -> bytes:
    descriptors, chunks = [], []
    raw_size = 0
    if len(tensors) > 8:
        raise ProtocolError("At most eight tensors are allowed per frame")
    for name, tensor in tensors.items():
        dtype = str(tensor.dtype).removeprefix("torch.")
        if not isinstance(name, str) or len(name) > 64 or dtype not in DTYPES or tensor.ndim > 8:
            raise ProtocolError("Invalid tensor name, dtype or dimensionality")
        size = tensor.numel() * tensor.element_size()
        raw_size += size
        if raw_size > max_bytes - 36:
            raise ProtocolError("Tensor payload exceeds the frame budget")
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ProtocolError("Nonfinite tensors are not allowed on the LAN pipeline")
        raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        descriptors.append({"name": name, "dtype": dtype, "shape": list(tensor.shape), "bytes": size})
        chunks.append(raw)
    header = json.dumps({"sequence": sequence, "metadata": metadata, "tensors": descriptors},
                        separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(header) > MAX_HEADER or 4 + len(header) + raw_size + 32 > max_bytes:
        raise ProtocolError("Frame exceeds its header or payload budget")
    return struct.pack("!I", len(header)) + header + b"".join(chunks)


def _decode(body: bytes, sequence: int) -> tuple[dict, dict[str, torch.Tensor]]:
    if len(body) < 4:
        raise ProtocolError("Truncated frame")
    header_size = struct.unpack("!I", body[:4])[0]
    if header_size > MAX_HEADER or header_size > len(body) - 4:
        raise ProtocolError("Invalid metadata length")
    try:
        header = json.loads(body[4:4 + header_size])
    except (ValueError, UnicodeDecodeError, RecursionError) as error:
        raise ProtocolError("Invalid frame metadata") from error
    if not isinstance(header, dict) or type(header.get("sequence")) is not int or header.get("sequence") != sequence:
        raise ProtocolError("Invalid frame sequence (replayed or reordered frame)")
    descriptors, metadata = header.get("tensors"), header.get("metadata")
    if not isinstance(descriptors, list) or len(descriptors) > 8 or not isinstance(metadata, dict):
        raise ProtocolError("Invalid frame structure")
    offset, tensors = 4 + header_size, {}
    for description in descriptors:
        if not isinstance(description, dict):
            raise ProtocolError("Invalid tensor descriptor")
        name, dtype, shape, size = (description.get(key) for key in ("name", "dtype", "shape", "bytes"))
        if not isinstance(name, str) or len(name) > 64 or name in tensors or not isinstance(dtype, str) or dtype not in DTYPES:
            raise ProtocolError("Invalid or duplicate tensor name/dtype")
        if not isinstance(shape, list) or len(shape) > 8 or any(type(n) is not int or n < 0 for n in shape):
            raise ProtocolError("Invalid tensor shape")
        expected = math.prod(shape) * torch.empty((), dtype=DTYPES[dtype]).element_size()
        if type(size) is not int or size != expected or offset + size > len(body):
            raise ProtocolError("Tensor byte count does not match shape")
        if size:
            tensor = torch.frombuffer(bytearray(body[offset:offset + size]), dtype=DTYPES[dtype]).clone().reshape(shape)
        else:
            tensor = torch.empty(shape, dtype=DTYPES[dtype])
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ProtocolError("Nonfinite tensor payload")
        tensors[name] = tensor
        offset += size
    if offset != len(body):
        raise ProtocolError("Trailing frame bytes")
    return metadata, tensors


class _Channel:
    def __init__(self, connection: socket.socket, token: bytes, challenge: bytes, *,
                 server: bool, max_bytes: int, timeout: float):
        self.connection, self.token, self.challenge = connection, token, challenge
        self.send_direction = b"server" if server else b"client"
        self.receive_direction = b"client" if server else b"server"
        self.max_bytes, self.timeout = max_bytes, timeout
        self.sent = self.received = 0

    def send(self, metadata: dict, tensors: dict[str, torch.Tensor] | None = None):
        body = _encode(metadata, tensors or {}, self.sent, self.max_bytes)
        signature = hmac.digest(self.token, self.challenge + self.send_direction + body, "sha256")
        frame = body + signature
        self.connection.settimeout(self.timeout)
        self.connection.sendall(struct.pack("!I", len(frame)) + frame)
        self.sent += 1

    def receive(self):
        deadline = time.monotonic() + self.timeout
        size = struct.unpack("!I", _recv_exact(self.connection, 4, deadline))[0]
        if size < 36 or size > self.max_bytes:
            raise ProtocolError("Frame length exceeds the configured budget")
        frame = _recv_exact(self.connection, size, deadline)
        body, signature = frame[:-32], frame[-32:]
        expected = hmac.digest(self.token, self.challenge + self.receive_direction + body, "sha256")
        if not hmac.compare_digest(signature, expected):
            raise AuthenticationError("Invalid frame authentication")
        result = _decode(body, self.received)
        self.received += 1
        return result


@dataclass
class _Pending:
    owner: str
    inputs: torch.Tensor
    outputs: torch.Tensor
    created: float


class PipelineStage:
    """One locally constructed stage, owned by one training connection at a time.

    Multiple microbatches may be queued within activation budgets. Backward
    accumulates local parameter gradients; step is legal only once all outstanding
    graphs have been consumed. Expired/disconnected leases discard pending work.
    """

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer | None = None, *,
                 max_pending: int = 4, max_activation_bytes: int = 64 * 1024 * 1024,
                 lease_seconds: float = 60, max_grad_norm: float = 1.0):
        if not 1 <= max_pending <= 64 or not 1024 <= max_activation_bytes <= 1024 * 1024 * 1024:
            raise ValueError("Invalid stage activation budget")
        if not 0 < lease_seconds <= 3600 or not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
            raise ValueError("Invalid stage lease or gradient clipping configuration")
        self.model, self.optimizer = model, optimizer
        self.parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if optimizer is not None:
            optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
            if optimized != {id(p) for p in self.parameters}:
                raise ValueError("Stage optimizer must own exactly the model's trainable parameters")
        self.device = next(iter(model.parameters()), next(iter(model.buffers()), torch.empty(0))).device
        self.max_pending, self.max_activation_bytes = max_pending, max_activation_bytes
        self.lease_seconds, self.max_grad_norm = lease_seconds, max_grad_norm
        self.pending: dict[str, _Pending] = {}
        self.owner: str | None = None
        self.last_active = 0.0
        self.backward_count = 0
        self.steps = 0
        self.lock = threading.RLock()

    def _expire(self):
        if self.owner and time.monotonic() - self.last_active > self.lease_seconds:
            self.abort(self.owner)

    def _claim(self, owner: str):
        self._expire()
        if self.owner is not None and self.owner != owner:
            raise ProtocolError("Stage is leased by another training connection")
        self.owner = owner
        self.last_active = time.monotonic()

    def _pending_bytes(self):
        return sum(item.inputs.numel() * item.inputs.element_size() + item.outputs.numel() * item.outputs.element_size()
                   for item in self.pending.values())

    def forward(self, owner: str, inputs: torch.Tensor, *, training: bool = False):
        with self.lock:
            self._expire()
            if self.owner is not None and self.owner != owner:
                raise ProtocolError("Stage is leased by another training connection")
            if inputs.numel() * inputs.element_size() > self.max_activation_bytes:
                raise ProtocolError("Stage input exceeds the activation budget")
            if training and len(self.pending) >= self.max_pending:
                raise ProtocolError("Stage pending microbatch budget reached")
            previous = self.model.training
            self.model.train(training)
            try:
                with torch.set_grad_enabled(training):
                    leaf = inputs.detach().to(self.device).clone()
                    if training and leaf.is_floating_point():
                        leaf.requires_grad_(True)
                    output = self.model(leaf)
                if not isinstance(output, torch.Tensor):
                    raise ProtocolError("Pipeline stages must return a single tensor")
                if not bool(torch.isfinite(output).all()):
                    raise ProtocolError("Stage produced nonfinite activations")
                size = leaf.numel() * leaf.element_size() + output.numel() * output.element_size()
                if size + self._pending_bytes() > self.max_activation_bytes:
                    raise ProtocolError("Stage output exceeds the activation budget")
                ticket = None
                if training:
                    if not output.requires_grad:
                        raise ProtocolError("Training stage output has no differentiable path")
                    self._claim(owner)
                    ticket = uuid.uuid4().hex
                    self.pending[ticket] = _Pending(owner, leaf, output, time.monotonic())
                return output.detach().cpu(), ticket
            finally:
                self.model.train(previous)

    def backward(self, owner: str, ticket: str, gradient: torch.Tensor):
        with self.lock:
            self._expire()
            item = self.pending.get(ticket)
            if item is None or item.owner != owner:
                raise ProtocolError("Unknown, expired or foreign forward ticket")
            if gradient.shape != item.outputs.shape or gradient.dtype != item.outputs.dtype:
                raise ProtocolError("Output gradient must match the forward output shape and dtype")
            if not bool(torch.isfinite(gradient).all()):
                raise ProtocolError("Output gradient must be finite")
            self._claim(owner)
            differentiable = ([item.inputs] if item.inputs.requires_grad else []) + self.parameters
            try:
                gradients = torch.autograd.grad(item.outputs, differentiable, gradient.to(self.device), allow_unused=True)
                if any(value is not None and not bool(torch.isfinite(value).all()) for value in gradients):
                    raise ProtocolError("Stage produced nonfinite gradients")
                input_gradient = gradients[0] if item.inputs.requires_grad else None
                parameter_gradients = gradients[1:] if item.inputs.requires_grad else gradients
                for parameter, value in zip(self.parameters, parameter_gradients, strict=True):
                    if value is not None:
                        if parameter.grad is None:
                            parameter.grad = value.detach()
                        else:
                            parameter.grad.add_(value.detach())
                self.backward_count += 1
                del self.pending[ticket]
                return input_gradient.detach().cpu() if input_gradient is not None else None
            except BaseException:
                self.abort(owner)
                raise

    def step(self, owner: str) -> dict:
        with self.lock:
            self._expire()
            if self.owner != owner or not self.backward_count:
                raise ProtocolError("No completed backward work belongs to this connection")
            if self.pending:
                raise ProtocolError("Consume every outstanding forward graph before stepping")
            norm = None
            if self.optimizer is not None:
                norm = float(nn.utils.clip_grad_norm_(self.parameters, self.max_grad_norm, error_if_nonfinite=True))
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.steps += 1
            else:
                for parameter in self.parameters:
                    parameter.grad = None
            count = self.backward_count
            self.backward_count = 0
            self.owner = None
            return {"optimizer_step": self.steps, "microbatches": count, "gradient_norm": norm,
                    "optimizer_applied": self.optimizer is not None}

    def abort(self, owner: str):
        with self.lock:
            if self.owner == owner:
                self.pending.clear()
                for parameter in self.parameters:
                    parameter.grad = None
                self.owner = None
                self.backward_count = 0


class _BoundedTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True
    block_on_close = False

    def process_request(self, request, client_address):
        if not self.slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()


class _Handler(socketserver.BaseRequestHandler):
    def handle(self):
        server = self.server
        owner = uuid.uuid4().hex
        connection = self.request
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        server.connections.add(connection)
        try:
            challenge = secrets.token_bytes(32)
            connection.settimeout(server.timeout_seconds)
            connection.sendall(MAGIC + challenge)
            proof = _recv_exact(connection, 32, time.monotonic() + server.timeout_seconds)
            if not hmac.compare_digest(proof, hmac.digest(server.token, b"client" + challenge, "sha256")):
                return
            connection.sendall(hmac.digest(server.token, b"server" + challenge, "sha256"))
            channel = _Channel(connection, server.token, challenge, server=True,
                               max_bytes=server.max_bytes, timeout=server.timeout_seconds)
            while True:
                metadata, tensors = channel.receive()
                try:
                    command = metadata.get("command")
                    if command == "forward":
                        if set(tensors) != {"inputs"} or type(metadata.get("training", False)) is not bool:
                            raise ProtocolError("Forward requires one input tensor and a boolean training flag")
                        outputs, ticket = server.stage.forward(owner, tensors["inputs"], training=metadata.get("training", False))
                        channel.send({"ok": True, "ticket": ticket}, {"outputs": outputs})
                    elif command == "backward":
                        ticket = metadata.get("ticket")
                        if not isinstance(ticket, str) or len(ticket) != 32 or set(tensors) != {"gradient"}:
                            raise ProtocolError("Backward requires a forward ticket and one output gradient")
                        result = server.stage.backward(owner, ticket, tensors["gradient"])
                        channel.send({"ok": True}, {"input_gradient": result} if result is not None else {})
                    elif command == "step" and not tensors:
                        channel.send({"ok": True, "result": server.stage.step(owner)})
                    elif command == "abort" and not tensors:
                        server.stage.abort(owner)
                        channel.send({"ok": True})
                    elif command == "status" and not tensors:
                        with server.stage.lock:
                            server.stage._expire()
                            channel.send({"ok": True, "result": {"optimizer_step": server.stage.steps,
                                         "pending_microbatches": len(server.stage.pending),
                                         "training_leased": server.stage.owner is not None}})
                    else:
                        raise ProtocolError("Unsupported pipeline command")
                except (ValueError, RuntimeError) as error:
                    # A failed response serialization must not retain a graph
                    # for a ticket that the coordinator never received.
                    server.stage.abort(owner)
                    channel.send({"ok": False, "error": str(error)[:512]})
        except (ConnectionError, OSError, TimeoutError, ProtocolError):
            pass
        finally:
            server.stage.abort(owner)
            server.connections.discard(connection)


class PipelineServer:
    def __init__(self, stage: PipelineStage, token: str | bytes, *, host: str = "127.0.0.1", port: int = 0,
                 max_frame_bytes: int = 16 * 1024 * 1024, max_connections: int = 8, timeout: float = 15):
        if not 1024 <= max_frame_bytes <= 64 * 1024 * 1024 or not 1 <= max_connections <= 64:
            raise ValueError("Invalid LAN frame/connection budget")
        if not 0 < timeout <= 300:
            raise ValueError("LAN timeout must be between zero and 300 seconds")
        secret = _secret(token)
        self._server = _BoundedTCPServer((host, port), _Handler)
        self._server.stage, self._server.token = stage, secret
        self._server.max_bytes, self._server.timeout_seconds = max_frame_bytes, timeout
        self._server.slots = threading.BoundedSemaphore(max_connections)
        self._server.connections = set()
        self._thread = None

    @property
    def address(self) -> tuple[str, int]:
        return self._server.server_address

    def start(self):
        if self._thread is not None:
            raise RuntimeError("Pipeline server is already started")
        self._thread = threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
        self._thread.start()
        return self

    def close(self):
        if self._thread is not None:
            self._server.shutdown()
            self._thread.join(timeout=2)
        for connection in tuple(self._server.connections):
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self._server.server_close()

    def __enter__(self):
        return self.start()

    def __exit__(self, *args):
        self.close()


class PipelineClient:
    def __init__(self, host: str, port: int, token: str | bytes, *,
                 timeout: float = 15, max_frame_bytes: int = 16 * 1024 * 1024):
        if not 0 < timeout <= 300 or not 1024 <= max_frame_bytes <= 64 * 1024 * 1024:
            raise ValueError("Invalid client timeout or frame budget")
        token = _secret(token)
        connection = socket.create_connection((host, port), timeout=timeout)
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            deadline = time.monotonic() + timeout
            hello = _recv_exact(connection, 36, deadline)
            if hello[:4] != MAGIC:
                raise ProtocolError("Peer is not a Neuralis LAN pipeline stage")
            challenge = hello[4:]
            connection.sendall(hmac.digest(token, b"client" + challenge, "sha256"))
            proof = _recv_exact(connection, 32, deadline)
            if not hmac.compare_digest(proof, hmac.digest(token, b"server" + challenge, "sha256")):
                raise AuthenticationError("Worker authentication failed")
        except BaseException:
            connection.close()
            raise
        self.channel = _Channel(connection, token, challenge, server=False,
                                max_bytes=max_frame_bytes, timeout=timeout)
        self.lock = threading.RLock()
        self.closed = False

    def _request(self, metadata: dict, tensors: dict | None = None):
        with self.lock:
            if self.closed:
                raise ConnectionError("Pipeline client is closed")
            try:
                self.channel.send(metadata, tensors)
                result, values = self.channel.receive()
            except (OSError, ConnectionError, ProtocolError):
                self.close()
                raise
            if not result.get("ok"):
                raise ProtocolError(result.get("error", "Pipeline request failed"))
            return result, values

    def forward(self, inputs: torch.Tensor, *, training: bool = False) -> tuple[torch.Tensor, str | None]:
        metadata, tensors = self._request({"command": "forward", "training": training}, {"inputs": inputs})
        return tensors["outputs"], metadata.get("ticket")

    def backward(self, ticket: str, gradient: torch.Tensor) -> torch.Tensor | None:
        _, tensors = self._request({"command": "backward", "ticket": ticket}, {"gradient": gradient})
        return tensors.get("input_gradient")

    def step(self) -> dict:
        return self._request({"command": "step"})[0]["result"]

    def status(self) -> dict:
        return self._request({"command": "status"})[0]["result"]

    def abort(self):
        self._request({"command": "abort"})

    def close(self):
        if not self.closed:
            self.closed = True
            try:
                self.channel.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.channel.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class LANPipeline:
    """Sequential stage coordinator with real split autograd.

    Gradients are passed in reverse stage order, then each local optimizer steps.
    Steps across machines are not atomic. A network failure during stepping can
    partially apply an update; callers must restore their coordinated checkpoints
    before retrying. This implementation does not auto-retry a training request.
    """

    def __init__(self, stages: list[PipelineClient]):
        if not 1 <= len(stages) <= 16 or len({id(stage) for stage in stages}) != len(stages):
            raise ValueError("Pipeline needs between 1 and 16 distinct stage clients")
        self.stages = stages
        self.lock = threading.RLock()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        with self.lock:
            for stage in self.stages:
                inputs, _ = stage.forward(inputs)
            return inputs

    def train_batch(self, inputs: torch.Tensor, targets: torch.Tensor, loss_fn: Callable) -> dict:
        with self.lock:
            tickets = []
            try:
                for stage in self.stages:
                    inputs, ticket = stage.forward(inputs, training=True)
                    tickets.append(ticket)
                outputs = inputs.detach().requires_grad_(True)
                loss = loss_fn(outputs, targets.to(outputs.device))
                if loss.ndim != 0 or not bool(torch.isfinite(loss)):
                    raise ValueError("Pipeline loss must be a finite scalar")
                gradient, = torch.autograd.grad(loss, outputs)
                for stage, ticket in reversed(list(zip(self.stages, tickets, strict=True))):
                    if gradient is None:
                        raise ProtocolError("Pipeline gradient path was interrupted between stages")
                    gradient = stage.backward(ticket, gradient)
                steps = [stage.step() for stage in self.stages]
                return {"loss": float(loss.detach()), "stages": steps}
            except BaseException:
                for stage in self.stages:
                    try:
                        stage.abort()
                    except (OSError, ConnectionError, ProtocolError):
                        pass
                raise

    def close(self):
        for stage in self.stages:
            stage.close()
