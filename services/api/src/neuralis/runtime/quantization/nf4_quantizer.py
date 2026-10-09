"""Portable block NF4 storage. Linear execution reconstructs floating weights.

This reference implementation is useful on CPU and MPS. It is not a fused
four-bit GEMM and makes no throughput or training-memory claim.
"""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


NF4_VALUES = (-1.0, -0.6961928009986877, -0.5250730514526367,
              -0.39491748809814453, -0.28444138169288635, -0.18477343022823334,
              -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
              0.24611230194568634, 0.33791524171829224, 0.44070982933044434,
              0.5626170039176941, 0.7229568362236023, 1.0)
MAX_ELEMENTS = 100_000_000
CHUNK_ELEMENTS = 1_048_576


@dataclass(frozen=True)
class PackedNF4:
    packed: torch.Tensor
    scales: torch.Tensor
    shape: tuple[int, ...]
    block_size: int
    original_dtype: torch.dtype = torch.float32

    def __post_init__(self):
        if not isinstance(self.block_size, int) or not 2 <= self.block_size <= 4096:
            raise ValueError("NF4 block_size must be between 2 and 4096")
        if any(not isinstance(n, int) or n < 0 for n in self.shape):
            raise ValueError("Invalid NF4 shape")
        count = math.prod(self.shape)
        if count > MAX_ELEMENTS:
            raise ValueError("NF4 tensor exceeds the element budget")
        if self.packed.dtype != torch.uint8 or self.packed.ndim != 1 or self.packed.numel() != (count + 1) // 2:
            raise ValueError("Invalid packed NF4 payload")
        if self.scales.dtype != torch.float32 or self.scales.ndim != 1:
            raise ValueError("NF4 scales must be a flat float32 tensor")
        if self.scales.numel() != (count + self.block_size - 1) // self.block_size:
            raise ValueError("Invalid NF4 scale count")
        if self.packed.device != self.scales.device:
            raise ValueError("NF4 payload and scales must share a device")
        if not bool(torch.isfinite(self.scales).all()) or bool((self.scales < 0).any()):
            raise ValueError("NF4 scales must be finite and nonnegative")
        if self.original_dtype not in {torch.float16, torch.bfloat16, torch.float32, torch.float64}:
            raise ValueError("Unsupported NF4 source dtype")

    @property
    def storage_bytes(self) -> int:
        return self.packed.numel() + self.scales.numel() * 4

    @property
    def compression_ratio(self) -> float:
        original = math.prod(self.shape) * torch.empty((), dtype=self.original_dtype).element_size()
        return original / self.storage_bytes if self.storage_bytes else 1.0

    def to(self, device) -> "PackedNF4":
        return PackedNF4(self.packed.to(device), self.scales.to(device), self.shape, self.block_size, self.original_dtype)


def quantize_nf4(tensor: torch.Tensor, block_size: int = 64) -> PackedNF4:
    if not isinstance(block_size, int) or not 2 <= block_size <= 4096:
        raise ValueError("NF4 block_size must be between 2 and 4096")
    if tensor.dtype not in {torch.float16, torch.bfloat16, torch.float32, torch.float64}:
        raise ValueError("NF4 quantization requires a real floating tensor")
    if tensor.numel() > MAX_ELEMENTS:
        raise ValueError("NF4 tensor exceeds the element budget")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("Cannot quantize NaN or infinite values")
    flat = tensor.detach().contiguous().view(-1).float()
    count = flat.numel()
    blocks = (count + block_size - 1) // block_size
    scales = torch.empty(blocks, dtype=torch.float32, device=flat.device)
    indices = torch.empty(count + count % 2, dtype=torch.uint8, device=flat.device)
    table = torch.tensor(NF4_VALUES, device=flat.device)
    boundaries = (table[:-1] + table[1:]) / 2
    blocks_per_chunk = max(1, CHUNK_ELEMENTS // block_size)
    for start in range(0, blocks, blocks_per_chunk):
        stop = min(blocks, start + blocks_per_chunk)
        begin, end = start * block_size, min(count, stop * block_size)
        chunk = flat[begin:end]
        if chunk.numel() % block_size:
            chunk = F.pad(chunk, (0, block_size - chunk.numel() % block_size))
        chunk = chunk.view(-1, block_size)
        scale = chunk.abs().amax(-1)
        scales[start:stop] = scale
        normalized = chunk / torch.where(scale > 0, scale, 1).unsqueeze(-1)
        code = torch.bucketize(normalized, boundaries).to(torch.uint8).flatten()
        indices[begin:end] = code[:end - begin]
    if count % 2:
        indices[-1] = 7  # Exact zero in the unused nibble.
    packed = indices[0::2] | (indices[1::2] << 4)
    return PackedNF4(packed, scales, tuple(tensor.shape), block_size, tensor.dtype)


def dequantize_nf4(value: PackedNF4, *, dtype: torch.dtype | None = None, device=None) -> torch.Tensor:
    dtype = dtype or value.original_dtype
    if dtype not in {torch.float16, torch.bfloat16, torch.float32, torch.float64}:
        raise ValueError("NF4 reconstruction requires a real floating dtype")
    target = torch.device(device) if device is not None else value.packed.device
    count = math.prod(value.shape)
    output = torch.empty(count, dtype=dtype, device=target)
    table = torch.tensor(NF4_VALUES, device=target)
    scales = value.scales.to(target)
    # Even-sized chunks keep the two-nibbles-per-byte boundaries aligned.
    for begin in range(0, count, CHUNK_ELEMENTS):
        end = min(count, begin + CHUNK_ELEMENTS)
        packed = value.packed[begin // 2:(end + 1) // 2].to(target)
        codes = torch.stack((packed & 15, packed >> 4), -1).flatten()[:end - begin].long()
        positions = torch.arange(begin, end, device=target) // value.block_size
        output[begin:end] = (table[codes] * scales[positions]).to(dtype)
    return output.reshape(value.shape)


class NF4Linear(nn.Module):
    def __init__(self, linear: nn.Linear, block_size: int = 64):
        super().__init__()
        value = quantize_nf4(linear.weight, block_size)
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.block_size = block_size
        self.register_buffer("packed_weight", value.packed)
        self.register_buffer("weight_scales", value.scales)
        self.register_buffer("bias", linear.bias.detach().clone() if linear.bias is not None else None)

    def _apply(self, fn, recurse=True):
        # Device/dtype casts must not round the scale metadata to fp16/bf16.
        scales = self._buffers.pop("weight_scales")
        try:
            super()._apply(fn, recurse=recurse)
            probe = fn(scales.new_empty(0))
            self._buffers["weight_scales"] = scales.to(device=probe.device, dtype=torch.float32)
        except BaseException:
            self._buffers["weight_scales"] = scales
            raise
        return self

    @property
    def packed(self) -> PackedNF4:
        return PackedNF4(self.packed_weight, self.weight_scales.float(),
                         (self.out_features, self.in_features), self.block_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weight = dequantize_nf4(self.packed, dtype=inputs.dtype, device=inputs.device)
        bias = self.bias.to(dtype=inputs.dtype, device=inputs.device) if self.bias is not None else None
        return F.linear(inputs, weight, bias)


def bitsandbytes_linear(linear: nn.Linear, device: str = "cuda", *, block_size: int = 64) -> nn.Module:
    """Opt-in fused adapter. No import or GPU allocation during CPU startup."""
    if block_size != 64:
        raise ValueError("The bitsandbytes Linear4bit adapter uses its own block size (64)")
    if not torch.cuda.is_available() or torch.version.hip:
        raise RuntimeError("This adapter requires a CUDA-capable PyTorch runtime; use NF4Linear on CPU/MPS/ROCm")
    target = torch.device(device)
    if target.type != "cuda":
        raise ValueError("bitsandbytes adapter device must be CUDA")
    try:
        import bitsandbytes as bnb
    except ImportError as error:
        raise RuntimeError("bitsandbytes is optional and is not installed") from error
    result = bnb.nn.Linear4bit(linear.in_features, linear.out_features, bias=linear.bias is not None,
                              compute_dtype=torch.float32, quant_type="nf4", compress_statistics=False)
    result.weight = bnb.nn.Params4bit(linear.weight.detach().cpu(), requires_grad=False,
                                     quant_type="nf4", compress_statistics=False)
    if linear.bias is not None:
        result.bias = nn.Parameter(linear.bias.detach().cpu().clone(), requires_grad=False)
    return result.to(target)
