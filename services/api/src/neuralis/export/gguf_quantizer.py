"""Public external llama.cpp conversion API; q8_0 uses the supplied converter.

Additional quantization formats require a compatible external llama.cpp tool;
this API does not invent an in-process quantizer for Neuralis toy models.
"""

from .gguf import GGUFConverter, inspect_gguf

__all__ = ["GGUFConverter", "inspect_gguf"]
