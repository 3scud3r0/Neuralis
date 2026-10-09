"""Stable local lexical baseline and an opt-in sentence-transformer adapter."""

import hashlib
from pathlib import Path
import re

import numpy as np


def normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    if vector.ndim != 1 or not 1 <= vector.size <= 4096 or not np.isfinite(vector).all():
        raise ValueError("Embedding must be a finite vector with one to 4096 dimensions")
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


class HashedEmbedder:
    """Signed feature hashing of Unicode words and adjacent word pairs.

    This is a deterministic lexical baseline, not a semantic neural model.
    BLAKE2 prevents process-randomized Python hashes from changing a saved index.
    """

    def __init__(self, dimensions: int = 384):
        if not 32 <= dimensions <= 4096:
            raise ValueError("Hash embedding dimensions must fit 32 to 4096")
        self.dimensions = dimensions
        self.signature = f"blake2b-word-bigram-v1:{dimensions}"

    def embed(self, text: str) -> np.ndarray:
        if not isinstance(text, str) or len(text) > 2 * 1024 * 1024:
            raise ValueError("Embedding text exceeds the input budget")
        tokens = re.findall(r"\w+", text.casefold(), flags=re.UNICODE)
        result = np.zeros(self.dimensions, dtype=np.float32)
        features = [(token, 1.0) for token in tokens]
        features.extend((left + "\0" + right, 0.5) for left, right in zip(tokens, tokens[1:]))
        for feature, weight in features:
            digest = hashlib.blake2b(feature.encode(), digest_size=16).digest()
            index = int.from_bytes(digest[:8], "little") % self.dimensions
            result[index] += weight if digest[8] & 1 else -weight
        return normalize(result)


class SentenceTransformerEmbedder:
    """Opt-in adapter loading only an existing local model directory by default."""

    def __init__(self, model_path: str | Path, *, allow_download: bool = False):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError("Neural embeddings require optional sentence-transformers") from error
        path = Path(model_path)
        if not allow_download and not path.is_dir():
            raise ValueError("Supply an existing local sentence-transformer model directory")
        self.model = SentenceTransformer(str(model_path), device="cpu", local_files_only=not allow_download)
        self.dimensions = int(self.model.get_sentence_embedding_dimension())
        if not 1 <= self.dimensions <= 4096:
            raise ValueError("Model embeddings exceed the index dimension budget")
        self.signature = f"sentence-transformers:{path.resolve()}:{self.dimensions}"

    def embed(self, text: str) -> np.ndarray:
        if not isinstance(text, str) or len(text) > 2 * 1024 * 1024:
            raise ValueError("Embedding text exceeds the input budget")
        return normalize(self.model.encode(text, normalize_embeddings=True, show_progress_bar=False))
