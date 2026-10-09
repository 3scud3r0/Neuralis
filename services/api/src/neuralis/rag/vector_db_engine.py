"""Public encrypted vector-store API with lexical and opt-in neural embeddings."""

from .local import LocalRAG
from .embeddings import HashedEmbedder, SentenceTransformerEmbedder

__all__ = ["LocalRAG", "HashedEmbedder", "SentenceTransformerEmbedder"]
