"""Local cosine retrieval with encrypted document, metadata and vector payloads."""

from collections.abc import Callable
import importlib.util
import json
from pathlib import Path
import sqlite3
from threading import RLock
import uuid

import numpy as np

from .chunking import chunk_text
from .embeddings import HashedEmbedder, normalize


class LocalRAG:
    """Fernet authenticated encryption protects every content-bearing payload.

    A caller-supplied key is required. There is no plaintext storage mode or
    fallback. Document/chunk IDs, count and ciphertext lengths remain visible.
    Search decrypts a bounded local collection in memory; vectors themselves are
    encrypted at rest, so this baseline does not pretend to be an ANN database.
    Caller controls filesystem access, key management, and request authorization.
    """

    def __init__(self, path: str | Path, *, key: bytes, embedder=None,
                 max_chunks: int = 10000, max_total_bytes: int = 64 * 1024 * 1024):
        try:
            from cryptography.fernet import Fernet, InvalidToken
        except ImportError as error:
            raise RuntimeError("Encrypted retrieval requires cryptography; plaintext fallback is disabled") from error
        if not isinstance(key, bytes):
            raise ValueError("Supply a Fernet key as bytes from secure caller-controlled storage")
        try:
            self._cipher = Fernet(key)
        except (ValueError, TypeError) as error:
            raise ValueError("Invalid Fernet key") from error
        if not 1 <= max_chunks <= 100000 or not 1024 <= max_total_bytes <= 1024**3:
            raise ValueError("Collection budgets are outside supported bounds")
        self.embedder = embedder or HashedEmbedder()
        if not isinstance(self.embedder.signature, str) or not 1 <= self.embedder.dimensions <= 4096:
            raise ValueError("Embedder must expose a signature and bounded dimensions")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_chunks, self.max_total_bytes = max_chunks, max_total_bytes
        self._lock = RLock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("CREATE TABLE IF NOT EXISTS rag_meta (id INTEGER PRIMARY KEY, payload BLOB NOT NULL)")
        self._connection.execute("""CREATE TABLE IF NOT EXISTS rag_chunks (
            document_id TEXT NOT NULL, chunk_index INTEGER NOT NULL, payload BLOB NOT NULL,
            PRIMARY KEY(document_id, chunk_index))""")
        self._connection.commit()
        meta = self._connection.execute("SELECT payload FROM rag_meta WHERE id=1").fetchone()
        try:
            if meta:
                saved = json.loads(self._cipher.decrypt(bytes(meta[0])))
                if saved.get("signature") != self.embedder.signature or saved.get("dimensions") != self.embedder.dimensions:
                    raise ValueError("Saved index uses a different embedding adapter; rebuild a separate collection")
            else:
                value = {"format": "neuralis-encrypted-rag-v1", "signature": self.embedder.signature,
                         "dimensions": self.embedder.dimensions}
                self._connection.execute("INSERT INTO rag_meta VALUES (1, ?)", (self._encrypt(value),))
                self._connection.commit()
        except InvalidToken as error:
            self._connection.close()
            raise ValueError("Collection key does not authenticate this encrypted index") from error
        except Exception:
            self._connection.close()
            raise

    def _encrypt(self, value: dict) -> bytes:
        return self._cipher.encrypt(json.dumps(value, ensure_ascii=False, allow_nan=False).encode())

    def _decrypt(self, value: bytes) -> dict:
        return json.loads(self._cipher.decrypt(bytes(value)))

    @staticmethod
    def capabilities() -> dict:
        return {"encryption_available": importlib.util.find_spec("cryptography") is not None,
                "encrypted_at_rest": True, "plaintext_fallback": False,
                "baseline_embeddings": "deterministic signed word and bigram hashing",
                "pdf_available": importlib.util.find_spec("pypdf") is not None,
                "semantic_adapter_available": importlib.util.find_spec("sentence_transformers") is not None}

    def add_document(self, text: str, source: str = "local-document", *, chunk_size: int = 800,
                     overlap: int = 120, metadata: dict | None = None) -> dict:
        if not isinstance(source, str) or not 1 <= len(source) <= 2000:
            raise ValueError("Source label must fit one to 2000 characters")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("Document metadata must be a JSON object")
        details = {} if metadata is None else metadata
        if len(json.dumps(details, allow_nan=False).encode()) > 16384:
            raise ValueError("Document metadata exceeds the 16 KiB budget")
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        identifier = uuid.uuid4().hex
        rows = []
        for chunk in chunks:
            vector = normalize(self.embedder.embed(chunk["text"]))
            if vector.size != self.embedder.dimensions:
                raise ValueError("Embedding adapter returned an inconsistent vector dimension")
            payload = {**chunk, "source": source, "metadata": details, "vector": vector.tolist()}
            rows.append((identifier, chunk["index"], self._encrypt(payload)))
        byte_count = sum(len(row[2]) for row in rows)
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                count, size = self._connection.execute(
                    "SELECT COUNT(*), COALESCE(SUM(LENGTH(payload)), 0) FROM rag_chunks").fetchone()
                if count + len(rows) > self.max_chunks or size + byte_count > self.max_total_bytes:
                    raise ValueError("Encrypted collection storage budget exhausted")
                self._connection.executemany("INSERT INTO rag_chunks VALUES (?, ?, ?)", rows)
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
        return {"document_id": identifier, "chunks": len(chunks), "characters": len(text),
                "encrypted_bytes": byte_count, "encrypted_at_rest": True,
                "embedding_adapter": self.embedder.signature}

    def search(self, query: str, limit: int = 5, *, minimum_score: float = -1.0) -> dict:
        if not isinstance(query, str) or not query.strip() or len(query) > 8000:
            raise ValueError("Query must contain one to 8000 characters")
        if not 1 <= limit <= 100 or not np.isfinite(minimum_score) or not -1 <= minimum_score <= 1:
            raise ValueError("Search limit or cosine score threshold is outside bounds")
        query_vector = normalize(self.embedder.embed(query))
        if query_vector.size != self.embedder.dimensions:
            raise ValueError("Embedding adapter returned an inconsistent query dimension")
        with self._lock:
            rows = self._connection.execute("SELECT document_id, chunk_index, payload FROM rag_chunks").fetchall()
        ranked = []
        for identifier, index, encrypted in rows:
            payload = self._decrypt(encrypted)
            vector = normalize(np.asarray(payload.pop("vector"), dtype=np.float32))
            if vector.size != query_vector.size:
                raise ValueError("Persisted vector dimension does not match the index")
            score = float(np.clip(np.dot(query_vector, vector), -1.0, 1.0))
            if score >= minimum_score:
                ranked.append({"document_id": identifier, "chunk_index": index, "score": score, **payload})
        ranked.sort(key=lambda x: (-x["score"], x["document_id"], x["chunk_index"]))
        return {"query": query, "matches": ranked[:limit], "searched_chunks": len(rows),
                "embedding_adapter": self.embedder.signature, "metric": "cosine",
                "encrypted_at_rest": True}

    def retrieve_and_answer(self, query: str, answerer: Callable[[str, list[dict]], str], *, limit: int = 5) -> dict:
        retrieval = self.search(query, limit)
        answer = answerer(query, retrieval["matches"])
        if not isinstance(answer, str) or len(answer) > 32000:
            raise ValueError("Answer provider must return public text within the response budget")
        return {**retrieval, "answer": answer, "answer_source": "caller-provided-generator"}

    def delete_document(self, document_id: str) -> dict:
        with self._lock:
            count = self._connection.execute("SELECT COUNT(*) FROM rag_chunks WHERE document_id=?", (document_id,)).fetchone()[0]
            self._connection.execute("DELETE FROM rag_chunks WHERE document_id=?", (document_id,))
            self._connection.commit()
        return {"document_id": document_id, "deleted_chunks": count,
                "secure_erasure": False}

    def status(self) -> dict:
        with self._lock:
            documents, chunks, size = self._connection.execute(
                "SELECT COUNT(DISTINCT document_id), COUNT(*), COALESCE(SUM(LENGTH(payload)), 0) FROM rag_chunks").fetchone()
        return {"documents": documents, "chunks": chunks, "encrypted_bytes": size,
                "max_chunks": self.max_chunks, "max_total_bytes": self.max_total_bytes,
                "encrypted_at_rest": True, "embedding_adapter": self.embedder.signature,
                "dimensions": self.embedder.dimensions, "storage_backend": "sqlite-encrypted-payloads"}

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
