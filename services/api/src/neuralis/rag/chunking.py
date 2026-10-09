"""Bounded text ingestion; PDF extraction is an optional local adapter."""

from pathlib import Path


def chunk_text(text: str, *, chunk_size: int = 800, overlap: int = 120) -> list[dict]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Document must contain nonempty text")
    if len(text.encode()) > 2 * 1024 * 1024:
        raise ValueError("Document exceeds the two-MiB ingestion budget")
    if not 64 <= chunk_size <= 8000 or not 0 <= overlap < chunk_size // 2:
        raise ValueError("Chunk size must fit 64 to 8000; overlap must be less than half the chunk")
    result = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        # Prefer a word boundary while preserving an explicit character range.
        if end < len(text):
            boundary = text.rfind(" ", start + chunk_size // 2, end)
            if boundary > start:
                end = boundary
        result.append({"text": text[start:end], "start": start, "end": end, "index": len(result)})
        if len(result) > 10000:
            raise ValueError("Document exceeds the 10000-chunk budget")
        if end == len(text):
            break
        start = end - overlap
    return result


def read_pdf(path: str | Path, *, max_pages: int = 200, max_bytes: int = 20 * 1024 * 1024) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise RuntimeError("PDF ingestion requires optional pypdf") from error
    source = Path(path)
    if not source.is_file() or source.stat().st_size > max_bytes:
        raise ValueError("PDF must be an existing file within the upload byte budget")
    if not 1 <= max_pages <= 1000:
        raise ValueError("PDF page budget must fit one to 1000")
    reader = PdfReader(source)
    if reader.is_encrypted:
        raise ValueError("Decrypt password-protected PDFs before ingestion")
    if len(reader.pages) > max_pages:
        raise ValueError("PDF exceeds the page budget")
    parts = []
    total = 0
    for page in reader.pages:
        value = page.extract_text() or ""
        total += len(value.encode())
        if total > 2 * 1024 * 1024:
            raise ValueError("Extracted PDF text exceeds the two-MiB ingestion budget")
        parts.append(value)
    result = "\n\n".join(parts)
    if not result.strip():
        raise ValueError("PDF has no extractable text; OCR is not configured")
    return result
