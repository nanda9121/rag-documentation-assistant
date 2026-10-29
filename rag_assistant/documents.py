"""Document extraction and overlapping text chunking."""

from dataclasses import dataclass
from io import BytesIO
import re


@dataclass(frozen=True)
class DocumentChunk:
    content: str
    source: str
    page: int | None = None


def extract_chunks(
    filename: str,
    data: bytes,
    chunk_size: int = 1_000,
    chunk_overlap: int = 150,
) -> list[DocumentChunk]:
    """Extract supported documents and split them into overlapping chunks."""
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap smaller than chunk_size")

    extension = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if extension == "pdf":
        from pypdf import PdfReader

        pages = [
            (page.extract_text() or "", page_number)
            for page_number, page in enumerate(PdfReader(BytesIO(data)).pages, start=1)
        ]
    elif extension in {"txt", "md"}:
        pages = [(data.decode("utf-8-sig"), None)]
    else:
        raise ValueError(f"Unsupported file type: {filename}")

    chunks: list[DocumentChunk] = []
    for text, page in pages:
        normalized = re.sub(r"\s+", " ", text).strip()
        start = 0
        while start < len(normalized):
            end = min(start + chunk_size, len(normalized))
            if end < len(normalized):
                boundary = normalized.rfind(" ", start + chunk_size // 2, end)
                if boundary > start:
                    end = boundary
            content = normalized[start:end].strip()
            if content:
                chunks.append(DocumentChunk(content=content, source=filename, page=page))
            if end >= len(normalized):
                break
            start = max(start + 1, end - chunk_overlap)
    return chunks
