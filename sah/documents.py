"""PDF / text extraction and chunking (Unit: RAG – document ingestion)."""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from . import config
from .preprocess import clean_text


@dataclass
class Chunk:
    id: str
    doc: str        # file name
    page: int       # 1-based page number
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


def extract_pages(path: str | Path) -> list[tuple[int, str]]:
    """Return [(page_number, text), ...]. Supports .pdf, .txt and .md."""
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        try:
            import pymupdf as fitz  # PyMuPDF >= 1.24
        except ImportError:
            import fitz

        with fitz.open(path) as pdf:
            return [(i + 1, mark_headings(page.get_text("text"))) for i, page in enumerate(pdf)]
    text = path.read_text(encoding="utf-8", errors="ignore")
    # Treat every ~3000 characters of a plain-text file as a "page" for citations.
    return [(i // 3000 + 1, text[i : i + 3000]) for i in range(0, max(len(text), 1), 3000)]


def mark_headings(text: str) -> str:
    """PDF headings have no full stop, so after whitespace is collapsed they glue
    onto the next sentence ("Unit 1 Tokenization is ..."). End short, unpunctuated
    lines that are followed by a capitalised line with a period."""
    lines = text.splitlines()
    out = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        nxt = next((l.strip() for l in lines[i + 1 :] if l.strip()), "")
        if (stripped and len(stripped.split()) <= 8 and not re.search(r"[.!?:;,]$", stripped)
                and nxt[:1].isupper()):
            line = stripped + "."
        out.append(line)
    return "\n".join(out)


def chunk_pages(doc_name: str, pages: list[tuple[int, str]],
                size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP) -> list[Chunk]:
    """Sliding-window chunking by words. Overlap keeps ideas that straddle a
    boundary retrievable from both neighbouring chunks."""
    step = max(size - overlap, 1)
    chunks: list[Chunk] = []
    for page_no, raw in pages:
        words = clean_text(raw).split()
        if not words:
            continue
        for start in range(0, len(words), step):
            window = words[start : start + size]
            if len(window) < 15 and start > 0:  # tiny tail already covered by overlap
                break
            text = " ".join(window)
            cid = hashlib.md5(f"{doc_name}|{page_no}|{start}".encode()).hexdigest()[:12]
            chunks.append(Chunk(cid, doc_name, page_no, text))
            if start + size >= len(words):
                break
    return chunks


def load_document(path: str | Path) -> list[Chunk]:
    path = Path(path)
    return chunk_pages(path.name, extract_pages(path))


def full_text(path: str | Path) -> str:
    return "\n".join(t for _, t in extract_pages(path))
