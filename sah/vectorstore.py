"""A small persistent vector database (Unit: semantic search).

Stores chunks as JSON and dense vectors as .npy. Uses FAISS (IndexFlatIP) when
installed, otherwise an exact NumPy/SciPy dot-product search — same results,
FAISS just scales better.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config
from .documents import Chunk
from .embeddings import make_embedder


@dataclass
class Hit:
    chunk: Chunk
    score: float


class VectorStore:
    def __init__(self, index_dir: Path = config.INDEX_DIR, backend: str | None = None) -> None:
        self.dir = Path(index_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.embedder = make_embedder(backend)
        self.chunks: list[Chunk] = []
        self.matrix = None
        self._faiss = None
        self._load()

    # ---------- persistence ----------
    @property
    def _chunks_file(self) -> Path:
        return self.dir / "chunks.json"

    @property
    def _vec_file(self) -> Path:
        return self.dir / f"vectors_{self.embedder.name}.npy"

    def _load(self) -> None:
        if self._chunks_file.exists():
            self.chunks = [Chunk(**c) for c in json.loads(self._chunks_file.read_text(encoding="utf-8"))]
        if not self.chunks:
            return
        if self.embedder.dense and self._vec_file.exists():
            mat = np.load(self._vec_file)
            if len(mat) == len(self.chunks):
                self.matrix = mat
                self._build_faiss()
                return
        self._reindex()
        if self.embedder.dense:
            self._save()  # cache transformer vectors so the next start is instant

    def _save(self) -> None:
        self._chunks_file.write_text(json.dumps([c.to_dict() for c in self.chunks], ensure_ascii=False), encoding="utf-8")
        if self.embedder.dense and self.matrix is not None:
            np.save(self._vec_file, self.matrix)

    # ---------- indexing ----------
    def _reindex(self) -> None:
        texts = [c.text for c in self.chunks]
        if not texts:
            self.matrix = None
            return
        cache_file = self.dir / "tfidf_norm_cache.json"
        if hasattr(self.embedder, "load_cache") and not self.embedder.cache:
            self.embedder.load_cache(cache_file)
        n_before = len(getattr(self.embedder, "cache", {}))
        self.embedder.fit(texts)
        self.matrix = self.embedder.encode(texts)
        if hasattr(self.embedder, "save_cache"):
            live = {hashlib.md5(t.encode("utf-8", "ignore")).hexdigest() for t in texts}
            self.embedder.cache = {k: v for k, v in self.embedder.cache.items() if k in live}
            if len(self.embedder.cache) != n_before or not cache_file.exists():
                self.embedder.save_cache(cache_file)
        self._build_faiss()

    def _build_faiss(self) -> None:
        self._faiss = None
        if not self.embedder.dense or self.matrix is None:
            return
        try:
            import faiss

            idx = faiss.IndexFlatIP(self.matrix.shape[1])
            idx.add(np.ascontiguousarray(self.matrix, dtype="float32"))
            self._faiss = idx
        except ImportError:
            pass

    def add(self, chunks: list[Chunk]) -> int:
        known = {c.id for c in self.chunks}
        new = [c for c in chunks if c.id not in known]
        if not new:
            return 0
        if self.embedder.dense and self.matrix is not None:
            # Pretrained embeddings: only encode the new chunks.
            self.chunks.extend(new)
            self.matrix = np.vstack([self.matrix, self.embedder.encode([c.text for c in new])])
            self._build_faiss()
        else:
            self.chunks.extend(new)
            self._reindex()  # TF-IDF vocabulary depends on the whole corpus
        self._save()
        return len(new)

    def remove_document(self, doc: str) -> None:
        self.chunks = [c for c in self.chunks if c.doc != doc]
        self._reindex()
        self._save()

    def neighbours(self, chunk: Chunk, before: int = 1, after: int = 1) -> list[Chunk]:
        """The chunk with its neighbours from the same document (sentence-window retrieval)."""
        idx = getattr(self, "_pos", None)
        if idx is None or len(idx) != len(self.chunks):
            idx = self._pos = {c.id: i for i, c in enumerate(self.chunks)}
        i = idx.get(chunk.id)
        if i is None:
            return [chunk]
        lo, hi = max(0, i - before), min(len(self.chunks), i + after + 1)
        return [c for c in self.chunks[lo:hi] if c.doc == chunk.doc]

    def documents(self) -> list[str]:
        return sorted({c.doc for c in self.chunks})

    # ---------- search ----------
    def search(self, query: str, k: int = config.TOP_K, docs: list[str] | None = None) -> list[Hit]:
        if not self.chunks or self.matrix is None:
            return []
        q = self.embedder.encode([query])
        if self._faiss is not None and not docs:
            scores, idx = self._faiss.search(np.asarray(q, dtype="float32"), min(k, len(self.chunks)))
            return [Hit(self.chunks[i], float(s)) for s, i in zip(scores[0], idx[0]) if i >= 0]
        sims = self.matrix @ q.T
        sims = np.asarray(sims.toarray() if hasattr(sims, "toarray") else sims).ravel()
        order = np.argsort(-sims)
        hits = []
        for i in order:
            if docs and self.chunks[i].doc not in docs:
                continue
            hits.append(Hit(self.chunks[i], float(sims[i])))
            if len(hits) == k:
                break
        return hits

    def chunks_for(self, docs: list[str] | None = None, page_from: int | None = None,
                   page_to: int | None = None) -> list[Chunk]:
        out = [c for c in self.chunks if not docs or c.doc in docs]
        if page_from:
            out = [c for c in out if c.page >= page_from]
        if page_to:
            out = [c for c in out if c.page <= page_to]
        return out
