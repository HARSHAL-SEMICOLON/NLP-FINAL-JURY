"""Text representation (Units: TF-IDF, word/sentence embeddings, transformers).

Two interchangeable backends:
  * TfidfEmbedder  – sparse lexical vectors (classical NLP, no download needed)
  * SbertEmbedder  – dense semantic vectors from a pretrained transformer
Both return L2-normalised vectors, so a dot product equals cosine similarity.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from . import config
from .preprocess import normalize_for_search


class TfidfEmbedder:
    name = "tfidf"
    dense = False

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1,
                                          token_pattern=r"(?u)\b\w+\b")
        self.fitted = False
        self.cache: dict[str, str] = {}   # text hash -> normalised text (POS tagging is the slow part)

    def _norm(self, text: str) -> str:
        key = hashlib.md5(text.encode("utf-8", "ignore")).hexdigest()
        if key not in self.cache:
            self.cache[key] = normalize_for_search(text)
        return self.cache[key]

    def load_cache(self, path) -> None:
        try:
            self.cache = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.cache = {}

    def save_cache(self, path) -> None:
        Path(path).write_text(json.dumps(self.cache), encoding="utf-8")

    def fit(self, corpus: list[str]) -> None:
        docs = [self._norm(t) or "empty" for t in corpus] or ["empty"]
        self.vectorizer.fit(docs)
        self.fitted = True

    def encode(self, texts: list[str]):
        return normalize(self.vectorizer.transform([self._norm(t) if len(texts) > 1 else normalize_for_search(t)
                                                    for t in texts]))

    def top_terms(self, text: str, n: int = 10) -> list[tuple[str, float]]:
        vec = self.vectorizer.transform([normalize_for_search(text)]).toarray()[0]
        vocab = self.vectorizer.get_feature_names_out()
        idx = vec.argsort()[::-1][:n]
        return [(vocab[i], round(float(vec[i]), 3)) for i in idx if vec[i] > 0]


class SbertEmbedder:
    name = "sbert"
    dense = True
    _model = None  # loaded once per process

    def __init__(self, model_name: str = config.SBERT_MODEL) -> None:
        from sentence_transformers import SentenceTransformer

        if SbertEmbedder._model is None:
            SbertEmbedder._model = SentenceTransformer(model_name)
        self.model = SbertEmbedder._model
        self.fitted = True

    def fit(self, corpus: list[str]) -> None:  # pretrained: nothing to fit
        pass

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False).astype("float32")


def sbert_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401

        return True
    except Exception:
        return False


def make_embedder(backend: str | None = None):
    backend = (backend or config.EMBEDDING_BACKEND).lower()
    if backend == "tfidf":
        return TfidfEmbedder()
    if backend in ("sbert", "auto") and sbert_available():
        try:
            return SbertEmbedder()
        except Exception as exc:  # model download failed, offline, etc.
            if backend == "sbert":
                raise
            print(f"[embeddings] Sentence Transformers unavailable ({exc}); using TF-IDF.")
    return TfidfEmbedder()


def cosine(a: str, b: str, embedder=None) -> float:
    """Semantic similarity between two texts (used on the NLP Lab page)."""
    emb = embedder or make_embedder()
    if not emb.dense:
        emb.fit([a, b])
    va, vb = emb.encode([a]), emb.encode([b])
    if hasattr(va, "toarray"):
        va, vb = va.toarray(), vb.toarray()
    return float(np.dot(np.asarray(va)[0], np.asarray(vb)[0]))
