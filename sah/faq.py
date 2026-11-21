"""Frequently asked questions: find repeated student questions and match new questions to
answers the teacher has already published (so she answers each doubt once)."""
from __future__ import annotations

import re

from sklearn.feature_extraction.text import TfidfVectorizer

from . import db
from .preprocess import remove_stopwords, stem, tokenize


# Question words carry the Bloom level, not the topic – ignore them when grouping questions.
_QWORDS = {"what", "explain", "define", "describe", "difference", "differentiate", "compare", "how", "why",
           "meaning", "mean", "tell", "give", "example", "please", "briefly", "detail", "concept", "short", "note"}


def _norm(t: str) -> str:
    t = re.sub(r"isation", "ization", t.lower())   # British/Indian spelling → one form
    t = re.sub(r"(analy|paraly)se", r"\1ze", t)
    toks = [w for w in remove_stopwords(tokenize(t)) if w.lower() not in _QWORDS]
    return " ".join(stem(toks)) or "x"


def repeated_questions(questions: list[str], threshold: float = 0.55, min_count: int = 2) -> list[dict]:
    """Group near-duplicate questions (TF-IDF cosine on Porter stems ≥ threshold)."""
    qs = [q.strip() for q in questions if q and q.strip()]
    if len(qs) < 2:
        return []
    vec = TfidfVectorizer(ngram_range=(1, 2))
    m = vec.fit_transform([_norm(q) for q in qs])
    sims = (m @ m.T).toarray()
    seen, groups = set(), []
    for i in range(len(qs)):
        if i in seen:
            continue
        members = [j for j in range(len(qs)) if j not in seen and sims[i, j] >= threshold]
        seen.update(members)
        if len(members) >= min_count:
            groups.append({"question": qs[i], "count": len(members), "variants": sorted({qs[j] for j in members})})
    return sorted(groups, key=lambda g: -g["count"])


def match(question: str, threshold: float = 0.5) -> dict | None:
    """Best published FAQ for a new question, if similar enough."""
    f = db.faqs()
    if f.empty:
        return None
    vec = TfidfVectorizer(ngram_range=(1, 2))
    m = vec.fit_transform([_norm(q) for q in f.question] + [_norm(question)])
    sims = (m[:-1] @ m[-1].T).toarray().ravel()
    i = int(sims.argmax())
    if sims[i] < threshold:
        return None
    row = f.iloc[i]
    return {"question": row.question, "answer": row.answer, "score": round(float(sims[i]), 2)}
