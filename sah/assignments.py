"""AI-assisted assignment review – the teacher keeps the final mark.

* Rubric format (one criterion per line):   Criterion | marks | key terms, comma separated
* LLM review: per-criterion suggested marks + missing concepts, grounded in the rubric
* Offline review: key-term coverage per criterion (lemmatised match) + semantic similarity
* Similarity check: TF-IDF cosine between every pair of submissions (copy detection)
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .llm import LLM, LLMError
from .preprocess import normalize_for_search


def parse_rubric(text: str) -> list[dict]:
    crit = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 2 and parts[0]:
            try:
                marks = float(parts[1])
            except ValueError:
                continue
            terms = [t.strip() for t in parts[2].split(",")] if len(parts) > 2 and parts[2] else []
            crit.append({"criterion": parts[0], "marks": marks, "terms": [t for t in terms if t]})
    return crit


PROMPT = """You are helping a university teacher review a student assignment. Suggest marks only; the teacher
decides the final mark. Be fair, specific and kind.

ASSIGNMENT: {title}
TASK: {description}
RUBRIC (criterion – max marks – key concepts expected):
{rubric}

STUDENT SUBMISSION:
{text}

Return ONLY JSON: {{"criteria": [{{"criterion": str, "awarded": number, "max": number, "comment": str}}],
"missing_concepts": [str], "strengths": [str], "summary": "2 sentences of feedback addressed to the student"}}"""


def _offline_review(text: str, rubric: list[dict]) -> dict:
    norm = set(normalize_for_search(text).split())
    rows, missing = [], []
    for c in rubric:
        terms = c["terms"] or [c["criterion"]]
        hit = []
        for t in terms:
            toks = normalize_for_search(t).split()
            if toks and sum(x in norm for x in toks) >= max(1, len(toks) / 2):
                hit.append(t)
            else:
                missing.append(t)
        frac = len(hit) / len(terms)
        rows.append({"criterion": c["criterion"], "awarded": round(c["marks"] * frac * 2) / 2, "max": c["marks"],
                     "comment": f"Covers {len(hit)}/{len(terms)} expected concepts."})
    words = len(text.split())
    return {"criteria": rows, "missing_concepts": missing,
            "strengths": [f"{words} words submitted"] if words else [],
            "summary": ("Good coverage of the expected concepts." if not missing else
                        f"Add more on: {', '.join(missing[:5])}.")}


def review(text: str, title: str, description: str, rubric_text: str, llm: LLM) -> dict:
    rubric = parse_rubric(rubric_text)
    if not rubric:
        rubric = [{"criterion": "Overall quality", "marks": 10.0, "terms": []}]
    result, method = None, "offline (key-concept coverage)"
    if llm.available and text.strip():
        try:
            rub = "\n".join(f"- {c['criterion']} – {c['marks']:g} – {', '.join(c['terms']) or 'n/a'}" for c in rubric)
            data = llm.generate_json(PROMPT.format(title=title, description=description, rubric=rub,
                                                   text=text[:12000]), temperature=0.1)
            if isinstance(data, dict) and data.get("criteria"):
                maxes = {c["criterion"]: c["marks"] for c in rubric}
                for c in data["criteria"]:  # never exceed the rubric's maximum
                    cap = maxes.get(c.get("criterion"), float(c.get("max", 0) or 0))
                    c["max"] = cap
                    c["awarded"] = max(0.0, min(float(c.get("awarded", 0) or 0), cap))
                result, method = data, f"AI ({llm.label})"
        except (LLMError, ValueError, KeyError, TypeError):
            result = None
    if result is None:
        result = _offline_review(text, rubric)
    total = sum(float(c["awarded"]) for c in result["criteria"])
    maximum = sum(float(c["max"]) for c in result["criteria"])
    result.update(total=round(total, 1), max=maximum, method=method)
    return result


def to_json(r: dict) -> str:
    return json.dumps(r, ensure_ascii=False)


def similarity_pairs(texts: dict[str, str], threshold: float = 0.8) -> list[tuple[str, str, float]]:
    """Pairs of students whose submissions are suspiciously similar (TF-IDF cosine ≥ threshold)."""
    users = [u for u, t in texts.items() if t and len(t.split()) >= 20]
    if len(users) < 2:
        return []
    vec = TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True)
    m = vec.fit_transform([normalize_for_search(texts[u]) or "x" for u in users])
    sims = (m @ m.T).toarray()
    pairs = []
    for i in range(len(users)):
        for j in range(i + 1, len(users)):
            if sims[i, j] >= threshold:
                pairs.append((users[i], users[j], round(float(sims[i, j]), 3)))
    return sorted(pairs, key=lambda p: -p[2])


def max_similarity(texts: dict[str, str]) -> dict[str, float]:
    users = [u for u, t in texts.items() if t]
    if len(users) < 2:
        return {u: 0.0 for u in users}
    vec = TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True)
    m = vec.fit_transform([normalize_for_search(texts[u]) or "x" for u in users])
    sims = (m @ m.T).toarray()
    np.fill_diagonal(sims, 0)
    return {u: round(float(sims[i].max()), 3) for i, u in enumerate(users)}
