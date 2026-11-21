"""Syllabus awareness (text classification + information extraction).

* parse_units()    – extracts "Unit I … (9 Hours) <topics>" blocks from a syllabus PDF
* tag_unit()       – classifies any question/answer into the closest syllabus unit
                     (TF-IDF cosine similarity = nearest-centroid text classification)
* bloom_level()    – rule-based classifier for Bloom's taxonomy level of a question,
                     the same scale the End-Semester exam uses
"""
from __future__ import annotations

import json
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from . import config
from .preprocess import remove_stopwords, stem, tokenize


def _norm(text: str) -> str:
    """Porter stems: robust for short questions, where POS-aware lemmas can differ
    between 'Stemming' (noun in the syllabus) and 'stemming' (verb in a question)."""
    return " ".join(stem(remove_stopwords(tokenize(text))))

SYLLABUS_FILE = config.DATA_DIR / "syllabus.json"

_UNIT_RE = re.compile(
    r"Unit[\s\-–]+([IVX]+|\d+)\b\.?\s*:?\s*(.*?)\(\s*(\d+)\s*Hours?\s*\)(.*?)"
    r"(?=Pedagogy|Unit[\s\-–]+(?:[IVX]+|\d+)\b\.?\s*\n|Learning Resources|Text Books|$)",
    re.IGNORECASE | re.DOTALL,
)
_NOISE = re.compile(r"MIT SCHOOL OF COMPUTING,?", re.IGNORECASE)


def _clean(s: str) -> str:
    s = _NOISE.sub(" ", s)
    s = re.sub(r"\.\s*\n", " ", s)          # periods added to PDF line ends by mark_headings
    return re.sub(r"\s+", " ", s).strip(" .:-")


def parse_units(text: str) -> list[dict]:
    units, seen = [], set()
    for m in _UNIT_RE.finditer(text):
        num, title, hours, topics = m.group(1).upper(), _clean(m.group(2)), m.group(3), _clean(m.group(4))
        if num in seen or len(topics.split()) < 5:
            continue
        seen.add(num)
        units.append({"unit": f"Unit {num}", "title": title, "hours": int(hours), "topics": topics})
    return units


def looks_like_syllabus(text: str) -> bool:
    return len(parse_units(text)) >= 3


def save(units: list[dict], source: str) -> None:
    SYLLABUS_FILE.write_text(json.dumps({"source": source, "units": units}, indent=2), encoding="utf-8")
    _cache.clear()


def load() -> dict:
    if SYLLABUS_FILE.exists():
        return json.loads(SYLLABUS_FILE.read_text(encoding="utf-8"))
    return {"source": None, "units": []}


def clear() -> None:
    SYLLABUS_FILE.unlink(missing_ok=True)
    _cache.clear()


_cache: dict = {}


def _model():
    units = load()["units"]
    if not units:
        return None, []
    key = json.dumps(units)
    if _cache.get("key") != key:
        docs = [_norm(u["title"] + " " + u["topics"]) for u in units]
        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        mat = vec.fit_transform(docs)
        _cache.update(key=key, vec=vec, mat=mat)
    return _cache, units


def tag_unit(text: str, min_score: float = 0.05) -> dict | None:
    """Return {'unit','title','score'} for the closest syllabus unit, or None."""
    model, units = _model()
    if not units:
        return None
    q = model["vec"].transform([_norm(text)])
    sims = (model["mat"] @ q.T).toarray().ravel()
    i = int(np.argmax(sims))
    if sims[i] < min_score:
        return {"unit": "Outside syllabus", "title": "", "score": float(sims[i])}
    return {"unit": units[i]["unit"], "title": units[i]["title"], "score": round(float(sims[i]), 3)}


# ---------------------------------------------------------------- Bloom's taxonomy
BLOOM = {
    "Remember":   ["define", "list", "name", "state", "what is", "what are", "who", "when", "identify",
                   "recall", "label", "full form", "stands for"],
    "Understand": ["explain", "describe", "summarize", "summarise", "discuss", "illustrate", "interpret",
                   "classify", "outline", "how does", "how do", "meaning of", "why is", "why do"],
    "Apply":      ["apply", "use", "implement", "solve", "calculate", "compute", "demonstrate", "show how",
                   "perform", "tokenize", "find the", "example of", "give an example"],
    "Analyze":    ["compare", "contrast", "differentiate", "difference", "distinguish", "analyze", "analyse",
                   "examine", "relationship", "versus", " vs "],
    "Evaluate":   ["evaluate", "justify", "assess", "critique", "which is better", "pros and cons",
                   "advantages and disadvantages", "argue", "recommend", "should"],
    "Create":     ["design", "create", "build", "develop", "propose", "construct", "formulate", "plan a",
                   "write a program", "generate"],
}
BLOOM_ORDER = list(BLOOM)
BLOOM_HELP = {
    "Remember": "recall facts and definitions",
    "Understand": "explain ideas in your own words",
    "Apply": "use a method on a new example",
    "Analyze": "compare, contrast and break down",
    "Evaluate": "judge and justify a choice",
    "Create": "design or build something new",
}


def bloom_level(question: str) -> str:
    """Highest Bloom level whose cue words appear in the question (default: Understand)."""
    q = " " + question.lower() + " "
    found = [lvl for lvl, cues in BLOOM.items() if any(c in q for c in cues)]
    return max(found, key=BLOOM_ORDER.index) if found else "Understand"
