"""Notes summarisation (Units: extractive vs abstractive summarisation).

* Abstractive (LLM): map-reduce — summarise sections, then merge — so long
  PDFs fit the model's context window.
* Extractive (offline): rank sentences by TF-IDF weight and keep the top ones
  in their original order; definitions are found with lexical patterns.
"""
from __future__ import annotations

import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .llm import LLM, LLMError
from .preprocess import STOPWORDS, is_heading, sentences

MODES = {
    "Short notes": "Write concise revision notes with headings and bullet points (max ~300 words).",
    "Important points": "List the 10–15 most important points a student must remember, as bullets.",
    "Definitions": "Extract every important term and give a one-to-two line exam-ready definition, "
                   "as a Markdown table | Term | Definition |.",
    "Exam summary": "Write an exam-preparation summary: key concepts, likely 2/5/10-mark questions, "
                    "formulas or diagrams to remember, and common mistakes.",
    "Teaching notes": "Write concise teaching notes for a lecturer: learning objectives, a 50-minute lesson "
                      "flow with timings, one analogy and one worked example per key concept, two in-class "
                      "check questions, and common student misconceptions to address.",
}

SECTION_WORDS = 2500


def _split(text: str, size: int = SECTION_WORDS) -> list[str]:
    words = text.split()
    return [" ".join(words[i : i + size]) for i in range(0, len(words), size)] or [""]


def summarize(text: str, mode: str, llm: LLM, difficulty: str = "Intermediate") -> tuple[str, str]:
    """Return (markdown, method)."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return "Nothing to summarise.", "none"
    if llm.available:
        try:
            parts = _split(text)
            if len(parts) > 1:  # map step
                partials = [llm.generate(f"Summarise this section of lecture notes in detailed bullet points, "
                                         f"keeping all definitions and formulas:\n\n{p}") for p in parts[:12]]
                text = "\n\n".join(partials)
            out = llm.generate(f"{MODES[mode]}\nAudience level: {difficulty}.\n"
                               f"Use only the material below; do not add outside facts.\n\nMATERIAL:\n{text}")
            return out, f"abstractive ({llm.label})"
        except LLMError:
            pass
    if mode == "Definitions":
        defs = extract_definitions(text)
        if defs:
            rows = "\n".join(f"| {t} | {d} |" for t, d in defs)
            return f"| Term | Definition |\n|---|---|\n{rows}", "extractive (pattern matching)"
    n = {"Short notes": 8, "Important points": 12, "Exam summary": 10, "Teaching notes": 10}.get(mode, 8)
    picked = extractive_summary(text, n)
    if not picked:
        return "Not enough complete sentences to summarise (is this a scanned PDF?).", "none"
    return "\n".join(f"- {s}" for s in picked), "extractive (TF-IDF sentence ranking)"


def extractive_summary(text: str, n: int = 8) -> list[str]:
    sents = [s for s in sentences(text) if 6 <= len(s.split()) <= 60 and not is_heading(s)]
    if len(sents) <= n:
        return sents
    vec = TfidfVectorizer(stop_words=list(STOPWORDS))
    m = vec.fit_transform(sents)
    # sentence score = mean TF-IDF weight of its terms, small bonus for early sentences
    lengths = np.maximum((m > 0).sum(axis=1).A1, 1)
    scores = m.sum(axis=1).A1 / lengths
    scores += 0.05 * (1 - np.arange(len(sents)) / len(sents))
    keep = sorted(np.argsort(-scores)[:n])
    return [sents[i] for i in keep]


_DEF_PATTERNS = [
    r"^(?P<term>[A-Z][\w\-/() ]{1,50}?)\s+(?:is|are)\s+(?:a|an|the)\s+(?P<def>.{15,240})",
    r"^(?P<term>[A-Z][\w\-/() ]{1,50}?)\s+(?:refers to|is defined as|means)\s+(?P<def>.{10,240})",
]


def extract_definitions(text: str, limit: int = 25) -> list[tuple[str, str]]:
    out, seen = [], set()
    for s in sentences(text):
        for p in _DEF_PATTERNS:
            m = re.match(p, s)
            if m:
                term = m.group("term").strip()
                if term.lower() in seen or len(term.split()) > 6 or term.lower() in ("it", "this", "there"):
                    continue
                seen.add(term.lower())
                out.append((term, s))
                break
        if len(out) == limit:
            break
    return out
