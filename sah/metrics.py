"""Evaluation metrics (Unit IV: perplexity, BLEU, ROUGE, hallucination).

Implemented from scratch so every formula can be explained in the viva.
"""
from __future__ import annotations

import math
import re
from collections import Counter

from .preprocess import sentences, tokenize


def _toks(text: str) -> list[str]:
    return [t.lower() for t in tokenize(text) if re.search(r"\w", t)]


def _ngrams(toks: list[str], n: int) -> Counter:
    return Counter(tuple(toks[i : i + n]) for i in range(len(toks) - n + 1))


# ---------------------------------------------------------------- BLEU
def bleu(reference: str, candidate: str, max_n: int = 4) -> dict:
    """Sentence BLEU = BP · exp(mean log p_n) with modified (clipped) n-gram precision.
    Zero counts get a tiny epsilon (Chen & Cherry smoothing method 1) so one missing 4-gram
    doesn't zero the whole score."""
    ref, cand = _toks(reference), _toks(candidate)
    if not cand or not ref:
        return {"bleu": 0.0, "precisions": [], "brevity_penalty": 0.0}
    precisions = []
    for n in range(1, max_n + 1):
        c, r = _ngrams(cand, n), _ngrams(ref, n)
        overlap = sum(min(cnt, r[g]) for g, cnt in c.items())
        total = max(sum(c.values()), 0)
        if total == 0:
            p = 0.0
        else:
            p = overlap / total if overlap else 0.1 / total
        precisions.append(p)
    bp = 1.0 if len(cand) > len(ref) else math.exp(1 - len(ref) / len(cand))
    score = bp * math.exp(sum(math.log(p) for p in precisions if p > 0) / max_n) if precisions[0] > 0 else 0.0
    return {"bleu": round(score, 4), "precisions": [round(p, 3) for p in precisions], "brevity_penalty": round(bp, 3)}


# ---------------------------------------------------------------- ROUGE
def _prf(overlap: int, cand_total: int, ref_total: int) -> dict:
    p = overlap / cand_total if cand_total else 0.0
    r = overlap / ref_total if ref_total else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3)}


def rouge_n(reference: str, candidate: str, n: int = 1) -> dict:
    r, c = _ngrams(_toks(reference), n), _ngrams(_toks(candidate), n)
    overlap = sum(min(cnt, r[g]) for g, cnt in c.items())
    return _prf(overlap, sum(c.values()), sum(r.values()))


def _lcs(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def rouge_l(reference: str, candidate: str, max_tokens: int = 1500) -> dict:
    r, c = _toks(reference)[:max_tokens], _toks(candidate)[:max_tokens]
    return _prf(_lcs(r, c), len(c), len(r))


def rouge_all(reference: str, candidate: str) -> dict:
    return {"ROUGE-1": rouge_n(reference, candidate, 1), "ROUGE-2": rouge_n(reference, candidate, 2),
            "ROUGE-L": rouge_l(reference, candidate)}


# ---------------------------------------------------------------- n-gram language model
class BigramLM:
    """Bigram language model with add-k (Laplace) smoothing, trained on the student's notes.
    Perplexity = exp(-1/N · Σ log P(w_i | w_{i-1})). Lower = text is more 'expected' given the notes."""

    def __init__(self, corpus: list[str], k: float = 1.0) -> None:
        self.k = k
        self.uni, self.bi = Counter(), Counter()
        for doc in corpus:
            for s in sentences(doc):
                toks = ["<s>"] + _toks(s) + ["</s>"]
                self.uni.update(toks)
                self.bi.update(zip(toks, toks[1:]))
        self.V = len(self.uni) + 1  # +1 for unknown words

    def prob(self, prev: str, word: str) -> float:
        return (self.bi[(prev, word)] + self.k) / (self.uni[prev] + self.k * self.V)

    def perplexity(self, text: str) -> dict:
        logp, n, steps = 0.0, 0, []
        for s in sentences(text) or [text]:
            toks = ["<s>"] + _toks(s) + ["</s>"]
            for prev, w in zip(toks, toks[1:]):
                p = self.prob(prev, w)
                logp += math.log(p)
                n += 1
                steps.append((prev, w, p))
        return {"perplexity": round(math.exp(-logp / n), 2) if n else float("inf"), "tokens": n, "steps": steps}


# ---------------------------------------------------------------- hallucination / grounding
def grounding(answer: str, contexts: list[str], threshold: float = 0.25) -> dict:
    """Share of answer sentences supported by the retrieved context.
    A sentence is 'supported' when ≥ threshold of its content words appear in some context chunk
    (unigram recall, like ROUGE-1 recall). Unsupported sentences are possible hallucinations."""
    from .preprocess import normalize_for_search

    ctx_sets = [set(normalize_for_search(c).split()) for c in contexts]
    rows = []
    for s in sentences(re.sub(r"[#*|`>]", " ", answer)):
        words = set(normalize_for_search(s).split())
        if len(words) < 4 or s.lower().startswith(("exam tip", "general knowledge")):
            continue
        best = max((len(words & c) / len(words) for c in ctx_sets), default=0.0)
        rows.append({"sentence": s, "support": round(best, 2), "supported": best >= threshold})
    score = sum(r["supported"] for r in rows) / len(rows) if rows else 0.0
    return {"score": round(score, 2), "sentences": rows}
