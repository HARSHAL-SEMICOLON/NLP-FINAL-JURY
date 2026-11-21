"""Classical NLP preprocessing (Unit: text preprocessing).

Uses NLTK when its data packages are installed, and falls back to small
built-in implementations otherwise, so the app never breaks on a fresh machine.
Run `python -m sah.setup_nltk` once to download the NLTK data.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import nltk
from nltk.stem import PorterStemmer  # needs no downloaded data

from .config import ROOT

# NLTK data shipped inside the project (created by `python -m sah.setup_nltk`)
PROJECT_NLTK = ROOT / "nltk_data"
if str(PROJECT_NLTK) not in nltk.data.path:
    nltk.data.path.insert(0, str(PROJECT_NLTK))

_stemmer = PorterStemmer()

# Fallback English stop-word list (subset of NLTK's list).
_FALLBACK_STOPWORDS = set(
    """a about above after again against all am an and any are as at be because been before being
    below between both but by can could did do does doing down during each few for from further had
    has have having he her here hers herself him himself his how i if in into is it its itself just
    me more most my myself no nor not now of off on once only or other our ours ourselves out over own
    same she should so some such than that the their theirs them themselves then there these they
    this those through to too under until up very was we were what when where which while who whom
    why will with would you your yours yourself yourselves also may might must shall""".split()
)


def _nltk_has(resource: str) -> bool:
    try:
        nltk.data.find(resource)
        return True
    except LookupError:
        return False


HAS_PUNKT = _nltk_has("tokenizers/punkt_tab")
HAS_STOPWORDS = _nltk_has("corpora/stopwords")
HAS_WORDNET = _nltk_has("corpora/wordnet")
HAS_TAGGER = _nltk_has("taggers/averaged_perceptron_tagger_eng")

if HAS_STOPWORDS:
    from nltk.corpus import stopwords

    STOPWORDS = set(stopwords.words("english"))
else:
    STOPWORDS = _FALLBACK_STOPWORDS

if HAS_WORDNET:
    from nltk.stem import WordNetLemmatizer

    _lemmatizer = WordNetLemmatizer()
else:
    _lemmatizer = None


def clean_text(text: str) -> str:
    """Normalise whitespace, strip URLs/odd symbols, keep sentence punctuation."""
    text = re.sub(r"https?://\S+", " ", text)
    text = text.replace("­", "")              # soft hyphens from PDFs
    text = re.sub(r"-\n(\w)", r"\1", text)          # re-join hyphenated line breaks
    text = re.sub(r"[^\w\s.,;:!?()'\"%/+-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def sentences(text: str) -> list[str]:
    if HAS_PUNKT:
        from nltk.tokenize import sent_tokenize

        return [s.strip() for s in sent_tokenize(text) if s.strip()]
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text) if s.strip()]


def is_heading(sentence: str) -> bool:
    """Title-like fragments ("Unit 1 - Introduction to NLP.") are not real sentences."""
    words = sentence.rstrip(".").split()
    if not words or len(words) > 10:
        return False
    not_lower = sum(1 for w in words if not w[:1].islower())
    return not_lower / len(words) >= 0.6


def tokenize(text: str) -> list[str]:
    if HAS_PUNKT:
        from nltk.tokenize import word_tokenize

        return word_tokenize(text)
    return re.findall(r"[A-Za-z0-9]+(?:'[a-z]+)?|[^\sA-Za-z0-9]", text)


def remove_stopwords(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t.lower() not in STOPWORDS and re.search(r"\w", t)]


def stem(tokens: list[str]) -> list[str]:
    return [_stemmer.stem(t.lower()) for t in tokens]


def _rule_lemma(word: str) -> str:
    w = word.lower()
    for suffix, repl in (("ies", "y"), ("sses", "ss"), ("ing", ""), ("ed", ""), ("s", "")):
        if suffix == "s" and w.endswith(("ss", "us", "is")):
            continue
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            return w[: -len(suffix)] + repl
    return w


def pos_tag(tokens: list[str]) -> list[str]:
    """Penn Treebank POS tags (NN, VBG, JJR …). Falls back to 'NN' for every token."""
    if HAS_TAGGER and tokens:
        return [tag for _, tag in nltk.pos_tag(tokens)]
    return ["NN"] * len(tokens)


def _wordnet_pos(tag: str) -> str:
    # Penn Treebank tag -> WordNet POS: adjective, verb, adverb, noun (default)
    return {"J": "a", "V": "v", "R": "r"}.get(tag[:1], "n")


def lemmatize(tokens: list[str], tags: list[str] | None = None) -> list[str]:
    """POS-aware lemmatization: running/VBG -> run, better/JJR -> good, studies/NNS -> study.
    Without POS tags WordNet treats every word as a noun ("running" stays "running")."""
    if _lemmatizer is not None:
        tags = tags or pos_tag(tokens)
        return [_lemmatizer.lemmatize(t.lower(), _wordnet_pos(g)) for t, g in zip(tokens, tags)]
    return [_rule_lemma(t) for t in tokens]


def normalize_for_search(text: str) -> str:
    """Lower-case, drop stop words, POS-aware lemmatize. Used for TF-IDF retrieval."""
    toks = tokenize(clean_text(text))
    tags = pos_tag(toks)                       # tag the full sentence: context helps the tagger
    keep = [(t, g) for t, g in zip(toks, tags) if t.lower() not in STOPWORDS and re.search(r"\w", t)]
    return " ".join(lemmatize([t for t, _ in keep], [g for _, g in keep]))


@dataclass
class PipelineTrace:
    """Every stage of the pipeline, for the NLP Lab page (viva demonstration)."""

    original: str
    cleaned: str
    sentences: list[str] = field(default_factory=list)
    tokens: list[str] = field(default_factory=list)
    pos_tags: list[str] = field(default_factory=list)
    without_stopwords: list[str] = field(default_factory=list)
    stems: list[str] = field(default_factory=list)
    lemmas: list[str] = field(default_factory=list)
    backend: str = ""


def trace(text: str) -> PipelineTrace:
    cleaned = clean_text(text)
    toks = tokenize(cleaned)
    tags = pos_tag(toks)
    keep = [(t, g) for t, g in zip(toks, tags) if t.lower() not in STOPWORDS and re.search(r"\w", t)]
    no_sw, no_sw_tags = [t for t, _ in keep], [g for _, g in keep]
    backend = ", ".join(
        f"{name}: {'NLTK' if ok else 'built-in fallback'}"
        for name, ok in (("tokenizer", HAS_PUNKT), ("stop words", HAS_STOPWORDS),
                         ("POS tagger", HAS_TAGGER), ("lemmatizer", HAS_WORDNET))
    )
    return PipelineTrace(
        original=text,
        cleaned=cleaned,
        sentences=sentences(cleaned),
        tokens=toks,
        pos_tags=no_sw_tags,
        without_stopwords=no_sw,
        stems=stem(no_sw),
        lemmas=lemmatize(no_sw, no_sw_tags),
        backend=backend,
    )
