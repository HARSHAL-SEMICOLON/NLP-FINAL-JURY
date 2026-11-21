"""Classical NLP tasks (Unit II): NER, dependency parsing, sentiment, topic modelling.

* NER            – NLTK maxent chunker on POS tags (spaCy used instead when installed)
* Dependency     – spaCy only (optional: pip install spacy && python -m spacy download en_core_web_sm)
* Sentiment      – VADER lexicon + rules; used on student feedback for the faculty dashboard
* Topic modelling – LDA over the uploaded material or over the questions students ask
"""
from __future__ import annotations

import nltk
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from . import preprocess  # noqa: F401  (adds the project nltk_data path)
from .preprocess import STOPWORDS, tokenize


def _has(res: str) -> bool:
    try:
        nltk.data.find(res)
        return True
    except LookupError:
        return False


HAS_NER = _has("chunkers/maxent_ne_chunker_tab") and _has("corpora/words") and preprocess.HAS_TAGGER
HAS_VADER = _has("sentiment/vader_lexicon.zip") or _has("sentiment/vader_lexicon")

_spacy_nlp = None


def spacy_model():
    global _spacy_nlp
    if _spacy_nlp is None:
        try:
            import spacy

            _spacy_nlp = spacy.load("en_core_web_sm")
        except Exception:  # not installed
            _spacy_nlp = False
    return _spacy_nlp or None


# ---------------------------------------------------------------- NER
def entities(text: str) -> list[dict]:
    nlp = spacy_model()
    if nlp:
        return [{"entity": e.text, "type": e.label_} for e in nlp(text).ents]
    if not HAS_NER:
        return []
    tree = nltk.ne_chunk(nltk.pos_tag(tokenize(text)))
    return [{"entity": " ".join(w for w, _ in st.leaves()), "type": st.label()}
            for st in tree if hasattr(st, "label")]


def dependencies(text: str) -> list[dict]:
    nlp = spacy_model()
    if not nlp:
        return []
    return [{"token": t.text, "dep": t.dep_, "head": t.head.text, "POS": t.pos_} for t in nlp(text)]


# ---------------------------------------------------------------- sentiment
_vader = None


def sentiment(text: str) -> dict:
    """{'label': positive|negative|neutral, 'compound': -1..1}"""
    global _vader
    if not text.strip():
        return {"label": "neutral", "compound": 0.0}
    if HAS_VADER:
        if _vader is None:
            from nltk.sentiment import SentimentIntensityAnalyzer

            _vader = SentimentIntensityAnalyzer()
        c = _vader.polarity_scores(text)["compound"]
    else:  # tiny fallback lexicon
        pos = {"good", "great", "helpful", "clear", "thanks", "excellent", "useful", "understood", "nice"}
        neg = {"bad", "confusing", "confused", "wrong", "useless", "unclear", "difficult", "poor"}
        toks = [t.lower() for t in tokenize(text)]
        c = (sum(t in pos for t in toks) - sum(t in neg for t in toks)) / max(len(toks) ** 0.5, 1)
    label = "positive" if c >= 0.05 else "negative" if c <= -0.05 else "neutral"
    return {"label": label, "compound": round(c, 3)}


# ---------------------------------------------------------------- topic modelling
def topics(texts: list[str], n_topics: int = 4, n_words: int = 8) -> tuple[list[list[str]], list[int]]:
    """LDA topic model. Returns (top words per topic, dominant topic per text)."""
    texts = [t for t in texts if t.strip()]
    if len(texts) < 2:
        return [], []
    vec = CountVectorizer(stop_words=list(STOPWORDS), token_pattern=r"(?u)\b[A-Za-z][A-Za-z-]{2,}\b",
                          max_df=0.9, min_df=1)
    try:
        X = vec.fit_transform(texts)
    except ValueError:
        return [], []
    n_topics = max(1, min(n_topics, X.shape[0]))
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=0, learning_method="batch")
    doc_topic = lda.fit_transform(X)
    vocab = vec.get_feature_names_out()
    words = [[vocab[i] for i in comp.argsort()[::-1][:n_words]] for comp in lda.components_]
    return words, [int(r.argmax()) for r in doc_topic]
