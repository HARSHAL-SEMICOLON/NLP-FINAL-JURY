"""Quiz generation and grading (Units: prompt engineering, TF-IDF, semantic similarity).

* With an LLM: MCQs / short-answer questions as structured JSON.
* Offline: cloze-style MCQs — the most informative TF-IDF term in a sentence
  is blanked out and other key terms from the material become distractors.
* Short answers are graded by semantic similarity + keyword coverage, so
  grading works with or without an LLM.
"""
from __future__ import annotations

import random
import re

from sklearn.feature_extraction.text import TfidfVectorizer

from .embeddings import cosine
from .llm import LLM, LLMError
from .preprocess import STOPWORDS, is_heading, normalize_for_search, sentences, stem
from .vectorstore import VectorStore

MCQ_PROMPT = """Create {n} multiple-choice questions from the STUDY MATERIAL below for an
engineering exam. Difficulty: {difficulty}. Topic focus: {topic}.
Bloom's taxonomy level for every question: {bloom}.
Each question must be answerable from the material. Mix recall, understanding and application.
Return ONLY JSON: {{"questions": [{{"question": str, "options": [4 strings],
"answer_index": 0-3, "explanation": str (why the answer is right and the others wrong),
"topic": short topic name}}]}}

STUDY MATERIAL:
{context}
"""

SHORT_PROMPT = """Create {n} short-answer exam questions (2–5 mark style) from the STUDY MATERIAL.
Difficulty: {difficulty}. Topic focus: {topic}.
Bloom's taxonomy level for every question: {bloom}.
Return ONLY JSON: {{"questions": [{{"question": str, "model_answer": str (3–5 sentences),
"keywords": [3–6 key terms a good answer must mention], "topic": short topic name}}]}}

STUDY MATERIAL:
{context}
"""

GRADE_PROMPT = """Grade the student's answer out of 10 against the model answer.
Question: {q}
Model answer: {model}
Student answer: {student}
Return ONLY JSON: {{"score": 0-10, "feedback": "2-3 sentences: what was right, what was missing"}}"""


def _chunks(store: VectorStore, topic: str, docs: list[str] | None, max_words: int = 1800):
    if topic.strip():
        chunks = [h.chunk for h in store.search(topic, k=8, docs=docs)]
    else:
        chunks = store.chunks_for(docs)
        random.shuffle(chunks)
    out, words = [], 0
    for c in chunks:
        out.append(c)
        words += len(c.text.split())
        if words > max_words:
            break
    return out


def _context(chunks) -> str:
    return "\n\n".join(f"(from {c.doc}, p.{c.page}) {c.text}" for c in chunks)


def complete_sentences(chunks) -> list[str]:
    """Sentences from each chunk, dropping the cut-off fragments at chunk edges."""
    out = []
    for c in chunks:
        sents = sentences(c.text)
        if sents and not sents[0][:1].isupper():
            sents = sents[1:]
        if sents and not sents[-1].rstrip().endswith((".", "!", "?")):
            sents = sents[:-1]
        for s_ in sents:
            if s_ not in out and not is_heading(s_):
                out.append(s_)
    return out


def _bloom_text(bloom: str) -> str:
    from .syllabus import BLOOM_HELP

    return f"{bloom} – {BLOOM_HELP[bloom]}" if bloom in BLOOM_HELP else "mixed (Remember to Analyze)"


def generate_mcq(store: VectorStore, llm: LLM, n: int = 5, topic: str = "", docs=None,
                 difficulty: str = "Intermediate", bloom: str = "Mixed") -> list[dict]:
    chunks = _chunks(store, topic, docs)
    if not chunks:
        return []
    ctx = _context(chunks)
    if llm.available:
        try:
            data = llm.generate_json(MCQ_PROMPT.format(n=n, difficulty=difficulty, topic=topic or "whole material",
                                                       bloom=_bloom_text(bloom), context=ctx))
            qs = data["questions"] if isinstance(data, dict) else data
            return [q for q in qs if len(q.get("options", [])) == 4 and 0 <= int(q.get("answer_index", -1)) < 4][:n]
        except (LLMError, ValueError, KeyError, TypeError):
            pass
    return cloze_mcq(complete_sentences(chunks), n, topic)


def cloze_mcq(sents: list[str], n: int = 5, topic: str = "") -> list[dict]:
    """Offline MCQ generator using TF-IDF keyword extraction."""
    sents = [s for s in sents if 8 <= len(s.split()) <= 45]
    if len(sents) < 2:
        return []
    vec = TfidfVectorizer(stop_words=list(STOPWORDS), token_pattern=r"(?u)\b[A-Za-z][A-Za-z-]{3,}\b")
    m = vec.fit_transform(sents)
    vocab = vec.get_feature_names_out()
    global_rank = m.sum(axis=0).A1
    pool, stems = [], set()
    for i in global_rank.argsort()[::-1][:60]:  # top corpus terms, one per stem ("word"/"words")
        st = stem([vocab[i]])[0]
        if st not in stems:
            stems.add(st)
            pool.append(vocab[i])

    rng = random.Random()
    questions, used = [], set()
    order = sorted(range(len(sents)), key=lambda i: -m[i].sum())
    for i in order:
        row = m[i].toarray()[0]
        for j in row.argsort()[::-1][:3]:
            term = vocab[j]
            if row[j] == 0 or term in used:
                continue
            pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
            if not pattern.search(sents[i]):
                continue
            distractors = [w for w in pool if stem([w])[0] != stem([term])[0] and w not in sents[i].lower()]
            if len(distractors) < 3:
                continue
            opts = rng.sample(distractors[:15], 3) + [term]
            rng.shuffle(opts)
            questions.append({
                "question": "Fill in the blank: " + pattern.sub("_____", sents[i], count=1),
                "options": opts,
                "answer_index": opts.index(term),
                "explanation": f"The original sentence reads: “{sents[i]}”",
                "topic": topic or term,
            })
            used.add(term)
            break
        if len(questions) == n:
            break
    return questions


def generate_short(store: VectorStore, llm: LLM, n: int = 3, topic: str = "", docs=None,
                   difficulty: str = "Intermediate", bloom: str = "Mixed") -> list[dict]:
    chunks = _chunks(store, topic, docs)
    if not chunks:
        return []
    ctx = _context(chunks)
    if llm.available:
        try:
            data = llm.generate_json(SHORT_PROMPT.format(n=n, difficulty=difficulty, bloom=_bloom_text(bloom),
                                                         topic=topic or "whole material", context=ctx))
            return (data["questions"] if isinstance(data, dict) else data)[:n]
        except (LLMError, ValueError, KeyError, TypeError):
            pass
    # Offline: turn definition-like sentences into "Explain …" questions.
    out = []
    for s in complete_sentences(chunks):
        m = re.match(r"^([A-Z][\w\- ()]{2,60}?) (?:is|are|refers to|means) (.+)", s)
        if m and len(s.split()) > 8:
            keys = [w for w in re.findall(r"[A-Za-z][A-Za-z-]+", m.group(2))
                    if len(w) > 4 and w.lower() not in STOPWORDS][:5]
            out.append({"question": f"Explain {m.group(1).strip()}.", "model_answer": s,
                        "keywords": keys, "topic": m.group(1).strip()})
        if len(out) == n:
            break
    return out


def grade_short(question: dict, student: str, llm: LLM) -> dict:
    """Score 0–10. Hybrid: semantic similarity (60%) + keyword coverage (40%),
    replaced by LLM judgement when an LLM is available."""
    student = student.strip()
    if not student:
        return {"score": 0, "feedback": "No answer given.", "similarity": 0.0, "coverage": 0.0}
    sim = max(0.0, cosine(question["model_answer"], student))
    norm = set(normalize_for_search(student).split())
    keys = [normalize_for_search(k) for k in question.get("keywords", [])]
    covered = [k for k in keys if k and sum(t in norm for t in k.split()) >= max(1, len(k.split()) / 2)]
    coverage = len(covered) / len(keys) if keys else sim
    score = round(10 * (0.6 * min(sim * 1.6, 1.0) + 0.4 * coverage), 1)
    result = {"score": score, "similarity": round(sim, 3), "coverage": round(coverage, 2),
              "feedback": f"Covered key terms: {', '.join(covered) or 'none'}. "
                          f"Missing: {', '.join(k for k in keys if k not in covered) or 'none'}."}
    if llm.available:
        try:
            j = llm.generate_json(GRADE_PROMPT.format(q=question["question"], model=question["model_answer"],
                                                      student=student), temperature=0.0)
            result.update(score=float(j["score"]), feedback=j["feedback"])
        except (LLMError, ValueError, KeyError, TypeError):
            pass
    return result
