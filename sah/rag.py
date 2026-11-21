"""Retrieval-Augmented Generation (Units: RAG, prompt engineering, responsible AI).

Pipeline for one question:
  1. preprocess the query            (preprocess.py)
  2. embed it                        (embeddings.py)
  3. semantic search over chunks     (vectorstore.py)
  4. build a grounded prompt         (this file)
  5. generate with the LLM           (llm.py)
  6. return answer + source refs + whether it is grounded in the material
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import config
from . import metrics
from .llm import LLM, LLMError
from .preprocess import normalize_for_search, sentences
from .vectorstore import Hit, VectorStore

LANGUAGES = ["English", "Hindi", "Marathi"]


def _language_rule(language: str) -> str:
    if language == "English":
        return ""
    return (f"- Write the whole answer in {language} (machine translation of the explanation). Keep technical "
            f"terms such as TF-IDF, Transformer, embedding in English inside brackets after the {language} word.\n")

DIFFICULTY = {
    "Beginner": "Explain like a patient tutor to a first-year student: plain words, a simple analogy, "
                "define every technical term, short paragraphs.",
    "Intermediate": "Explain for a third-year engineering student: correct terminology, key steps or "
                    "formulas, one worked example where useful.",
    "Advanced": "Explain at exam-topper / interview depth: precise terminology, internals, trade-offs, "
                "complexity, and how it compares with related methods.",
}

SYSTEM = (
    "You are Smart Academic Help, an AI teaching assistant for engineering students. "
    "Be accurate, structured and exam-oriented. Use Markdown headings and bullet points where they help. "
    "Never invent citations."
)


@dataclass
class Answer:
    text: str
    sources: list[Hit] = field(default_factory=list)
    grounded: bool = False          # True when supported by uploaded material
    mode: str = "rag"               # "rag" | "general"
    model: str = ""
    grounding: dict | None = None   # share of answer sentences supported by the sources
    language: str = "English"

    @property
    def basis(self) -> str:
        if self.mode == "general":
            return "General LLM knowledge"
        return "Based on your uploaded material" if self.grounded else "Not found in your material – general knowledge"


def tidy(text: str) -> str:
    """Normalise model output: 【1】 citations → [1], <br> in tables → ' · ',
    and LaTeX \\[ … \\] / \\( … \\) → $$ … $$ / $ … $ so Streamlit renders the maths."""
    text = re.sub(r"\\\[(.+?)\\\]", lambda m: "$$" + m.group(1).strip() + "$$", text, flags=re.DOTALL)
    text = re.sub(r"\\\((.+?)\\\)", lambda m: "$" + m.group(1).strip() + "$", text, flags=re.DOTALL)
    text = re.sub(r"[【\[](\d+)[】\]]", r"[\1]", text)
    return re.sub(r"\s*<br\s*/?>\s*", " · ", text, flags=re.IGNORECASE)


def diversify(hits: list[Hit], k: int, per_doc: int = 2) -> list[Hit]:
    """Best-scoring passages, but at most `per_doc` from one document while others still
    have relevant passages – so a 600-page textbook can't crowd out the teacher's notes."""
    picked, count = [], {}
    for h in hits:
        if count.get(h.chunk.doc, 0) < per_doc:
            picked.append(h)
            count[h.chunk.doc] = count.get(h.chunk.doc, 0) + 1
        if len(picked) == k:
            return picked
    for h in hits:  # not enough variety – fill up with the next best
        if h not in picked:
            picked.append(h)
        if len(picked) == k:
            break
    return picked


def _window_text(h: Hit, store=None) -> str:
    """The passage plus its neighbours, de-duplicating the 30-word overlap between chunks."""
    if store is None:
        return h.chunk.text
    words: list[str] = []
    for c in store.neighbours(h.chunk):
        w = c.text.split()
        overlap = next((n for n in range(min(len(words), len(w), 40), 0, -1) if words[-n:] == w[:n]), 0)
        words.extend(w[overlap:])
    return " ".join(words)


def _context_block(hits: list[Hit], store=None) -> str:
    return "\n\n".join(f"[{i}] (source: {h.chunk.doc}, page {h.chunk.page})\n{_window_text(h, store)}"
                       for i, h in enumerate(hits, 1))


def build_rag_prompt(question: str, hits: list[Hit], difficulty: str, history: list[dict],
                     language: str = "English", store=None) -> str:
    convo = "\n".join(f"{m['role'].upper()}: {m['content'][:500]}" for m in history[-4:])
    return f"""Answer the student's question using the CONTEXT from their own study material.

Rules:
- Use the context first and cite it inline as [1], [2] … matching the context numbers.
- The context is extracted from PDFs, so formulas and tables may look garbled. If any passage is about
  the topic, it IS covered: explain it clearly (fix the formula from your knowledge) and cite that passage.
- Only if no passage is about the topic at all, begin with exactly
  "⚠️ This topic is not covered in your uploaded material." and then answer from general
  knowledge, under a heading "General knowledge", without citations.
- Write formulas in LaTeX between $…$ (inline) or $$…$$ (display).
- Do not cite a number that is not in the context. Use plain square brackets like [1].
- Use Markdown only (no HTML tags such as <br>).
{_language_rule(language)}- Style: {DIFFICULTY.get(difficulty, DIFFICULTY['Intermediate'])}
- End with a 2–3 line "Exam tip".

CONTEXT:
{_context_block(hits, store) if hits else '(no relevant passages found)'}

RECENT CONVERSATION:
{convo or '(none)'}

QUESTION: {question}
"""


def build_general_prompt(question: str, difficulty: str, history: list[dict], language: str = "English") -> str:
    convo = "\n".join(f"{m['role'].upper()}: {m['content'][:500]}" for m in history[-4:])
    return f"""{_language_rule(language)}Style: {DIFFICULTY.get(difficulty, DIFFICULTY['Intermediate'])}
End with a 2–3 line "Exam tip".

RECENT CONVERSATION:
{convo or '(none)'}

QUESTION: {question}
"""


def extractive_answer(question: str, hits: list[Hit], max_sentences: int = 5) -> str:
    """Offline fallback: pick the sentences from retrieved chunks that share the
    most (lemmatised, stop-word-free) terms with the question."""
    q_terms = set(normalize_for_search(question).split())
    scored = []
    for rank, h in enumerate(hits):
        for s in sentences(h.chunk.text):
            terms = set(normalize_for_search(s).split())
            overlap = len(q_terms & terms) / (len(q_terms) or 1)
            if overlap > 0:
                scored.append((overlap + h.score - 0.01 * rank, s, rank + 1))
    scored.sort(key=lambda x: -x[0])
    seen, picked = set(), []
    for _, s, ref in scored:
        key = s.lower()[:80]
        if key not in seen:
            seen.add(key)
            picked.append(f"- {s} [{ref}]")
        if len(picked) == max_sentences:
            break
    if not picked:
        return "I could not find this in your uploaded material, and no LLM is configured for a general answer."
    return ("**Key passages from your material** (offline extractive mode – add a Gemini API key "
            "for full explanations):\n\n" + "\n".join(picked))


class Assistant:
    def __init__(self, store: VectorStore | None = None, llm: LLM | None = None) -> None:
        self.store = store or VectorStore()
        self.llm = llm or LLM()

    def retrieve(self, question: str, k: int = config.TOP_K, docs: list[str] | None = None,
                 history: list[dict] | None = None) -> list[Hit]:
        query = question
        # Follow-up questions ("explain it again", "give an example") carry little
        # content on their own, so add the previous user question for retrieval.
        if history and len(question.split()) < 6:
            prev = [m["content"] for m in history if m["role"] == "user"]
            if prev:
                query = prev[-1] + " " + question
        k = max(k, 5)
        return diversify(self.store.search(query, k=k * 4, docs=docs), k)

    def ask(self, question: str, difficulty: str = "Intermediate", use_material: bool = True,
            docs: list[str] | None = None, history: list[dict] | None = None,
            language: str = "English") -> Answer:
        history = history or []
        if not use_material or not self.store.chunks:
            if not self.llm.available:
                return Answer("No LLM is configured and no material is uploaded. Add GROQ_API_KEY to .env "
                              "or upload a PDF.", mode="general")
            text = tidy(self.llm.generate(build_general_prompt(question, difficulty, history, language), SYSTEM))
            return Answer(text, mode="general", model=self.llm.label, language=language)

        hits = self.retrieve(question, docs=docs, history=history)
        relevant = [h for h in hits if h.score >= config.MIN_RELEVANCE[self.store.embedder.name]]
        grounded = bool(relevant)

        if self.llm.available:
            try:
                text = tidy(self.llm.generate(build_rag_prompt(question, relevant, difficulty, history, language,
                                                               self.store), SYSTEM))
                if text.startswith("⚠️"):
                    grounded = False
                # keep only sources the model actually cited
                cited = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
                sources = [h for i, h in enumerate(relevant, 1) if i in cited] or relevant
                ground = (metrics.grounding(text, [_window_text(h, self.store) for h in relevant])
                          if grounded and language == "English" else None)
                return Answer(text, sources if grounded else [], grounded, "rag", self.llm.label, ground, language)
            except LLMError:
                pass  # fall through to offline answer
        text = extractive_answer(question, relevant)
        return Answer(text, relevant, grounded, "rag", "offline-extractive",
                      metrics.grounding(text, [h.chunk.text for h in relevant]) if grounded else None)
