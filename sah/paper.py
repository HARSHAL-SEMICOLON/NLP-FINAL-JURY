"""Question-paper generator for faculty (prompt engineering + structured output).

Builds an exam paper from the syllabus units the teacher selects, with marks, Bloom's
level and Course Outcome (CO) for every question – the format ESA papers use.
* LLM mode: questions grounded in the uploaded material for each unit (JSON output).
* Offline mode: Bloom-level templates filled with the syllabus topics of each unit.
"""
from __future__ import annotations

import random
import re

from .llm import LLM, LLMError
from .syllabus import BLOOM_HELP
from .vectorstore import VectorStore

# marks -> default Bloom levels used for that section
SECTIONS = {
    2: ["Remember", "Understand"],
    5: ["Apply", "Analyze"],
    10: ["Analyze", "Evaluate", "Create"],
}

PROMPT = """You are setting an end-semester university exam paper for the course "{course}".
Write {n} questions worth {marks} marks each for {unit} – {title}.
Syllabus topics of this unit: {topics}
Bloom's taxonomy levels to use (spread across the questions): {blooms}.
Ground the questions in the COURSE MATERIAL below when it is relevant; otherwise use the syllabus topics.
A {marks}-mark question should need about {minutes} minutes to answer. No duplicate questions.
Return ONLY JSON: {{"questions": [{{"question": str, "bloom": one of {bloom_list}, "answer_key": "3-5 key points an examiner looks for"}}]}}

COURSE MATERIAL:
{context}
"""

_TEMPLATES = {
    "Remember": ["Define {a}.", "What is meant by {a}?", "List the key features of {a}."],
    "Understand": ["Explain {a} with a suitable example.", "Describe how {a} works.",
                   "Discuss the role of {a} in NLP applications."],
    "Apply": ["Demonstrate {a} on a sample sentence of your choice.",
              "Apply {a} to a real-world dataset and explain each step."],
    "Analyze": ["Compare and contrast {a} and {b}.", "Differentiate between {a} and {b} with examples."],
    "Evaluate": ["Critically evaluate {a} for a real-world application. Justify your answer.",
                 "Assess the advantages and limitations of {a} compared with {b}."],
    "Create": ["Design a system that uses {a} and {b} to build an intelligent application. Explain the architecture.",
               "Propose an approach combining {a} and {b} for a student helpdesk chatbot."],
}


def _topic_list(topics: str) -> list[str]:
    parts = re.split(r"[,;:]|\band\b", topics)
    return [p.strip(" .") for p in parts if 2 <= len(p.strip().split()) <= 6 or (len(p.strip()) > 3 and len(p.split()) == 1)]


def _co_for(unit_index: int, n_cos: int) -> str:
    return f"CO{min(unit_index + 1, n_cos)}" if n_cos else f"CO{unit_index + 1}"


def generate(units: list[dict], plan: dict[int, int], llm: LLM, store: VectorStore | None = None,
             course: str = "Natural Language Processing with Large Models", n_cos: int = 5) -> list[dict]:
    """units: syllabus units chosen by the teacher; plan: {marks: number of questions}.
    Returns a list of {section_marks, question, unit, bloom, co, answer_key}."""
    rng = random.Random()
    paper: list[dict] = []
    all_units = units
    for marks, count in sorted(plan.items()):
        if count <= 0:
            continue
        # spread the questions of this section across the chosen units
        per_unit = {i: 0 for i in range(len(all_units))}
        for k in range(count):
            per_unit[k % len(all_units)] += 1
        for idx, n in per_unit.items():
            if n == 0:
                continue
            u = all_units[idx]
            blooms = SECTIONS.get(marks, ["Understand"])
            co = _co_for(u.get("index", idx), n_cos)
            qs = []
            if llm.available:
                ctx = ""
                if store is not None and store.chunks:
                    ctx = "\n\n".join(h.chunk.text for h in store.search(u["title"] + " " + u["topics"], k=4))
                try:
                    data = llm.generate_json(PROMPT.format(
                        course=course, n=n, marks=marks, unit=u["unit"], title=u["title"], topics=u["topics"],
                        blooms=", ".join(f"{b} ({BLOOM_HELP[b]})" for b in blooms), bloom_list=blooms,
                        minutes={2: 3, 5: 8, 10: 18}.get(marks, 8), context=ctx[:6000] or "(none)"))
                    qs = (data["questions"] if isinstance(data, dict) else data)[:n]
                except (LLMError, ValueError, KeyError, TypeError):
                    qs = []
            if not qs:  # offline templates
                topics = _topic_list(u["topics"]) or [u["title"]]
                for _ in range(n):
                    bloom = rng.choice(blooms)
                    a, b = (rng.sample(topics, 2) if len(topics) > 1 else (topics[0], topics[0]))
                    qs.append({"question": rng.choice(_TEMPLATES[bloom]).format(a=a, b=b), "bloom": bloom,
                               "answer_key": f"Definition and key idea of {a}; example; link to {u['title']}."})
            for q in qs:
                paper.append({"marks": marks, "question": str(q.get("question", "")).strip(), "unit": u["unit"],
                              "bloom": q.get("bloom", blooms[0]), "co": co,
                              "answer_key": q.get("answer_key", "")})
    return paper


def to_markdown(paper: list[dict], course: str, code: str, duration: str, total: int,
                with_key: bool = False) -> str:
    lines = [f"# {course} ({code})", "", f"**Duration:** {duration}  |  **Max. Marks:** {total}", "",
             "*Instructions: Answer all questions. Figures to the right indicate marks. "
             "Bloom's level and CO are shown for each question.*", ""]
    qno = 0
    for marks in sorted({q["marks"] for q in paper}):
        section = [q for q in paper if q["marks"] == marks]
        lines += [f"## Section – {marks} marks each", "",
                  "| Q.No | Question | Marks | Unit | Bloom's level | CO |", "|---|---|---|---|---|---|"]
        for q in section:
            qno += 1
            text = q["question"].replace("|", "/")
            lines.append(f"| {qno} | {text} | {marks} | {q['unit']} | {q['bloom']} | {q['co']} |")
        lines.append("")
    if with_key:
        lines += ["## Answer key (for examiner)", ""]
        for i, q in enumerate(paper, 1):
            lines.append(f"**Q{i}.** {q['answer_key']}")
        lines.append("")
    return "\n".join(lines)
