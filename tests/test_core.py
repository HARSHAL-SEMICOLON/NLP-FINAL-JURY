"""Run: pytest -q   (works offline; no API key or Arduino needed)"""
import os
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["SAH_DATA_DIR"] = tempfile.mkdtemp()
os.environ["LLM_PROVIDER"] = "offline"
os.environ["EMBEDDING_BACKEND"] = "tfidf"
os.environ["ARDUINO_IP"] = ""

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from sah import arduino, db, documents, evaluation, planner, preprocess, quiz, summarize  # noqa: E402
from sah.llm import LLM, parse_json  # noqa: E402
from sah.rag import Assistant  # noqa: E402
from sah.vectorstore import VectorStore  # noqa: E402

SAMPLE = ROOT / "sample_data" / "NLP_Unit1_Notes.pdf"


@pytest.fixture(scope="module")
def store():
    if not SAMPLE.exists():
        import runpy

        runpy.run_path(str(ROOT / "sample_data" / "make_sample_pdf.py"))
    s = VectorStore(Path(os.environ["SAH_DATA_DIR"]) / "idx")
    s.add(documents.load_document(SAMPLE))
    return s


def test_preprocessing():
    t = preprocess.trace("The students were running experiments on tokenization.")
    assert "students" in t.tokens
    assert "the" not in [w.lower() for w in t.without_stopwords]
    assert "run" in t.stems
    if preprocess.HAS_TAGGER and preprocess.HAS_WORDNET:  # POS-aware lemmatization
        assert t.lemmas[t.without_stopwords.index("running")] == "run"


def test_chunking_and_headings(store):
    assert len(store.chunks) >= 4
    assert {c.page for c in store.chunks} == {1, 2, 3, 4}
    assert all(not c.text.startswith("Unit 1 - Text Preprocessing Tokenization") for c in store.chunks)


def test_rag_grounded_and_ungrounded(store):
    a = Assistant(store, LLM("offline"))
    ans = a.ask("What is the difference between stemming and lemmatization?")
    assert ans.grounded and ans.sources[0].chunk.page == 2
    off = a.ask("Explain photosynthesis in green plants")
    assert not off.grounded


def test_mcq_and_short(store):
    llm = LLM("offline")
    mcq = quiz.generate_mcq(store, llm, 3)
    assert mcq and all(len(q["options"]) == 4 for q in mcq)
    assert all("_____" in q["question"] for q in mcq)
    short = quiz.generate_short(store, llm, 2, topic="stemming")
    assert short
    good = quiz.grade_short(short[0], short[0]["model_answer"], llm)
    bad = quiz.grade_short(short[0], "I like cricket", llm)
    assert good["score"] > bad["score"]


def test_summaries():
    text = documents.full_text(SAMPLE)
    llm = LLM("offline")
    for mode in summarize.MODES:
        out, method = summarize.summarize(text, mode, llm)
        assert out and method.startswith("extractive")
    terms = [t for t, _ in summarize.extract_definitions(text)]
    assert "Tokenization" in terms and "TF-IDF" in terms


def test_planner_and_db():
    db.init()
    exam = (date.today() + timedelta(days=6)).isoformat()
    sid = db.upsert_subject("NLP", exam)
    for unit, name, d in [("U1", "Tokenization", 1), ("U1", "TF-IDF", 2), ("U2", "Transformers", 3)]:
        db.add_topic(sid, unit, name, d)
    sched, risk = planner.build_plan(db.topics(), hours_per_day=2)
    assert set(sched[sched.type == "study"].topic) == {"Tokenization", "TF-IDF", "Transformers"}
    assert (sched.type == "revision").any()
    assert risk.empty
    for _ in range(2):
        db.record_attempt("NLP", "TF-IDF", "mcq", "q", False, 0.0)
    assert "TF-IDF" in set(db.weak_topics().topic)


def test_evaluation(store):
    tests = pd.DataFrame({"question": ["What is tokenization?", "How is IDF calculated?"],
                          "doc": ["NLP_Unit1_Notes.pdf"] * 2, "page": [2, 3]})
    metrics, _ = evaluation.evaluate(store, tests, k=3)
    assert metrics["Hit@3"] == 1.0


def test_parse_json_and_arduino_disabled():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert arduino.set_status("ready") is False  # no board configured -> graceful no-op


# ---------------------------------------------------------------- syllabus, metrics, classical NLP, roles
SYLLABUS = ROOT / "sample_data" / "23CSE4021_NLP_Syllabus.pdf"


def test_syllabus_units_and_tagging():
    from sah import syllabus

    units = syllabus.parse_units(documents.full_text(SYLLABUS))
    assert [u["unit"] for u in units] == ["Unit I", "Unit II", "Unit III", "Unit IV", "Unit V"]
    syllabus.save(units, SYLLABUS.name)
    assert syllabus.tag_unit("What is stemming?")["unit"] == "Unit II"
    assert syllabus.tag_unit("Explain multi-head attention")["unit"] == "Unit III"
    assert syllabus.tag_unit("Who won the cricket world cup")["unit"] == "Outside syllabus"
    assert syllabus.bloom_level("Define TF-IDF") == "Remember"
    assert syllabus.bloom_level("Compare BERT and GPT") == "Analyze"
    assert syllabus.bloom_level("Design a RAG chatbot") == "Create"


def test_metrics():
    from sah import metrics

    assert metrics.bleu("the cat sat on the mat", "the cat sat on the mat")["bleu"] == 1.0
    assert metrics.bleu("the cat sat on the mat", "a dog ran in the park")["bleu"] < 0.1
    r = metrics.rouge_all("tokenization splits text into tokens", "tokenization splits text")
    assert r["ROUGE-1"]["precision"] == 1.0 and r["ROUGE-1"]["recall"] < 1.0
    lm = metrics.BigramLM([documents.full_text(SAMPLE)])
    assert lm.perplexity("Tokenization is the process of splitting text.")["perplexity"] < \
        lm.perplexity("Bananas fly purple over quantum cheese.")["perplexity"]
    g = metrics.grounding("Stemming chops word endings to reach a root form. The Moon is made of green cheese "
                          "according to scientists.", [documents.full_text(SAMPLE)])
    assert [s["supported"] for s in g["sentences"]] == [True, False]


def test_classical():
    from sah import classical

    assert classical.sentiment("This explanation was really helpful")["label"] == "positive"
    assert classical.sentiment("still confused, the answer was useless")["label"] == "negative"
    if classical.HAS_NER:
        assert any(e["entity"] == "Paris" for e in classical.entities("Meta trained Llama in Paris."))
    words, dom = classical.topics([c for _, c in documents.extract_pages(SAMPLE)], 2)
    assert len(words) == 2 and len(dom) == 4


def test_question_paper_offline():
    from sah import paper, syllabus

    units = syllabus.parse_units(documents.full_text(SYLLABUS))
    p = paper.generate(units[:2], {2: 2, 5: 1, 10: 1}, LLM("offline"))
    assert sum(q["marks"] for q in p) == 2 * 2 + 5 + 10
    assert {q["co"] for q in p} <= {"CO1", "CO2"}
    assert "| Q.No |" in paper.to_markdown(p, "NLP", "23CSE4021", "2 hours", 19, True)


def test_per_student_data():
    db.init()
    db.touch_user("15", "Harshal")
    db.record_attempt("NLP", "TF-IDF", "mcq", "q", True, 1.0, user="15", unit="Unit I", bloom="Remember")
    db.record_attempt("NLP", "TF-IDF", "mcq", "q", False, 0.0, user="22", unit="Unit I", bloom="Remember")
    assert len(db.attempts("15")) == 1 and len(db.attempts()) >= 2
    sid = db.upsert_subject("NLP", "2030-01-01", "15")
    assert db.subjects("15").id.tolist() == [sid] and sid not in db.subjects("22").id.tolist()


# ---------------------------------------------------------------- well-being & faculty automation
def test_wellbeing():
    from sah import wellbeing

    assert wellbeing.needs_urgent_support("honestly I want to die")
    assert not wellbeing.needs_urgent_support("this exam will kill me lol")
    db.init()
    db.add_checkin("77", 2, "Stressed", "private worry", shared=False)
    assert db.checkins("77").note.iloc[0] == ""          # private notes are never stored
    db.log_chat("What is NLP?", True, 0.3, "m", "77", "Unit I", "Remember")
    assert wellbeing.streak("77") == 1 and "first_question" in wellbeing.earned_badges("77")
    aid = db.add_assignment("A", "d", "Unit I", "2030-01-01", "Def | 2 | token", 2)
    assert aid not in {r["title"] for r in wellbeing.reminders("77")}  # far future → not in 14-day window


def test_assignment_review_and_similarity():
    from sah import assignments as A

    rub = "Definition | 4 | token, split text\nTypes | 4 | subword, byte pair encoding\nExample | 2 | example"
    good = "Tokenization is how we split text into tokens. Subword methods like byte pair encoding help. Example: hi."
    r = A.review(good, "T", "d", rub, LLM("offline"))
    assert r["total"] == r["max"] == 10
    weak = A.review("Tokens are words.", "T", "d", rub, LLM("offline"))
    assert weak["total"] < 5 and "byte pair encoding" in weak["missing_concepts"]
    pairs = A.similarity_pairs({"a": good * 3, "b": good * 3 + " extra words here", "c": "different " * 30})
    assert pairs and pairs[0][:2] == ("a", "b")


def test_faq_grouping_and_match():
    from sah import faq

    groups = faq.repeated_questions(["What is tokenization?", "what is tokenisation", "Explain tokenization",
                                     "What is LoRA?"])
    assert groups and groups[0]["count"] == 3
    db.add_faq("What is tokenization?", "Splitting text into tokens.")
    assert faq.match("explain tokenisation")["answer"].startswith("Splitting")


def test_reports_and_attendance():
    from sah import reports

    rows = reports.parse_attendance(b"roll,name,01-11-2026,02-11-2026\n15,H,P,A\n")
    assert rows == [("15", "2026-11-01", 1), ("15", "2026-11-02", 0)]
    assert reports.parse_attendance(b"Roll No,Date,Status\n15,2026-11-03,Present\n") == [("15", "2026-11-03", 1)]
    db.import_attendance(rows)
    assert float(reports.attendance_summary().query("roll == '15'")["attendance %"].iloc[0]) == 50.0
    st = reports.class_stats()
    assert "Key numbers" in reports.class_report_md(st)
    assert reports.ai_summary(st, LLM("offline"))


def test_autoload_defaults_and_skip(tmp_path, monkeypatch):
    from sah import autoload

    monkeypatch.setattr(autoload, "TEXTBOOK_DIRS", [tmp_path / "textbooks"])
    (tmp_path / "textbooks").mkdir()
    (tmp_path / "textbooks" / "my_textbook.txt").write_text("Tokenization splits text into tokens. " * 50)
    s = VectorStore(Path(os.environ["SAH_DATA_DIR"]) / "auto_idx")
    msgs = autoload.run(s)
    assert {"my_textbook.txt", "NLP_Unit1_Notes.pdf"} <= set(s.documents()), msgs
    s.remove_document("my_textbook.txt")
    autoload.skip("my_textbook.txt")
    autoload.run(s)
    assert "my_textbook.txt" not in s.documents()          # removed by the teacher → stays removed
    autoload.unskip("my_textbook.txt")
