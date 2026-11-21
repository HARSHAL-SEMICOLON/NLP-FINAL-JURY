"""SQLite persistence. Every student row carries the student's roll number in `user`,
so the faculty dashboard can aggregate the whole class."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime

import pandas as pd

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    roll TEXT PRIMARY KEY, name TEXT, role TEXT DEFAULT 'student', first_seen TEXT, last_seen TEXT
);
CREATE TABLE IF NOT EXISTS quiz_attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    taken_at TEXT, subject TEXT, topic TEXT, kind TEXT,
    question TEXT, correct INTEGER, score REAL, user TEXT DEFAULT '', unit TEXT DEFAULT '', bloom TEXT DEFAULT '',
    quiz_id INTEGER
);
CREATE TABLE IF NOT EXISTS subjects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user TEXT DEFAULT '', name TEXT, exam_date TEXT, UNIQUE(user, name)
);
CREATE TABLE IF NOT EXISTS topics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_id INTEGER, unit TEXT, name TEXT, difficulty INTEGER DEFAULT 2,
    done INTEGER DEFAULT 0, done_at TEXT,
    FOREIGN KEY(subject_id) REFERENCES subjects(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS help_flags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    flagged_at TEXT, source TEXT, question TEXT, resolved INTEGER DEFAULT 0,
    user TEXT DEFAULT '', unit TEXT DEFAULT '', reply TEXT DEFAULT '', anonymous INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS chat_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asked_at TEXT, question TEXT, grounded INTEGER, top_score REAL, model TEXT,
    user TEXT DEFAULT '', unit TEXT DEFAULT '', bloom TEXT DEFAULT '', grounding REAL, language TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    at TEXT, user TEXT, question TEXT, rating INTEGER, comment TEXT,
    sentiment TEXT, compound REAL, unit TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS announcements (
    id INTEGER PRIMARY KEY AUTOINCREMENT, at TEXT, text TEXT
);
CREATE TABLE IF NOT EXISTS checkins (
    id INTEGER PRIMARY KEY AUTOINCREMENT, at TEXT, user TEXT, mood INTEGER, label TEXT,
    note TEXT, shared INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS assignments (
    id INTEGER PRIMARY KEY AUTOINCREMENT, created TEXT, title TEXT, description TEXT, unit TEXT,
    due TEXT, rubric TEXT, max_marks REAL
);
CREATE TABLE IF NOT EXISTS submissions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, assignment_id INTEGER, user TEXT, at TEXT, text TEXT,
    filename TEXT, ai_score REAL, ai_feedback TEXT, final_mark REAL, teacher_comment TEXT,
    released INTEGER DEFAULT 0, UNIQUE(assignment_id, user)
);
CREATE TABLE IF NOT EXISTS class_quizzes (
    id INTEGER PRIMARY KEY AUTOINCREMENT, created TEXT, title TEXT, unit TEXT, bloom TEXT, due TEXT,
    questions TEXT, published INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS attendance (
    roll TEXT, date TEXT, present INTEGER, PRIMARY KEY(roll, date)
);
CREATE TABLE IF NOT EXISTS faq (
    id INTEGER PRIMARY KEY AUTOINCREMENT, at TEXT, question TEXT, answer TEXT, unit TEXT
);
"""

# Columns added after the first release – added to old databases on start-up.
_MIGRATE = {
    "quiz_attempts": {"user": "TEXT DEFAULT ''", "unit": "TEXT DEFAULT ''", "bloom": "TEXT DEFAULT ''",
                      "quiz_id": "INTEGER"},
    "help_flags": {"user": "TEXT DEFAULT ''", "unit": "TEXT DEFAULT ''", "reply": "TEXT DEFAULT ''",
                   "anonymous": "INTEGER DEFAULT 0"},
    "chat_log": {"user": "TEXT DEFAULT ''", "unit": "TEXT DEFAULT ''", "bloom": "TEXT DEFAULT ''",
                 "grounding": "REAL", "language": "TEXT DEFAULT ''"},
}


@contextmanager
def conn():
    c = sqlite3.connect(config.DB_PATH)
    c.execute("PRAGMA foreign_keys = ON")
    try:
        yield c
        c.commit()
    finally:
        c.close()


def _cols(c, table: str) -> set[str]:
    return {r[1] for r in c.execute(f"PRAGMA table_info({table})")}


def init() -> None:
    with conn() as c:
        # v1 'subjects' had UNIQUE(name) and no user column: rebuild it per-user.
        if "subjects" in {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")} \
                and "user" not in _cols(c, "subjects"):
            c.execute("PRAGMA foreign_keys = OFF")
            c.execute("ALTER TABLE subjects RENAME TO subjects_v1")
            c.execute("CREATE TABLE subjects (id INTEGER PRIMARY KEY AUTOINCREMENT, user TEXT DEFAULT '', "
                      "name TEXT, exam_date TEXT, UNIQUE(user, name))")
            c.execute("INSERT INTO subjects(id, user, name, exam_date) SELECT id, '', name, exam_date FROM subjects_v1")
            c.execute("DROP TABLE subjects_v1")
        c.executescript(SCHEMA)
        for table, cols in _MIGRATE.items():
            have = _cols(c, table)
            for col, decl in cols.items():
                if col not in have:
                    c.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _df(q: str, params=()) -> pd.DataFrame:
    with conn() as c:
        return pd.read_sql_query(q, c, params=params)


# ---------- users ----------
def touch_user(roll: str, name: str, role: str = "student") -> None:
    with conn() as c:
        c.execute("INSERT INTO users(roll, name, role, first_seen, last_seen) VALUES (?,?,?,?,?) "
                  "ON CONFLICT(roll) DO UPDATE SET name=excluded.name, last_seen=excluded.last_seen",
                  (roll, name, role, now(), now()))


def users() -> pd.DataFrame:
    return _df("SELECT * FROM users WHERE role='student' ORDER BY roll")


# ---------- quiz ----------
def record_attempt(subject: str, topic: str, kind: str, question: str, correct: bool, score: float,
                   user: str = "", unit: str = "", bloom: str = "", quiz_id: int | None = None) -> None:
    with conn() as c:
        c.execute("INSERT INTO quiz_attempts(taken_at,subject,topic,kind,question,correct,score,user,unit,bloom,"
                  "quiz_id) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                  (now(), subject, topic, kind, question, int(correct), score, user, unit, bloom, quiz_id))


def attempts(user: str | None = None) -> pd.DataFrame:
    if user is None:
        return _df("SELECT * FROM quiz_attempts ORDER BY taken_at")
    return _df("SELECT * FROM quiz_attempts WHERE user=? ORDER BY taken_at", (user,))


def weak_topics(user: str | None = None, min_attempts: int = 2, threshold: float = 0.6) -> pd.DataFrame:
    """Knowledge-gap identification: topics whose average score is below threshold."""
    df = attempts(user)
    if df.empty:
        return df
    g = df.groupby(["subject", "topic"]).agg(attempts=("score", "size"), avg_score=("score", "mean")).reset_index()
    return g[(g.attempts >= min_attempts) & (g.avg_score < threshold)].sort_values("avg_score")


# ---------- planner (per student) ----------
def upsert_subject(name: str, exam_date: str, user: str = "") -> int:
    with conn() as c:
        c.execute("INSERT INTO subjects(user, name, exam_date) VALUES (?,?,?) "
                  "ON CONFLICT(user, name) DO UPDATE SET exam_date=excluded.exam_date", (user, name, exam_date))
        return c.execute("SELECT id FROM subjects WHERE user=? AND name=?", (user, name)).fetchone()[0]


def add_topic(subject_id: int, unit: str, name: str, difficulty: int = 2) -> None:
    with conn() as c:
        c.execute("INSERT INTO topics(subject_id, unit, name, difficulty) VALUES (?,?,?,?)",
                  (subject_id, unit, name, difficulty))


def delete_subject(subject_id: int) -> None:
    with conn() as c:
        c.execute("DELETE FROM topics WHERE subject_id=?", (subject_id,))
        c.execute("DELETE FROM subjects WHERE id=?", (subject_id,))


def set_done(topic_id: int, done: bool) -> None:
    with conn() as c:
        c.execute("UPDATE topics SET done=?, done_at=? WHERE id=?", (int(done), now() if done else None, topic_id))


def subjects(user: str = "") -> pd.DataFrame:
    return _df("SELECT * FROM subjects WHERE user=? ORDER BY exam_date", (user,))


def topics(user: str = "") -> pd.DataFrame:
    return _df("SELECT t.*, s.name AS subject, s.exam_date FROM topics t JOIN subjects s ON s.id=t.subject_id "
               "WHERE s.user=? ORDER BY s.exam_date, t.unit, t.id", (user,))


def all_topics() -> pd.DataFrame:
    return _df("SELECT t.*, s.name AS subject, s.user FROM topics t JOIN subjects s ON s.id=t.subject_id")


# ---------- help flags ----------
def add_help_flag(source: str, question: str, user: str = "", unit: str = "", anonymous: bool = False) -> None:
    with conn() as c:
        c.execute("INSERT INTO help_flags(flagged_at, source, question, user, unit, anonymous) VALUES (?,?,?,?,?,?)",
                  (now(), source, question, user, unit, int(anonymous)))


def help_flags(include_resolved: bool = False, user: str | None = None) -> pd.DataFrame:
    q, params = "SELECT * FROM help_flags WHERE 1=1", []
    if not include_resolved:
        q += " AND resolved=0"
    if user is not None:
        q += " AND user=?"
        params.append(user)
    return _df(q + " ORDER BY flagged_at DESC", params)


def resolve_flag(flag_id: int, reply: str = "") -> None:
    with conn() as c:
        c.execute("UPDATE help_flags SET resolved=1, reply=? WHERE id=?", (reply, flag_id))


# ---------- chat log ----------
def log_chat(question: str, grounded: bool, top_score: float, model: str, user: str = "", unit: str = "",
             bloom: str = "", grounding: float | None = None, language: str = "") -> None:
    with conn() as c:
        c.execute("INSERT INTO chat_log(asked_at, question, grounded, top_score, model, user, unit, bloom, "
                  "grounding, language) VALUES (?,?,?,?,?,?,?,?,?,?)",
                  (now(), question, int(grounded), top_score, model, user, unit, bloom, grounding, language))


def chat_log(user: str | None = None) -> pd.DataFrame:
    if user is None:
        return _df("SELECT * FROM chat_log ORDER BY asked_at")
    return _df("SELECT * FROM chat_log WHERE user=? ORDER BY asked_at", (user,))


# ---------- feedback ----------
def add_feedback(user: str, question: str, rating: int, comment: str, sentiment: str, compound: float,
                 unit: str = "") -> None:
    with conn() as c:
        c.execute("INSERT INTO feedback(at, user, question, rating, comment, sentiment, compound, unit) "
                  "VALUES (?,?,?,?,?,?,?,?)", (now(), user, question, rating, comment, sentiment, compound, unit))


def feedback() -> pd.DataFrame:
    return _df("SELECT * FROM feedback ORDER BY at DESC")


# ---------- announcements (faculty → students) ----------
def add_announcement(text: str) -> None:
    with conn() as c:
        c.execute("INSERT INTO announcements(at, text) VALUES (?,?)", (now(), text))


def announcements(limit: int = 5) -> pd.DataFrame:
    return _df("SELECT * FROM announcements ORDER BY at DESC LIMIT ?", (limit,))


def delete_announcement(aid: int) -> None:
    with conn() as c:
        c.execute("DELETE FROM announcements WHERE id=?", (aid,))


# ---------- well-being check-ins ----------
def add_checkin(user: str, mood: int, label: str, note: str, shared: bool) -> None:
    with conn() as c:
        c.execute("INSERT INTO checkins(at, user, mood, label, note, shared) VALUES (?,?,?,?,?,?)",
                  (now(), user, mood, label, note if shared else "", int(shared)))


def checkins(user: str | None = None) -> pd.DataFrame:
    if user is None:
        return _df("SELECT * FROM checkins ORDER BY at")
    return _df("SELECT * FROM checkins WHERE user=? ORDER BY at", (user,))


# ---------- assignments ----------
def add_assignment(title: str, description: str, unit: str, due: str, rubric: str, max_marks: float) -> int:
    with conn() as c:
        cur = c.execute("INSERT INTO assignments(created, title, description, unit, due, rubric, max_marks) "
                        "VALUES (?,?,?,?,?,?,?)", (now(), title, description, unit, due, rubric, max_marks))
        return cur.lastrowid


def assignments() -> pd.DataFrame:
    return _df("SELECT * FROM assignments ORDER BY due")


def delete_assignment(aid: int) -> None:
    with conn() as c:
        c.execute("DELETE FROM submissions WHERE assignment_id=?", (aid,))
        c.execute("DELETE FROM assignments WHERE id=?", (aid,))


def submit(assignment_id: int, user: str, text: str, filename: str = "") -> None:
    with conn() as c:
        c.execute("INSERT INTO submissions(assignment_id, user, at, text, filename) VALUES (?,?,?,?,?) "
                  "ON CONFLICT(assignment_id, user) DO UPDATE SET at=excluded.at, text=excluded.text, "
                  "filename=excluded.filename, ai_score=NULL, ai_feedback=NULL",
                  (assignment_id, user, now(), text, filename))


def submissions(assignment_id: int | None = None, user: str | None = None) -> pd.DataFrame:
    q, params = "SELECT * FROM submissions WHERE 1=1", []
    if assignment_id is not None:
        q += " AND assignment_id=?"
        params.append(assignment_id)
    if user is not None:
        q += " AND user=?"
        params.append(user)
    return _df(q + " ORDER BY at", params)


def save_ai_review(sub_id: int, score: float, feedback: str) -> None:
    with conn() as c:
        c.execute("UPDATE submissions SET ai_score=?, ai_feedback=? WHERE id=?", (score, feedback, sub_id))


def release_mark(sub_id: int, mark: float, comment: str) -> None:
    with conn() as c:
        c.execute("UPDATE submissions SET final_mark=?, teacher_comment=?, released=1 WHERE id=?",
                  (mark, comment, sub_id))


# ---------- class quizzes (faculty-published) ----------
def add_class_quiz(title: str, unit: str, bloom: str, due: str, questions_json: str, published: bool) -> int:
    with conn() as c:
        cur = c.execute("INSERT INTO class_quizzes(created, title, unit, bloom, due, questions, published) "
                        "VALUES (?,?,?,?,?,?,?)", (now(), title, unit, bloom, due, questions_json, int(published)))
        return cur.lastrowid


def class_quizzes(published_only: bool = False) -> pd.DataFrame:
    q = "SELECT * FROM class_quizzes" + (" WHERE published=1" if published_only else "") + " ORDER BY due"
    return _df(q)


def set_quiz_published(qid: int, published: bool) -> None:
    with conn() as c:
        c.execute("UPDATE class_quizzes SET published=? WHERE id=?", (int(published), qid))


def delete_class_quiz(qid: int) -> None:
    with conn() as c:
        c.execute("DELETE FROM class_quizzes WHERE id=?", (qid,))


# ---------- attendance (imported) ----------
def import_attendance(rows: list[tuple[str, str, int]]) -> int:
    with conn() as c:
        c.executemany("INSERT INTO attendance(roll, date, present) VALUES (?,?,?) "
                      "ON CONFLICT(roll, date) DO UPDATE SET present=excluded.present", rows)
    return len(rows)


def attendance() -> pd.DataFrame:
    return _df("SELECT * FROM attendance ORDER BY date")


def clear_attendance() -> None:
    with conn() as c:
        c.execute("DELETE FROM attendance")


# ---------- FAQ ----------
def add_faq(question: str, answer: str, unit: str = "") -> None:
    with conn() as c:
        c.execute("INSERT INTO faq(at, question, answer, unit) VALUES (?,?,?,?)", (now(), question, answer, unit))


def faqs() -> pd.DataFrame:
    return _df("SELECT * FROM faq ORDER BY at DESC")


def delete_faq(fid: int) -> None:
    with conn() as c:
        c.execute("DELETE FROM faq WHERE id=?", (fid,))
