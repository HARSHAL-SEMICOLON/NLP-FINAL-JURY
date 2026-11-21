"""Student well-being support: academic stress check-ins, exam readiness, reminders,
streaks and badges, and today's small tasks.

This is academic-stress support, NOT a mental-health diagnosis or treatment tool.
* No mood score or private note is ever shown to faculty unless the student opts in.
* If a note suggests serious distress, the student is pointed to people who can help
  (Tele-MANAS 14416, a trusted person, the college counsellor, 112 in an emergency).
"""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta

import pandas as pd

from . import db

MOODS = {5: ("😊", "Feeling good"), 4: ("🙂", "Okay"), 3: ("😐", "Tired"), 2: ("😟", "Stressed"),
         1: ("😫", "Overwhelmed")}

HELPLINE = ("**Tele-MANAS: 14416** or **1-800-891-4416** – free, confidential, 24×7, many Indian languages "
            "(Government of India). In an emergency call **112**.")

# Phrases that suggest a student may be at risk – shown support immediately (never stored unless shared).
_CRISIS = re.compile(
    r"\b(kill(ing)? myself|suicid\w*|end (my|it all|my life)|want to die|wanna die|don'?t want to (live|be alive)|"
    r"self[- ]?harm|hurt(ing)? myself|cut(ting)? myself|no reason to live|better off dead|take my (own )?life)\b",
    re.IGNORECASE)


def needs_urgent_support(text: str) -> bool:
    return bool(text and _CRISIS.search(text))


def coping_steps(mood: int) -> list[str]:
    """Practical, study-focused suggestions – not clinical advice."""
    if mood >= 4:
        return ["Keep the momentum: finish one topic from today's list.",
                "Teach one idea to a friend – it's the fastest way to check you understand it."]
    if mood == 3:
        return ["Try one 25-minute focus session, then a 5-minute break away from screens.",
                "Pick the smallest task below and finish just that.",
                "Sleep counts as revision – your brain consolidates while you rest."]
    return ["You don't have to finish the whole syllabus today. Do just the first task below.",
            "Write down what's worrying you, then ask one doubt – privately if you prefer.",
            "Talk to someone you trust: a friend, family member, mentor or the college counsellor.",
            "Take a short walk or drink some water before you start again."]


# ---------------------------------------------------------------- readiness
def readiness(user: str, units: list[dict]) -> dict:
    """Per-unit readiness = 50% topics completed + 50% quiz accuracy (topics only if no quiz yet)."""
    tp, att = db.topics(user), db.attempts(user)
    rows = []
    for u in units:
        t = tp[tp.unit == u["unit"]] if not tp.empty else tp
        topic_pct = float(t.done.mean()) if not t.empty else None
        a = att[att.unit == u["unit"]] if not att.empty else att
        quiz_pct = float(a.score.mean()) if not a.empty else None
        parts = [x for x in (topic_pct, quiz_pct) if x is not None]
        rows.append({"unit": u["unit"], "title": u["title"], "topics": topic_pct, "quiz": quiz_pct,
                     "readiness": sum(parts) / len(parts) if parts else 0.0,
                     "remaining": int((~t.done.astype(bool)).sum()) if not t.empty else 0})
    overall = sum(r["readiness"] for r in rows) / len(rows) if rows else 0.0
    subs = db.subjects(user)
    days_left = None
    if not subs.empty:
        future = [(pd.to_datetime(d).date() - date.today()).days for d in subs.exam_date]
        future = [d for d in future if d >= 0]
        days_left = min(future) if future else None
    remaining = sum(r["remaining"] for r in rows)
    pace = round(remaining / max(days_left - 1, 1), 1) if days_left and remaining else 0
    return {"units": rows, "overall": overall, "days_left": days_left, "remaining_topics": remaining,
            "topics_per_day": pace}


# ---------------------------------------------------------------- streaks & badges
def active_days(user: str) -> set[date]:
    days: set[date] = set()
    for df, col in ((db.chat_log(user), "asked_at"), (db.attempts(user), "taken_at"),
                    (db.checkins(user), "at")):
        if not df.empty:
            days |= set(pd.to_datetime(df[col]).dt.date)
    tp = db.topics(user)
    if not tp.empty:
        days |= set(pd.to_datetime(tp.done_at.dropna()).dt.date)
    return days


def streak(user: str) -> int:
    days, n, d = active_days(user), 0, date.today()
    if d not in days:
        d -= timedelta(days=1)  # today not done yet – streak still alive from yesterday
    while d in days:
        n += 1
        d -= timedelta(days=1)
    return n


BADGES = [
    ("first_question", "💡", "Curious mind", "Asked your first question"),
    ("first_quiz", "🎯", "Quiz starter", "Answered your first quiz question"),
    ("quiz_25", "📚", "Practice habit", "Answered 25 quiz questions"),
    ("sharp", "🏹", "Sharp shooter", "80%+ accuracy over at least 10 questions"),
    ("higher_order", "🧠", "Deep thinker", "Correct answer at Analyze level or above"),
    ("streak_3", "🔥", "3-day streak", "Studied 3 days in a row"),
    ("streak_7", "🌟", "7-day streak", "Studied 7 days in a row"),
    ("unit_done", "✅", "Unit finisher", "Completed every topic of a unit"),
    ("brave", "🙋", "Brave asker", "Raised a doubt with your teacher"),
]


def earned_badges(user: str) -> set[str]:
    log, att, tp = db.chat_log(user), db.attempts(user), db.topics(user)
    got = set()
    if len(log):
        got.add("first_question")
    if len(att):
        got.add("first_quiz")
    if len(att) >= 25:
        got.add("quiz_25")
    if len(att) >= 10 and att.score.mean() >= 0.8:
        got.add("sharp")
    if not att.empty and ((att.bloom.isin(["Analyze", "Evaluate", "Create"])) & (att.score >= 0.6)).any():
        got.add("higher_order")
    s = streak(user)
    if s >= 3:
        got.add("streak_3")
    if s >= 7:
        got.add("streak_7")
    if not tp.empty and tp.groupby("unit").done.min().max() == 1:
        got.add("unit_done")
    if len(db.help_flags(include_resolved=True, user=user)):
        got.add("brave")
    return got


def improvement(user: str) -> str | None:
    att = db.attempts(user)
    if len(att) < 6:
        return None
    att = att.assign(t=pd.to_datetime(att.taken_at))
    cut = datetime.now() - timedelta(days=7)
    recent, before = att[att.t >= cut].score, att[att.t < cut].score
    if len(recent) >= 3 and len(before) >= 3 and recent.mean() > before.mean() + 0.05:
        return (f"Your quiz accuracy went from {before.mean():.0%} to {recent.mean():.0%} this week. "
                "That's real progress.")
    if len(att) >= 6:
        first, last = att.score.iloc[: len(att) // 2].mean(), att.score.iloc[len(att) // 2:].mean()
        if last > first + 0.05:
            return f"You've improved from {first:.0%} to {last:.0%} since your first quizzes."
    return None


# ---------------------------------------------------------------- reminders & today's tasks
def reminders(user: str) -> list[dict]:
    """Upcoming deadlines for this student: assignments, class quizzes and exams (next 14 days)."""
    out, today = [], date.today()
    subs = db.submissions(user=user)
    done_ids = set(subs.assignment_id) if not subs.empty else set()
    for a in db.assignments().itertuples():
        days = (pd.to_datetime(a.due).date() - today).days
        if a.id not in done_ids and -3 <= days <= 14:
            out.append({"kind": "Assignment", "title": a.title, "days": days, "unit": a.unit})
    att = db.attempts(user)
    taken = set(att.quiz_id.dropna().astype(int)) if not att.empty else set()
    for q in db.class_quizzes(published_only=True).itertuples():
        days = (pd.to_datetime(q.due).date() - today).days
        if q.id not in taken and -1 <= days <= 14:
            out.append({"kind": "Class quiz", "title": q.title, "days": days, "unit": q.unit})
    for s in db.subjects(user).itertuples():
        days = (pd.to_datetime(s.exam_date).date() - today).days
        if 0 <= days <= 14:
            out.append({"kind": "Exam", "title": s.name, "days": days, "unit": ""})
    return sorted(out, key=lambda r: r["days"])


def today_tasks(user: str, units: list[dict], hours: float = 3.0, limit: int = 4) -> list[dict]:
    """A short, doable list: today's planner topics + a practice quiz on the weakest unit."""
    from . import planner

    tasks: list[dict] = []
    tp = db.topics(user)
    if not tp.empty:
        w = db.weak_topics(user)
        sched, _ = planner.build_plan(tp, hours, weak=set(w.topic) if not w.empty else set())
        if not sched.empty:
            todays = sched[sched.date == sched.date.min()]
            for r in todays.head(limit - 1).itertuples():
                tasks.append({"title": r.topic, "meta": f"{r.unit} · {r.hours:g} h", "kind": r.type})
    att = db.attempts(user)
    if not att.empty and (att.unit != "").any():
        by = att[att.unit != ""].groupby("unit").score.mean().sort_values()
        if by.iloc[0] < 0.75:
            tasks.append({"title": f"10-minute practice quiz on {by.index[0]}",
                          "meta": f"your accuracy there is {by.iloc[0]:.0%}", "kind": "practice",
                          "unit": by.index[0]})
    elif units:
        tasks.append({"title": f"Warm-up quiz on {units[0]['unit']}", "meta": "5 questions", "kind": "practice",
                      "unit": units[0]["unit"]})
    return tasks[:limit]
