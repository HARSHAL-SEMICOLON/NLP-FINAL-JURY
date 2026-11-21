"""Smart study planner — a deterministic scheduling algorithm (no LLM needed).

Rules
* Each pending topic needs hours = HOURS_BY_DIFFICULTY[difficulty] (+50% if it is a
  known weak topic from quiz results).
* Earliest-deadline-first: topics of the nearest exam are scheduled first,
  harder topics earlier in the window.
* The day before each exam is reserved for revision of that subject.
* Topics that cannot fit before their exam are reported as "at risk".
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

HOURS_BY_DIFFICULTY = {1: 1.0, 2: 1.5, 3: 2.5}


def build_plan(topics: pd.DataFrame, hours_per_day: float = 4.0, start: date | None = None,
               weak: set[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """topics needs columns: id, subject, unit, name, difficulty, done, exam_date (YYYY-MM-DD).
    Returns (schedule, at_risk)."""
    start = start or date.today()
    weak = {w.lower() for w in (weak or set())}
    cols = ["date", "subject", "unit", "topic", "hours", "type"]
    if topics.empty:
        return pd.DataFrame(columns=cols), pd.DataFrame(columns=["subject", "unit", "topic", "exam_date"])

    pending = topics[topics.done == 0].copy()
    pending["exam"] = pd.to_datetime(pending.exam_date).dt.date
    pending = pending[pending.exam > start]
    pending["need"] = pending.difficulty.map(HOURS_BY_DIFFICULTY).fillna(1.5)
    pending.loc[pending.name.str.lower().isin(weak), "need"] *= 1.5
    pending = pending.sort_values(["exam", "difficulty"], ascending=[True, False])

    queue = pending.to_dict("records")
    exams = {(r["subject"], r["exam"]) for r in queue} | {
        (r.subject, pd.to_datetime(r.exam_date).date()) for r in topics.itertuples()
        if pd.to_datetime(r.exam_date).date() > start}
    revision_days = {exam - timedelta(days=1): subj for subj, exam in exams}

    rows = []
    day = start
    last_exam = max((e for _, e in exams), default=start)
    while day < last_exam:
        left = hours_per_day
        if day in revision_days:
            rows.append([day, revision_days[day], "—", "Full revision + previous-year questions",
                         min(hours_per_day, 3.0), "revision"])
            left -= min(hours_per_day, 3.0)
        for item in queue:
            if left <= 0:
                break
            if item["need"] <= 0 or item["exam"] - timedelta(days=1) <= day:
                continue  # done, or no study days left before its exam
            h = round(min(item["need"], left), 2)
            rows.append([day, item["subject"], item["unit"], item["name"], h, "study"])
            item["need"] -= h
            left -= h
        day += timedelta(days=1)

    schedule = pd.DataFrame(rows, columns=cols)
    at_risk = pd.DataFrame([{"subject": r["subject"], "unit": r["unit"], "topic": r["name"],
                             "exam_date": r["exam"], "hours_short": round(r["need"], 2)}
                            for r in queue if r["need"] > 0.01])
    return schedule, at_risk
