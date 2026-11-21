"""Auto-generated performance reports for faculty: class report, per-student report,
attendance import, and an AI-written summary with suggested actions."""
from __future__ import annotations

import io
import re
from datetime import date, datetime, timedelta

import pandas as pd

from . import config, db
from .llm import LLM, LLMError


# ---------------------------------------------------------------- attendance import
def parse_attendance(raw: bytes) -> list[tuple[str, str, int]]:
    """Accepts long format (roll, date, status) or wide format (roll, [name], <date columns>).
    Status: P/A, Present/Absent, 1/0, Y/N."""
    df = pd.read_csv(io.BytesIO(raw), dtype=str).fillna("")
    df.columns = [c.strip() for c in df.columns]
    low = {c.lower(): c for c in df.columns}
    roll_col = next((low[k] for k in ("roll", "roll no", "roll_no", "rollno", "roll number") if k in low), df.columns[0])

    def present(v: str) -> int | None:
        v = str(v).strip().lower()
        if v in ("p", "present", "1", "y", "yes", "true"):
            return 1
        if v in ("a", "absent", "0", "n", "no", "false"):
            return 0
        return None

    def to_date(v: str) -> str:
        v = str(v).strip()
        iso = bool(re.match(r"^\d{4}-\d{1,2}-\d{1,2}", v))
        return str(pd.to_datetime(v, dayfirst=not iso).date())

    rows = []
    if "date" in low and any(k in low for k in ("status", "present", "attendance")):
        st = next(low[k] for k in ("status", "present", "attendance") if k in low)
        for _, rec in df.iterrows():
            p = present(rec[st])
            if p is not None and rec[low["date"]]:
                rows.append((rec[roll_col].strip(), to_date(rec[low["date"]]), p))
        return rows
    for c in df.columns:
        if c == roll_col or c.lower() in ("name", "student", "student name"):
            continue
        try:
            d = to_date(c)
        except (ValueError, TypeError):
            continue
        for _, rec in df.iterrows():
            p = present(rec[c])
            if p is not None and rec[roll_col].strip():
                rows.append((rec[roll_col].strip(), d, p))
    return rows


def attendance_summary() -> pd.DataFrame:
    a = db.attendance()
    if a.empty:
        return pd.DataFrame(columns=["roll", "classes", "present", "attendance %"])
    g = a.groupby("roll").present.agg(["size", "sum"]).reset_index()
    g.columns = ["roll", "classes", "present"]
    g["attendance %"] = (g.present / g.classes * 100).round(1)
    return g


# ---------------------------------------------------------------- class report
def class_stats(since: date | None = None) -> dict:
    since = since or (date.today() - timedelta(days=7))
    s = str(since)
    log = db.chat_log()
    att = db.attempts()
    flags = db.help_flags(include_resolved=True)
    fb = db.feedback()
    log = log[(log.user != "faculty") & (log.asked_at >= s)]
    att = att[(att.user != "faculty") & (att.taken_at >= s)]
    flags = flags[flags.flagged_at >= s]
    fb = fb[fb["at"] >= s] if not fb.empty else fb
    by_unit = (att[att.unit != ""].groupby("unit").score.agg(["mean", "size"]).round(3)
               if not att.empty else pd.DataFrame())
    hardest = (att.groupby("question").score.agg(["mean", "size"]).query("size >= 2").sort_values("mean").head(5)
               if not att.empty else pd.DataFrame())
    asked = log.groupby("unit").size().sort_values(ascending=False) if not log.empty else pd.Series(dtype=int)
    ci = db.checkins()
    ci = ci[ci["at"] >= s] if not ci.empty else ci
    att_sum = attendance_summary()
    return {
        "since": s, "students": len(db.users()), "active_students": int(pd.concat([log.user, att.user]).nunique()),
        "questions": len(log), "quiz_answers": len(att),
        "quiz_avg": float(att.score.mean()) if not att.empty else None,
        "by_unit": by_unit, "hardest": hardest, "asked_by_unit": asked,
        "doubts": len(flags), "open_doubts": int((flags.resolved == 0).sum()) if not flags.empty else 0,
        "helpful": float(fb.rating.mean()) if not fb.empty else None,
        "grounded": float(log.grounded.mean()) if not log.empty else None,
        "stressed_checkins": int((ci.mood <= 2).sum()) if not ci.empty else 0, "checkins": len(ci),
        "attendance_avg": float(att_sum["attendance %"].mean()) if not att_sum.empty else None,
        "low_attendance": att_sum[att_sum["attendance %"] < 75].roll.tolist() if not att_sum.empty else [],
    }


def _pct(x) -> str:
    return f"{x:.0%}" if x is not None else "—"


def class_report_md(st: dict, ai_summary: str = "") -> str:
    lines = [f"# Weekly class report – {config.COURSE_NAME} ({config.COURSE_CODE})",
             f"Period: {st['since']} to {date.today()} · generated {datetime.now():%d %b %Y %H:%M}", ""]
    if ai_summary:
        lines += ["## Summary", ai_summary, ""]
    lines += ["## Key numbers", "",
              "| Measure | Value |", "|---|---|",
              f"| Students registered / active | {st['students']} / {st['active_students']} |",
              f"| Questions asked | {st['questions']} |",
              f"| Quiz answers / accuracy | {st['quiz_answers']} / {_pct(st['quiz_avg'])} |",
              f"| Doubts raised / still open | {st['doubts']} / {st['open_doubts']} |",
              f"| Answers rated helpful | {_pct(st['helpful'])} |",
              f"| Questions answered from course material | {_pct(st['grounded'])} |",
              f"| Average attendance | {st['attendance_avg']:.1f}% |" if st["attendance_avg"] is not None
              else "| Average attendance | not imported |",
              f"| Well-being check-ins (stressed/overwhelmed) | {st['checkins']} ({st['stressed_checkins']}) |", ""]
    if not st["by_unit"].empty:
        lines += ["## Quiz accuracy by unit", "", "| Unit | Accuracy | Answers |", "|---|---|---|"]
        lines += [f"| {u} | {r['mean']:.0%} | {int(r['size'])} |" for u, r in st["by_unit"].iterrows()]
        lines.append("")
    if not st["hardest"].empty:
        lines += ["## Most-missed questions", ""]
        lines += [f"- {q} — {r['mean']:.0%} correct ({int(r['size'])} answers)" for q, r in st["hardest"].iterrows()]
        lines.append("")
    if st["low_attendance"]:
        lines += ["## Attendance below 75%", "", ", ".join(st["low_attendance"]), ""]
    return "\n".join(lines)


def ai_summary(st: dict, llm: LLM) -> str:
    facts = class_report_md(st)
    if llm.available:
        try:
            return llm.generate(
                "You are a teaching assistant. From these class statistics, write a 120-word summary for the "
                "teacher in plain English, then 3 short, specific recommended actions as bullets (what to re-teach, "
                "who to check on, which practice to assign). Do not invent numbers.\n\n" + facts, temperature=0.3)
        except LLMError:
            pass
    tips = []
    if not st["by_unit"].empty:
        weakest = st["by_unit"]["mean"].idxmin()
        tips.append(f"- Re-teach **{weakest}** (lowest quiz accuracy, {st['by_unit'].loc[weakest, 'mean']:.0%}) "
                    "and publish a short practice quiz on it.")
    if st["open_doubts"]:
        tips.append(f"- Answer the {st['open_doubts']} open doubt(s) – or publish repeated ones as FAQs.")
    if st["low_attendance"]:
        tips.append(f"- Check in with students below 75% attendance: {', '.join(st['low_attendance'][:5])}.")
    if st["grounded"] is not None and st["grounded"] < 0.7:
        tips.append("- Many questions weren't covered by the uploaded notes – see Content gaps and add material.")
    return (f"This period {st['active_students']} of {st['students']} students were active, asking "
            f"{st['questions']} questions and answering {st['quiz_answers']} quiz questions "
            f"(accuracy {_pct(st['quiz_avg'])}).\n\n" + ("\n".join(tips) or "- No action needed this week."))


# ---------------------------------------------------------------- student report
def student_report_md(roll: str, name: str) -> str:
    att, log, tp = db.attempts(roll), db.chat_log(roll), db.topics(roll)
    subs = db.submissions(user=roll)
    a_sum = attendance_summary()
    arow = a_sum[a_sum.roll == roll]
    lines = [f"# Student report – {name} ({roll})", f"{config.COURSE_NAME} ({config.COURSE_CODE}) · "
             f"{datetime.now():%d %b %Y}", "",
             "| Measure | Value |", "|---|---|",
             f"| Questions asked | {len(log)} |",
             f"| Quiz answers / accuracy | {len(att)} / {_pct(att.score.mean() if not att.empty else None)} |",
             f"| Syllabus topics completed | {int(tp.done.sum())}/{len(tp)} |" if not tp.empty
             else "| Syllabus topics completed | no plan yet |",
             f"| Assignments submitted | {len(subs)} |",
             f"| Attendance | {arow['attendance %'].iloc[0]}% |" if not arow.empty else "| Attendance | not imported |",
             ""]
    if not att.empty and (att.unit != "").any():
        lines += ["## Accuracy by unit", ""]
        for u, v in att[att.unit != ""].groupby("unit").score.mean().sort_values().items():
            lines.append(f"- {u}: {v:.0%}")
        lines.append("")
    w = db.weak_topics(roll)
    if not w.empty:
        lines += ["## Topics to revise", ""] + [f"- {t}" for t in w.topic] + [""]
    if not subs.empty and subs.released.any():
        lines += ["## Assignment marks", ""]
        amap = dict(zip(db.assignments().id, db.assignments().title))
        for s in subs[subs.released == 1].itertuples():
            lines.append(f"- {amap.get(s.assignment_id, s.assignment_id)}: {s.final_mark:g}")
    return "\n".join(lines)
