"""Faculty pages: class overview (with the re-teach loop), reports, class quizzes, assignments,
question paper, doubts & feedback, attendance, class well-being, FAQ."""
from __future__ import annotations

import json
from collections import Counter
from datetime import date, timedelta

import pandas as pd
import streamlit as st

from sah import arduino, assignments as asg, classical, config, db, faq, paper, quiz, reports, syllabus, wellbeing
from sah.llm import LLMError

from . import ctx
from . import theme as ui

ss = st.session_state


def _names() -> dict:
    u = db.users()
    return dict(zip(u.roll, u.name))


def _who(roll: str, anonymous: int = 0) -> str:
    if anonymous:
        return "Anonymous student"
    n = _names().get(roll)
    return f"{n} ({roll})" if n else (roll or "unknown")


def _draft_quiz(unit: dict | None, n: int = 5, bloom: str = "Understand", title: str = "") -> int | None:
    focus = (unit["title"] + " " + unit["topics"]) if unit else ""
    qs = quiz.generate_mcq(ctx.store(), ctx.llm(), n, focus, None, "Intermediate", bloom)
    if not qs:
        return None
    for q in qs:
        q.setdefault("bloom", bloom)
    name = title or f"{unit['unit'] if unit else 'Course'} practice – {bloom}"
    return db.add_class_quiz(name, unit["unit"] if unit else "", bloom,
                             str(date.today() + timedelta(days=7)), json.dumps(qs), False)


# ---------------------------------------------------------------- overview
def dashboard_page() -> None:
    llm = ctx.llm()
    ui.title("Class overview", f"{config.COURSE_NAME} · {config.COURSE_CODE}")
    stu, log, att, flags = db.users(), db.chat_log(), db.attempts(), db.help_flags()
    log, att = log[log.user != "faculty"], att[att.user != "faculty"]
    us = ctx.units()
    if stu.empty and log.empty:
        ui.notebook("No students yet",
                    "Students open the same app, choose Student and log in with their roll number. "
                    "Upload the syllabus and notes in Materials first so their questions are answered from it.")
        _announcements()
        return

    week = str(date.today() - timedelta(days=7))
    wl, wa = log[log.asked_at >= week], att[att.taken_at >= week]
    by_unit = att[att.unit.isin([u["unit"] for u in us])].groupby("unit").score.agg(["mean", "size"])
    weakest = by_unit[by_unit["size"] >= 3]["mean"].sort_values()
    summary = (f"{pd.concat([wl.user, wa.user]).nunique()} of {len(stu)} students were active this week, "
               f"asking {len(wl)} questions and answering {len(wa)} quiz questions"
               + (f" at {wa.score.mean():.0%} accuracy." if not wa.empty else "."))
    if not weakest.empty:
        summary += f" The class is weakest in <strong>{weakest.index[0]}</strong> ({weakest.iloc[0]:.0%})."
    ui.notebook("This week", summary)
    a_sum = reports.attendance_summary()
    ui.stats([("students", len(stu)), ("questions asked", len(log)),
              ("quiz accuracy", f"{att.score.mean():.0%}" if not att.empty else "—"),
              ("open doubts", len(flags), "warn" if len(flags) else ""),
              ("answered from your notes", f"{log.grounded.mean():.0%}" if not log.empty else "—"),
              ("avg attendance", f"{a_sum['attendance %'].mean():.0f}%" if not a_sum.empty else "—")])

    # ---- the re-teach loop
    st.subheader("Needs re-teaching")
    doubted = Counter(flags.unit[flags.unit != ""])
    focus = list(weakest[weakest < 0.7].index[:2]) + [u for u, _ in doubted.most_common(2)]
    focus = list(dict.fromkeys(focus))[:3]
    if not focus:
        st.caption("No unit stands out yet – it appears here once quiz accuracy drops below 70% or doubts pile up.")
    for uname in focus:
        u = next((x for x in us if x["unit"] == uname), None)
        acc = by_unit["mean"].get(uname)
        meta = (f"quiz accuracy {acc:.0%}" if acc is not None else "no quiz data") + \
               f" · {doubted.get(uname, 0)} open doubts · {int((log.unit == uname).sum())} questions asked"
        ui.tasks([ui.task(f"{uname} – {u['title'] if u else ''}", meta, "re-teach")])
        c1, c2, _ = st.columns([1.3, 1.3, 2])
        if c1.button("Draft a practice quiz", key=f"dq_{uname}", icon=":material/quiz:"):
            with st.spinner("Writing questions…"):
                qid = _draft_quiz(u, 5, "Understand")
            st.toast("Draft saved in Class quizzes – review and publish it." if qid else "Couldn't generate questions.")
        if c2.button("Teaching ideas", key=f"ti_{uname}", icon=":material/lightbulb:"):
            topics = u["topics"] if u else uname
            missed = att[(att.unit == uname) & (att.score < 0.5)].question.value_counts().head(5).index.tolist()
            if llm.available:
                try:
                    ss[f"ideas_{uname}"] = llm.generate(
                        f"A class is struggling with {uname}: {topics}. Questions they got wrong: {missed}. "
                        "Suggest, briefly: (1) the likely misconception, (2) one everyday analogy, (3) one worked "
                        "example, (4) a 5-minute in-class activity. Use short bullets.", temperature=0.4)
                except LLMError as e:
                    ss[f"ideas_{uname}"] = f"The AI model didn't respond: {e}"
            else:
                ss[f"ideas_{uname}"] = ("Most-missed questions:\n" + "\n".join(f"- {m}" for m in missed)) if missed \
                    else "Add GROQ_API_KEY for AI teaching suggestions."
        if ss.get(f"ideas_{uname}"):
            with st.container(border=True):
                st.markdown(ss[f"ideas_{uname}"])

    if not att.empty and (att.unit != "").any():
        st.subheader("Is understanding improving?")
        w = att[att.unit != ""].assign(week=pd.to_datetime(att.taken_at).dt.to_period("W").dt.start_time.dt.date)
        pv = w.pivot_table(index="unit", columns="week", values="score", aggfunc="mean")
        st.dataframe(pv.map(lambda v: f"{v:.0%}" if pd.notna(v) else ""), width="stretch")
        st.caption("Accuracy per unit per week – check a unit again after you re-teach it.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Questions per unit")
        if not log.empty:
            ui.bars(log.assign(unit=log.unit.replace("", "Untagged")).groupby("unit").size(), "unit", "questions")
    with c2:
        st.subheader("Bloom's level of questions")
        if not log.empty:
            order = [b for b in syllabus.BLOOM_ORDER if b in set(log.bloom)]
            ui.bars(log.groupby("bloom").size().reindex(order), "level", "questions")
            st.caption("Mostly Remember/Understand? Push higher-order questions in class.")

    st.subheader("Students")
    subs, asg_df = db.submissions(), db.assignments()
    due_ids = set(asg_df[pd.to_datetime(asg_df.due).dt.date < date.today()].id) if not asg_df.empty else set()
    rows = []
    for s in stu.itertuples():
        a = att[att.user == s.roll]
        arow = a_sum[a_sum.roll == s.roll] if not a_sum.empty else a_sum
        missing = len(due_ids - set(subs[subs.user == s.roll].assignment_id)) if due_ids else 0
        rows.append({"roll": s.roll, "name": s.name, "questions": int((log.user == s.roll).sum()),
                     "quiz %": round(a.score.mean() * 100) if not a.empty else None, "answers": len(a),
                     "open doubts": int(((flags.user == s.roll) & (flags.anonymous == 0)).sum()),
                     "missing work": missing,
                     "attendance %": float(arow["attendance %"].iloc[0]) if not arow.empty else None,
                     "last seen": (s.last_seen or "")[:10]})
    sdf = pd.DataFrame(rows)
    if not sdf.empty:
        sdf["needs attention"] = (((sdf["quiz %"].fillna(100) < 50) & (sdf.answers >= 3)) | (sdf["open doubts"] >= 2)
                                  | (sdf["missing work"] > 0) | (sdf["attendance %"].fillna(100) < 75))
        st.dataframe(sdf.sort_values("needs attention", ascending=False), hide_index=True, width="stretch",
                     column_config={"needs attention": st.column_config.CheckboxColumn("needs attention")})

    if len(log) >= 5:
        st.subheader("What the class is asking about")
        words, dom = classical.topics(log.question.tolist(), min(4, max(2, len(log) // 5)), 7)
        for i, wds in enumerate(words):
            st.markdown(f"**Theme {i + 1}** · {dom.count(i)} questions")
            ui.chips([ui.chip(x) for x in wds])
    gaps = log[(log.grounded == 0) | (log.unit == "Outside syllabus")]
    if not gaps.empty:
        st.subheader("Content gaps")
        st.caption("Students asked these but your uploaded material couldn't answer them.")
        st.dataframe(gaps[["asked_at", "unit", "question"]].tail(15), hide_index=True, width="stretch")
    _announcements()


def _announcements() -> None:
    st.subheader("Announcements")
    with st.form("ann", clear_on_submit=True, border=False):
        text = st.text_area("Message to all students", placeholder="Unit III quiz on Monday – revise positional "
                                                                   "encoding.", height=80)
        if st.form_submit_button("Post announcement", icon=":material/campaign:") and text.strip():
            db.add_announcement(text.strip())
            arduino.set_status("ready", "New announcement")
            st.rerun()
    for a in db.announcements(10).itertuples():
        c1, c2 = st.columns([8, 1])
        c1.caption(f"{a.at[:16]} — {a.text}")
        if c2.button("Delete", key=f"ann{a.id}"):
            db.delete_announcement(a.id)
            st.rerun()


# ---------------------------------------------------------------- reports
def reports_page() -> None:
    llm = ctx.llm()
    ui.title("Reports", "Weekly class reports and individual student reports, generated from the app's data.")
    days = st.segmented_control("Period", [7, 14, 30, 120], default=7,
                                format_func=lambda d: f"Last {d} days") or 7
    stats = reports.class_stats(date.today() - timedelta(days=days))
    c1, c2 = st.columns([1, 3])
    if c1.button("Write AI summary", type="primary", icon=":material/auto_awesome:"):
        with st.spinner("Summarising the class…"):
            ss.report_summary = reports.ai_summary(stats, llm)
    md = reports.class_report_md(stats, ss.get("report_summary", ""))
    with st.container(border=True):
        st.markdown(md)
    st.download_button("Download class report", md, f"class_report_{date.today()}.md", icon=":material/download:")
    st.subheader("Student report")
    stu = db.users()
    if stu.empty:
        st.caption("No students yet.")
        return
    pick = st.selectbox("Student", [f"{r.name} ({r.roll})" for r in stu.itertuples()])
    roll = pick.rsplit("(", 1)[1].rstrip(")")
    smd = reports.student_report_md(roll, pick.rsplit(" (", 1)[0])
    with st.container(border=True):
        st.markdown(smd)
    c1, c2 = st.columns(2)
    c1.download_button("Download this report", smd, f"report_{roll}.md", icon=":material/download:")
    every = "\n\n---\n\n".join(reports.student_report_md(r.roll, r.name) for r in stu.itertuples())
    c2.download_button("Download all student reports", every, "all_student_reports.md", icon=":material/folder_zip:")


# ---------------------------------------------------------------- class quizzes
def class_quizzes_page() -> None:
    ui.title("Class quizzes", "Generate a quiz from your notes, check it, publish it. Marking is automatic.")
    us = ctx.units()
    with st.expander("Create a quiz", expanded=db.class_quizzes().empty, icon=":material/add_circle:"):
        c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
        uopt = ["Whole course"] + [ctx.unit_label(u) for u in us]
        upick = c1.selectbox("Unit", uopt)
        bloom = c2.selectbox("Bloom's level", ["Mixed"] + syllabus.BLOOM_ORDER, index=2)
        n = c3.number_input("Questions", 3, 20, 8)
        due = c4.date_input("Due", date.today() + timedelta(days=7))
        title = st.text_input("Title", placeholder="Unit II weekly quiz")
        if st.button("Generate draft", type="primary", icon=":material/auto_awesome:"):
            u = next((x for x in us if ctx.unit_label(x) == upick), None)
            with st.spinner("Writing questions from your notes…"):
                qid = _draft_quiz(u, int(n), bloom, title)
            if qid:
                with db.conn() as c:
                    c.execute("UPDATE class_quizzes SET due=? WHERE id=?", (str(due), qid))
                st.toast("Draft ready below – review, then publish.")
                st.rerun()
            st.error("Couldn't generate questions – add more material for this unit.")
    cq = db.class_quizzes()
    if cq.empty:
        return
    att = db.attempts()
    n_students = max(len(db.users()), 1)
    for q in cq.itertuples():
        qs = json.loads(q.questions)
        taken = att[att.quiz_id == q.id] if not att.empty else att
        who = taken.user.nunique() if not taken.empty else 0
        status = "published" if q.published else "draft"
        ui.tasks([ui.task(q.title, f"{q.unit or 'Whole course'} · {len(qs)} questions · {q.bloom} · due {q.due}",
                          f"{status} · {who}/{n_students} done", "calm" if q.published else "ink")])
        with st.expander("Review" if not q.published else "Results", icon=":material/fact_check:"):
            if not q.published:
                keep = []
                for i, x in enumerate(qs):
                    k = st.checkbox(f"Q{i + 1}. {x['question']}", value=True, key=f"keep_{q.id}_{i}")
                    st.caption("Options: " + " · ".join(("✅ " if j == int(x["answer_index"]) else "") + o
                                                        for j, o in enumerate(x["options"])))
                    if k:
                        keep.append(x)
                c1, c2 = st.columns(2)
                if c1.button("Publish to students", key=f"pub_{q.id}", type="primary", icon=":material/send:"):
                    with db.conn() as c:
                        c.execute("UPDATE class_quizzes SET questions=?, published=1 WHERE id=?",
                                  (json.dumps(keep), q.id))
                    db.add_announcement(f"New quiz: {q.title} – due {q.due}. Find it in Quiz.")
                    st.rerun()
                if c2.button("Delete draft", key=f"delq_{q.id}"):
                    db.delete_class_quiz(int(q.id))
                    st.rerun()
            else:
                if taken.empty:
                    st.caption("No submissions yet.")
                else:
                    ui.stats([("students done", f"{who}/{n_students}"), ("average", f"{taken.score.mean():.0%}")])
                    per_q = taken.groupby("question").score.mean().sort_values()
                    st.markdown("**Most-missed questions** – likely misconceptions")
                    ui.bars(per_q.rename(index=lambda s: s[:70] + ("…" if len(s) > 70 else "")), "question",
                            "correct", pct=True)
                missing = sorted(set(db.users().roll) - set(taken.user)) if not taken.empty else list(db.users().roll)
                if missing:
                    st.caption("Not attempted: " + ", ".join(_who(r) for r in missing))
                if st.button("Unpublish", key=f"unpub_{q.id}"):
                    db.set_quiz_published(int(q.id), False)
                    st.rerun()


# ---------------------------------------------------------------- assignments
RUBRIC_TEMPLATE = ("Definition and concept | 3 | tokenization, tokens, split text\n"
                   "Types / methods | 4 | word tokenization, subword, byte pair encoding\n"
                   "Example | 3 | example, sentence")


def assignments_page() -> None:
    llm = ctx.llm()
    ui.title("Assignments", "AI suggests marks against your rubric and flags similar submissions. "
                            "You decide every final mark.")
    us = ctx.units()
    with st.expander("Create an assignment", expanded=db.assignments().empty, icon=":material/add_circle:"):
        with st.form("new_asg", clear_on_submit=True):
            c1, c2, c3 = st.columns([2, 1, 1])
            title = c1.text_input("Title", placeholder="Tokenization techniques")
            unit = c2.selectbox("Unit", [""] + [u["unit"] for u in us])
            due = c3.date_input("Due", date.today() + timedelta(days=7))
            desc = st.text_area("Task for students", placeholder="Explain tokenization and compare word and subword "
                                                                 "methods with an example.")
            rubric = st.text_area("Rubric – one criterion per line:  criterion | marks | key concepts",
                                  RUBRIC_TEMPLATE, height=110)
            if st.form_submit_button("Create assignment", type="primary") and title.strip():
                crit = asg.parse_rubric(rubric)
                db.add_assignment(title.strip(), desc, unit, str(due), rubric, sum(c["marks"] for c in crit) or 10)
                db.add_announcement(f"New assignment: {title.strip()} – due {due}.")
                st.rerun()
    a = db.assignments()
    if a.empty:
        return
    names = _names()
    for row in a.itertuples():
        subs = db.submissions(int(row.id))
        late = subs[pd.to_datetime(subs["at"]).dt.date > pd.to_datetime(row.due).date()] if not subs.empty else subs
        missing = sorted(set(names) - set(subs.user)) if names else []
        ui.tasks([ui.task(row.title, f"{row.unit} · due {row.due} · {row.max_marks:g} marks",
                          f"{len(subs)}/{len(names)} submitted", "ink")])
        with st.expander("Submissions", icon=":material/inbox:"):
            if missing:
                st.caption("Not submitted: " + ", ".join(f"{names[r]} ({r})" for r in missing))
            if subs.empty:
                st.caption("No submissions yet.")
                if st.button("Delete assignment", key=f"dela_{row.id}"):
                    db.delete_assignment(int(row.id))
                    st.rerun()
                continue
            sims = asg.similarity_pairs(dict(zip(subs.user, subs.text)))
            for u1, u2, s in sims:
                st.warning(f"Very similar submissions: {_who(u1)} and {_who(u2)} – cosine {s:.2f}. Worth a look.",
                           icon=":material/content_copy:")
            if st.button("AI-review all unreviewed", key=f"rv_all_{row.id}", icon=":material/auto_awesome:"):
                with st.spinner("Reviewing…"):
                    for s in subs[subs.ai_score.isna()].itertuples():
                        r = asg.review(s.text, row.title, row.description, row.rubric, llm)
                        db.save_ai_review(int(s.id), r["total"], asg.to_json(r))
                st.rerun()
            for s in subs.itertuples():
                is_late = s.id in set(late.id)
                head = f"{_who(s.user)} · {s.at[:16]}" + (" · late" if is_late else "") + \
                       (f" · marked {s.final_mark:g}" if s.released else "")
                with st.container(border=True):
                    st.markdown(f"**{head}**")
                    st.caption(s.text[:400] + ("…" if len(s.text) > 400 else ""))
                    if s.ai_feedback:
                        r = json.loads(s.ai_feedback)
                        st.dataframe(pd.DataFrame(r["criteria"]), hide_index=True, width="stretch")
                        if r.get("missing_concepts"):
                            ui.chips([ui.chip("missing: " + m, "red") for m in r["missing_concepts"][:8]])
                        st.caption(f"{r.get('summary', '')}  ·  suggested {r['total']:g}/{r['max']:g} · {r['method']}")
                    elif st.button("AI review", key=f"rv_{s.id}", icon=":material/auto_awesome:"):
                        r = asg.review(s.text, row.title, row.description, row.rubric, llm)
                        db.save_ai_review(int(s.id), r["total"], asg.to_json(r))
                        st.rerun()
                    c1, c2, c3 = st.columns([1, 3, 1.2])
                    default = s.final_mark if pd.notna(s.final_mark) else (s.ai_score if pd.notna(s.ai_score) else 0.0)
                    mark = c1.number_input("Final mark", 0.0, float(row.max_marks), float(default), 0.5,
                                           key=f"mk_{s.id}")
                    comment = c2.text_input("Comment to student", s.teacher_comment or "", key=f"cm_{s.id}")
                    if c3.button("Release mark", key=f"rel_{s.id}", type="primary"):
                        db.release_mark(int(s.id), mark, comment)
                        st.toast(f"Mark released to {_who(s.user)}.")
                        st.rerun()


# ---------------------------------------------------------------- question paper
def paper_page() -> None:
    llm, store = ctx.llm(), ctx.store()
    ui.title("Question paper", "Pick units and marks – every question comes with its Bloom's level, CO and an answer key.")
    us = ctx.units()
    if not us:
        st.info("Load the course syllabus in Materials first.")
        return
    c1, c2, c3 = st.columns([3, 1, 1])
    course = c1.text_input("Course", config.COURSE_NAME)
    code = c2.text_input("Code", config.COURSE_CODE)
    duration = c3.text_input("Duration", "2 hours")
    chosen = st.multiselect("Units", [ctx.unit_label(u) for u in us], default=[ctx.unit_label(u) for u in us])
    c1, c2, c3, c4 = st.columns(4)
    n2 = c1.number_input("2-mark questions", 0, 20, 5, help="Remember / Understand")
    n5 = c2.number_input("5-mark questions", 0, 10, 4, help="Apply / Analyze")
    n10 = c3.number_input("10-mark questions", 0, 6, 2, help="Analyze / Evaluate / Create")
    key = c4.checkbox("Answer key", True)
    total = 2 * n2 + 5 * n5 + 10 * n10
    ui.chips([ui.chip(f"Total {total} marks")])
    if st.button("Generate paper", type="primary", icon=":material/auto_awesome:") and chosen:
        sel = [u for u in us if ctx.unit_label(u) in chosen]
        with st.spinner("Setting the paper…"), arduino.working("Paper ready"):
            ss.paper = paper.generate(sel, {2: n2, 5: n5, 10: n10}, llm, store, course)
        ss.paper_meta = (course, code, duration, total, key)
    if ss.get("paper"):
        course, code, duration, total, key = ss.paper_meta
        df = pd.DataFrame(ss.paper)
        st.dataframe(df[["marks", "question", "unit", "bloom", "co"]], hide_index=True, width="stretch")
        c1, c2 = st.columns(2)
        with c1:
            ui.bars(df.groupby("bloom").marks.sum(), "Bloom's level", "marks")
        with c2:
            ui.bars(df.groupby("unit").marks.sum(), "unit", "marks")
        md = paper.to_markdown(ss.paper, course, code, duration, total, key)
        st.download_button("Download paper (Markdown)", md, f"{code}_question_paper.md", icon=":material/download:")
        with st.expander("Preview"):
            st.markdown(md)


# ---------------------------------------------------------------- doubts & feedback
def doubts_page() -> None:
    ui.title("Doubts & feedback", "Reply once – publish common answers to the FAQ so the class finds them.")
    flags = db.help_flags()
    if flags.empty:
        st.caption("No open doubts. Students send them from the Ask page, anonymously if they prefer.")
    for f in flags.itertuples():
        with st.container(border=True):
            ui.chips([ui.chip(_who(f.user, f.anonymous)), ui.chip(f.unit or "untagged"),
                      ui.chip(f.flagged_at[:16]), ui.chip(f"via {f.source}")])
            st.markdown(f"**{f.question}**")
            reply = st.text_area("Reply", key=f"rep{f.id}", height=80, label_visibility="collapsed",
                                 placeholder="Your reply – the student sees it on their home page")
            c1, c2, c3 = st.columns([1.2, 1.4, 3])
            if c1.button("Reply", key=f"res{f.id}", type="primary"):
                db.resolve_flag(f.id, reply)
                st.rerun()
            if c2.button("Reply + add to FAQ", key=f"faq{f.id}") and reply.strip():
                db.resolve_flag(f.id, reply)
                db.add_faq(f.question, reply, f.unit or "")
                st.toast("Added to FAQ.")
                st.rerun()
    st.subheader("Answer ratings")
    fb = db.feedback()
    if fb.empty:
        st.caption("Students rate answers 👍/👎 with an optional comment; VADER scores each comment.")
        return
    ui.stats([("ratings", len(fb)), ("found helpful", f"{fb.rating.mean():.0%}"),
              ("average sentiment", f"{fb.compound.mean():+.2f}")])
    ui.bars(fb.groupby("sentiment").size(), "sentiment", "comments")
    neg = fb[(fb.sentiment == "negative") | (fb.rating == 0)]
    if not neg.empty:
        st.markdown("**Unhelpful answers** – check whether the notes cover these")
        st.dataframe(neg[["at", "unit", "question", "comment", "compound"]], hide_index=True, width="stretch")


# ---------------------------------------------------------------- attendance
def attendance_page() -> None:
    ui.title("Attendance", "Import the register as CSV. The app only reads it – corrections stay with you.")
    up = st.file_uploader("Attendance CSV", type=["csv"])
    st.caption("Either one row per student with a column per date (P/A), or columns roll, date, status.")
    st.download_button("Download a sample CSV", "roll,name,01-11-2026,03-11-2026,05-11-2026\n15,Harshal,P,P,A\n"
                                                "22,Asha,P,A,A\n", "attendance_sample.csv", icon=":material/download:")
    if up is not None and st.button("Import", type="primary"):
        rows = reports.parse_attendance(up.getvalue())
        if rows:
            db.import_attendance(rows)
            st.success(f"Imported {len(rows)} records.")
            st.rerun()
        else:
            st.error("No attendance found. Check the columns: roll + date columns with P/A, or roll, date, status.")
    s = reports.attendance_summary()
    if s.empty:
        return
    names = _names()
    s.insert(1, "name", s.roll.map(names).fillna(""))
    low = s[s["attendance %"] < 75]
    ui.stats([("students", len(s)), ("average", f"{s['attendance %'].mean():.0f}%"),
              ("below 75%", len(low), "warn" if len(low) else "good")])
    st.dataframe(s.sort_values("attendance %"), hide_index=True, width="stretch",
                 column_config={"attendance %": st.column_config.ProgressColumn("attendance %", min_value=0,
                                                                                max_value=100, format="%.0f%%")})
    if st.button("Clear attendance data"):
        db.clear_attendance()
        st.rerun()


# ---------------------------------------------------------------- class well-being
def wellbeing_page() -> None:
    ui.title("Class well-being", "Anonymous totals from students' optional check-ins. Names appear only when a "
                                 "student chooses to share.")
    ci = db.checkins()
    if ci.empty:
        st.info("No check-ins yet. Students check in from their home page.")
        return
    week = ci[ci["at"] >= str(date.today() - timedelta(days=7))]
    ui.stats([("check-ins this week", len(week)),
              ("feeling stressed or overwhelmed", int((week.mood <= 2).sum()),
               "warn" if (week.mood <= 2).any() else "good"),
              ("students checking in", week.user.nunique())])
    if not week.empty:
        dist = week.mood.map(lambda m: f"{wellbeing.MOODS[m][0]} {wellbeing.MOODS[m][1]}").value_counts()
        order = [f"{e} {l}" for _, (e, l) in sorted(wellbeing.MOODS.items(), reverse=True)]
        ui.bars(dist.reindex([o for o in order if o in dist.index]), "mood", "check-ins")
    by_day = ci.assign(day=pd.to_datetime(ci["at"]).dt.normalize()).groupby("day").mood.mean().reset_index()
    by_day["avg"] = by_day.mood / 5
    st.subheader("Class mood over time")
    ui.line(by_day, "day", "avg", "average mood")
    st.caption("Stress usually rises before deadlines – compare with your assignment and quiz dates.")
    shared = ci[ci.shared == 1].sort_values("at", ascending=False)
    st.subheader("Shared with you")
    if shared.empty:
        st.caption("No student has shared a check-in.")
    for r in shared.head(20).itertuples():
        e, l = wellbeing.MOODS[int(r.mood)]
        ui.tasks([ui.task(f"{_who(r.user)} – {e} {l}", r.note or "(no note)", r.at[:10],
                          "" if r.mood <= 2 else "calm")])
    ui.support("If a student seems to be struggling",
               "<p>This is a check-in, not a diagnosis. Reach out personally, listen, and involve the college "
               "counsellor or mentor. Tele-MANAS (<strong>14416</strong> / <strong>1-800-891-4416</strong>) is free, "
               "confidential and 24×7.</p>")


# ---------------------------------------------------------------- FAQ
def faq_page() -> None:
    ui.title("FAQ manager", "Questions students keep asking – answer each one once.")
    log = db.chat_log()
    log = log[log.user != "faculty"]
    published = set(db.faqs().question.str.lower()) if not db.faqs().empty else set()
    groups = [g for g in faq.repeated_questions(log.question.tolist()) if g["question"].lower() not in published]
    st.subheader("Asked repeatedly")
    if not groups:
        st.caption("Nothing repeated yet.")
    for i, g in enumerate(groups[:8]):
        with st.container(border=True):
            ui.chips([ui.chip(f"asked {g['count']}×", "amber")])
            st.markdown(f"**{g['question']}**")
            st.caption("Variants: " + " · ".join(g["variants"][:5]))
            key = f"ta_{i}"
            if st.button("Draft answer from course material", key=f"d_{i}", icon=":material/auto_awesome:"):
                with st.spinner("Drafting…"):
                    try:
                        ans = ctx.assistant().ask(g["question"], "Beginner", bool(ctx.store().chunks))
                        ss[key] = ans.text          # fills the text box below (set before it is drawn)
                    except LLMError as e:
                        ss[key] = f"(AI unavailable: {e})"
            answer = st.text_area("Answer", key=key, height=140)
            if st.button("Publish to FAQ", key=f"p_{i}", type="primary") and answer.strip():
                db.add_faq(g["question"], answer.strip(), (syllabus.tag_unit(g["question"]) or {}).get("unit", ""))
                st.rerun()
    st.subheader("Add your own")
    with st.form("faq_add", clear_on_submit=True):
        q = st.text_input("Question", placeholder="When is the Unit III assignment due?")
        a = st.text_area("Answer", height=90)
        if st.form_submit_button("Publish") and q.strip() and a.strip():
            db.add_faq(q.strip(), a.strip(), (syllabus.tag_unit(q) or {}).get("unit", ""))
            st.rerun()
    st.subheader("Published")
    for r in db.faqs().itertuples():
        c1, c2 = st.columns([8, 1])
        with c1.expander(r.question):
            st.markdown(r.answer)
        if c2.button("Delete", key=f"delf_{r.id}"):
            db.delete_faq(int(r.id))
            st.rerun()
