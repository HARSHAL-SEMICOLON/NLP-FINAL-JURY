"""Student pages: home (check-in, today's plan, deadlines, readiness), well-being, quiz,
assignments, FAQ, study planner, progress."""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta

import pandas as pd
import streamlit as st

from sah import arduino, config, db, documents, paper, planner, quiz, syllabus, wellbeing
from sah.rag import DIFFICULTY

from . import ctx
from . import theme as ui

ss = st.session_state


def _due_text(days: int) -> str:
    if days < 0:
        return f"{-days} day{'s' if days != -1 else ''} overdue"
    return {0: "today", 1: "tomorrow"}.get(days, f"in {days} days")


def _greeting() -> str:
    h = datetime.now().hour
    return "Good morning" if h < 12 else "Good afternoon" if h < 17 else "Good evening"


# ---------------------------------------------------------------- check-in widget
def checkin_widget(compact: bool = True) -> None:
    me = ctx.me()
    today = str(date.today())
    done = db.checkins(me)
    done_today = not done.empty and (done["at"].str[:10] == today).any()
    labels = [f"{e} {l}" for _, (e, l) in sorted(wellbeing.MOODS.items(), reverse=True)]
    with st.container(border=True):
        st.markdown("**How are you feeling about your studies today?**")
        if done_today and compact and not ss.get("checkin_open"):
            last = done.iloc[-1]
            e, l = wellbeing.MOODS[int(last.mood)]
            st.caption(f"You checked in today: {e} {l}. Thanks – it helps you notice patterns.")
            if st.button("Check in again", key="ci_again"):
                ss.checkin_open = True
                st.rerun()
            return
        with st.form("checkin", border=False, clear_on_submit=True):
            pick = st.pills("Mood", labels, label_visibility="collapsed")
            note = st.text_input("Anything on your mind? (optional, private)",
                                 placeholder="e.g. Unit III feels like too much")
            share = st.checkbox("Share this check-in with my teacher so they can support me",
                                help="Off by default. Without it, your teacher only sees anonymous class totals.")
            sent = st.form_submit_button("Save check-in")
        if sent and pick:
            mood = 5 - labels.index(pick)
            db.add_checkin(me, mood, wellbeing.MOODS[mood][1], note.strip(), share)
            ss.checkin_result = {"mood": mood, "urgent": wellbeing.needs_urgent_support(note)}
            ss.checkin_open = False
            st.rerun()
        elif sent:
            st.caption("Pick how you feel first.")
    res = ss.pop("checkin_result", None)
    if res:
        show_support(res["mood"], res["urgent"])


def show_support(mood: int, urgent: bool) -> None:
    if urgent:
        ui.support("You don't have to handle this alone",
                   "<p>What you wrote sounds really heavy. Please talk to someone today – a friend, family "
                   "member, your mentor or the college counsellor.</p>"
                   f"<p>{_md(wellbeing.HELPLINE)}</p>", urgent=True)
        return
    steps = "".join(f"<li>{ui.esc(s)}</li>" for s in wellbeing.coping_steps(mood))
    extra = (f"<p style='margin-top:.5rem'>If it keeps feeling like too much, talking helps. "
             f"{_md(wellbeing.HELPLINE)}</p>") if mood <= 2 else ""
    head = {5: "Great – let's use that energy", 4: "Nice, steady progress", 3: "Go easy, keep it small",
            2: "Let's make today lighter", 1: "One small step is enough today"}[mood]
    ui.support(head, f"<ul style='margin:.2rem 0 0 1.1rem'>{steps}</ul>{extra}")


def _md(text: str) -> str:
    """Tiny markdown → HTML for **bold** in trusted strings."""
    import re

    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", ui.esc(text))


# ---------------------------------------------------------------- home
def home_page() -> None:
    me, user = ctx.me(), ctx.user()
    us = ctx.units()
    ui.title(f"{_greeting()}, {user['name'].split()[0]}", f"{config.COURSE_NAME} · {config.COURSE_CODE}")

    r = wellbeing.readiness(me, us) if us else None
    if r and r["days_left"] is not None:
        pace = (f"About <strong>{r['topics_per_day']:g} topics a day</strong> keeps you on track."
                if r["remaining_topics"] else "Every planned topic is done – switch to revision and practice.")
        ui.notebook(f"Exam in {r['days_left']} days · {r['overall']:.0%} ready",
                    f"{r['remaining_topics']} topics left in your plan. {pace}")
    elif us:
        ui.notebook("Set your exam date",
                    "Add the course to your study planner and the app will break the syllabus into small daily tasks.")
        if st.button("Open study planner", icon=":material/calendar_month:"):
            ctx.goto("Study planner")

    for a in db.announcements(3).itertuples():
        st.info(f"**From your teacher** · {a.at[:10]} — {a.text}", icon=":material/campaign:")
    replies = db.help_flags(include_resolved=True, user=me)
    replies = replies[(replies.resolved == 1) & (replies.reply.fillna("") != "")]
    for rp in replies.head(3).itertuples():
        st.success(f"**Your teacher replied** to “{rp.question[:80]}” — {rp.reply}", icon=":material/forum:")

    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.subheader("Today's plan")
        tasks = wellbeing.today_tasks(me, us)
        if not tasks:
            st.caption("No plan yet. Add the syllabus in the study planner to get daily tasks.")
        ui.tasks([ui.task(t["title"], t["meta"], "practice" if t["kind"] == "practice" else
                          ("revise" if t["kind"] == "revision" else ""),
                          "ink" if t["kind"] == "practice" else "calm") for t in tasks])
        practice = next((t for t in tasks if t["kind"] == "practice"), None)
        if practice and st.button(f"Start practice on {practice['unit']}", type="primary",
                                  icon=":material/play_arrow:"):
            ss.quiz_prefill = practice["unit"]
            ctx.goto("Quiz")
        checkin_widget()
    with right:
        st.subheader("Coming up")
        rem = wellbeing.reminders(me)
        if not rem:
            st.caption("No deadlines in the next two weeks.")
        ui.tasks([ui.task(f"{x['kind']}: {x['title']}", x["unit"], _due_text(x["days"]),
                          "" if x["days"] <= 2 else "ink") for x in rem[:6]])
        if rem and rem[0]["days"] <= 1 and not ss.get("led_reminded"):
            arduino.set_status("ready", f"{rem[0]['kind']} due {_due_text(rem[0]['days'])}")
            ss.led_reminded = True
        if r:
            st.subheader("Readiness by unit")
            st.markdown("".join(ui.meter(f"{u['unit']} · {u['title'][:34]}", u["readiness"])
                                for u in r["units"]), unsafe_allow_html=True)

    st.subheader("Your progress")
    s = wellbeing.streak(me)
    got = wellbeing.earned_badges(me)
    att = db.attempts(me)
    ui.stats([("day streak", f"{s} 🔥" if s else "0", "good" if s >= 3 else ""),
              ("badges earned", f"{len(got)}/{len(wellbeing.BADGES)}"),
              ("quiz accuracy", f"{att.score.mean():.0%}" if not att.empty else "—"),
              ("questions asked", len(db.chat_log(me)))])
    msg = wellbeing.improvement(me)
    if msg:
        ui.chips([ui.chip("📈 " + msg, "sage")])
    ui.badges(wellbeing.BADGES, got)


# ---------------------------------------------------------------- well-being
def wellbeing_page() -> None:
    me = ctx.me()
    ui.title("Check-in", "A private space to notice how studying feels. Not a medical tool – "
                         "just a way to keep stress manageable.")
    checkin_widget(compact=False)
    hist = db.checkins(me)
    if not hist.empty:
        st.subheader("Your check-ins")
        h = hist.assign(day=pd.to_datetime(hist["at"]).dt.normalize()).groupby("day").mood.mean().reset_index()
        h["mood level"] = h.mood / 5
        ui.line(h, "day", "mood level", "mood (higher = better)")
        st.caption("Shared with your teacher: " +
                   (f"{int(hist.shared.sum())} of {len(hist)} check-ins" if hist.shared.any() else "none"))
    st.subheader("When exams feel like too much")
    c1, c2 = st.columns(2)
    with c1:
        ui.tasks([ui.task("Shrink the task", "Open Today's plan and do only the first item.", tone="calm"),
                  ui.task("Ask instead of worrying", "Send a doubt – anonymously if you like.", tone="calm"),
                  ui.task("Protect sleep", "Sleep is when memory consolidates. Late-night cramming costs marks.",
                          tone="calm")])
    with c2:
        ui.support("People who can help",
                   "<p>Talk to a friend, family member, your mentor or the college counsellor.</p>"
                   f"<p>{_md(wellbeing.HELPLINE)}</p>")


# ---------------------------------------------------------------- quiz
def _take_class_quiz(qrow) -> None:
    me = ctx.me()
    qs = json.loads(qrow.questions)
    with st.form(f"cq_{qrow.id}"):
        picks = [st.radio(f"**Q{i}.** {q['question']}", q["options"], index=None, key=f"cq{qrow.id}_{i}")
                 for i, q in enumerate(qs, 1)]
        if st.form_submit_button("Submit quiz", type="primary"):
            correct = 0
            for q, p in zip(qs, picks):
                ok = p == q["options"][int(q["answer_index"])]
                correct += ok
                db.record_attempt(config.COURSE_CODE, q.get("topic") or qrow.unit or "class quiz", "class",
                                  q["question"], ok, float(ok), me, qrow.unit or
                                  (syllabus.tag_unit(q["question"]) or {}).get("unit", ""),
                                  q.get("bloom") or qrow.bloom, int(qrow.id))
            ss[f"cq_result_{qrow.id}"] = correct
            arduino.set_status("success" if correct / len(qs) >= .6 else "error", f"Score {correct}/{len(qs)}")
            st.rerun()


def quiz_page() -> None:
    store, llm, me = ctx.store(), ctx.llm(), ctx.me()
    ui.title("Quiz", "Practise by unit and by Bloom's level – Remember up to Create, like the end-semester paper.")
    att = db.attempts(me)
    taken = set(att.quiz_id.dropna().astype(int)) if not att.empty else set()
    cq = db.class_quizzes(published_only=True)
    if not cq.empty:
        st.subheader("From your teacher")
        for q in cq.itertuples():
            n = len(json.loads(q.questions))
            days = (pd.to_datetime(q.due).date() - date.today()).days
            if q.id in taken:
                mine = att[att.quiz_id == q.id]
                ui.tasks([ui.task(q.title, f"{q.unit} · {n} questions", f"done · {mine.score.mean():.0%}", "calm")])
            else:
                ui.tasks([ui.task(q.title, f"{q.unit} · {n} questions · {q.bloom}", f"due {_due_text(days)}",
                                  "" if days <= 2 else "ink")])
                with st.expander(f"Take “{q.title}”", icon=":material/quiz:"):
                    _take_class_quiz(q)
            if f"cq_result_{q.id}" in ss:
                st.success(f"Submitted “{q.title}”: {ss[f'cq_result_{q.id}']}/{n} correct.")
        st.divider()
    st.subheader("Practice quiz")
    if not store.chunks:
        st.info("No course material yet – your teacher (or you) can add it in Materials.")
        return
    us = ctx.units()
    c1, c2, c3, c4 = st.columns(4)
    kind = c1.radio("Type", ["MCQ", "Short answer"], horizontal=True)
    n = c2.number_input("Questions", 1, 15, 5)
    difficulty = c3.selectbox("Difficulty", list(DIFFICULTY), index=1)
    bloom = c4.selectbox("Bloom's level", ["Mixed"] + syllabus.BLOOM_ORDER)
    labels = ["All units"] + [ctx.unit_label(u) for u in us]
    pre = ss.pop("quiz_prefill", None)
    idx = next((i for i, l in enumerate(labels) if pre and l.startswith(pre + " ")), 0)
    unit_choice = st.selectbox("Syllabus unit", labels, index=idx) if us else "All units"
    topic = st.text_input("Specific topic (optional)", placeholder="e.g. positional encoding")
    unit = next((u for u in us if ctx.unit_label(u) == unit_choice), None)
    focus = topic or (unit["title"] + " " + unit["topics"] if unit else "")
    ss.setdefault("mcq", [])
    ss.setdefault("short", [])
    if st.button("Generate quiz", type="primary", icon=":material/auto_awesome:"):
        with st.spinner("Writing questions from your notes…"), arduino.working("Quiz ready"):
            if kind == "MCQ":
                ss.mcq, ss.short = quiz.generate_mcq(store, llm, n, focus, None, difficulty, bloom), []
            else:
                ss.short, ss.mcq = quiz.generate_short(store, llm, n, focus, None, difficulty, bloom), []
        ss.quiz_meta = {"unit": unit["unit"] if unit else "", "bloom": bloom, "topic": topic}
        if not (ss.mcq or ss.short):
            st.error("Couldn't make questions from this material. Try another unit or add more notes.")
        if not llm.available:
            st.caption("Offline mode: fill-in-the-blank questions built from TF-IDF keywords.")
    meta = ss.get("quiz_meta", {"unit": "", "bloom": bloom, "topic": topic})

    def q_unit(q: dict) -> str:
        return meta["unit"] or (syllabus.tag_unit(q["question"]) or {}).get("unit", "")

    if ss.mcq:
        with st.form("mcq_form"):
            picks = [st.radio(f"**Q{i}.** {q['question']}", q["options"], index=None, key=f"mcq{i}")
                     for i, q in enumerate(ss.mcq, 1)]
            submitted = st.form_submit_button("Check answers", type="primary")
        if submitted:
            correct = 0
            for i, (q, p) in enumerate(zip(ss.mcq, picks), 1):
                ok = p == q["options"][int(q["answer_index"])]
                correct += ok
                db.record_attempt(config.COURSE_CODE, q.get("topic") or meta["topic"] or "general", "mcq",
                                  q["question"], ok, float(ok), me, q_unit(q), q.get("bloom") or meta["bloom"])
                with st.expander(f"{'✅' if ok else '❌'} Q{i}", expanded=not ok):
                    st.write(f"Correct answer: **{q['options'][int(q['answer_index'])]}**")
                    st.caption(q.get("explanation", ""))
            ui.stats([("score", f"{correct}/{len(ss.mcq)}", "good" if correct / len(ss.mcq) >= .6 else "warn")])
            arduino.set_status("success" if correct / len(ss.mcq) >= 0.6 else "error", f"Score {correct}/{len(ss.mcq)}")
    if ss.short:
        with st.form("short_form"):
            answers = [st.text_area(f"**Q{i}.** {q['question']}", key=f"sa{i}") for i, q in enumerate(ss.short, 1)]
            submitted = st.form_submit_button("Grade my answers", type="primary")
        if submitted:
            total = 0.0
            with st.spinner("Grading…"):
                for i, (q, a) in enumerate(zip(ss.short, answers), 1):
                    g = quiz.grade_short(q, a, llm)
                    total += g["score"]
                    db.record_attempt(config.COURSE_CODE, q.get("topic") or meta["topic"] or "general", "short",
                                      q["question"], g["score"] >= 6, g["score"] / 10, me, q_unit(q),
                                      q.get("bloom") or meta["bloom"])
                    with st.expander(f"Q{i}: {g['score']}/10", expanded=True):
                        st.write(g["feedback"])
                        st.markdown(f"**Model answer:** {q['model_answer']}")
            ui.stats([("total", f"{total:.1f}/{10 * len(ss.short)}")])


# ---------------------------------------------------------------- assignments
def assignments_page() -> None:
    me = ctx.me()
    ui.title("Assignments", "Submit before the deadline. Your teacher reviews every submission and decides the marks.")
    a = db.assignments()
    if a.empty:
        st.info("No assignments yet.")
        return
    mine = db.submissions(user=me)
    for row in a.itertuples():
        sub = mine[mine.assignment_id == row.id]
        days = (pd.to_datetime(row.due).date() - date.today()).days
        if not sub.empty and sub.released.iloc[0]:
            tone, status = "calm", f"marked · {sub.final_mark.iloc[0]:g}/{row.max_marks:g}"
        elif not sub.empty:
            tone, status = "ink", "submitted"
        else:
            tone, status = ("" if days <= 3 else "ink"), f"due {_due_text(days)}"
        ui.tasks([ui.task(row.title, f"{row.unit} · {row.max_marks:g} marks · due {row.due}", status, tone)])
        with st.expander("Open", icon=":material/assignment:"):
            st.markdown(row.description or "")
            from sah.assignments import parse_rubric

            rub = parse_rubric(row.rubric or "")
            if rub:
                st.caption("Marking criteria: " + " · ".join(f"{c['criterion']} ({c['marks']:g})" for c in rub))
            if not sub.empty and sub.released.iloc[0]:
                s0 = sub.iloc[0]
                st.success(f"**Mark: {s0.final_mark:g}/{row.max_marks:g}**  \n{s0.teacher_comment or ''}")
                if s0.ai_feedback:
                    fb = json.loads(s0.ai_feedback)
                    if fb.get("missing_concepts"):
                        st.caption("To strengthen: " + ", ".join(fb["missing_concepts"][:6]))
                continue
            if days < 0 and sub.empty:
                st.warning("The deadline has passed. You can still submit – it will be marked late.")
            with st.form(f"sub_{row.id}"):
                text = st.text_area("Your answer", value=sub.text.iloc[0] if not sub.empty else "", height=200)
                up = st.file_uploader("…or upload a PDF/TXT", type=["pdf", "txt", "md"], key=f"up_{row.id}")
                if st.form_submit_button("Submit" if sub.empty else "Update submission", type="primary"):
                    fname = ""
                    if up is not None:
                        path = config.UPLOAD_DIR / f"sub_{row.id}_{me}_{up.name}"
                        path.write_bytes(up.getbuffer())
                        text = (text + "\n\n" if text.strip() else "") + documents.full_text(path)
                        fname = up.name
                    if text.strip():
                        db.submit(int(row.id), me, text.strip(), fname)
                        st.toast("Submitted.")
                        st.rerun()
                    else:
                        st.error("Write an answer or upload a file first.")


# ---------------------------------------------------------------- FAQ
def faq_page() -> None:
    ui.title("FAQ", "Answers your teacher has already written for common questions.")
    f = db.faqs()
    if f.empty:
        st.info("No FAQs yet. Ask in the Ask page – common questions get answered here.")
        return
    q = st.text_input("Search", placeholder="e.g. tokenization", label_visibility="collapsed",
                      icon=":material/search:")
    if q:
        f = f[f.question.str.contains(q, case=False) | f.answer.str.contains(q, case=False)]
    for r in f.itertuples():
        with st.expander(r.question, icon=":material/help:"):
            if r.unit:
                ui.chips([ui.chip(r.unit)])
            st.markdown(r.answer)


# ---------------------------------------------------------------- planner
def planner_page() -> None:
    me = ctx.me()
    ui.title("Study planner", "Breaks the syllabus into daily tasks, harder topics first, with a revision day "
                              "before each exam.")
    us = ctx.units()
    tp = db.topics(me)
    if us:
        with st.expander("Add the whole course from the syllabus", expanded=tp.empty, icon=":material/bolt:"):
            exam = st.date_input("End-semester exam date", date.today() + timedelta(days=21), key="syl_exam")
            if st.button(f"Add {config.COURSE_CODE} topics", type="primary"):
                sid = db.upsert_subject(config.COURSE_CODE, exam.isoformat(), me)
                existing = set(tp[tp.subject_id == sid].name) if not tp.empty else set()
                for u in us:
                    for t in paper._topic_list(u["topics"])[:12]:
                        if t not in existing:
                            db.add_topic(sid, u["unit"], t, 3 if u["index"] in (2, 3) else 2)
                st.rerun()
    with st.expander("Add another subject", icon=":material/add:"):
        with st.form("add_subject", clear_on_submit=True):
            name = st.text_input("Subject", placeholder="Deep Learning for Computer Vision")
            exam = st.date_input("Exam date", date.today() + timedelta(days=14))
            raw = st.text_area("Topics – one per line as  Unit | Topic | difficulty(1-3)",
                               placeholder="Unit 1 | CNN basics | 1\nUnit 2 | Object detection | 3", height=120)
            if st.form_submit_button("Save subject") and name.strip():
                sid = db.upsert_subject(name.strip(), exam.isoformat(), me)
                for line in raw.splitlines():
                    parts = [p.strip() for p in line.split("|")]
                    if len(parts) >= 2 and parts[1]:
                        diff = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 2
                        db.add_topic(sid, parts[0], parts[1], min(max(diff, 1), 3))
                st.rerun()
    if tp.empty:
        return
    st.subheader("Topic checklist")
    for subj, grp in tp.groupby("subject", sort=False):
        done = int(grp.done.sum())
        st.markdown(ui.meter(f"{subj} · exam {grp.exam_date.iloc[0]}", done / len(grp), f"{done}/{len(grp)} topics"),
                    unsafe_allow_html=True)
        for unit, ugrp in grp.groupby("unit", sort=False):
            with st.expander(f"{unit} · {int(ugrp.done.sum())}/{len(ugrp)}"):
                cols = st.columns(3)
                for i, r in enumerate(ugrp.itertuples()):
                    val = cols[i % 3].checkbox(r.name, value=bool(r.done), key=f"t{r.id}")
                    if val != bool(r.done):
                        db.set_done(r.id, val)
                        st.rerun()
    with st.expander("Remove a subject", icon=":material/delete:"):
        subs = db.subjects(me)
        pick = st.selectbox("Subject", subs.name)
        if st.button("Delete subject"):
            db.delete_subject(int(subs[subs.name == pick].id.iloc[0]))
            st.rerun()
    st.subheader("Schedule")
    hrs = st.slider("Study hours per day", 1.0, 10.0, 3.0, 0.5,
                    help="Be realistic – a plan you can keep beats an ambitious one you abandon.")
    w = db.weak_topics(me)
    sched, risk = planner.build_plan(tp, hrs, weak=set(w.topic) if not w.empty else set())
    if sched.empty:
        st.success("Nothing pending before your exams.")
    else:
        st.dataframe(sched, hide_index=True, width="stretch")
        st.download_button("Download plan (CSV)", sched.to_csv(index=False), "study_plan.csv",
                           icon=":material/download:")
    if not risk.empty:
        st.warning(f"{len(risk)} topics don't fit before the exam at {hrs:g} h/day. Add a little time, "
                   "or focus on the units with the most marks first.")
        with st.expander("Topics that don't fit"):
            st.dataframe(risk, hide_index=True)


# ---------------------------------------------------------------- progress
def progress_page() -> None:
    me = ctx.me()
    ui.title("My progress", "Where you're strong, where to revise next.")
    att = db.attempts(me)
    if att.empty:
        st.info("Take a quiz to see your progress here.")
    else:
        att["day"] = pd.to_datetime(att.taken_at).dt.normalize()
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Accuracy over time")
            ui.line(att.groupby("day").score.mean().reset_index(), "day", "score", "accuracy")
        with c2:
            st.subheader("By syllabus unit")
            u = att[att.unit != ""]
            if not u.empty:
                ui.bars(u.groupby("unit").score.mean(), "unit", "accuracy", pct=True)
        b = att[att.bloom.isin(syllabus.BLOOM_ORDER)]
        if not b.empty:
            st.subheader("By Bloom's level")
            order = [x for x in syllabus.BLOOM_ORDER if x in set(b.bloom)]
            ui.bars(b.groupby("bloom").score.mean().reindex(order), "level", "accuracy", pct=True)
        weak = db.weak_topics(me)
        if not weak.empty:
            st.subheader("Revise these")
            ui.tasks([ui.task(r.topic, f"{r.attempts} attempts", f"{r.avg_score:.0%}") for r in weak.itertuples()])
    st.subheader("Badges")
    ui.badges(wellbeing.BADGES, wellbeing.earned_badges(me))
    log = db.chat_log(me)
    if not log.empty:
        st.subheader("Questions I asked")
        st.dataframe(log[["asked_at", "question", "unit", "bloom"]].tail(20), hide_index=True, width="stretch")
