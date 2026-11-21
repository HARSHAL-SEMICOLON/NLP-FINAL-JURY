"""Smart Academic Help – Streamlit front end.  Run:  streamlit run app.py

Two roles share one app and one database:
  Student – AI study assistant, quizzes, assignments, planner, progress, well-being check-ins
  Faculty – class overview with re-teach loop, reports, class quizzes, assignment review,
            question papers, doubts & FAQ, attendance, class well-being
"""
from __future__ import annotations

import streamlit as st

from sah import arduino, config, db, syllabus
from ui import common, ctx, faculty, student
from ui import theme as ui

st.set_page_config(page_title="Smart Academic Help", page_icon=":material/school:", layout="wide")
ui.inject()
db.init()
ss = st.session_state


# ================================================================ login
def login_screen() -> None:
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.markdown('<div class="sah-login-mark">Smart<br>Academic Help</div>', unsafe_allow_html=True)
        st.markdown(f"An AI study companion for **{config.COURSE_NAME} ({config.COURSE_CODE})**.")
        ui.tasks([
            ui.task("For students", "Answers from your own course notes, quizzes by unit and Bloom's level, "
                                    "a daily plan, and a private place to say when it's too much.", tone="calm"),
            ui.task("For faculty", "See where the class is stuck, auto-graded quizzes, AI-suggested assignment "
                                   "marks you approve, reports and question papers in minutes.", tone="ink"),
        ])
    with right:
        t_student, t_faculty = st.tabs(["Student", "Faculty"])
        with t_student, st.form("student_login"):
            name = st.text_input("Your name")
            roll = st.text_input("Roll number")
            if st.form_submit_button("Start studying", type="primary", width="stretch"):
                if name.strip() and roll.strip():
                    ss.user = {"name": name.strip(), "roll": roll.strip(), "role": "student"}
                    db.touch_user(roll.strip(), name.strip())
                    st.rerun()
                else:
                    st.error("Enter your name and roll number.")
        with t_faculty, st.form("faculty_login"):
            name = st.text_input("Name", "Faculty")
            pin = st.text_input("Faculty PIN", type="password", help="Set FACULTY_PIN in .env (default 1234).")
            if st.form_submit_button("Open faculty view", type="primary", width="stretch"):
                if pin == config.FACULTY_PIN:
                    ss.user = {"name": name.strip() or "Faculty", "roll": "faculty", "role": "faculty"}
                    st.rerun()
                else:
                    st.error("That PIN doesn't match FACULTY_PIN in .env.")


if "user" not in ss:
    login_screen()
    st.stop()

user = ctx.user()
FAC = ctx.is_faculty()

# ================================================================ navigation
M = ":material/"
if FAC:
    sections = {
        "Overview": [(faculty.dashboard_page, "Class overview", "space_dashboard"),
                     (faculty.reports_page, "Reports", "summarize")],
        "Teach": [(faculty.class_quizzes_page, "Class quizzes", "quiz"),
                  (faculty.assignments_page, "Assignments", "assignment"),
                  (faculty.paper_page, "Question paper", "description"),
                  (common.summaries_page, "Teaching notes", "edit_note"),
                  (lambda: common.materials_page(True), "Materials & syllabus", "library_books")],
        "Students": [(faculty.doubts_page, "Doubts & feedback", "forum"),
                     (faculty.faq_page, "FAQ manager", "help"),
                     (faculty.attendance_page, "Attendance", "how_to_reg"),
                     (faculty.wellbeing_page, "Class well-being", "favorite")],
        "Tools": [(common.ask_page, "Ask", "chat"), (common.nlp_lab_page, "NLP Lab", "science"),
                  (common.evaluation_page, "Evaluation", "monitoring"),
                  (common.hardware_page, "Arduino", "developer_board")],
    }
else:
    sections = {
        "Today": [(student.home_page, "Home", "home"), (student.wellbeing_page, "Check-in", "self_improvement")],
        "Learn": [(common.ask_page, "Ask", "chat"), (student.quiz_page, "Quiz", "quiz"),
                  (common.summaries_page, "Summaries", "edit_note"), (student.faq_page, "FAQ", "help")],
        "Plan": [(student.planner_page, "Study planner", "calendar_month"),
                 (student.assignments_page, "Assignments", "assignment"),
                 (student.progress_page, "My progress", "trending_up")],
        "Explore": [(lambda: common.materials_page(False), "Materials", "library_books"),
                    (common.nlp_lab_page, "NLP Lab", "science")],
    }

pages: dict[str, list] = {}
for sec, items in sections.items():
    pages[sec] = []
    for fn, title, icon in items:
        pg = st.Page(fn, title=title, icon=f"{M}{icon}:", url_path=title.lower().replace(" ", "-").replace("&", "and"))
        ctx.PAGES[title] = pg
        pages[sec].append(pg)

# ================================================================ sidebar
with st.sidebar:
    st.markdown('<div class="sah-side-brand">Smart Academic Help</div>', unsafe_allow_html=True)
    who = f"{user['name']} · faculty" if FAC else f"{user['name']} · roll {user['roll']}"
    st.markdown(f'<div class="sah-side-user">{ui.esc(who)}</div>', unsafe_allow_html=True)

if ss.get("_test_page"):          # test harness: render a page directly by title
    flat = {t: fn for items in sections.values() for fn, t, _ in items}
    flat[ss._test_page]()
else:
    st.navigation(pages, position="sidebar", expanded=True).run()

with st.sidebar:
    st.divider()
    if st.button("Log out", icon=":material/logout:", width="stretch"):
        for k in list(ss.keys()):
            del ss[k]
        st.rerun()
    syl = syllabus.load()
    llm, store = ctx.llm(), ctx.store()
    lines = [f"Syllabus: {len(syl['units'])} units" if syl["units"] else "Syllabus: not loaded",
             f"AI model: {llm.label}",
             f"Knowledge base: {len(store.documents())} docs · {len(store.chunks)} passages",
             f"Arduino: {config.ARDUINO_IP if arduino.enabled() else 'off'}"]
    if llm.last_error:
        lines.append("⚠ " + llm.last_error)
    st.markdown('<div class="sah-side-status">' + "<br>".join(ui.esc(x) for x in lines) + "</div>",
                unsafe_allow_html=True)
    if arduino.enabled():
        @st.fragment(run_every="3s")
        def hardware_watch():
            presses = arduino.read_button_presses()
            if presses:
                last_q = next((m["content"] for m in reversed(ss.get("chat", [])) if m["role"] == "user"),
                              "(help requested from the Arduino button)")
                for _ in range(presses):
                    db.add_help_flag("arduino-button", last_q, ctx.me(),
                                     (syllabus.tag_unit(last_q) or {}).get("unit", ""))
                st.toast("Help request sent from the Arduino button.")

        hardware_watch()
