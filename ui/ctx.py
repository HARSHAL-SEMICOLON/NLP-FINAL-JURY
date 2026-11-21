"""Shared state for all pages: engine objects, the logged-in user, syllabus units, navigation."""
from __future__ import annotations

import streamlit as st

from sah import syllabus
from sah.llm import LLM
from sah.rag import Assistant
from sah.vectorstore import VectorStore


@st.cache_resource(show_spinner="Loading the course syllabus, notes and textbooks… (first start only)")
def get_store() -> VectorStore:
    from sah import autoload

    store = VectorStore()
    STARTUP_MESSAGES.extend(autoload.run(store))   # default material, indexed once and remembered
    return store


STARTUP_MESSAGES: list[str] = []


@st.cache_resource
def get_llm() -> LLM:
    return LLM()


def store() -> VectorStore:
    return get_store()


def llm() -> LLM:
    return get_llm()


def assistant() -> Assistant:
    return Assistant(get_store(), get_llm())


def user() -> dict:
    return st.session_state.user


def me() -> str:
    return st.session_state.user["roll"]


def is_faculty() -> bool:
    return st.session_state.user["role"] == "faculty"


def units() -> list[dict]:
    us = syllabus.load()["units"]
    for i, u in enumerate(us):
        u["index"] = i
    return us


def unit_label(u: dict) -> str:
    return f"{u['unit']} – {u['title']}"


# ---------- navigation ----------
PAGES: dict[str, object] = {}  # title -> st.Page, filled by app.py


def goto(title: str) -> None:
    """Switch page (works with st.navigation and with the test harness)."""
    if st.session_state.get("_test_page"):
        st.session_state._test_page = title
        st.rerun()
    st.switch_page(PAGES[title])
