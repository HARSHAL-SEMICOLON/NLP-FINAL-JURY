"""Visual identity: "the annotated notebook".

Ink-blue type on clean paper, a highlighter-yellow sweep behind each page title (the one
memorable flourish), a red notebook-margin rule on things with a deadline, and Atkinson
Hyperlegible for body text – a typeface designed for readability.

Tokens
  ink        #22357A  primary, headings, sidebar cover
  graphite   #23262F  body text           muted  #5D6372
  paper      #F8F9FC  background          rule   #DCE3F0  (notebook ruling)
  highlight  #FFE45C  title sweep         margin #D9443F  (deadline rule)
  sage       #3E8E6E  calm / well-being / success
"""
from __future__ import annotations

import html

import altair as alt
import pandas as pd
import streamlit as st

INK, GRAPHITE, MUTED, PAPER, RULE = "#22357A", "#23262F", "#5D6372", "#F8F9FC", "#DCE3F0"
HIGHLIGHT, MARGIN, SAGE, AMBER = "#FFE45C", "#D9443F", "#3E8E6E", "#C9861A"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:ital,wght@0,400;0,700;1,400&family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&display=swap');

html, body, [class*="st-"], .stMarkdown, button, input, textarea, select {{
  font-family: 'Atkinson Hyperlegible', system-ui, sans-serif;
}}
/* keep Streamlit's icon font – the rule above must not touch Material Symbols */
[data-testid="stIconMaterial"], span[data-testid="stIconMaterial"], .material-symbols-rounded {{
  font-family: 'Material Symbols Rounded' !important; font-weight: normal; font-style: normal;
}}
h1, h2, h3, h4, .sah-display {{
  font-family: 'Bricolage Grotesque', 'Atkinson Hyperlegible', sans-serif !important;
  color: {INK}; letter-spacing: -0.01em;
}}
.block-container {{ padding-top: 2.2rem; max-width: 1180px; }}

/* ---------- sidebar: the notebook cover ---------- */
[data-testid="stSidebar"] {{ background: {INK}; }}
[data-testid="stSidebar"] * {{ color: #E9EDF8; }}
[data-testid="stSidebar"] [data-testid="stSidebarNavSeparator"] {{ border-color: rgba(255,255,255,.18); }}
[data-testid="stSidebarNav"] a {{ border-radius: 8px; }}
[data-testid="stSidebarNav"] a:hover {{ background: rgba(255,255,255,.08); }}
[data-testid="stSidebarNav"] a[aria-current="page"] {{ background: rgba(255,228,92,.18); }}
[data-testid="stSidebarNav"] a[aria-current="page"] span {{ color: {HIGHLIGHT} !important; font-weight: 700; }}
[data-testid="stSidebarNav"] header, [data-testid="stSidebarNavItems"] header span {{
  color: rgba(233,237,248,.6) !important; }}
[data-testid="stSidebar"] .stButton button {{
  background: transparent; border: 1.5px solid rgba(255,255,255,.35); color: #fff;
}}
[data-testid="stSidebar"] .stButton button:hover {{ border-color: {HIGHLIGHT}; color: {HIGHLIGHT}; }}
.sah-side-brand {{ font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800; font-size: 1.35rem;
  color: #fff; line-height: 1.1; margin: .2rem 0 .1rem; }}
.sah-side-user {{ font-size: .9rem; color: rgba(233,237,248,.8); margin-bottom: .6rem; }}
.sah-side-status {{ font-size: .78rem; color: rgba(233,237,248,.7); line-height: 1.6; }}

/* ---------- page title: the highlighter sweep ---------- */
.sah-title {{ margin: 0 0 1.4rem; }}
.sah-title h1 {{ font-size: clamp(1.9rem, 3.2vw, 2.6rem); font-weight: 800; margin: 0; padding: 0;
  display: inline; line-height: 1.25;
  background-image: linear-gradient(100deg, rgba(255,228,92,0) 0%, rgba(255,228,92,.85) 2%,
                                    rgba(255,228,92,.75) 96%, rgba(255,228,92,0) 100%);
  background-repeat: no-repeat; background-size: 100% 42%; background-position: 0 88%;
  animation: sah-sweep .7s cubic-bezier(.2,.7,.2,1) both; }}
@keyframes sah-sweep {{ from {{ background-size: 0% 42%; }} to {{ background-size: 100% 42%; }} }}
@media (prefers-reduced-motion: reduce) {{ .sah-title h1 {{ animation: none; }} }}
.sah-title p {{ color: {MUTED}; font-size: 1.05rem; margin: .55rem 0 0; max-width: 68ch; }}

/* ---------- notebook page (hero) ---------- */
.sah-page {{ position: relative; background: #fff; border: 1.5px solid {RULE}; border-radius: 14px;
  padding: 26px 1.6rem 14px 3.1rem; margin-bottom: 1.2rem;
  background-image: repeating-linear-gradient(to bottom, transparent 0 31px, {RULE} 31px 32px);
  background-position: 0 26px; background-repeat: repeat-y; }}
.sah-page::before {{ content: ""; position: absolute; left: 2.1rem; top: 0; bottom: 0; width: 2px;
  background: {MARGIN}; opacity: .55; }}
.sah-page h2 {{ font-size: 1.45rem; margin: 0 !important; padding: 0 !important; line-height: 32px !important; }}
.sah-page p {{ margin: 0; color: {GRAPHITE}; line-height: 32px; font-size: 1.02rem; }}

/* ---------- stats ---------- */
.sah-stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(135px, 1fr)); gap: .8rem;
  margin: .3rem 0 1.2rem; }}
.sah-stat {{ background: #fff; border: 1.5px solid {RULE}; border-radius: 12px; padding: .85rem 1rem; }}
.sah-stat .v {{ font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800; font-size: 1.9rem;
  color: {INK}; line-height: 1.1; }}
.sah-stat .l {{ color: {MUTED}; font-size: .88rem; margin-top: .15rem; }}
.sah-stat.warn .v {{ color: {MARGIN}; }} .sah-stat.good .v {{ color: {SAGE}; }}

/* ---------- tasks & deadlines: margin rule ---------- */
.sah-task {{ background: #fff; border: 1.5px solid {RULE}; border-left: 4px solid {MARGIN};
  border-radius: 6px 12px 12px 6px; padding: .65rem .9rem; margin-bottom: .55rem; }}
.sah-task.calm {{ border-left-color: {SAGE}; }} .sah-task.ink {{ border-left-color: {INK}; }}
.sah-task .t {{ font-weight: 700; color: {GRAPHITE}; }}
.sah-task .m {{ color: {MUTED}; font-size: .88rem; }}
.sah-task .due {{ float: right; font-weight: 700; color: {MARGIN}; font-size: .9rem; }}
.sah-task.calm .due {{ color: {SAGE}; }} .sah-task.ink .due {{ color: {INK}; }}

/* ---------- chips & badges ---------- */
.sah-chip {{ display: inline-block; padding: .12rem .6rem; border-radius: 999px; font-size: .8rem;
  font-weight: 700; margin: 0 .3rem .3rem 0; background: #EDF1F8; color: {INK}; }}
.sah-chip.sage {{ background: #E3F2EC; color: {SAGE}; }} .sah-chip.red {{ background: #FBE7E6; color: {MARGIN}; }}
.sah-chip.amber {{ background: #FBF1DC; color: #8A5A0B; }}
.sah-badges {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: .6rem; }}
.sah-badge {{ border: 1.5px dashed {RULE}; border-radius: 12px; padding: .7rem; text-align: center;
  color: {MUTED}; background: transparent; }}
.sah-badge .e {{ font-size: 1.6rem; filter: grayscale(1); opacity: .35; }}
.sah-badge.got {{ border: 1.5px solid {HIGHLIGHT}; background: #FFFBE0; color: {GRAPHITE}; }}
.sah-badge.got .e {{ filter: none; opacity: 1; }}
.sah-badge .n {{ font-weight: 700; font-size: .9rem; }} .sah-badge .d {{ font-size: .75rem; }}

/* ---------- readiness bars ---------- */
.sah-meter {{ margin: .35rem 0 .7rem; }}
.sah-meter .row {{ display: flex; justify-content: space-between; font-size: .9rem; }}
.sah-meter .bar {{ height: 10px; background: #EDF1F8; border-radius: 999px; overflow: hidden; margin-top: .25rem; }}
.sah-meter .fill {{ height: 100%; border-radius: 999px; background: {INK}; }}
.sah-meter .fill.low {{ background: {MARGIN}; }} .sah-meter .fill.mid {{ background: {AMBER}; }}
.sah-meter .fill.high {{ background: {SAGE}; }}

/* ---------- support card (well-being) ---------- */
.sah-support {{ background: #F1F8F4; border: 1.5px solid #CFE6DA; border-radius: 14px; padding: 1rem 1.2rem;
  margin: .6rem 0 1rem; }}
.sah-support h4 {{ color: {SAGE} !important; margin: 0 0 .4rem; }}
.sah-urgent {{ background: #FDF0EF; border: 2px solid {MARGIN}; border-radius: 14px; padding: 1rem 1.2rem;
  margin: .6rem 0 1rem; }}
.sah-urgent h4 {{ color: {MARGIN} !important; margin: 0 0 .4rem; }}

/* ---------- login ---------- */
.sah-login-mark {{ font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800;
  font-size: clamp(2.4rem, 6vw, 4.2rem); color: {INK}; line-height: 1.02; margin: .4rem 0 .6rem; }}

/* ---------- streamlit details ---------- */
[data-testid="stChatMessage"] {{ background: #fff; border: 1.5px solid {RULE}; border-radius: 14px; }}
[data-testid="stExpander"] details {{ border: 1.5px solid {RULE}; border-radius: 12px; background: #fff; }}
[data-testid="stMetric"] {{ background: #fff; border: 1.5px solid {RULE}; border-radius: 12px; padding: .7rem 1rem; }}
.stButton button[kind="primary"], .stFormSubmitButton button[kind="primary"] {{ font-weight: 700; }}
div[data-testid="stTabs"] button[aria-selected="true"] p {{ color: {INK}; font-weight: 700; }}
:focus-visible {{ outline: 3px solid {HIGHLIGHT} !important; outline-offset: 2px; }}
</style>
"""


def esc(x) -> str:
    return html.escape(str(x if x is not None else ""))


def inject() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def title(text: str, subtitle: str = "") -> None:
    sub = f"<p>{esc(subtitle)}</p>" if subtitle else ""
    st.markdown(f'<div class="sah-title"><h1>{esc(text)}</h1>{sub}</div>', unsafe_allow_html=True)


def notebook(heading: str, body_html: str) -> None:
    """Ruled 'notebook page' card – used once per page for the most important message."""
    st.markdown(f'<div class="sah-page"><h2>{esc(heading)}</h2><p>{body_html}</p></div>', unsafe_allow_html=True)


def stats(items: list[tuple]) -> None:
    """items: (label, value[, tone]) with tone in {'', 'warn', 'good'}"""
    cells = "".join(f'<div class="sah-stat {it[2] if len(it) > 2 else ""}"><div class="v">{esc(it[1])}</div>'
                    f'<div class="l">{esc(it[0])}</div></div>' for it in items)
    st.markdown(f'<div class="sah-stats">{cells}</div>', unsafe_allow_html=True)


def task(title_: str, meta: str = "", due: str = "", tone: str = "") -> str:
    d = f'<span class="due">{esc(due)}</span>' if due else ""
    m = f'<div class="m">{esc(meta)}</div>' if meta else ""
    return f'<div class="sah-task {tone}">{d}<div class="t">{esc(title_)}</div>{m}</div>'


def tasks(items: list[str]) -> None:
    st.markdown("".join(items), unsafe_allow_html=True)


def chip(text: str, tone: str = "") -> str:
    return f'<span class="sah-chip {tone}">{esc(text)}</span>'


def chips(items: list[str]) -> None:
    st.markdown("".join(items), unsafe_allow_html=True)


def meter(label: str, pct: float, note: str = "") -> str:
    pct = max(0.0, min(1.0, pct or 0.0))
    tone = "low" if pct < .4 else "mid" if pct < .7 else "high"
    return (f'<div class="sah-meter"><div class="row"><span>{esc(label)}</span><span>{pct:.0%}'
            f'{" · " + esc(note) if note else ""}</span></div><div class="bar"><div class="fill {tone}" '
            f'style="width:{pct * 100:.0f}%"></div></div></div>')


def badges(all_badges: list[tuple], got: set[str]) -> None:
    cells = "".join(f'<div class="sah-badge {"got" if key in got else ""}"><div class="e">{e}</div>'
                    f'<div class="n">{esc(n)}</div><div class="d">{esc(d)}</div></div>'
                    for key, e, n, d in all_badges)
    st.markdown(f'<div class="sah-badges">{cells}</div>', unsafe_allow_html=True)


def support(heading: str, body_html: str, urgent: bool = False) -> None:
    cls = "sah-urgent" if urgent else "sah-support"
    st.markdown(f'<div class="{cls}"><h4>{esc(heading)}</h4>{body_html}</div>', unsafe_allow_html=True)


def bars(data: pd.Series | pd.DataFrame, x_title: str = "", y_title: str = "", pct: bool = False,
         horizontal: bool = True, height: int | None = None) -> None:
    """Altair bar chart in the notebook palette; the weakest bar is marked in margin red when pct."""
    df = data.rename("value").reset_index() if isinstance(data, pd.Series) else data.copy()
    df.columns = ["label", "value"]
    df = df.dropna()
    if df.empty:
        return
    color = alt.value(INK)
    if pct and len(df) > 1:
        lowest = df.value.min()
        color = alt.condition(alt.datum.value == lowest, alt.value(MARGIN), alt.value(INK))
    fmt = ".0%" if pct else ",.0f"
    axis = alt.Axis(format=fmt, grid=True, gridColor=RULE, tickMinStep=0.1 if pct else 1)
    enc_val = alt.X("value:Q", title=y_title, axis=axis) if horizontal else alt.Y("value:Q", title=y_title, axis=axis)
    enc_lab = alt.Y("label:N", title=x_title, sort=None) if horizontal else alt.X("label:N", title=x_title, sort=None)
    chart = (alt.Chart(df).mark_bar(cornerRadiusEnd=4, size=18)
             .encode(enc_val, enc_lab, color=color,
                     tooltip=[alt.Tooltip("label:N", title=x_title or "item"),
                              alt.Tooltip("value:Q", title=y_title or "value", format=fmt)])
             .properties(height=height or max(120, 34 * len(df)))
             .configure_view(strokeWidth=0)
             .configure_axis(labelFont="Atkinson Hyperlegible", titleFont="Atkinson Hyperlegible",
                             labelColor=MUTED, titleColor=MUTED, domain=False, ticks=False))
    st.altair_chart(chart, width="stretch")


def line(df: pd.DataFrame, x: str, y: str, y_title: str = "", pct: bool = True) -> None:
    if df.empty:
        return
    fmt = ".0%" if pct else ",.0f"
    chart = (alt.Chart(df).mark_line(color=INK, strokeWidth=2.5, point=alt.OverlayMarkDef(color=INK, size=50))
             .encode(alt.X(f"{x}:T", title=""), alt.Y(f"{y}:Q", title=y_title,
                                                      axis=alt.Axis(format=fmt, gridColor=RULE)),
                     tooltip=[alt.Tooltip(f"{x}:T"), alt.Tooltip(f"{y}:Q", format=fmt)])
             .properties(height=220).configure_view(strokeWidth=0)
             .configure_axis(labelFont="Atkinson Hyperlegible", labelColor=MUTED, titleColor=MUTED, domain=False))
    st.altair_chart(chart, width="stretch")
