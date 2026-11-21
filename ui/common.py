"""Pages shared by students and faculty: Ask, NLP Lab, Materials, Summaries, Evaluation, Hardware."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from sah import (arduino, autoload, classical, config, db, documents, evaluation, faq, metrics, preprocess,
                 summarize, syllabus)
from sah.embeddings import TfidfEmbedder, cosine, sbert_available
from sah.llm import LLMError
from sah.rag import DIFFICULTY, LANGUAGES
from sah.vectorstore import VectorStore

from . import ctx
from . import theme as ui

ss = st.session_state
SAMPLES = {"NLP_Unit1_Notes.pdf": "sample Unit 1 notes",
           "23CSE4021_NLP_Syllabus.pdf": f"course syllabus ({config.COURSE_CODE})"}


# ---------------------------------------------------------------- answer helpers
def basis_badge(ans) -> None:
    if ans.mode == "general":
        ui.chips([ui.chip("General knowledge – not from course material", "amber")])
    elif ans.grounded:
        ui.chips([ui.chip("✓ From your course material", "sage")])
    else:
        ui.chips([ui.chip("Not in your course material – general knowledge", "red")])


def show_sources(hits) -> None:
    if not hits:
        return
    with st.expander(f"Sources ({len(hits)})", icon=":material/menu_book:"):
        for i, h in enumerate(hits, 1):
            st.markdown(f"**[{i}] {h.chunk.doc} — page {h.chunk.page}** · similarity `{h.score:.2f}`")
            st.caption(h.chunk.text[:400] + ("…" if len(h.chunk.text) > 400 else ""))


def show_tags(m: dict) -> None:
    items = []
    if m.get("unit"):
        items.append(ui.chip(m["unit"], "red" if m["unit"] == "Outside syllabus" else ""))
    if m.get("bloom"):
        items.append(ui.chip(f"Bloom: {m['bloom']}"))
    g = m["answer"].grounding if m.get("answer") else None
    if g and g["sentences"]:
        items.append(ui.chip(f"Grounding {g['score']:.0%}", "sage" if g["score"] >= .7 else "amber"))
    if items:
        ui.chips(items)
    if g and g["sentences"]:
        weak = [r for r in g["sentences"] if not r["supported"]]
        if weak:
            with st.expander(f"{len(weak)} sentence(s) not found in your notes – double-check these",
                             icon=":material/fact_check:"):
                for r in weak:
                    st.caption(f"• {r['sentence']}  _(support {r['support']:.0%})_")


def doc_picker(key: str, label: str = "Limit to documents (optional)"):
    return st.multiselect(label, ctx.store().documents(), key=key) or None


# ---------------------------------------------------------------- Ask
def ask_page() -> None:
    store, me = ctx.store(), ctx.me()
    ui.title("Ask anything", "Answers come from your course material first, with page references. "
                             "Pick how deep the explanation should go.")
    ss.setdefault("chat", [])
    ss.setdefault("fb_done", set())
    c1, c2, c3 = st.columns([2, 1.3, 1.9])
    difficulty = c1.select_slider("Explanation level", list(DIFFICULTY), value="Intermediate")
    language = c2.selectbox("Answer language", LANGUAGES, help="Hindi/Marathi = machine translation by the LLM")
    use_material = c3.toggle("Answer from course material", value=bool(store.chunks), disabled=not store.chunks,
                             help="Upload PDFs in Materials first.")
    docs = None
    if use_material and len(store.documents()) > 1:
        docs = doc_picker("ask_docs")

    if not ss.chat:
        ui.chips([ui.chip("Try: Compare stemming and lemmatization"),
                  ui.chip("Try: Explain positional encoding simply"),
                  ui.chip("Try: How does RAG reduce hallucination?")])

    for m in ss.chat:
        with st.chat_message(m["role"], avatar=":material/school:" if m["role"] == "assistant" else None):
            if m.get("faq"):
                ui.chips([ui.chip("📌 Your teacher answered a similar question", "sage")])
            st.markdown(m["content"])
            if m.get("answer"):
                basis_badge(m["answer"])
                show_tags(m)
                show_sources(m["answer"].sources)

    def run(question: str, level: str):
        tag = syllabus.tag_unit(question) or {}
        bloom = syllabus.bloom_level(question)
        ss.chat.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        hit = faq.match(question)
        if hit:
            ss.chat.append({"role": "assistant", "faq": True,
                            "content": f"**{hit['question']}**\n\n{hit['answer']}"})
            with st.chat_message("assistant", avatar=":material/school:"):
                ui.chips([ui.chip("📌 Your teacher answered a similar question", "sage")])
                st.markdown(f"**{hit['question']}**\n\n{hit['answer']}")
        with st.chat_message("assistant", avatar=":material/school:"), st.spinner("Reading your notes…"):
            try:
                with arduino.working("Answer ready"):
                    ans = ctx.assistant().ask(question, level, use_material, docs,
                                              [m for m in ss.chat[:-1] if not m.get("faq")], language)
            except LLMError as e:
                st.error(f"The AI model didn't respond: {e}. Try again, or switch LLM_PROVIDER in .env.")
                return
            msg = {"role": "assistant", "content": ans.text, "answer": ans, "unit": tag.get("unit", ""),
                   "bloom": bloom, "question": question}
            st.markdown(ans.text)
            basis_badge(ans)
            show_tags(msg)
            show_sources(ans.sources)
        top = max((h.score for h in ans.sources), default=0.0)
        db.log_chat(question, ans.grounded, top, ans.model, me, tag.get("unit", ""), bloom,
                    ans.grounding["score"] if ans.grounding else None, language)
        ss.chat.append(msg)

    if q := st.chat_input("Ask about any topic in your syllabus…"):
        run(q, difficulty)

    last = next((i for i in range(len(ss.chat) - 1, -1, -1) if ss.chat[i].get("answer")), None)
    if last is not None and not ctx.is_faculty():
        m = ss.chat[last]
        if last not in ss.fb_done:
            with st.form(f"fb_{last}", clear_on_submit=True, border=False):
                f1, f2, f3 = st.columns([1.4, 4, 1])
                rating = f1.radio("Was this helpful?", ["👍 Yes", "👎 No"], horizontal=True)
                comment = f2.text_input("Comment for your teacher (optional)")
                if f3.form_submit_button("Send rating"):
                    sent = classical.sentiment(comment or ("helpful" if rating.startswith("👍") else "not helpful"))
                    db.add_feedback(me, m["question"], 1 if rating.startswith("👍") else 0, comment,
                                    sent["label"], sent["compound"], m.get("unit", ""))
                    ss.fb_done.add(last)
                    st.toast("Rating sent to your teacher.")
                    st.rerun()
        with st.container(border=True):
            st.markdown("**Still stuck?** Explain it differently, or send this doubt to your teacher.")
            b1, b2, b3 = st.columns([1.6, 1.6, 1])
            anon = b3.checkbox("Ask anonymously", help="Your teacher sees the doubt but not your name.")
            if b1.button("Explain it more simply", icon=":material/lightbulb:"):
                run(f"I still don't understand: {m['question']}. Explain it again more simply with an analogy.",
                    "Beginner")
            if b2.button("Send doubt to teacher", icon=":material/front_hand:"):
                db.add_help_flag("app", m["question"], me, m.get("unit", ""), anon)
                st.toast("Doubt sent" + (" anonymously." if anon else "."))
    if ss.chat and st.button("Clear chat", icon=":material/delete_sweep:"):
        ss.chat, ss.fb_done = [], set()
        st.rerun()


# ---------------------------------------------------------------- Materials
def index_file(path: Path) -> str:
    with arduino.working("Indexed"):
        added = ctx.store().add(documents.load_document(path))
    msg = f"{path.name}: {added} new chunks indexed"
    text = documents.full_text(path)
    if syllabus.looks_like_syllabus(text):
        us = syllabus.parse_units(text)
        syllabus.save(us, path.name)
        msg += f" · syllabus detected ({len(us)} units)"
    return msg


def materials_page(can_delete: bool) -> None:
    store = ctx.store()
    if can_delete:
        ui.title("Materials & syllabus", "Upload the syllabus once – units are detected automatically and "
                                         "every student question is tagged with its unit.")
        us = ctx.units()
        if us:
            ui.notebook(f"{len(us)} units loaded from {syllabus.load()['source']}",
                        " ".join(ui.chip(u["unit"] + " · " + u["title"]) for u in us))
            with st.expander("Unit topics"):
                st.dataframe(pd.DataFrame(us)[["unit", "title", "hours", "topics"]], hide_index=True, width="stretch")
            if st.button("Clear syllabus"):
                syllabus.clear()
                st.rerun()
    else:
        ui.title("Course materials", "Everything the assistant answers from. You can add your own notes too.")
    if can_delete:
        st.caption(f"Loaded automatically at every start: the syllabus, the sample notes and every PDF in "
                   f"`{autoload.textbook_dir()}`. Put textbooks there once – no uploading needed.")
    for m in ctx.STARTUP_MESSAGES:
        st.toast(m)
    ctx.STARTUP_MESSAGES.clear()
    files = st.file_uploader("Upload PDF, TXT or MD", type=["pdf", "txt", "md"], accept_multiple_files=True)
    if files and st.button("Add to course material", type="primary"):
        for f in files:
            path = config.UPLOAD_DIR / Path(f.name).name
            path.write_bytes(f.getbuffer())
            with st.spinner(f"Reading {f.name}…"):
                autoload.unskip(f.name)
                st.success(index_file(path))
        st.rerun()
    pending = [(f, l) for f, l in SAMPLES.items()
               if (Path(config.ROOT) / "sample_data" / f).exists() and f not in store.documents()]
    if pending:
        cols = st.columns(len(pending))
        for col, (fname, label) in zip(cols, pending):
            if col.button(f"Load {label}", icon=":material/upload_file:"):
                dest = config.UPLOAD_DIR / fname
                dest.write_bytes((Path(config.ROOT) / "sample_data" / fname).read_bytes())
                autoload.unskip(fname)
                st.toast(index_file(dest))
                st.rerun()
    st.subheader("In the knowledge base")
    if not store.documents():
        st.caption("Nothing yet. Upload a PDF or load the samples above.")
    for d in store.documents():
        chunks = [c for c in store.chunks if c.doc == d]
        a, b = st.columns([6, 1])
        with a.expander(f"{d} — {len(chunks)} passages, {len({c.page for c in chunks})} pages",
                        icon=":material/description:"):
            for c in chunks[:30]:
                st.caption(f"p.{c.page} · {c.text[:220]}…")
        if can_delete and b.button("Remove", key=f"del_{d}"):
            store.remove_document(d)
            autoload.skip(d)              # don't re-add it on the next start
            (config.UPLOAD_DIR / d).unlink(missing_ok=True)
            if syllabus.load().get("source") == d:
                syllabus.clear()
            st.rerun()


# ---------------------------------------------------------------- Summaries
def summaries_page() -> None:
    store, llm = ctx.store(), ctx.llm()
    fac = ctx.is_faculty()
    ui.title("Teaching notes" if fac else "Summaries",
             "Turn long material into lesson-ready notes." if fac else
             "Turn long notes into short notes, key points, definitions or an exam summary.")
    us = ctx.units()
    src = st.radio("Source", ["A syllabus unit", "A document", "Pasted text"], horizontal=True)
    text = ""
    if src == "A document":
        docs = store.documents()
        if not docs:
            st.info("No material yet – add some in Materials.")
            return
        d = st.selectbox("Document", docs)
        pages = sorted({c.page for c in store.chunks if c.doc == d})
        p1, p2 = st.select_slider("Pages", options=pages, value=(pages[0], pages[-1])) if len(pages) > 1 else (1, 1)
        path = config.UPLOAD_DIR / d
        if not path.exists():
            path = Path("sample_data") / d
        text = "\n".join(t for p, t in documents.extract_pages(path) if p1 <= p <= p2)
    elif src == "A syllabus unit":
        if not us:
            st.info("Load the course syllabus first (Materials).")
            return
        u = next(x for x in us if ctx.unit_label(x) == st.selectbox("Unit", [ctx.unit_label(x) for x in us]))
        hits = store.search(u["title"] + " " + u["topics"], k=10)
        text = "\n".join(h.chunk.text for h in hits)
        st.caption(f"Uses the {len(hits)} passages from your material closest to {u['unit']}'s topics.")
    else:
        text = st.text_area("Paste notes", height=220)
    modes = list(summarize.MODES)
    mode = st.selectbox("Output", modes, index=modes.index("Teaching notes") if fac else 0)
    level = st.select_slider("Level", list(DIFFICULTY), value="Intermediate")
    if st.button("Summarise", type="primary", icon=":material/auto_awesome:") and text.strip():
        with st.spinner("Summarising…"), arduino.working("Summary ready"):
            out, method = summarize.summarize(text, mode, llm, level)
        st.caption(f"{method} · {len(text.split())} words → {len(out.split())} words")
        with st.container(border=True):
            st.markdown(out)
        r = metrics.rouge_all(text, out)
        with st.expander("Quality check – ROUGE against the source", icon=":material/rule:"):
            st.dataframe(pd.DataFrame(r).T, width="stretch")
            st.caption("Precision = how much of the summary comes from the source (faithfulness). "
                       "Recall = how much of the source it covers (low by design).")
        st.download_button("Download as Markdown", out, file_name=f"{mode.lower().replace(' ', '_')}.md",
                           icon=":material/download:")


# ---------------------------------------------------------------- NLP Lab
def nlp_lab_page() -> None:
    store, llm = ctx.store(), ctx.llm()
    ui.title("NLP Lab", "Every syllabus concept, running live on real text.")
    tabs = st.tabs(["Preprocessing", "NER & parsing", "Sentiment", "TF-IDF", "Topics (LDA)", "Similarity",
                    "Language model", "Translation + BLEU", "Syllabus classifier", "Retrieval"])
    with tabs[0]:
        st.caption("Unit II – cleaning, tokenization, stop words, POS tagging, stemming vs lemmatization")
        txt = st.text_area("Text", "Students are studying the transformers' attention mechanisms and were "
                                   "running better experiments on tokenization.")
        t = preprocess.trace(txt)
        st.caption(t.backend)
        st.markdown(f"**Sentences** {len(t.sentences)} · **Tokens** {len(t.tokens)} · "
                    f"**After stop-word removal** {len(t.without_stopwords)}")
        st.code(" | ".join(t.tokens), language=None)
        st.dataframe(pd.DataFrame({"token": t.without_stopwords, "POS tag": t.pos_tags,
                                   "stem (Porter)": t.stems, "lemma (WordNet + POS)": t.lemmas}), hide_index=True,
                     width="stretch")
        st.caption("The lemmatizer needs the POS tag: 'running' (VBG) → 'run', 'better' (JJR) → 'good'.")
    with tabs[1]:
        st.caption("Unit II – Named Entity Recognition and dependency parsing")
        txt = st.text_input("Sentence", "Google released Gemini in California while Meta trained Llama in Paris.")
        ents = classical.entities(txt)
        if ents:
            ui.chips([ui.chip(f"{e['entity']} · {e['type']}") for e in ents])
        else:
            st.caption("No entities found.")
        deps = classical.dependencies(txt)
        if deps:
            st.dataframe(pd.DataFrame(deps), hide_index=True)
        else:
            st.caption("Dependency parsing is optional: `pip install spacy` then "
                       "`python -m spacy download en_core_web_sm`.")
    with tabs[2]:
        st.caption("Unit II – sentiment analysis (VADER). The same model scores student feedback for faculty.")
        txt = st.text_area("One comment per line", "The explanation of attention was really clear, thanks!\n"
                                                   "Still confused about positional encoding, the answer was too long.")
        rows = [{"comment": s, **classical.sentiment(s)} for s in txt.splitlines() if s.strip()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    with tabs[3]:
        if not store.chunks:
            st.caption("Add material to compute TF-IDF over your corpus.")
        else:
            d = st.selectbox("Document", store.documents(), key="tfidf_doc")
            tf = TfidfEmbedder()
            tf.fit([c.text for c in store.chunks])
            terms = pd.Series(dict(tf.top_terms(" ".join(c.text for c in store.chunks if c.doc == d), 15)))
            ui.bars(terms, "term", "TF-IDF weight")
            st.caption("High weight = frequent in this document, rare across the whole corpus.")
    with tabs[4]:
        st.caption("Unit II – topic modelling with Latent Dirichlet Allocation over your material")
        if len(store.chunks) < 4:
            st.caption("Add more material first.")
        else:
            k = st.slider("Number of topics", 2, 8, 4)
            words, dom = classical.topics([c.text for c in store.chunks], k)
            for i, w in enumerate(words):
                st.markdown(f"**Topic {i + 1}** · {dom.count(i)} passages")
                ui.chips([ui.chip(x) for x in w])
    with tabs[5]:
        a = st.text_input("Sentence A", "How does a transformer use self-attention?")
        b = st.text_input("Sentence B", "Explain the attention mechanism in transformer models.")
        c = st.text_input("Sentence C", "What is the syllabus for the chemistry practical?")
        rows = [("TF-IDF (lexical)", cosine(a, b, TfidfEmbedder()), cosine(a, c, TfidfEmbedder()))]
        if store.embedder.dense:
            rows.append(("Sentence-BERT (semantic)", cosine(a, b, store.embedder), cosine(a, c, store.embedder)))
        st.dataframe(pd.DataFrame(rows, columns=["method", "sim(A,B)", "sim(A,C)"]).round(3), hide_index=True)
        if not sbert_available():
            st.caption("Install sentence-transformers to compare against transformer embeddings.")
    with tabs[6]:
        st.caption("Unit I/IV – bigram language model with Laplace smoothing, trained on your notes. "
                   "Perplexity = exp(−1/N Σ log P(wᵢ|wᵢ₋₁)); lower = more predictable.")
        if not store.chunks:
            st.caption("Add material first.")
        else:
            lm = metrics.BigramLM([c.text for c in store.chunks])
            s1 = st.text_input("In-domain sentence", "Tokenization is the process of splitting text into tokens.")
            s2 = st.text_input("Out-of-domain sentence", "Bananas fly purple over quantum cheese.")
            r1, r2 = lm.perplexity(s1), lm.perplexity(s2)
            ui.stats([("Perplexity – in-domain", r1["perplexity"], "good"),
                      ("Perplexity – out-of-domain", r2["perplexity"], "warn")])
            st.dataframe(pd.DataFrame(r1["steps"], columns=["previous", "word", "P(word|previous)"]).round(5),
                         hide_index=True)
    with tabs[7]:
        st.caption("Unit III – machine translation with an LLM, evaluated by back-translation BLEU/ROUGE")
        src = st.text_area("English sentence", "Retrieval augmented generation reduces hallucination by adding "
                                               "relevant passages from trusted documents to the prompt.")
        target = st.selectbox("Translate to", ["Hindi", "Marathi", "German", "French"])
        if st.button("Translate and evaluate"):
            if not llm.available:
                st.warning("Translation needs an LLM – add GROQ_API_KEY to .env.")
            else:
                with st.spinner("Translating…"):
                    tr = llm.generate(f"Translate into {target}. Return only the translation.\n\n{src}", temperature=0)
                    back = llm.generate(f"Translate into English. Return only the translation.\n\n{tr}", temperature=0)
                st.markdown(f"**{target}:** {tr}")
                st.markdown(f"**Back to English:** {back}")
                b = metrics.bleu(src, back)
                r = metrics.rouge_all(src, back)
                st.dataframe(pd.DataFrame([{"metric": "BLEU", "score": b["bleu"]}] +
                                          [{"metric": k, "score": v["f1"]} for k, v in r.items()]), hide_index=True)
    with tabs[8]:
        st.caption("Text classification: which syllabus unit and which Bloom's level does a question belong to?")
        q = st.text_input("Question", "Compare BERT and GPT pretraining objectives")
        tag = syllabus.tag_unit(q)
        ui.stats([("Syllabus unit", tag["unit"] if tag else "no syllabus"), ("Bloom's level", syllabus.bloom_level(q)),
                  ("Similarity to unit", f"{tag['score']:.2f}" if tag else "—")])
    with tabs[9]:
        q = st.text_input("Query", "difference between stemming and lemmatization")
        if q and store.chunks:
            thr = config.MIN_RELEVANCE[store.embedder.name]
            st.caption(f"Relevance threshold for '{store.embedder.name}': {thr}")
            for h in store.search(q, k=6):
                st.markdown(f"{'🟢' if h.score >= thr else '⚪'} `{h.score:.3f}` **{h.chunk.doc} p.{h.chunk.page}** — "
                            f"{h.chunk.text[:200]}…")


# ---------------------------------------------------------------- Evaluation & hardware
def evaluation_page() -> None:
    store = ctx.store()
    ui.title("Evaluation", "Measure the retriever and the generated text – numbers for your report.")
    t1, t2 = st.tabs(["Retrieval: Hit@k and MRR", "Generation: BLEU and ROUGE"])
    with t1:
        default = pd.DataFrame({
            "question": ["What is tokenization?", "How is IDF calculated?",
                         "What causes hallucination and how does RAG help?",
                         "Difference between stemming and lemmatization", "Main challenge in NLP"],
            "doc": ["NLP_Unit1_Notes.pdf"] * 5, "page": [2, 3, 4, 2, 1]})
        tests = st.data_editor(ss.get("eval_tests", default), num_rows="dynamic", width="stretch")
        ss.eval_tests = tests
        k = st.slider("k", 1, 10, 4)
        if st.button("Run evaluation", type="primary"):
            backends = ["tfidf"] + (["sbert"] if sbert_available() else [])
            results = []
            for b in backends:
                vs = store if store.embedder.name == b else VectorStore(config.INDEX_DIR, backend=b)
                m, detail = evaluation.evaluate(vs, tests.dropna(subset=["question", "page"]), k)
                results.append(m)
                with st.expander(f"Details – {b}"):
                    st.dataframe(detail, hide_index=True)
            st.dataframe(pd.DataFrame(results), hide_index=True)
    with t2:
        ref = st.text_area("Reference (e.g. textbook definition)",
                           "Tokenization is the process of splitting text into smaller units called tokens.")
        cand = st.text_area("Candidate (e.g. model or student answer)",
                            "Tokenization splits the text into small units known as tokens.")
        b = metrics.bleu(ref, cand)
        r = metrics.rouge_all(ref, cand)
        st.dataframe(pd.DataFrame([{"metric": "BLEU-4", "precision": None, "recall": None, "f1/score": b["bleu"]}] +
                                  [{"metric": k, "precision": v["precision"], "recall": v["recall"],
                                    "f1/score": v["f1"]} for k, v in r.items()]), hide_index=True)
        st.caption(f"BLEU n-gram precisions {b['precisions']} · brevity penalty {b['brevity_penalty']}. "
                   "BLEU is precision-oriented (translation); ROUGE is recall-oriented (summarisation).")


def hardware_page() -> None:
    ui.title("Arduino status display", "The UNO R4 WiFi shows what the assistant is doing on its LED matrix.")
    if not arduino.enabled():
        st.info("Set ARDUINO_IP in .env to the IP shown on the Serial Monitor after the board joins Wi-Fi, "
                "then restart the app.")
        return
    info = arduino.ping()
    ui.stats([("Board", "online" if info else "offline", "good" if info else "warn"),
              ("IP address", config.ARDUINO_IP)])
    st.caption("Each button shows its icon instantly. Scrolling text takes a few seconds, and the board "
               "finishes scrolling before it takes the next command.")
    icons = {"ready": "🙂", "processing": "⏳", "success": "✔", "error": "✖"}
    cols = st.columns(4)
    for col, state in zip(cols, arduino.STATES):
        if col.button(f"{icons[state]} {state.title()}", width="stretch"):
            arduino.reset_backoff()
            st.toast(f"Board shows {icons[state]}" if arduino.set_status(state)   # icon only – no scrolling
                     else "The board didn't answer – is it on the same Wi-Fi?")
    msg = st.text_input("Scroll a message on the LED matrix", "Exam in 3 days")
    if st.button("Send message"):
        arduino.reset_backoff()
        st.toast("Scrolling on the board." if arduino.set_status("ready", msg)
                 else "The board didn't answer – is it on the same Wi-Fi?")
