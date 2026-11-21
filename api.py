"""REST backend (FastAPI) exposing the same engine the Streamlit app uses.

Run:  uvicorn api:app --host 0.0.0.0 --port 8000      Docs: http://localhost:8000/docs
Useful for a separate frontend/mobile app, and the Arduino can call
POST /help to log a help request directly.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from sah import arduino, config, db, documents, quiz, summarize
from sah.llm import LLM, LLMError
from sah.rag import Assistant
from sah.vectorstore import VectorStore

app = FastAPI(title="Smart Academic Help API", version="1.0")
db.init()
store, llm = VectorStore(), LLM()
assistant = Assistant(store, llm)


class AskIn(BaseModel):
    question: str
    difficulty: str = "Intermediate"
    use_material: bool = True
    docs: list[str] | None = None


class QuizIn(BaseModel):
    kind: str = "mcq"          # "mcq" | "short"
    n: int = 5
    topic: str = ""
    difficulty: str = "Intermediate"
    docs: list[str] | None = None


class SummaryIn(BaseModel):
    text: str
    mode: str = "Short notes"
    difficulty: str = "Intermediate"


class HelpIn(BaseModel):
    question: str = "(help requested from device)"
    source: str = "arduino"


@app.get("/health")
def health():
    return {"llm": llm.label, "retriever": store.embedder.name, "documents": store.documents(),
            "chunks": len(store.chunks), "arduino": bool(arduino.ping())}


@app.post("/documents")
async def upload(file: UploadFile = File(...)):
    if Path(file.filename).suffix.lower() not in (".pdf", ".txt", ".md"):
        raise HTTPException(400, "Only PDF, TXT or MD files are supported")
    path = config.UPLOAD_DIR / Path(file.filename).name
    path.write_bytes(await file.read())
    return {"document": path.name, "new_chunks": store.add(documents.load_document(path))}


@app.get("/documents")
def list_documents():
    return store.documents()


@app.post("/ask")
def ask(body: AskIn):
    try:
        with arduino.working("Answer ready"):
            ans = assistant.ask(body.question, body.difficulty, body.use_material, body.docs)
    except LLMError as e:
        raise HTTPException(503, str(e)) from e
    db.log_chat(body.question, ans.grounded, max((h.score for h in ans.sources), default=0.0), ans.model)
    return {"answer": ans.text, "basis": ans.basis, "grounded": ans.grounded, "model": ans.model,
            "sources": [{"doc": h.chunk.doc, "page": h.chunk.page, "score": round(h.score, 3)} for h in ans.sources]}


@app.post("/quiz")
def make_quiz(body: QuizIn):
    fn = quiz.generate_mcq if body.kind == "mcq" else quiz.generate_short
    return fn(store, llm, body.n, body.topic, body.docs, body.difficulty)


@app.post("/summarize")
def make_summary(body: SummaryIn):
    if body.mode not in summarize.MODES:
        raise HTTPException(400, f"mode must be one of {list(summarize.MODES)}")
    out, method = summarize.summarize(body.text, body.mode, llm, body.difficulty)
    return {"summary": out, "method": method}


@app.post("/help")
def help_request(body: HelpIn):
    db.add_help_flag(body.source, body.question)
    return {"ok": True}


@app.get("/progress")
def progress():
    att = db.attempts()
    return {"attempts": len(att), "average": float(att.score.mean()) if not att.empty else None,
            "weak_topics": db.weak_topics().to_dict("records")}
