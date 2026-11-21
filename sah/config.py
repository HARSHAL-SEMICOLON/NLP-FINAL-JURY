"""Central configuration. Values come from environment variables / a .env file."""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("SAH_DATA_DIR", ROOT / "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
INDEX_DIR = DATA_DIR / "index"
DB_PATH = DATA_DIR / "smart_academic_help.db"

for _d in (DATA_DIR, UPLOAD_DIR, INDEX_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---- LLM -------------------------------------------------------------------
# "groq" (cloud, free tier), "gemini" (cloud), "ollama" (local) or "offline" (no LLM: extractive fallbacks)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
# Groq rate limits are per model, so on a 429 / retired model the next one is tried.
GROQ_FALLBACK_MODELS = ["llama-3.3-70b-versatile", "openai/gpt-oss-120b", "openai/gpt-oss-20b",
                        "llama-3.1-8b-instant"]
GROQ_URL = os.getenv("GROQ_URL", "https://api.groq.com/openai/v1/chat/completions")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
# Tried in order if the configured model name is rejected by the API.
GEMINI_FALLBACK_MODELS = ["gemini-3.5-flash", "gemini-3-flash-preview", "gemini-2.5-flash"]
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

# ---- Embeddings / retrieval -------------------------------------------------
# "auto" = Sentence Transformers if installed, else TF-IDF. Or force "sbert" / "tfidf".
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "auto").lower()
SBERT_MODEL = os.getenv("SBERT_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "120"))       # words per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "30"))  # words shared between chunks
TOP_K = int(os.getenv("TOP_K", "4"))
# Below this cosine similarity the answer is flagged as general knowledge,
# not grounded in the uploaded material (Responsible AI feature).
# TF-IDF cosine scores run much lower than transformer-embedding scores.
MIN_RELEVANCE = {
    "tfidf": float(os.getenv("MIN_RELEVANCE_TFIDF", "0.08")),
    "sbert": float(os.getenv("MIN_RELEVANCE_SBERT", "0.30")),
}

# ---- Roles -------------------------------------------------------------------
# PIN for the Faculty view (class analytics, doubts, question papers). Change it in .env.
FACULTY_PIN = os.getenv("FACULTY_PIN", "1234")
COURSE_NAME = os.getenv("COURSE_NAME", "Natural Language Processing with Large Models")
COURSE_CODE = os.getenv("COURSE_CODE", "23CSE4021")

# ---- Arduino UNO R4 WiFi ----------------------------------------------------
ARDUINO_IP = os.getenv("ARDUINO_IP", "")  # e.g. 192.168.1.50 ; empty = hardware disabled
# The board can't answer while it scrolls text (up to ~6 s), so wait long enough.
ARDUINO_TIMEOUT = float(os.getenv("ARDUINO_TIMEOUT", "7.0"))
