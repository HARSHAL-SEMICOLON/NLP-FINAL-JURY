"""Default course material, loaded automatically when the app starts.

Indexed once and remembered in data/index, so later starts are instant:
  * sample_data/23CSE4021_NLP_Syllabus.pdf   (units detected automatically)
  * sample_data/NLP_Unit1_Notes.pdf
  * every PDF/TXT/MD in a `textbooks` folder – inside the project or next to it
    (e.g. E:\\NLP JURY FINAL\\textbooks)
If the free Jurafsky & Martin draft (SLP3) is sitting in the user's Downloads folder,
it is copied into the textbooks folder first.
A document the teacher removes in the app is remembered and not re-added.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from . import config, documents, syllabus

ROOT = config.ROOT
DEFAULT_FILES = [ROOT / "sample_data" / "23CSE4021_NLP_Syllabus.pdf", ROOT / "sample_data" / "NLP_Unit1_Notes.pdf"]
TEXTBOOK_DIRS = [ROOT / "textbooks", ROOT.parent / "textbooks"]
SKIP_FILE = config.DATA_DIR / "autoload_skip.json"
SUFFIXES = {".pdf", ".txt", ".md"}
# Names Chrome may have saved the SLP3 draft under
DOWNLOAD_PATTERNS = ["Jurafsky_Martin_SLP3*.pdf", "ed3book*.pdf"]


def enabled() -> bool:
    return os.getenv("AUTOLOAD", "1") != "0"


def _skipped() -> set[str]:
    try:
        return set(json.loads(SKIP_FILE.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return set()


def skip(doc_name: str) -> None:
    """Called when the teacher removes a document, so it isn't loaded again on restart."""
    s = _skipped() | {doc_name}
    SKIP_FILE.write_text(json.dumps(sorted(s)), encoding="utf-8")


def unskip(doc_name: str) -> None:
    s = _skipped() - {doc_name}
    SKIP_FILE.write_text(json.dumps(sorted(s)), encoding="utf-8")


def textbook_dir() -> Path:
    """The textbooks folder to use: an existing one next to the project wins, else inside it."""
    for d in (ROOT.parent / "textbooks", ROOT / "textbooks"):
        if d.is_dir():
            return d
    d = ROOT / "textbooks"
    d.mkdir(exist_ok=True)
    return d


def collect_downloads() -> list[str]:
    """Copy the SLP3 PDF from ~/Downloads into the textbooks folder if it's there."""
    dl = Path.home() / "Downloads"
    if not dl.is_dir():
        return []
    dest_dir, copied = textbook_dir(), []
    have = {p.name for p in dest_dir.iterdir()} if dest_dir.exists() else set()
    for pat in DOWNLOAD_PATTERNS:
        for f in sorted(dl.glob(pat)):
            if f.name not in have and f.stat().st_size > 100_000:
                shutil.copy2(f, dest_dir / f.name)
                copied.append(f.name)
    return copied


def candidates() -> list[Path]:
    files = [f for f in DEFAULT_FILES if f.exists()]
    seen = {f.name for f in files}
    for d in TEXTBOOK_DIRS:
        if d.is_dir():
            for f in sorted(d.iterdir()):
                if f.is_file() and f.suffix.lower() in SUFFIXES and f.name.lower() != "readme.md" \
                        and f.name not in seen:
                    files.append(f)
                    seen.add(f.name)
    return files


def run(store) -> list[str]:
    """Index every default file the knowledge base doesn't have yet. Returns messages."""
    if not enabled():
        return []
    msgs = [f"Copied {n} from Downloads into the textbooks folder" for n in collect_downloads()]
    have, skipped = set(store.documents()), _skipped()
    for f in candidates():
        if f.name in have or f.name in skipped:
            continue
        dest = config.UPLOAD_DIR / f.name
        if not dest.exists():
            shutil.copy2(f, dest)          # the app keeps its own copy (summaries read from here)
        try:
            added = store.add(documents.load_document(dest))
        except Exception as exc:  # noqa: BLE001 – one bad file must not stop the app
            msgs.append(f"Couldn't read {f.name}: {exc}")
            continue
        msgs.append(f"Loaded {f.name} ({added} passages)")
        if not syllabus.load()["units"] and f.suffix.lower() == ".pdf":
            text = documents.full_text(dest)
            if syllabus.looks_like_syllabus(text):
                syllabus.save(syllabus.parse_units(text), f.name)
                msgs.append(f"Syllabus detected in {f.name}")
    return msgs
