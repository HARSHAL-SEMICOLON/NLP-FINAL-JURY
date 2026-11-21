"""Retrieval evaluation (Unit: model evaluation).

Given test questions with the page (and document) where the answer lives,
measure how well each retriever finds it:
  Hit@k – fraction of questions whose correct page appears in the top-k results
  MRR   – mean reciprocal rank of the first correct result
"""
from __future__ import annotations

import pandas as pd

from .vectorstore import VectorStore


def evaluate(store: VectorStore, tests: pd.DataFrame, k: int = 4) -> tuple[dict, pd.DataFrame]:
    """tests columns: question, doc (optional, blank = any), page."""
    rows, rr = [], []
    for t in tests.itertuples():
        hits = store.search(str(t.question), k=k)
        rank = None
        for i, h in enumerate(hits, 1):
            doc_ok = not getattr(t, "doc", "") or str(t.doc).strip() in ("", "nan") or h.chunk.doc == t.doc
            if doc_ok and int(h.chunk.page) == int(t.page):
                rank = i
                break
        rr.append(1 / rank if rank else 0.0)
        rows.append({"question": t.question, "expected_page": t.page, "found_at_rank": rank or "—",
                     "top_result": f"{hits[0].chunk.doc} p.{hits[0].chunk.page} ({hits[0].score:.2f})" if hits else "—"})
    detail = pd.DataFrame(rows)
    n = max(len(tests), 1)
    metrics = {"retriever": store.embedder.name, f"Hit@{k}": round(sum(r > 0 for r in rr) / n, 3),
               "MRR": round(sum(rr) / n, 3), "questions": len(tests)}
    return metrics, detail
