"""
RAG pipeline — chunks docs.txt into sections, builds a TF-IDF index,
and retrieves the most relevant sections for a given query.
No external embedding API required.
"""

import os
import re
from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_DOCS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "docs.txt")


def _load_and_chunk(path: str) -> List[Tuple[str, str]]:
    with open(path) as f:
        raw = f.read()

    chunks = []
    for part in re.split(r"SECTION:\s*", raw):
        part = part.strip()
        if not part:
            continue
        lines = part.split("\n", 1)
        title = lines[0].strip()
        body = lines[1].strip() if len(lines) > 1 else ""
        # Prepend title to body so title keywords count during retrieval
        chunks.append((title, f"{title}\n{body}"))
    return chunks


_CHUNKS = _load_and_chunk(_DOCS_PATH)
_TITLES = [c[0] for c in _CHUNKS]
_TEXTS = [c[1] for c in _CHUNKS]

_VECTORIZER = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
_CHUNK_VECTORS = _VECTORIZER.fit_transform(_TEXTS)


def retrieve_docs(query: str, top_k: int = 2, min_score: float = 0.05) -> List[dict]:
    """Return up to top_k policy sections most relevant to the query."""
    if not query.strip():
        return []

    query_vec = _VECTORIZER.transform([query])
    scores = cosine_similarity(query_vec, _CHUNK_VECTORS).flatten()

    results = []
    for idx in np.argsort(scores)[::-1][:top_k]:
        score = float(scores[idx])
        if score < min_score:
            break
        body = _TEXTS[idx].split("\n", 1)
        results.append({
            "title": _TITLES[idx],
            "content": body[1].strip() if len(body) > 1 else _TEXTS[idx],
            "score": round(score, 4),
        })

    return results
