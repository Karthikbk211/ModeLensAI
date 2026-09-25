"""
Retrieval-Augmented Generation layer.

The bundled ML knowledge base is converted into fixed-size text vectors and
stored in the SQL database. A query is converted using the same vectorizer
and the highest cosine-similarity chunks are retrieved. The retrieved context
is then passed to the LLM with the measured ModelLens findings.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sqlalchemy.orm import Session

from src.database.models import KnowledgeChunk

KB_DIR = Path(__file__).resolve().parents[2] / "knowledge_base"
N_FEATURES = 1024


def _vectorize(texts: list[str]) -> np.ndarray:
    # The small, deterministic vector representation keeps deployment light
    # while still providing retrieval over the knowledge base.
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=N_FEATURES)
    matrix = vectorizer.fit_transform(texts)
    return normalize(matrix).toarray()


def _chunk_text(text: str, max_chars: int = 1200) -> list[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    parts, current = [], []
    size = 0
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            continue
        if size + len(paragraph) > max_chars and current:
            parts.append("\n".join(current))
            current, size = [], 0
        current.append(paragraph)
        size += len(paragraph)
    if current:
        parts.append("\n".join(current))
    return parts


def ingest_knowledge_base(session: Session) -> int:
    """Vectorize and persist the bundled knowledge base once."""
    if session.query(KnowledgeChunk).count() > 0:
        return 0

    texts, sources = [], []
    for path in sorted(KB_DIR.glob("*.md")):
        for chunk in _chunk_text(path.read_text(encoding="utf-8")):
            texts.append(chunk)
            sources.append(path.name)

    if not texts:
        return 0

    vectors = _vectorize(texts)
    for source, content, vector in zip(sources, texts, vectors):
        session.add(
            KnowledgeChunk(
                source=source,
                content=content,
                embedding=json.dumps(vector.tolist()),
            )
        )
    session.commit()
    return len(texts)


def retrieve(session: Session, query: str, top_k: int = 4) -> list[dict]:
    """Retrieve the most relevant knowledge chunks for a ModelLens finding."""
    rows = session.query(KnowledgeChunk).all()
    if not rows:
        return []

    # Refit on the stored corpus so query and corpus use the same vocabulary.
    texts = [row.content for row in rows]
    vectorizer = TfidfVectorizer(
        stop_words="english", ngram_range=(1, 2), max_features=N_FEATURES
    )
    corpus = normalize(vectorizer.fit_transform(texts))
    query_vec = normalize(vectorizer.transform([query]))
    scores = (corpus @ query_vec.T).toarray().ravel()

    order = np.argsort(scores)[::-1][:top_k]
    return [
        {
            "source": rows[i].source,
            "content": rows[i].content,
            "score": round(float(scores[i]), 4),
        }
        for i in order
    ]
