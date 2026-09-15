"""
RAG Engine — Retrieval-Augmented Generation for Lernix.

Builds a lightweight TF-IDF vector store from curriculum content stored in
the database. When a student asks a question, the engine retrieves the most
relevant course chunks and prepends them to the LLM prompt so the model
answers from actual curriculum data rather than generic knowledge.

No external vector-DB dependency — uses numpy cosine similarity.
"""

import json
import math
import re
from collections import Counter

import database


# ---------------------------------------------------------------------------
# Tokenisation & TF-IDF helpers
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _tf(tokens: list[str]) -> dict[str, float]:
    counts = Counter(tokens)
    total = len(tokens) or 1
    return {t: c / total for t, c in counts.items()}


def _idf(corpus_tokens: list[list[str]]) -> dict[str, float]:
    N = len(corpus_tokens)
    df: dict[str, int] = {}
    for tokens in corpus_tokens:
        for t in set(tokens):
            df[t] = df.get(t, 0) + 1
    return {t: math.log((N + 1) / (d + 1)) + 1 for t, d in df.items()}


def _tfidf_vec(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    tf = _tf(tokens)
    return {t: tf[t] * idf.get(t, 1.0) for t in tf}


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    common = set(a) & set(b)
    if not common:
        return 0.0
    dot = sum(a[t] * b[t] for t in common)
    mag_a = math.sqrt(sum(v * v for v in a.values()))
    mag_b = math.sqrt(sum(v * v for v in b.values()))
    return dot / (mag_a * mag_b) if mag_a and mag_b else 0.0


# ---------------------------------------------------------------------------
# VectorStore — in-memory, rebuilt per request (fast enough for SQLite scale)
# ---------------------------------------------------------------------------

class VectorStore:
    """Holds chunked curriculum documents with TF-IDF vectors."""

    def __init__(self):
        self.chunks: list[dict] = []   # {"text": str, "meta": dict}
        self.vectors: list[dict[str, float]] = []
        self._idf: dict[str, float] = {}

    def add_documents(self, docs: list[dict]):
        """docs: list of {"text": str, "meta": dict}"""
        corpus_tokens = [_tokenise(d["text"]) for d in docs]
        self._idf = _idf(corpus_tokens)
        self.chunks = docs
        self.vectors = [_tfidf_vec(t, self._idf) for t in corpus_tokens]

    def query(self, question: str, top_k: int = 3) -> list[dict]:
        """Return top_k most relevant chunks for the question."""
        if not self.chunks:
            return []
        q_tokens = _tokenise(question)
        q_vec = _tfidf_vec(q_tokens, self._idf)
        scores = [(_cosine(q_vec, v), i) for i, v in enumerate(self.vectors)]
        scores.sort(reverse=True)
        return [
            {**self.chunks[i], "score": round(s, 4)}
            for s, i in scores[:top_k]
            if s > 0
        ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_store_from_curriculum(curriculum_id: int) -> VectorStore:
    """
    Load a curriculum from the DB and chunk it into course-level documents.
    Each chunk = one course's full text (name + description + outcomes + skills).
    """
    store = VectorStore()
    curr = database.get_curriculum(curriculum_id)
    if not curr:
        return store

    data = json.loads(curr["raw_json"])
    docs = []
    for sem in data.get("semesters", []):
        for course in sem.get("courses", []):
            text_parts = [
                course.get("name", ""),
                course.get("description", ""),
                " ".join(course.get("learning_outcomes", [])),
                " ".join(course.get("key_skills", [])),
                " ".join(course.get("career_opportunities", [])),
                " ".join(course.get("recommended_projects", [])),
            ]
            docs.append({
                "text": " ".join(filter(None, text_parts)),
                "meta": {
                    "curriculum_title": data.get("title", ""),
                    "semester": sem.get("semester_number"),
                    "course_code": course.get("code", ""),
                    "course_name": course.get("name", ""),
                    "description": course.get("description", ""),
                    "outcomes": course.get("learning_outcomes", []),
                    "skills": course.get("key_skills", []),
                },
            })
    store.add_documents(docs)
    return store


def retrieve_context(question: str, curriculum_id: int, top_k: int = 3) -> str:
    """
    Returns a formatted context string of the top-k relevant course chunks
    to be injected into an LLM prompt.
    """
    store = build_store_from_curriculum(curriculum_id)
    results = store.query(question, top_k=top_k)
    if not results:
        return ""

    lines = ["Relevant curriculum context:"]
    for r in results:
        m = r["meta"]
        lines.append(
            f"- [{m['course_code']}] {m['course_name']} (Semester {m['semester']}): "
            f"{m['description']} | Skills: {', '.join(m['skills'])}"
        )
    return "\n".join(lines)


def retrieve_context_from_store(question: str, store: VectorStore, top_k: int = 3) -> str:
    """Same as retrieve_context but accepts a pre-built store (avoids DB re-read)."""
    results = store.query(question, top_k=top_k)
    if not results:
        return ""
    lines = ["Relevant curriculum context:"]
    for r in results:
        m = r["meta"]
        lines.append(
            f"- [{m['course_code']}] {m['course_name']} (Semester {m['semester']}): "
            f"{m['description']} | Skills: {', '.join(m['skills'])}"
        )
    return "\n".join(lines)
