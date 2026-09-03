"""
ISB Call Bot — Knowledge Base
Loads ISB knowledge entries and provides relevance-based retrieval.
Uses TF-IDF + cosine similarity for retrieval (no external API needed).
"""

import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Optional
from pathlib import Path

from .config import KNOWLEDGE_BASE_PATH


class KnowledgeBase:
    def __init__(self, kb_path: Path = KNOWLEDGE_BASE_PATH):
        self.entries: List[Dict] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self._load(kb_path)

    def _load(self, path: Path):
        with open(path, "r", encoding="utf-8") as f:
            self.entries = json.load(f)
        docs = [f"{e['title']} {e.get('category', '')} {e['content']}" for e in self.entries]
        self.vectorizer = TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), stop_words="english", max_features=5000,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(docs)

    def search(self, query: str, top_k: int = 3, min_score: float = 0.05) -> List[Dict]:
        if not self.vectorizer or self.tfidf_matrix is None:
            return []
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in ranked_indices:
            if scores[idx] >= min_score:
                entry = self.entries[idx].copy()
                entry["score"] = float(scores[idx])
                results.append(entry)
        return results

    def get_context(self, query: str, top_k: int = 3) -> str:
        results = self.search(query, top_k=top_k)
        if not results:
            return "No specific information found. Please advise the caller to visit isb.edu or contact ISB admissions."
        context_parts = []
        for r in results:
            context_parts.append(f"[{r['title']}] ({r.get('category', '')})\n{r['content']}")
        return "\n\n---\n\n".join(context_parts)


_kb_instance: Optional[KnowledgeBase] = None


def get_knowledge_base() -> KnowledgeBase:
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = KnowledgeBase()
    return _kb_instance
