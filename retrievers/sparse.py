from typing import Any, List, Tuple

import bm25s

from .base import BaseRetriever


class BM25Retriever(BaseRetriever):
    """BM25-based sparse retriever."""

    def __init__(self, documents: List[str]):
        super().__init__(documents)
        self.bm25 = bm25s.BM25(corpus=self.documents)
        tokenized_docs = bm25s.tokenize(self.documents, show_progress=False)
        self.bm25.index(tokenized_docs, show_progress=False)

    def retrieve(
        self, query: str, top_k: int = 5, **kwargs: Any
    ) -> Tuple[List[str], List[float]]:
        """Retrieve top-k documents using BM25"""
        tokenized_query = bm25s.tokenize(query, show_progress=False)
        results, scores = self.bm25.retrieve(
            tokenized_query, k=top_k, show_progress=False
        )
        return list(results[0]), scores
