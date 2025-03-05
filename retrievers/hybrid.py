from typing import List, Tuple

import numpy as np

from .base import BaseRetriever
from .dense import DenseRetriever
from .sparse import BM25Retriever


class HybridRetriever(BaseRetriever):
    """Combines BM25 + FAISS."""

    def __init__(self, documents: List[str], alpha: float = 0.5):
        super().__init__(documents)
        self.sparse_retriever = BM25Retriever(documents)
        self.dense_retriever = DenseRetriever(documents)
        self.alpha = alpha  # Balances BM25 & Dense scores

    def retrieve(self, query: str, top_k: int = 5) -> Tuple[List[str], List[float]]:
        """Retrieve top-k documents using hybrid retrieval."""
        sparse_docs, sparse_scores = self.sparse_retriever.retrieve(query, top_k=top_k)
        dense_docs, dense_scores = self.dense_retriever.retrieve(query, top_k=top_k)

        doc_scores = {}

        sparse_scores = np.array(sparse_scores)
        dense_scores = np.array(dense_scores)

        if sparse_scores.max() > 0:
            sparse_scores = sparse_scores / sparse_scores.max()

        if dense_scores.max() > 0:
            dense_scores = dense_scores / dense_scores.max()

        for doc, score in zip(sparse_docs, sparse_scores):
            doc_scores[doc] = self.alpha * score

        for doc, score in zip(dense_docs, dense_scores):
            doc_scores[doc] = doc_scores.get(doc, 0) + (1 - self.alpha) * score

        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[
            :top_k
        ]
        return list(dict(sorted_docs).keys()), list(dict(sorted_docs).values())
