from typing import List, Tuple

import faiss
from sentence_transformers import SentenceTransformer

from .base import BaseRetriever


class DenseRetriever(BaseRetriever):
    """Dense retriever using FAISS for fast vector search."""

    def __init__(self, documents: List[str]):
        super().__init__(documents)
        self.documents = list(documents)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = self._build_index(self.documents)

    def _build_index(self, documents: List[str]) -> faiss.IndexFlatL2:
        """Encodes documents and stores them in FAISS index."""
        embeddings = self.model.encode(documents, convert_to_numpy=True)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        return index

    def retrieve(self, query: str, top_k: int = 5) -> Tuple[List[str], List[float]]:
        """Retrieve top-k documents using FAISS."""
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, top_k)
        return [self.documents[i] for i in indices[0]], distances[0].tolist()
