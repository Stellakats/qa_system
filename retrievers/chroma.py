from typing import List, Tuple

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

from .base import BaseRetriever


class ChromaRetriever(BaseRetriever):
    """Dense retriever using ChromaDB for vector search."""

    def __init__(
        self, documents: List[str], collection_name="squad_passages", **kwargs
    ):
        super().__init__(documents)

        use_server = kwargs.get(
            "use_server", False
        )  # ✅ Extract `use_server` from kwargs

        if use_server:
            print("connecting to ChromadB server...")
            self.client = chromadb.HttpClient(host="localhost", port=8000)
        else:
            print("using local ChromaDB storage (./chroma_db)...")
            self.client = chromadb.PersistentClient(path="./chroma_db")  # Local storage

        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

        self.embedder = SentenceTransformer("BAAI/bge-base-en")

        if self.collection.count() == 0:
            self._index_documents()
        else:
            print(f"✅ Found {self.collection.count()} existing documents in ChromaDB.")

    def _index_documents(self):
        """Encodes documents and stores them in ChromaDB."""
        print("Indexing documents in ChromadB...")

        embeddings = self.embedder.encode(self.documents, normalize_embeddings=True)

        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()

        self.collection.add(
            ids=[str(i) for i in range(len(self.documents))],
            documents=self.documents,
            embeddings=embeddings,
        )

        print(f"Indexed {len(self.documents)} documents in ChromaDb.")

    def retrieve(self, query: str, top_k: int = 5) -> Tuple[List[str], List[float]]:
        """Retrieve top-k documents using ChromaDB."""
        query_embedding = self.embedder.encode(query, normalize_embeddings=True)

        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "distances"],
        )

        if results and results.get("documents") and results.get("distances"):
            top_docs = results["documents"][0]
            top_scores = results["distances"][0]
        else:
            print("No passages retrieved")
            top_docs, top_scores = [], []

        return top_docs, top_scores
