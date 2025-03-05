from typing import List

from retrievers.dense import DenseRetriever
from retrievers.hybrid import HybridRetriever
from retrievers.sparse import BM25Retriever

from .chroma import ChromaRetriever

RETRIEVERS = {
    "sparse": BM25Retriever,
    "dense": DenseRetriever,
    "hybrid": HybridRetriever,
    "chroma": ChromaRetriever,
}


def get_retriever(name: str, documents: List[str], **kwargs) -> object:
    """Returns the retriever instance based on the selected retriever type."""
    if name == "chroma":
        return ChromaRetriever(documents, **kwargs)
    elif name in RETRIEVERS:
        return RETRIEVERS[name](documents)
    else:
        raise ValueError(
            f"Retriever '{name}' not found. Available: {list(RETRIEVERS.keys())}"
        )
