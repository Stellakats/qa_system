from abc import ABC, abstractmethod
from typing import Any, List, Tuple


class BaseRetriever(ABC):
    """Abstract base class for retrievers."""

    def __init__(self, documents: List[str]):
        """Initialize retriever with a list of documents."""
        self.documents = list(set(documents))

    @abstractmethod
    def retrieve(
        self, query: str, top_k: int = 5, **kwargs: Any
    ) -> Tuple[List[str], List[float]]:
        """Retrieve top-k relevant documents.

        Args:
            query (str): The search query.
            top_k (int): Number of top documents to retrieve.
            **kwargs: Additional parameters for specialized retrievers.

        Returns:
            Tuple[List[str], List[float]]: Retrieved documents and their relevance scores.
        """
        raise NotImplementedError
