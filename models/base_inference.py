import abc
from typing import List


class BaseQAInference(abc.ABC):
    """Abstract base class for Question Answering models."""

    def __init__(self, batch_size: int = 8) -> None:
        """Initialize base class with batch size."""
        self.batch_size = batch_size

    @abc.abstractmethod
    def answer(self, questions: List[str], contexts: List[str]) -> List[str]:
        """Generates answers given lists of questions and contexts.

        Args:
            questions (List[str]): List of questions.
            contexts (List[str]): List of corresponding contexts.

        Returns:
            List[str]: List of generated answers.
        """
