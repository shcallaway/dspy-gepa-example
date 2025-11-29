"""Dataset definitions for all tasks."""

from .sentiment import get_sentiment_data
from .qa import get_qa_data

__all__ = [
    "get_sentiment_data",
    "get_qa_data",
]
