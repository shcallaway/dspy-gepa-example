"""Dataset definitions for all tasks."""

from .sentiment import get_data as get_sentiment_data
from .qa import get_data as get_qa_data
from .math import get_data as get_math_data

__all__ = [
    "get_sentiment_data",
    "get_qa_data",
    "get_math_data",
]
