"""Evaluation metrics for all tasks."""

from .sentiment import accuracy as sentiment_accuracy
from .qa import accuracy as qa_accuracy
from .math import accuracy as math_accuracy
from .common import exact_match, evaluate_model

__all__ = [
    "sentiment_accuracy",
    "qa_accuracy",
    "math_accuracy",
    "exact_match",
    "evaluate_model",
]
