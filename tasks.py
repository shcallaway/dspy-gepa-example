"""Task registry and configuration for DSPy GEPA examples."""

from datasets import get_sentiment_data, get_qa_data, get_math_data
from models import SentimentClassifier, QAModule, MathSolver
from metrics import sentiment_accuracy, qa_accuracy, math_accuracy


# Task Configuration Registry
TASKS = {
    "sentiment": {
        "name": "Sentiment Classification",
        "get_data": get_sentiment_data,
        "model_class": SentimentClassifier,
        "metric": sentiment_accuracy,
        "gepa_auto": "light",  # Light optimization for simple task
        "input_fields": ["text"],
        "output_field": "sentiment",
    },
    "qa": {
        "name": "Question Answering",
        "get_data": get_qa_data,
        "model_class": QAModule,
        "metric": qa_accuracy,
        "gepa_auto": "medium",  # Medium optimization for multi-input task
        "input_fields": ["question", "context"],
        "output_field": "answer",
    },
    "math": {
        "name": "Math Word Problems (ReAct)",
        "get_data": get_math_data,
        "model_class": MathSolver,
        "metric": math_accuracy,
        "gepa_auto": "light",  # Light optimization to reduce LLM call volume
        "input_fields": ["problem"],
        "output_field": "answer",
    },
}
