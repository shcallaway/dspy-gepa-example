"""Task registry and configuration for DSPy GEPA examples."""

from datasets import get_sentiment_data, get_qa_data
from models import SentimentClassifier, QAModule
from metrics import sentiment_accuracy, qa_accuracy


# Task Configuration Registry
TASKS = {
    "sentiment": {
        "name": "Sentiment Classification",
        "get_data": get_sentiment_data,
        "model_class": SentimentClassifier,
        "metric": sentiment_accuracy,
        "gepa_breadth": 2,
        "gepa_depth": 1,
        "input_fields": ["text"],
        "output_field": "sentiment",
    },
    "qa": {
        "name": "Question Answering",
        "get_data": get_qa_data,
        "model_class": QAModule,
        "metric": qa_accuracy,
        "gepa_breadth": 3,  # Higher: multi-input optimization
        "gepa_depth": 2,    # More iterations: complex task
        "input_fields": ["question", "context"],
        "output_field": "answer",
    },
}
