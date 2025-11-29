"""
Dataset definitions and utilities for DSPy GEPA examples.
"""

import dspy
from typing import List, Tuple


# Sentiment Classification Dataset
# Format: (text, expected_sentiment)
SENTIMENT_TRAIN_DATA = [
    ("This movie was absolutely fantastic! I loved every minute.", "positive"),
    ("Terrible experience. Would not recommend to anyone.", "negative"),
    ("Best purchase I've made all year! Highly recommend.", "positive"),
    ("Complete waste of time and money. Very disappointed.", "negative"),
    ("Amazing quality and fast delivery. Very happy!", "positive"),
    ("Poor customer service and broken product.", "negative"),
    ("Exceeded all my expectations. Will buy again!", "positive"),
    ("Worst meal I've ever had. Don't go there.", "negative"),
]

SENTIMENT_DEV_DATA = [
    ("This product is incredible! Worth every penny.", "positive"),
    ("Not good at all. Returned it immediately.", "negative"),
    ("Absolutely love it! Five stars!", "positive"),
    ("Horrible quality. Very upset with this purchase.", "negative"),
]


def create_examples(
    data: List[Tuple[str, str]],
    input_fields: List[str] = None
) -> List[dspy.Example]:
    """
    Convert tuples to DSPy Examples.

    Args:
        data: List of tuples containing (input_text, output_text)
        input_fields: List of field names to mark as inputs (defaults to first field)

    Returns:
        List of DSPy Example objects
    """
    if input_fields is None:
        # For sentiment: (text, sentiment) -> input is 'text'
        input_fields = ["text"]

    examples = []
    for item in data:
        if len(item) == 2:
            # Assume (text, label) format for now
            example = dspy.Example(text=item[0], sentiment=item[1])
            examples.append(example.with_inputs(*input_fields))
        else:
            raise ValueError(f"Unexpected data format: {item}")

    return examples


def get_sentiment_data():
    """
    Get sentiment classification train and dev datasets.

    Returns:
        Tuple of (train_examples, dev_examples)
    """
    train_examples = create_examples(SENTIMENT_TRAIN_DATA)
    dev_examples = create_examples(SENTIMENT_DEV_DATA)
    return train_examples, dev_examples


# Question Answering Dataset
# Format: (question, context, answer)
QA_TRAIN_DATA = [
    ("What is the capital of France?", "France is a country in Western Europe. Its capital is Paris.", "Paris"),
    ("Who wrote Romeo and Juliet?", "William Shakespeare wrote the famous play Romeo and Juliet.", "William Shakespeare"),
    ("What is the largest planet?", "Jupiter is the largest planet in our solar system.", "Jupiter"),
    ("When was Python created?", "Python was created by Guido van Rossum in 1991.", "1991"),
    ("What does DNA stand for?", "DNA stands for deoxyribonucleic acid.", "deoxyribonucleic acid"),
    ("How many continents are there?", "There are seven continents on Earth.", "seven"),
]

QA_DEV_DATA = [
    ("What is the smallest country?", "Vatican City is the smallest country in the world.", "Vatican City"),
    ("What year did the Titanic sink?", "The Titanic sank in 1912.", "1912"),
]


def get_qa_data():
    """
    Get question answering train and dev datasets.

    Returns:
        Tuple of (train_examples, dev_examples)
    """
    train = []
    for q, ctx, ans in QA_TRAIN_DATA:
        ex = dspy.Example(question=q, context=ctx, answer=ans)
        train.append(ex.with_inputs("question", "context"))

    dev = []
    for q, ctx, ans in QA_DEV_DATA:
        ex = dspy.Example(question=q, context=ctx, answer=ans)
        dev.append(ex.with_inputs("question", "context"))

    return train, dev


# Template for adding new datasets:
#
# YOUR_TASK_TRAIN_DATA = [
#     (input1, output1),
#     (input2, output2),
#     ...
# ]
#
# YOUR_TASK_DEV_DATA = [
#     (input1, output1),
#     ...
# ]
#
# def get_your_task_data():
#     """Get your task train and dev datasets."""
#     train = create_examples(YOUR_TASK_TRAIN_DATA, input_fields=["your_input"])
#     dev = create_examples(YOUR_TASK_DEV_DATA, input_fields=["your_input"])
#     return train, dev
