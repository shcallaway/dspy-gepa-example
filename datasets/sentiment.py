"""Sentiment classification dataset."""

import dspy
from typing import List, Tuple


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


def get_sentiment_data():
    """Get sentiment classification train and dev datasets."""
    train = []
    for text, sentiment in SENTIMENT_TRAIN_DATA:
        ex = dspy.Example(text=text, sentiment=sentiment)
        train.append(ex.with_inputs("text"))

    dev = []
    for text, sentiment in SENTIMENT_DEV_DATA:
        ex = dspy.Example(text=text, sentiment=sentiment)
        dev.append(ex.with_inputs("text"))

    return train, dev
