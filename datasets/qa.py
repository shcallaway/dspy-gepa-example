"""Question answering dataset."""

import dspy
from typing import List, Tuple


TRAIN_DATA = [
    ("What is the capital of France?", "France is a country in Western Europe. Its capital is Paris.", "Paris"),
    ("Who wrote Romeo and Juliet?", "William Shakespeare wrote the famous play Romeo and Juliet.", "William Shakespeare"),
    ("What is the largest planet?", "Jupiter is the largest planet in our solar system.", "Jupiter"),
    ("When was Python created?", "Python was created by Guido van Rossum in 1991.", "1991"),
    ("What does DNA stand for?", "DNA stands for deoxyribonucleic acid.", "deoxyribonucleic acid"),
    ("How many continents are there?", "There are seven continents on Earth.", "seven"),
]

DEV_DATA = [
    ("What is the smallest country?", "Vatican City is the smallest country in the world.", "Vatican City"),
    ("What year did the Titanic sink?", "The Titanic sank in 1912.", "1912"),
]


def get_data():
    """Get question answering train and dev datasets."""
    train = []
    for q, ctx, ans in TRAIN_DATA:
        ex = dspy.Example(question=q, context=ctx, answer=ans)
        train.append(ex.with_inputs("question", "context"))

    dev = []
    for q, ctx, ans in DEV_DATA:
        ex = dspy.Example(question=q, context=ctx, answer=ans)
        dev.append(ex.with_inputs("question", "context"))

    return train, dev
