"""Sentiment classification models."""

import dspy


class SentimentClassification(dspy.Signature):
    """Classify the sentiment of a text as positive or negative."""

    text: str = dspy.InputField(desc="The text to classify")
    sentiment: str = dspy.OutputField(desc="Either 'positive' or 'negative'")


class SentimentClassifier(dspy.Module):
    """
    A simple sentiment classifier using Chain of Thought reasoning.

    This module takes text as input and predicts whether the sentiment
    is positive or negative.
    """

    def __init__(self):
        super().__init__()
        self.classify = dspy.ChainOfThought(SentimentClassification)

    def forward(self, text):
        """
        Classify the sentiment of the given text.

        Args:
            text: The text to classify

        Returns:
            Prediction with sentiment field
        """
        return self.classify(text=text)
