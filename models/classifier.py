"""
Classifier models and signatures for DSPy GEPA examples.
"""

import dspy


# ============================================================================
# Sentiment Classification
# ============================================================================

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


# ============================================================================
# Question Answering
# ============================================================================

class QuestionAnswering(dspy.Signature):
    """Answer a question based on provided context."""

    question: str = dspy.InputField(desc="The question to answer")
    context: str = dspy.InputField(desc="Context containing the answer")
    answer: str = dspy.OutputField(desc="Concise answer to the question")


class QAModule(dspy.Module):
    """
    Question answering using Chain of Thought reasoning.

    This module takes a question and context as inputs and generates
    a concise answer based on the provided context.
    """

    def __init__(self):
        super().__init__()
        self.qa = dspy.ChainOfThought(QuestionAnswering)

    def forward(self, question, context):
        """
        Answer a question based on context.

        Args:
            question: The question to answer
            context: The context passage

        Returns:
            Prediction with answer field
        """
        return self.qa(question=question, context=context)


# ============================================================================
# Template for adding new tasks
# ============================================================================

# class YourTaskSignature(dspy.Signature):
#     """Description of your task."""
#
#     input_field: str = dspy.InputField(desc="Description of input")
#     output_field: str = dspy.OutputField(desc="Description of output")
#
#
# class YourTaskModule(dspy.Module):
#     """Your task module with Chain of Thought reasoning."""
#
#     def __init__(self):
#         super().__init__()
#         self.predictor = dspy.ChainOfThought(YourTaskSignature)
#
#     def forward(self, input_field):
#         return self.predictor(input_field=input_field)
