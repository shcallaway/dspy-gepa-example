"""Question answering models."""

import dspy


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
