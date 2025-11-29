"""Question answering metrics."""


def qa_accuracy(example, prediction, trace=None) -> bool:
    """
    Check if predicted answer matches expected answer.
    Uses case-insensitive exact match.

    Args:
        example: DSPy Example with expected answer
        prediction: Model prediction with answer field
        trace: Optional trace (unused)

    Returns:
        True if answers match (case-insensitive), False otherwise
    """
    expected = str(example.answer).lower().strip()
    predicted = str(prediction.answer).lower().strip()
    return expected == predicted
