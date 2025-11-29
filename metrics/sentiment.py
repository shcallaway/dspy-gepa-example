"""Sentiment classification metrics."""


def accuracy(example, prediction, trace=None) -> bool:
    """
    Check if predicted sentiment matches expected sentiment.

    Args:
        example: DSPy Example with expected sentiment
        prediction: Model prediction with sentiment field
        trace: Optional trace (unused)

    Returns:
        True if sentiments match (case-insensitive), False otherwise
    """
    return example.sentiment.lower() == prediction.sentiment.lower()
