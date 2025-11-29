"""Sentiment classification metrics."""


def accuracy(gold, pred, trace=None, pred_name=None, pred_trace=None) -> bool:
    """
    Check if predicted sentiment matches expected sentiment.

    Args:
        gold: DSPy Example with expected sentiment
        pred: Model prediction with sentiment field
        trace: Optional trace (unused)
        pred_name: Name of the prediction (unused)
        pred_trace: Trace of the prediction (unused)

    Returns:
        True if sentiments match (case-insensitive), False otherwise
    """
    return gold.sentiment.lower() == pred.sentiment.lower()
