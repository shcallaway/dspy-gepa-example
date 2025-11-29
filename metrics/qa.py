"""Question answering metrics."""


def accuracy(gold, pred, trace=None, pred_name=None, pred_trace=None) -> bool:
    """
    Check if predicted answer matches expected answer.
    Uses case-insensitive exact match.

    Args:
        gold: DSPy Example with expected answer
        pred: Model prediction with answer field
        trace: Optional trace (unused)
        pred_name: Name of the prediction (unused)
        pred_trace: Trace of the prediction (unused)

    Returns:
        True if answers match (case-insensitive), False otherwise
    """
    expected = str(gold.answer).lower().strip()
    predicted = str(pred.answer).lower().strip()
    return expected == predicted
