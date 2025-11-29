"""Math word problem metrics."""


def accuracy(gold, pred, trace=None, pred_name=None, pred_trace=None) -> bool:
    """
    Check if predicted answer matches expected answer numerically.

    Handles both integer and decimal answers with tolerance for floating point.

    Args:
        gold: DSPy Example with expected answer
        pred: Model prediction with answer field
        trace: Optional trace (unused)
        pred_name: Name of the prediction (unused)
        pred_trace: Trace of the prediction (unused)

    Returns:
        True if answers match numerically, False otherwise
    """
    try:
        # Extract and clean the answers
        expected = str(gold.answer).strip()
        predicted = str(pred.answer).strip()

        # Try exact string match first (fastest)
        if expected.lower() == predicted.lower():
            return True

        # Try numerical comparison (handles "12.0" vs "12", etc.)
        expected_num = float(expected)
        predicted_num = float(predicted)

        # Use small tolerance for floating point comparison
        return abs(expected_num - predicted_num) < 1e-6

    except (ValueError, AttributeError):
        # If conversion fails, fall back to string comparison
        return str(gold.answer).lower().strip() == str(pred.answer).lower().strip()
