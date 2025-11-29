"""
Evaluation metrics for DSPy GEPA examples.
"""

from typing import Callable, List
import dspy


def sentiment_accuracy(example, prediction, trace=None) -> bool:
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


def exact_match(example, prediction, trace=None) -> bool:
    """
    Generic exact match metric for any output field.

    Automatically detects the output field name from the example.

    Args:
        example: DSPy Example with expected output
        prediction: Model prediction
        trace: Optional trace (unused)

    Returns:
        True if outputs match exactly, False otherwise
    """
    # Get the first non-input field as the output field
    for key in example.__dict__:
        if not key.startswith('_'):
            expected = getattr(example, key, None)
            predicted = getattr(prediction, key, None)
            if expected is not None and predicted is not None:
                return str(expected).lower() == str(predicted).lower()

    return False


def evaluate_model(
    model: dspy.Module,
    examples: List[dspy.Example],
    metric: Callable,
    verbose: bool = False
) -> float:
    """
    Evaluate a model on a dataset using a given metric.

    Args:
        model: DSPy Module to evaluate
        examples: List of examples to evaluate on
        metric: Metric function to use
        verbose: Whether to print per-example results

    Returns:
        Accuracy score (fraction correct)
    """
    correct = 0
    total = len(examples)

    for i, example in enumerate(examples):
        # Get input fields
        input_dict = {k: v for k, v in example.__dict__.items()
                      if not k.startswith('_') and k in example._input_keys}

        # Run prediction
        prediction = model(**input_dict)

        # Evaluate
        is_correct = metric(example, prediction)
        correct += is_correct

        if verbose:
            print(f"Example {i+1}/{total}: {'✓' if is_correct else '✗'}")
            print(f"  Input: {input_dict}")
            print(f"  Expected: {example.__dict__}")
            print(f"  Predicted: {prediction.__dict__}")
            print()

    accuracy = correct / total if total > 0 else 0.0
    return accuracy


# Template for custom metrics:
#
# def your_custom_metric(example, prediction, trace=None) -> bool:
#     """
#     Custom metric description.
#
#     Args:
#         example: DSPy Example with expected output
#         prediction: Model prediction
#         trace: Optional trace
#
#     Returns:
#         True if prediction is correct, False otherwise
#     """
#     # Your custom logic here
#     return example.your_field == prediction.your_field
