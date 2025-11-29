"""Common evaluation utilities shared across tasks."""

from typing import Callable, List
import dspy


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
