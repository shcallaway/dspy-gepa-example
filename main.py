"""
DSPy GEPA Tutorial - Multi-Task Examples
=========================================

Demonstrates GEPA optimization on multiple tasks:
- Sentiment Classification: Classify text as positive/negative
- Question Answering: Answer questions from context

Usage:
    python main.py --task sentiment
    python main.py --task qa
"""

import argparse
import dspy
from dspy.teleprompt import GEPA

from config import get_default_lm
from tasks import TASKS


def run_baseline_evaluation(task_config, model, dev_examples):
    """Evaluate baseline model (generic for all tasks)."""
    print("Step 1: Testing BASELINE (unoptimized) model...")
    print("-" * 60)

    metric = task_config["metric"]
    correct = 0

    for example in dev_examples:
        # Get inputs dynamically based on task
        inputs = {field: getattr(example, field) for field in task_config["input_fields"]}
        prediction = model(**inputs)
        is_correct = metric(example, prediction)
        correct += is_correct

        # Display results (task-specific formatting)
        print_example_result(example, prediction, is_correct, task_config)

    score = correct / len(dev_examples)
    print(f"Baseline Accuracy: {score:.1%} ({correct}/{len(dev_examples)})")
    print()
    return score


def run_gepa_optimization(task_config, train_examples, dev_examples):
    """Run GEPA optimization (generic for all tasks)."""
    print("Step 2: Running GEPA optimization...")
    print("-" * 60)
    print(f"GEPA Config: auto={task_config['gepa_auto']}")
    print("This will take a few moments as GEPA evolves the prompts...")
    print()

    # Create a reflection LM for GEPA to use for generating new instructions
    reflection_lm = dspy.LM(model='gpt-4o-mini', temperature=1.0, max_tokens=4000)

    optimizer = GEPA(
        metric=task_config["metric"],
        auto=task_config["gepa_auto"],
        reflection_lm=reflection_lm,
    )

    optimized = optimizer.compile(
        student=task_config["model_class"](),
        trainset=train_examples,
        valset=dev_examples,
    )

    print("Optimization complete!")
    print()
    return optimized


def run_optimized_evaluation(task_config, model, dev_examples):
    """Evaluate optimized model (generic for all tasks)."""
    print("Step 3: Testing OPTIMIZED model...")
    print("-" * 60)

    metric = task_config["metric"]
    correct = 0

    for example in dev_examples:
        inputs = {field: getattr(example, field) for field in task_config["input_fields"]}
        prediction = model(**inputs)
        is_correct = metric(example, prediction)
        correct += is_correct

        print_example_result(example, prediction, is_correct, task_config)

    score = correct / len(dev_examples)
    print(f"Optimized Accuracy: {score:.1%} ({correct}/{len(dev_examples)})")
    print()
    return score


def print_example_result(example, prediction, is_correct, task_config):
    """Print example result with task-specific formatting."""
    check = '✓' if is_correct else '✗'

    if task_config["input_fields"] == ["text"]:
        # Sentiment task
        print(f"Text: {example.text[:50]}...")
        print(f"Expected: {example.sentiment} | Predicted: {prediction.sentiment} | {check}")
    else:
        # QA task
        print(f"Q: {example.question}")
        print(f"Context: {example.context[:60]}...")
        print(f"Expected: {example.answer} | Predicted: {prediction.answer} | {check}")
    print()


def print_results_summary(baseline_score, optimized_score):
    """Print comparison summary."""
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Baseline Accuracy:  {baseline_score:.1%}")
    print(f"Optimized Accuracy: {optimized_score:.1%}")
    improvement = ((optimized_score - baseline_score) / baseline_score * 100) if baseline_score > 0 else 0
    print(f"Improvement: {improvement:+.1f}%")
    print()


def demo_optimized_model(task_config, model):
    """Demo model on new examples (task-specific)."""
    print("Step 4: Try the optimized model on new examples...")
    print("-" * 60)

    if task_config["input_fields"] == ["text"]:
        # Sentiment examples
        test_texts = [
            "This is the best thing ever!",
            "I'm very disappointed with this.",
        ]
        for text in test_texts:
            result = model(text=text)
            print(f"Text: {text}")
            print(f"Sentiment: {result.sentiment}")
            print()
    else:
        # QA examples
        test_examples = [
            {"question": "What is the largest ocean?", "context": "The Pacific Ocean is the largest ocean on Earth."},
            {"question": "When was the internet invented?", "context": "The internet was developed in the 1960s-1970s."},
        ]
        for ex in test_examples:
            result = model(question=ex["question"], context=ex["context"])
            print(f"Q: {ex['question']}")
            print(f"Answer: {result.answer}")
            print()


def main():
    """Main workflow with task selection."""
    parser = argparse.ArgumentParser(
        description="DSPy GEPA Multi-Task Examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--task",
        choices=list(TASKS.keys()),
        default="sentiment",
        help="Task to run (default: sentiment)"
    )
    args = parser.parse_args()

    # Get task configuration
    task_config = TASKS[args.task]

    print("=" * 60)
    print(f"DSPy GEPA: {task_config['name']}")
    print("=" * 60)
    print()

    # Configure LM
    get_default_lm()

    # Load data
    train_examples, dev_examples = task_config["get_data"]()

    # Baseline evaluation
    baseline_model = task_config["model_class"]()
    baseline_score = run_baseline_evaluation(task_config, baseline_model, dev_examples)

    # GEPA optimization
    optimized_model = run_gepa_optimization(task_config, train_examples, dev_examples)

    # Optimized evaluation
    optimized_score = run_optimized_evaluation(task_config, optimized_model, dev_examples)

    # Results summary
    print_results_summary(baseline_score, optimized_score)

    # Demo on new examples
    demo_optimized_model(task_config, optimized_model)


if __name__ == "__main__":
    main()
