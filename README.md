# DSPy GEPA Project

A modular, production-ready example of using **GEPA (Generative Evolutionary Prompt Adaptation)** to optimize prompts in DSPy.

## Project Structure

```
dspy-gepa-example/
├── config.py              # Language model configuration
├── datasets/              # Dataset definitions (per-task organization)
│   ├── __init__.py
│   ├── sentiment.py       # Sentiment classification data
│   └── qa.py              # Question answering data
├── models/                # Model signatures and modules (per-task)
│   ├── __init__.py
│   ├── sentiment.py       # Sentiment models
│   └── qa.py              # QA models
├── metrics/               # Evaluation metrics (per-task)
│   ├── __init__.py
│   ├── sentiment.py       # Sentiment metrics
│   ├── qa.py              # QA metrics
│   └── common.py          # Shared utilities
├── main.py                # Main tutorial orchestration
├── requirements.txt       # Project dependencies
└── README.md              # This file
```

## What This Project Demonstrates

This project uses GEPA to optimize prompts for **multiple tasks**:

### Sentiment Classification
- Classify text as positive or negative
- Single-input task demonstrating basic GEPA usage
- GEPA params: breadth=2, depth=1

### Question Answering
- Answer questions based on context
- Multi-input task (question + context)
- Demonstrates higher GEPA optimization (breadth=3, depth=2)

### Workflow for Each Task

1. **Baseline Evaluation** - Test unoptimized Chain of Thought model
2. **GEPA Optimization** - Automatically improve prompts through evolution
3. **Optimized Evaluation** - Measure performance gains
4. **Comparison** - Quantify improvement

The per-task file organization makes it easy to:
- Understand what code belongs to which task
- Add new tasks without touching existing ones
- Experiment with different models and datasets
- Scale to production use cases

## Prerequisites

- Python 3.9 or higher
- OpenAI API key (or another LLM provider supported by DSPy)

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set your API key:**
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```

   Or for other providers:
   ```bash
   export ANTHROPIC_API_KEY='your-api-key-here'
   ```

## Usage

### Run Sentiment Classification (Default)

```bash
python main.py
```

Or explicitly:
```bash
python main.py --task sentiment
```

### Run Question Answering

```bash
python main.py --task qa
```

### Customize the LM Provider

Edit `config.py` or modify the `get_default_lm()` function:

```python
from config import configure_lm

# Use Anthropic Claude
configure_lm(provider="anthropic", model="claude-3-5-sonnet-20241022")

# Use Together AI
configure_lm(provider="together", model="meta-llama/Llama-3-70b-chat-hf")
```

## Adding New Tasks

The per-task file organization makes adding new tasks straightforward. Each task needs 3 files:

### 1. Add Your Dataset

Create `datasets/your_task.py`:

```python
"""Your task dataset."""

import dspy

YOUR_TASK_TRAIN_DATA = [
    ("input 1", "output 1"),
    ("input 2", "output 2"),
    # ...
]

YOUR_TASK_DEV_DATA = [
    ("input 1", "output 1"),
    # ...
]

def get_your_task_data():
    """Get your task train and dev datasets."""
    train = []
    for input_val, output_val in YOUR_TASK_TRAIN_DATA:
        ex = dspy.Example(input=input_val, output=output_val)
        train.append(ex.with_inputs("input"))

    dev = []
    for input_val, output_val in YOUR_TASK_DEV_DATA:
        ex = dspy.Example(input=input_val, output=output_val)
        dev.append(ex.with_inputs("input"))

    return train, dev
```

Update `datasets/__init__.py`:
```python
from .your_task import get_your_task_data

__all__ = [..., "get_your_task_data"]
```

### 2. Define Your Model

Create `models/your_task.py`:

```python
"""Your task models."""

import dspy

class YourTaskSignature(dspy.Signature):
    """Description of your task."""

    input: str = dspy.InputField(desc="Input description")
    output: str = dspy.OutputField(desc="Output description")

class YourTaskModule(dspy.Module):
    """Your task module with Chain of Thought reasoning."""

    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought(YourTaskSignature)

    def forward(self, input):
        return self.predictor(input=input)
```

Update `models/__init__.py`:
```python
from .your_task import YourTaskSignature, YourTaskModule

__all__ = [..., "YourTaskSignature", "YourTaskModule"]
```

### 3. Add Evaluation Metric

Create `metrics/your_task.py`:

```python
"""Your task metrics."""

def your_task_accuracy(example, prediction, trace=None) -> bool:
    """Check if prediction is correct."""
    return example.output.lower() == prediction.output.lower()
```

Update `metrics/__init__.py`:
```python
from .your_task import your_task_accuracy

__all__ = [..., "your_task_accuracy"]
```

### 4. Register in main.py

Add to the `TASKS` dictionary in `main.py`:

```python
TASKS = {
    # ... existing tasks ...
    "your_task": {
        "name": "Your Task Name",
        "get_data": get_your_task_data,
        "model_class": YourTaskModule,
        "metric": your_task_accuracy,
        "gepa_breadth": 3,
        "gepa_depth": 2,
        "input_fields": ["input"],
        "output_field": "output",
    },
}
```

Then run:
```bash
python main.py --task your_task
```

## Key GEPA Parameters

- `metric`: Function to evaluate prompt quality
- `breadth`: Number of prompt variations per iteration (higher = more exploration)
- `depth`: Number of optimization iterations (higher = more refinement)
- `init_temperature`: Creativity in generating variations (0.0-2.0)

### Task-Specific Parameters

| Task | Breadth | Depth | Rationale |
|------|---------|-------|-----------|
| Sentiment | 2 | 1 | Simple task, single input field |
| QA | 3 | 2 | Complex task, multiple inputs need more optimization |

## Module Reference

### `config.py`
- `configure_lm()`: Configure DSPy with any LLM provider
- `get_default_lm()`: Quick setup with OpenAI GPT-4o-mini
- `PROVIDER_CONFIGS`: Pre-configured settings for common providers

### `datasets/`
Each task has its own dataset file:
- `sentiment.py`: Sentiment classification data and loader
- `qa.py`: Question answering data and loader
- Add new tasks by creating new files

### `models/`
Each task has its own model file:
- `sentiment.py`: `SentimentClassification` signature and `SentimentClassifier` module
- `qa.py`: `QuestionAnswering` signature and `QAModule`
- Add new tasks by creating new files

### `metrics/`
Each task has its own metrics file:
- `sentiment.py`: `accuracy()` metric
- `qa.py`: `accuracy()` metric
- `common.py`: Shared utilities (`exact_match()`, `evaluate_model()`)

### `main.py`
- Task configuration registry (`TASKS` dictionary)
- Generic evaluation functions that work with all tasks
- Command-line interface for task selection
- Complete tutorial workflow

## Expected Output

Running `python main.py --task sentiment` will show:

1. Baseline model performance on dev set
2. GEPA optimization progress (breadth=2, depth=1)
3. Optimized model performance on dev set
4. Performance comparison and improvement metrics
5. Demo predictions on new examples

Running `python main.py --task qa` will show the same workflow but with:
- Multi-field inputs (question + context)
- More intensive GEPA optimization (breadth=3, depth=2)

## Learn More

- [DSPy Documentation](https://dspy.ai/)
- [GEPA for AIME Tutorial](https://dspy.ai/tutorials/gepa_aime/)
- [DSPy GitHub](https://github.com/stanfordnlp/dspy)

## Contributing

This is a starter template. Feel free to:
- Add new tasks and datasets (just create 3 new files!)
- Experiment with different models (ReAct, ProgramOfThought, etc.)
- Try different optimizers (BootstrapFewShot, COPRO, MIPROv2)
- Extend evaluation metrics
- Share your improvements!
