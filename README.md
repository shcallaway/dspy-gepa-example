# DSPy GEPA Project

A modular, production-ready example of using **GEPA (Generative Evolutionary Prompt Adaptation)** to optimize prompts in DSPy.

## Quick Start

```bash
# 1. Create and activate virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your OpenAI API key
export OPENAI_API_KEY='your-api-key-here'

# 4. Run the sentiment classification example
python main.py --task sentiment

# Or run the question answering example
python main.py --task qa
```

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
├── tasks.py               # Task registry (glues everything together)
├── main.py                # Main tutorial orchestration
└── requirements.txt       # Project dependencies
```

## What This Project Demonstrates

This project uses GEPA to optimize prompts for **multiple tasks**:

### Sentiment Classification
- Classify text as positive or negative
- Single-input task demonstrating basic GEPA usage
- GEPA optimization level: "light"

### Question Answering
- Answer questions based on context
- Multi-input task (question + context)
- GEPA optimization level: "medium"

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
  - Get one at: https://platform.openai.com/api-keys

## Setup

1. **Create a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set your API key (required):**
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```

   Or for other providers:
   ```bash
   export ANTHROPIC_API_KEY='your-api-key-here'
   ```

   **Note:** The script will not work without an API key set.

## Usage

**Note:** Make sure you've activated your virtual environment and set your API key before running!

```bash
source venv/bin/activate  # Activate virtual environment
export OPENAI_API_KEY='your-api-key-here'  # Set API key
```

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

def get_data():
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
from .your_task import get_data as get_your_task_data

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

def accuracy(gold, pred, trace=None, pred_name=None, pred_trace=None) -> bool:
    """Check if prediction is correct."""
    return gold.output.lower() == pred.output.lower()
```

Update `metrics/__init__.py`:
```python
from .your_task import accuracy as your_task_accuracy

__all__ = [..., "your_task_accuracy"]
```

### 4. Register in tasks.py

Add to the `TASKS` dictionary in `tasks.py`:

```python
TASKS = {
    # ... existing tasks ...
    "your_task": {
        "name": "Your Task Name",
        "get_data": get_your_task_data,
        "model_class": YourTaskModule,
        "metric": your_task_accuracy,
        "gepa_auto": "medium",  # or "light", "heavy"
        "input_fields": ["input"],
        "output_field": "output",
    },
}
```

Then run:
```bash
python main.py --task your_task
# Or: python3 main.py --task your_task
```

## Key GEPA Parameters

- `metric`: Function to evaluate prompt quality
- `auto`: Optimization intensity level ("light", "medium", "heavy")
- `reflection_lm`: Separate LM for generating instruction variations
- Higher `auto` levels = more exploration and refinement iterations

### Task-Specific Parameters

| Task | Auto Level | Rationale |
|------|---------|-----------|
| Sentiment | "light" | Simple task, single input field |
| QA | "medium" | Complex task, multiple inputs need more optimization |

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

### `tasks.py`
- Task configuration registry (`TASKS` dictionary)
- Imports and organizes all task components

### `main.py`
- Generic evaluation functions that work with all tasks
- Command-line interface for task selection
- Complete tutorial workflow

## Expected Output

Running the sentiment task will show:

1. Baseline model performance on dev set
2. GEPA optimization progress (breadth=2, depth=1)
3. Optimized model performance on dev set
4. Performance comparison and improvement metrics
5. Demo predictions on new examples

Running the QA task will show the same workflow but with:
- Multi-field inputs (question + context)
- More intensive GEPA optimization (breadth=3, depth=2)

## Troubleshooting

### `ModuleNotFoundError: No module named 'dspy'`
Make sure you've installed dependencies and activated your virtual environment:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### `command not found: python`
Use `python3` instead:
```bash
python3 main.py --task sentiment
```

### API Key Errors
Ensure your API key is set as an environment variable:
```bash
# Check if it's set
echo $OPENAI_API_KEY

# Set it if needed
export OPENAI_API_KEY='your-api-key-here'
```

### Script runs but no output
The GEPA optimization can take a few minutes. Be patient and watch for the progress bars.

### Rate Limit and Quota Errors

If you encounter `RateLimitError` or quota exceeded errors:

**Error: "You exceeded your current quota"**
This means you've hit your OpenAI billing/usage cap:
1. Check your usage at https://platform.openai.com/usage
2. Verify you have credits or add more at https://platform.openai.com/settings/organization/billing
3. Create a new API key if needed at https://platform.openai.com/api-keys

**Error: "Rate limit exceeded"**
You're making too many requests per minute. Solutions:

1. **The code already includes retry logic** with exponential backoff (configured in `config.py` and `main.py`)

2. **Reduce LLM call volume** by optimizing task parameters:

   ```python
   # In models/math.py (or your task file)
   # Reduce ReAct iterations
   max_iters=2  # Instead of 5

   # In tasks.py
   # Use lighter GEPA optimization
   "gepa_auto": "light",  # Instead of "medium" or "heavy"

   # In datasets/your_task.py
   # Use fewer training examples
   TRAIN_DATA = [...]  # Reduce from 10 to 5 examples
   ```

3. **Increase retry parameters** in `config.py`:
   ```python
   configure_lm(
       model="gpt-5-mini",
       num_retries=10,  # Increase from 5
       timeout=120.0    # Increase timeout
   )
   ```

4. **Switch to faster/cheaper models**:
   - Use `gpt-5-nano` instead of `gpt-5-mini` for even faster inference
   - Or use `gpt-4o-mini` for lower costs

**Understanding LLM Call Volume:**
- ReAct tasks (like math) make multiple calls per example (up to `max_iters` iterations)
- GEPA optimization tests multiple prompt variations on your training set
- Total calls ≈ (GEPA variations) × (training examples) × (max_iters)
- Example: 3 variations × 5 examples × 2 iters = 30 calls during optimization

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
