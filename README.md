# DSPy GEPA Project

A modular, production-ready example of using **GEPA (Generative Evolutionary Prompt Adaptation)** to optimize prompts in DSPy.

## Project Structure

```
dspy-gepa-example/
├── config.py              # Language model configuration
├── datasets.py            # Dataset definitions and utilities
├── metrics.py             # Evaluation metrics
├── models/
│   ├── __init__.py
│   └── classifier.py      # Model signatures and modules
├── main.py                # Main tutorial orchestration
├── requirements.txt       # Project dependencies
└── README.md              # This file
```

## What This Project Demonstrates

This project uses GEPA to optimize prompts for various tasks, starting with **sentiment classification**:

1. **Baseline Evaluation** - Test unoptimized Chain of Thought classifier
2. **GEPA Optimization** - Automatically improve prompts through evolution
3. **Optimized Evaluation** - Measure performance gains
4. **Comparison** - Quantify improvement

The modular structure makes it easy to:
- Add new tasks and datasets
- Experiment with different models
- Customize evaluation metrics
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

### Run the Tutorial

```bash
python main.py
```

### Customize the LM Provider

Edit `config.py` or pass parameters to `configure_lm()`:

```python
from config import configure_lm

# Use Anthropic Claude
configure_lm(provider="anthropic", model="claude-3-5-sonnet-20241022")

# Use Together AI
configure_lm(provider="together", model="meta-llama/Llama-3-70b-chat-hf")
```

## Adding New Tasks

The modular structure makes it easy to add new tasks:

### 1. Add Your Dataset

In `datasets.py`:

```python
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
    train = create_examples(YOUR_TASK_TRAIN_DATA, input_fields=["input_field"])
    dev = create_examples(YOUR_TASK_DEV_DATA, input_fields=["input_field"])
    return train, dev
```

### 2. Define Your Model

In `models/classifier.py` (or create a new file in `models/`):

```python
class YourTaskSignature(dspy.Signature):
    """Description of your task."""

    input_field: str = dspy.InputField(desc="Input description")
    output_field: str = dspy.OutputField(desc="Output description")

class YourTaskModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought(YourTaskSignature)

    def forward(self, input_field):
        return self.predictor(input_field=input_field)
```

### 3. Add Evaluation Metric

In `metrics.py`:

```python
def your_task_metric(example, prediction, trace=None) -> bool:
    """Check if prediction is correct."""
    return example.output_field == prediction.output_field
```

### 4. Create Your Main Script

Copy and modify `main.py`, or add to the existing one:

```python
from datasets import get_your_task_data
from models import YourTaskModule
from metrics import your_task_metric

train_examples, dev_examples = get_your_task_data()
optimizer = GEPA(metric=your_task_metric, breadth=3, depth=2)
optimized = optimizer.compile(
    student=YourTaskModule(),
    trainset=train_examples,
    valset=dev_examples
)
```

## Key GEPA Parameters

- `metric`: Function to evaluate prompt quality
- `breadth`: Number of prompt variations per iteration (higher = more exploration)
- `depth`: Number of optimization iterations (higher = more refinement)
- `init_temperature`: Creativity in generating variations (0.0-2.0)

## Module Reference

### `config.py`
- `configure_lm()`: Configure DSPy with any LLM provider
- `get_default_lm()`: Quick setup with OpenAI GPT-4o-mini
- `PROVIDER_CONFIGS`: Pre-configured settings for common providers

### `datasets.py`
- `create_examples()`: Convert tuples to DSPy Examples
- `get_sentiment_data()`: Load sentiment classification data
- Templates for adding new datasets

### `models/classifier.py`
- `SentimentClassification`: Signature for sentiment task
- `SentimentClassifier`: Chain of Thought module for sentiment
- Templates for adding new models

### `metrics.py`
- `sentiment_accuracy()`: Metric for sentiment classification
- `exact_match()`: Generic exact match metric
- `evaluate_model()`: Batch evaluation utility
- Templates for custom metrics

### `main.py`
- Complete tutorial workflow
- Modular functions for each step
- Easy to customize and extend

## Expected Output

Running `python main.py` will show:

1. Baseline model performance on dev set
2. GEPA optimization progress
3. Optimized model performance on dev set
4. Performance comparison and improvement metrics
5. Demo predictions on new examples

## Learn More

- [DSPy Documentation](https://dspy.ai/)
- [GEPA for AIME Tutorial](https://dspy.ai/tutorials/gepa_aime/)
- [DSPy GitHub](https://github.com/stanfordnlp/dspy)

## Contributing

This is a starter template. Feel free to:
- Add new tasks and datasets
- Experiment with different models (ReAct, ProgramOfThought, etc.)
- Try different optimizers (BootstrapFewShot, COPRO, MIPROv2)
- Extend evaluation metrics
- Share your improvements!
# dspy-gepa-example
