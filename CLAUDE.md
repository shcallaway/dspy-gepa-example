# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a modular DSPy project demonstrating **GEPA (Generative Evolutionary Prompt Adaptation)** for automatic prompt optimization across multiple NLP tasks. The project uses a per-task file organization where each task has dedicated dataset, model, and metric files.

## Common Commands

### Setup and Installation
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set API key (required)
export OPENAI_API_KEY='your-api-key-here'
```

### Running Tasks
```bash
# Run sentiment classification (default)
python main.py
python main.py --task sentiment

# Run question answering
python main.py --task qa

# Use python3 if python command not available
python3 main.py --task sentiment
```

## Architecture

### Core Pattern: Task Registry System

The entire application is built around a **task registry pattern** (`tasks.py`). Each task is a self-contained unit with:
- Dataset loader function
- Model class (DSPy Module)
- Evaluation metric function
- GEPA optimization parameters
- Input/output field definitions

This registry drives the generic workflow in `main.py`, allowing new tasks to be added without modifying the core evaluation logic.

### Key Architectural Components

**1. Task Configuration (`tasks.py`)**
- Central registry (`TASKS` dict) maps task names to configurations
- Each task config specifies: data loader, model class, metric, GEPA params, and field names
- Main workflow uses this registry to run tasks generically

**2. Generic Workflow (`main.py`)**
- Single workflow handles all tasks using dynamic field access
- Uses `task_config["input_fields"]` to extract inputs from examples
- Formats output based on task type (checks `input_fields` to distinguish tasks)
- GEPA optimization runs identically for all tasks with task-specific parameters

**3. Modular Organization**
```
datasets/    # Per-task data loaders (sentiment.py, qa.py)
models/      # Per-task DSPy Modules and Signatures (sentiment.py, qa.py)
metrics/     # Per-task evaluation functions + shared utilities (common.py)
config.py    # LLM provider configuration
tasks.py     # Task registry (glues everything together)
main.py      # Generic workflow orchestration
```

**4. DSPy Integration**
- All models inherit from `dspy.Module` and use `dspy.Signature` for input/output schemas
- Models use `dspy.ChainOfThought` predictor pattern
- GEPA optimizer takes: metric function, trainset, valset, and optimization level (`auto` param)
- Separate reflection LM (`gpt-4o-mini` with temp=1.0) used for GEPA instruction generation

### GEPA Optimization Flow

1. Baseline model created from `task_config["model_class"]()`
2. GEPA optimizer initializes with task metric and `auto` setting ("light", "medium", etc.)
3. Reflection LM generates instruction variations during optimization
4. Optimizer compiles optimized model using trainset and valset
5. Optimized model evaluated on same valset for comparison

### Adding New Tasks

To add a task, create 3 files:

1. `datasets/your_task.py` - Define data and `get_your_task_data()` loader
2. `models/your_task.py` - Define Signature and Module classes
3. `metrics/your_task.py` - Define accuracy/metric function

Then register in `tasks.py`:
```python
TASKS = {
    "your_task": {
        "name": "Your Task Name",
        "get_data": get_your_task_data,
        "model_class": YourTaskModule,
        "metric": your_task_accuracy,
        "gepa_auto": "medium",  # or "light", "heavy"
        "input_fields": ["field1", "field2"],
        "output_field": "output",
    },
}
```

Update `__init__.py` files in each directory to export your new functions/classes.

## Important Implementation Details

### DSPy Example Format
- Dataset loaders return lists of `dspy.Example` objects
- Use `.with_inputs()` to mark which fields are inputs vs labels
- Example: `ex.with_inputs("text")` marks "text" as input, other fields as labels

### GEPA Parameters
- `auto` parameter controls optimization intensity: "light" (simple tasks) to "heavy" (complex)
- `reflection_lm` must be separate from main LM, used only for instruction generation
- Higher temperature (1.0) for reflection LM encourages exploration

### LM Configuration
- `config.py` provides `configure_lm()` for flexible provider setup
- DSPy uses model strings like `"openai/gpt-4o-mini"` or `"anthropic/claude-3-5-sonnet-20241022"`
- API keys read from environment variables by default

### Metric Functions
- Signature: `metric(example, prediction, trace=None) -> bool`
- Must compare `example` (ground truth) with `prediction` (model output)
- Return `True` for correct, `False` for incorrect
- See `metrics/common.py` for reusable utilities like `exact_match()`

## Dependencies

- `dspy>=2.5.0` - DSPy framework with GEPA support
- `openai>=1.0.0` - OpenAI API client (or other provider SDKs as needed)

Requires Python 3.9+.
