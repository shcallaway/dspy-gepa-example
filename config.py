"""
Configuration for DSPy language models.
"""

import os
import dspy


def configure_lm(
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    api_key: str = None,
    **kwargs
):
    """
    Configure the DSPy language model.

    Args:
        provider: LLM provider (e.g., 'openai', 'anthropic', 'together')
        model: Model name (e.g., 'gpt-4o-mini', 'claude-3-5-sonnet-20241022')
        api_key: Optional API key (defaults to environment variable)
        **kwargs: Additional arguments for the LM

    Returns:
        Configured DSPy LM instance
    """
    # Construct the model string
    if "/" in model:
        model_string = model
    else:
        model_string = f"{provider}/{model}"

    # Create and configure the LM
    lm = dspy.LM(model_string, **kwargs)
    dspy.configure(lm=lm)

    return lm


def get_default_lm():
    """
    Get the default language model configuration.
    Uses OpenAI GPT-5-mini by default with retry logic for rate limits.

    Set OPENAI_API_KEY environment variable before running.
    """
    return configure_lm(
        provider="openai",
        model="gpt-5-mini",
        num_retries=5,  # Retry up to 5 times on rate limit errors
        timeout=60.0    # 60 second timeout per request
    )


# Example configurations for different providers
PROVIDER_CONFIGS = {
    "openai": {
        "model": "gpt-5-mini",
        "env_var": "OPENAI_API_KEY"
    },
    "anthropic": {
        "model": "claude-3-5-sonnet-20241022",
        "env_var": "ANTHROPIC_API_KEY"
    },
    "together": {
        "model": "meta-llama/Llama-3-70b-chat-hf",
        "env_var": "TOGETHER_API_KEY"
    }
}
