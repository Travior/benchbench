"""
Model configuration and instructor client access via litellm.

Usage:
    from benchbench.models import get_async_client, Model

    client = get_async_client()
    response = await client.chat.completions.create(
        model=Model.GPT_4O,
        response_model=MySchema,
        messages=[...]
    )

API keys are read from environment variables:
    - OPENAI_API_KEY for OpenAI models
    - ANTHROPIC_API_KEY for Anthropic models
    - OPENROUTER_API_KEY for OpenRouter models
    - etc. (see litellm docs for full list)
"""

from enum import StrEnum

import instructor
import litellm


class Model(StrEnum):
    # OpenRouter
    GPT_51_NANO_OR = "openrouter/openai/gpt-5-nano"


def get_async_client() -> instructor.AsyncInstructor:
    """Get an async instructor-wrapped litellm client."""
    return instructor.from_litellm(litellm.acompletion)
