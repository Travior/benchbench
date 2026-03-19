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
    GROK_41_FAST_OR = "openrouter/x-ai/grok-4.1-fast"
    SONNET_45_OR = "openrouter/anthropic/claude-sonnet-4.5"
    SONNET_46_OR = "openrouter/anthropic/claude-sonnet-4.6"
    GEMINI_3_OR = "openrouter/google/gemini-3-pro-preview"
    GEMINI_3_FLASH_OR = "openrouter/google/gemini-3-flash-preview"
    GEMINI_31_PRO_OR = "openrouter/google/gemini-3.1-pro-preview"
    GEMINI_31_FLASH_LITE_OR = "openrouter/google/gemini-3.1-flash-lite-preview"

    OPUS_45_OR = "openrouter/anthropic/claude-opus-4.5"
    OPUS_46_OR = "openrouter/anthropic/claude-opus-4.6"

    GPT_52_OR = "openrouter/openai/gpt-5.2"
    GPT_54_OR = "openrouter/openai/gpt-5.4"
    GPT_54_MINI_OR = "openrouter/openai/gpt-5.4-mini"

    GLM_47_OR = "openrouter/z-ai/glm-4.7"
    GLM_5_OR = "openrouter/z-ai/glm-5"

    MINIMAX_21_OR = "openrouter/minimax/minimax-m2.1"
    MINIMAX_25_OR = "openrouter/minimax/minimax-m2.5"

    QWEN_3_MAX = "openrouter/qwen/qwen3-max-thinking"


MODEL_CONFIG: dict[Model, dict] = {
    Model.GLM_47_OR: {
        "extra_body": {
            "provider": {
                "order": ["z-ai"],
                "allow_fallbacks": False,
            }
        }
    },
    Model.GLM_5_OR: {
        "extra_body": {
            "provider": {
                "order": ["z-ai"],
                "allow_fallbacks": False,
            }
        }
    },
    Model.GPT_54_OR: {"extra_body": {"reasoning": {"effort": "high"}}},
}


def get_model_config(model: Model) -> dict:
    """Get the model-specific configuration dict for a given model.

    Returns an empty dict if the model has no specific configuration.
    """
    return MODEL_CONFIG.get(model, {})


def get_async_client() -> instructor.AsyncInstructor:
    """Get an async instructor-wrapped litellm client."""
    return instructor.from_litellm(litellm.acompletion)
