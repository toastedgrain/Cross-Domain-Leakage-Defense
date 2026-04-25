"""Generic OpenAI-compatible provider for arbitrary endpoints."""

import os

from openai import AsyncOpenAI

from benchmark.config import ModelEntry
from benchmark.exceptions import FatalBenchmarkError
from benchmark.types import GenerateResult
from benchmark.utils import api_retry, openai_compat_generate


@api_retry()
async def openai_compatible_generate(
    model: ModelEntry, system_prompt: str, user_message: str
) -> GenerateResult:
    """Generate response using any OpenAI-compatible API."""
    if not model.base_url:
        raise FatalBenchmarkError("base_url is required for openai_compatible provider")

    api_key_env = model.api_key_env or "OPENAI_API_KEY"
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise FatalBenchmarkError(f"{api_key_env} environment variable not set")

    async with AsyncOpenAI(base_url=model.base_url, api_key=api_key) as client:
        return await openai_compat_generate(client, model, system_prompt, user_message)
