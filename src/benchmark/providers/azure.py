"""Azure AI Foundry provider for sequential inference.

Azure AI Foundry endpoints use the OpenAI-compatible format:
  https://<resource>.services.ai.azure.com/openai/v1/

This is different from the legacy Azure OpenAI Service (*.openai.azure.com),
which requires AsyncAzureOpenAI. Foundry endpoints work with the standard
AsyncOpenAI client using base_url + api_key.
"""

import os

from openai import AsyncOpenAI

from benchmark.config import ModelEntry
from benchmark.exceptions import FatalBenchmarkError
from benchmark.types import GenerateResult
from benchmark.utils import api_retry, openai_compat_generate


@api_retry()
async def azure_generate(
    model: ModelEntry, system_prompt: str, user_message: str
) -> GenerateResult:
    """Generate response using an Azure AI Foundry endpoint.

    Requires:
      - base_url: full Azure AI Foundry endpoint ending in /openai/v1/
                  e.g. https://<resource>.services.ai.azure.com/openai/v1/
      - api_key_env: env var holding your key (defaults to AZURE_OPENAI_API_KEY)
      - name: the deployment name (e.g. DeepSeek-V3.2)
    """
    if not model.base_url:
        raise FatalBenchmarkError("base_url is required for azure provider")

    api_key_env = model.api_key_env or "AZURE_OPENAI_API_KEY"
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise FatalBenchmarkError(f"{api_key_env} environment variable not set")

    async with AsyncOpenAI(base_url=model.base_url, api_key=api_key) as client:
        return await openai_compat_generate(client, model, system_prompt, user_message)
