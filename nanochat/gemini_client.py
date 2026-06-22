"""Gemini API client — optional inference backend for nanochat.

Usage:
    export GEMINI_API_KEY=your_key
    from nanochat.gemini_client import GeminiClient
    client = GeminiClient()
    response = client.complete("Explain how AI works in a few words")
    print(response)
"""

import os
from google import genai

DEFAULT_MODEL = "gemini-2.5-flash"


class GeminiClient:
    """Thin wrapper around google-genai for text completion."""

    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None):
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError(
                "No Gemini API key found. Set the GEMINI_API_KEY environment variable "
                "or pass api_key= to GeminiClient()."
            )
        self._client = genai.Client(api_key=key)
        self.model = model

    def complete(self, prompt: str) -> str:
        """Return a single text completion for the given prompt."""
        response = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
        )
        return response.text
