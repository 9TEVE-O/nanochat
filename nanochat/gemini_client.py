"""Gemini API client — optional inference backend for nanochat.

Requires the gemini extra: pip install nanochat[gemini]

Usage:
    export GEMINI_API_KEY=your_key
    from nanochat.gemini_client import GeminiClient
    client = GeminiClient()
    response = client.complete("Explain how AI works in a few words")
    print(response)
"""

import os

try:
    from google import genai as _genai
except ImportError:
    _genai = None

DEFAULT_MODEL = "gemini-2.5-flash"


class GeminiClient:
    """Thin wrapper around google-genai for text completion."""

    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None):
        if _genai is None:
            raise ImportError(
                "google-genai is not installed. Run: pip install nanochat[gemini]"
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None):
        if genai is None:
            raise ImportError(
                "The 'google-genai' package is required to use GeminiClient. "
                "Please install it with: pip install 'nanochat[gemini]'"
            )
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError(
                "No Gemini API key found. Set the GEMINI_API_KEY environment variable "
                "or pass api_key= to GeminiClient()."
            )
        self._client = _genai.Client(api_key=key)
        self.model = model

    def complete(self, prompt: str) -> str:
        """Return a single text completion for the given prompt."""
        response = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
        )
        return response.text
