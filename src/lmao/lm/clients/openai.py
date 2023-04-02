import os
from typing import Optional

from .base import Client


class OpenAI(Client):
    base_url = "https://api.openai.com/v1"

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key or os.environ.get("OPENAI_API_KEY"))

    def chat(self, prompt: str, **kwargs) -> str:
        return ""

    def generate(self, prompt: str, **kwargs) -> str:
        request = {"prompt": prompt, "model": "text-davinci-003"}
        status_code, response = self._post_request("completions", request)
        return response["choices"][0]["text"]
