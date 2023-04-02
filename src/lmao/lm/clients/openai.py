import os
from typing import NamedTuple, Optional

from lmao.lm.clients.base import SUCCESS_STATUS_CODE, BaseGenerateResponse, Client
from lmao.lm.schemas.openai import OpenAIGenerateSchema

__all__ = ["OpenAI"]


class Schema(NamedTuple):
    generate: dict


class OpenAI(Client):
    base_url = "https://api.openai.com/v1"

    schema = Schema(generate=OpenAIGenerateSchema.schema()["properties"])

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key or os.environ.get("OPENAI_API_KEY"))

    def chat(self, prompt: str, **kwargs) -> str:
        return ""

    def generate(self, prompt: str, **kwargs) -> BaseGenerateResponse:
        status_code, response = self._post_request(
            "completions", OpenAIGenerateSchema(prompt=prompt, **kwargs).to_request_dict()
        )
        return BaseGenerateResponse(
            text=response["choices"][0]["text"] if status_code == SUCCESS_STATUS_CODE else None,
            raw_response=response,
            status_code=status_code,
        )
