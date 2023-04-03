from collections import deque
from typing import Deque, Dict, NamedTuple, Optional

from lmao.lm.clients.base import SUCCESS_STATUS_CODE, BaseClientResponse, Client
from lmao.lm.schemas.openai import OpenAIChatSchema, OpenAIGenerateSchema

__all__ = ["OpenAI"]


DEFAULT_SYSTEM_MESSAGE = "You are a helpful assistant."


class Schema(NamedTuple):
    generate: dict


class OpenAI(Client):
    base_url = "https://api.openai.com/v1"
    api_env_name = "OPENAI_API_KEY"

    schema = Schema(generate=OpenAIGenerateSchema.schema()["properties"])

    def __init__(self, api_key: Optional[str] = None, chat_history_length: int = 5):
        super().__init__(api_key)
        self.chat_history_length = chat_history_length
        self.chat_history: Deque[Dict[str, str]] = deque(maxlen=chat_history_length)

    def clear_chat_history(self):
        self.chat_history.clear()

    def chat(
        self, message_content: str, is_user: bool = True, system_message: str = DEFAULT_SYSTEM_MESSAGE, **kwargs
    ) -> BaseClientResponse:
        self.chat_history.append({"role": "user" if is_user else "assistant", "content": message_content})
        messages = [{"role": "system", "content": system_message}] + list(self.chat_history)
        status_code, response = self._post_request(
            "chat/completions", OpenAIChatSchema(messages=messages, **kwargs).to_request_dict()
        )
        assistant_message = response["choices"][0]["message"]["content"] if status_code == SUCCESS_STATUS_CODE else None
        if assistant_message:
            self.chat_history.append({"role": "assistant", "content": assistant_message})
        return BaseClientResponse(
            text=assistant_message,
            raw_response=response,
            status_code=status_code,
        )

    def generate(self, prompt: str, **kwargs) -> BaseClientResponse:
        status_code, response = self._post_request(
            "completions", OpenAIGenerateSchema(prompt=prompt, **kwargs).to_request_dict()
        )
        return BaseClientResponse(
            text=response["choices"][0]["text"] if status_code == SUCCESS_STATUS_CODE else None,
            raw_response=response,
            status_code=status_code,
        )
