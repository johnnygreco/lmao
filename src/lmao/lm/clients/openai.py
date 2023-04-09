from typing import Literal, NamedTuple, Optional

from lmao.lm.clients.base import SUCCESS_STATUS_CODE, BaseClient, ChatHistory, ClientResponse
from lmao.lm.schemas.openai import OpenAIChatSchema, OpenAIGenerateSchema

__all__ = ["OpenAI"]


DEFAULT_SYSTEM_MESSAGE = "You are a helpful assistant."


class Schema(NamedTuple):
    complete: dict
    chat: dict


class OpenAIChatHistory(ChatHistory):
    def append(self, role: str, content: str):
        message = self.check_message_format({"role": role, "content": content})
        self._messages.append(message)

    def check_message_format(self, message):
        if not isinstance(message, dict):
            raise ValueError(f"Message must be a dict, not {type(message)}.")
        if "role" not in message:
            raise ValueError("Message must have a 'role' key.")
        if "content" not in message:
            raise ValueError("Message must have a 'content' key.")
        if message["role"] not in ["user", "assistant"]:
            raise ValueError("Message role must be 'user' or 'assistant'.")
        return message


class OpenAI(BaseClient):
    base_url = "https://api.openai.com/v1"
    api_env_name = "OPENAI_API_KEY"

    schema = Schema(complete=OpenAIGenerateSchema.schema()["properties"], chat=OpenAIChatSchema.schema()["properties"])

    def __init__(self, api_key: Optional[str] = None, chat_history_length: int = 5):
        super().__init__(api_key)
        self.chat_history = OpenAIChatHistory(chat_history_length)

    def chat(
        self,
        message_content: str,
        role: Literal["user", "assistant"] = "user",
        system_message: str = DEFAULT_SYSTEM_MESSAGE,
        **kwargs,
    ) -> ClientResponse:
        self.chat_history.append(role=role, content=message_content)
        messages = [{"role": "system", "content": system_message}] + list(self.chat_history)
        status_code, response = self._post_request(
            "chat/completions", OpenAIChatSchema(messages=messages, **kwargs).to_request_dict()
        )
        assistant_message = response["choices"][0]["message"]["content"] if status_code == SUCCESS_STATUS_CODE else None
        if assistant_message:
            self.chat_history.append(role="assistant", content=assistant_message)
        return ClientResponse(
            text=assistant_message,
            raw_response=response,
            status_code=status_code,
        )

    def complete(self, prompt: str, **kwargs) -> ClientResponse:
        status_code, response = self._post_request(
            "completions", OpenAIGenerateSchema(prompt=prompt, **kwargs).to_request_dict()
        )
        return ClientResponse(
            text=response["choices"][0]["text"] if status_code == SUCCESS_STATUS_CODE else None,
            raw_response=response,
            status_code=status_code,
        )
