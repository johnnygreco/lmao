from typing import Optional

from lmao.adapters.base import BaseChatbotAdapter
from lmao.lm.clients.anthropic import AnthropicChatHistory, AnthropicClient
from lmao.lm.clients.base import ClientResponse
from lmao.lm.clients.openai import OpenAIChatHistory, OpenAIClient

__all__ = ["AnthropicChatbotAdapter", "OpenAIChatbotAdapter"]


class AnthropicChatbotAdapter(BaseChatbotAdapter):
    def __init__(self, api_key: Optional[str] = None, chat_history_length: int = 5):
        super().__init__(
            client=AnthropicClient(api_key),
            endpoint_method_name="complete",
            chat_history=AnthropicChatHistory(max_length=chat_history_length),
        )

    def postprocess_response(self, response: ClientResponse) -> ClientResponse:
        if response.text is not None:
            response.text = response.text.strip()
        return response

    def to_endpoint_kwargs(self, request: str) -> dict:
        return {"prompt": request}


class OpenAIChatbotAdapter(BaseChatbotAdapter):
    def __init__(self, api_key: Optional[str] = None, chat_history_length: int = 5):
        super().__init__(
            client=OpenAIClient(api_key),
            endpoint_method_name="chat",
            chat_history=OpenAIChatHistory(max_length=chat_history_length),
        )

    def postprocess_response(self, response: ClientResponse) -> ClientResponse:
        if response.text is not None:
            response.text = response.text.strip()
        return response

    def to_endpoint_kwargs(self, request) -> dict:
        return {"messages": request}
