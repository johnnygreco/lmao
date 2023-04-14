from abc import ABC, abstractmethod

from lmao.lm.clients.base import BaseClient, ChatHistory, ClientResponse
from lmao.prompters.base import Prompter

__all__ = ["BaseAdapter", "BaseChatbotAdapter", "BaseTaskAdapter"]


class BaseAdapter(ABC):
    def __init__(self, client: BaseClient, endpoint_method_name: str):
        self.client = client
        self.endpoint_method_name = endpoint_method_name

    def postprocess_response(self, response: ClientResponse) -> ClientResponse:
        return response

    @abstractmethod
    def to_endpoint_kwargs(self, request) -> dict:
        pass


class BaseChatbotAdapter(BaseAdapter):
    def __init__(self, client: BaseClient, endpoint_method_name: str, chat_history: ChatHistory):
        self.chat_history = chat_history
        super().__init__(client=client, endpoint_method_name=endpoint_method_name)


class BaseTaskAdapter(BaseAdapter):
    def __init__(self, client: BaseClient, endpoint_method_name: str, prompter: Prompter):
        self.prompter = prompter
        super().__init__(client=client, endpoint_method_name=endpoint_method_name)
