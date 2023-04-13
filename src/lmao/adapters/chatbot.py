from lmao.lm.clients import BaseClient, ChatHistory
from lmao.lm.clients.base import SUCCESS_STATUS_CODE

__all__ = ["Chatbot"]


class Chatbot:
    def __init__(self, lm_client: BaseClient, chat_history: ChatHistory, client_method_name: str):
        self.lm_client = lm_client
        self.history = chat_history
        self.client_method_name = client_method_name

    def chat(self, message: str, **kwargs) -> str:
        self.history.add_human_message(message)
        response = getattr(self.lm_client, self.client_method_name)(self.history.to_request_format(), **kwargs)
        if response.status_code == SUCCESS_STATUS_CODE:
            self.history.add_assistant_message(response.text)
        return response
