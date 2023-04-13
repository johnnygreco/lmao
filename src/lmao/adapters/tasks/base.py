from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import NamedTuple

from lmao.lm.clients.base import BaseClient, ClientResponse
from lmao.lm.prompts.base import Prompter

__all__ = ["adapter_errors", "TaskAdapter"]


class TaskAdapterErrors(NamedTuple):
    CLIENT_ERROR: str
    PREDICTION_ERROR: str


@dataclass
class TaskAdapterResponse:
    prediction: str
    lm_response: ClientResponse
    success: bool


class TaskAdapter(ABC):
    def __init__(self, lm_client: BaseClient, client_method_name: str, prompter: Prompter):
        self.lm_client = lm_client
        self.prompter = prompter
        self.client_method_name = client_method_name

    @abstractmethod
    def predict(self, text: str) -> TaskAdapterResponse:
        pass


adapter_errors = TaskAdapterErrors(CLIENT_ERROR="CLIENT ERROR", PREDICTION_ERROR="PREDICTION ERROR")
