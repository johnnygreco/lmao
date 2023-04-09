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
    llm_response: ClientResponse
    success: bool


class TaskAdapter(ABC):
    def __init__(self, lm: BaseClient, lm_method_name: str, prompter: Prompter):
        self.lm = lm
        self.prompter = prompter
        self.lm_method_name = lm_method_name

    @abstractmethod
    def predict(self, text: str) -> TaskAdapterResponse:
        pass


adapter_errors = TaskAdapterErrors(CLIENT_ERROR="CLIENT ERROR", PREDICTION_ERROR="PREDICTION ERROR")
