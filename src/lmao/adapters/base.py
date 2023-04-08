from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import NamedTuple

from lmao.lm.clients.base import BaseClient, BaseClientResponse
from lmao.lm.prompts.base import Prompter

__all__ = ["adapter_errors", "BaseAdapter"]


class AdapterErrors(NamedTuple):
    CLIENT_ERROR: str
    PREDICTION_ERROR: str


@dataclass
class AdapterResponse:
    prediction: str
    llm_response: BaseClientResponse


class BaseAdapter(ABC):
    def __init__(self, lm: BaseClient, lm_method_name: str, prompter: Prompter):
        self.lm = lm
        self.prompter = prompter
        self.lm_method_name = lm_method_name

    @abstractmethod
    def predict(self, text: str) -> AdapterResponse:
        pass


adapter_errors = AdapterErrors(CLIENT_ERROR="CLIENT ERROR", PREDICTION_ERROR="PREDICTION ERROR")
