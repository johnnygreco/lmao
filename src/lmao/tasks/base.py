from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import NamedTuple

from lmao.adapters.base import BaseAdapter
from lmao.lm.clients.base import ClientResponse

__all__ = ["adapter_errors", "BaseTask"]


class TaskErrors(NamedTuple):
    CLIENT_ERROR: str
    PREDICTION_ERROR: str


@dataclass
class TaskResponse:
    prediction: str
    client_response: ClientResponse
    success: bool


class BaseTask(ABC):
    def __init__(self, adapter: BaseAdapter):
        self.adapter = adapter

    @abstractmethod
    def predict(self, text: str) -> TaskResponse:
        pass


adapter_errors = TaskErrors(CLIENT_ERROR="CLIENT ERROR", PREDICTION_ERROR="PREDICTION ERROR")
