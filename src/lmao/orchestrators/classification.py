from typing import Iterable, Union

from lmao.adapters import SentimentAnalysisAdapter
from lmao.lm.clients.base import BaseClient

__all__ = ["SentimentAnalysisOrchestrator"]


class SentimentAnalysisOrchestrator:
    def __init__(self, lm: BaseClient, lm_method_name: str, include_neutral: bool = True, **kwargs):
        self.lm_method_name = lm_method_name
        self.clf = SentimentAnalysisAdapter(lm, lm_method_name, include_neutral=include_neutral)
        self.categories = ["positive", "negative"] + (["neutral"] if include_neutral else [])

    def predict(self, data: Union[Iterable, str], **kwargs):
        temperature = kwargs.pop("temperature", 0)
        data = [data] if isinstance(data, str) else data
        predictions = []
        for text in data:
            predictions.append(self.clf.predict(text, temperature=temperature, **kwargs).prediction)
            if hasattr(self.clf.lm, "chat_history"):
                self.clf.lm.chat_history.clear()
        return predictions
