from typing import Iterable, List, Union

from lmao.adapters import SentimentAnalysisAdapter
from lmao.lm.clients.base import BaseClient
from lmao.orchestrators.base import BaseOrchestrator

__all__ = ["SentimentAnalysisOrchestrator"]


class SentimentAnalysisOrchestrator(BaseOrchestrator):
    def __init__(self, lm_client: BaseClient, client_method_name: str, include_neutral: bool = True):
        self.client_method_name = client_method_name
        self.adapter = SentimentAnalysisAdapter(lm_client, client_method_name, include_neutral=include_neutral)
        self.categories = ["positive", "negative"] + (["neutral"] if include_neutral else [])
        super().__init__()

    def run_pipeline(self, data: Union[Iterable, str], **kwargs) -> List[str]:
        temperature = kwargs.pop("temperature", 0)
        data = [data] if isinstance(data, str) else data
        predictions = []
        for text in data:
            predictions.append(self.adapter.predict(text, temperature=temperature, **kwargs).prediction)
            if hasattr(self.adapter.lm_client, "chat_history"):
                self.adapter.lm_client.chat_history.clear()
        return predictions
