from typing import List

from lmao.adapters.base import AdapterResponse, BaseAdapter, adapter_errors
from lmao.lm.clients.base import SUCCESS_STATUS_CODE, BaseClient
from lmao.lm.prompts.classification import ClassificationPrompter, SentimentAnalysisPrompter

__all__ = ["TextClassificationAdapter", "SentimentAnalysisAdapter"]


class TextClassificationAdapter(BaseAdapter):
    def __init__(self, lm: BaseClient, lm_method_name: str, categories: List[str], lowercase: bool = True, **kwargs):
        self.lowercase = lowercase
        self.categories = [c.lower() for c in categories] if lowercase else categories
        super().__init__(
            lm=lm, lm_method_name=lm_method_name, prompter=ClassificationPrompter(categories=self.categories, **kwargs)
        )

    def predict(self, text: str) -> AdapterResponse:
        response = getattr(self.lm, self.lm_method_name)(self.prompter.create_prompt(text))
        if response.status_code == SUCCESS_STATUS_CODE:
            prediction = response.text.strip().lower() if self.lowercase else response.text.strip()
            if prediction not in self.categories:
                prediction = adapter_errors.PREDICTION_ERROR
        else:
            prediction = adapter_errors.CLIENT_ERROR
        return AdapterResponse(prediction=prediction, llm_response=response)


class SentimentAnalysisAdapter(BaseAdapter):
    def __init__(self, lm: BaseClient, lm_method_name: str, **kwargs):
        self.categories = ["positive", "negative", "neutral"]
        super().__init__(lm=lm, lm_method_name=lm_method_name, prompter=SentimentAnalysisPrompter(**kwargs))

    def predict(self, text: str) -> AdapterResponse:
        response = getattr(self.lm, self.lm_method_name)(self.prompter.create_prompt(text))
        if response.status_code == SUCCESS_STATUS_CODE:
            prediction = response.text.strip().lower()
            if prediction not in self.categories:
                prediction = adapter_errors.PREDICTION_ERROR
        else:
            prediction = adapter_errors.CLIENT_ERROR
        return AdapterResponse(prediction=prediction, llm_response=response)
