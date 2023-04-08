from typing import List

from lmao.adapters.base import AdapterResponse, BaseAdapter, adapter_errors
from lmao.lm.clients.base import SUCCESS_STATUS_CODE, BaseClient
from lmao.lm.prompts.classification import ClassificationPrompter, SentimentAnalysisPrompter

__all__ = ["TextClassificationAdapter", "SentimentAnalysisAdapter"]


class TextClassificationAdapter(BaseAdapter):
    def __init__(self, lm: BaseClient, lm_method_name: str, categories: List[str], lowercase: bool = True, **kwargs):
        self.lowercase = lowercase
        self.categories = [c.lower() for c in categories] if lowercase else categories
        prompter = kwargs.pop("prompter") or ClassificationPrompter(categories=self.categories, **kwargs)
        super().__init__(lm=lm, lm_method_name=lm_method_name, prompter=prompter)

    def predict(self, text: str) -> AdapterResponse:
        response = getattr(self.lm, self.lm_method_name)(self.prompter.create_prompt(text))
        success = True
        if response.status_code == SUCCESS_STATUS_CODE:
            prediction = response.text.strip().lower() if self.lowercase else response.text.strip()
            prediction = prediction.replace(".", "")
            if prediction not in self.categories:
                prediction = adapter_errors.PREDICTION_ERROR
                success = False
        else:
            prediction = adapter_errors.CLIENT_ERROR
            success = False
        return AdapterResponse(prediction=prediction, llm_response=response, success=success)


class SentimentAnalysisAdapter(TextClassificationAdapter):
    def __init__(self, lm: BaseClient, lm_method_name: str, **kwargs):
        super().__init__(
            lm=lm,
            lm_method_name=lm_method_name,
            categories=["positive", "negative", "neutral"],
            lowercase=True,
            prompter=SentimentAnalysisPrompter(**kwargs),
        )
