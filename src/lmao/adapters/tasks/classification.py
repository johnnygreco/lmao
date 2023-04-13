from typing import List

from lmao.adapters.tasks.base import TaskAdapter, TaskAdapterResponse, adapter_errors
from lmao.lm.clients.base import SUCCESS_STATUS_CODE, BaseClient
from lmao.lm.prompts.classification import ClassificationPrompter, SentimentAnalysisPrompter

__all__ = ["TextClassificationAdapter", "SentimentAnalysisAdapter"]


class TextClassificationAdapter(TaskAdapter):
    def __init__(
        self, lm_client: BaseClient, client_method_name: str, categories: List[str], lowercase: bool = True, **kwargs
    ):
        self.lowercase = lowercase
        self.categories = [c.lower() for c in categories] if lowercase else categories
        prompter = kwargs.pop("prompter", None) or ClassificationPrompter(categories=self.categories, **kwargs)
        super().__init__(lm_client=lm_client, client_method_name=client_method_name, prompter=prompter)

    def predict(self, text: str, **kwargs) -> TaskAdapterResponse:
        response = getattr(self.lm_client, self.client_method_name)(self.prompter.create_prompt(text), **kwargs)
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
        return TaskAdapterResponse(prediction=prediction, lm_response=response, success=success)


class SentimentAnalysisAdapter(TextClassificationAdapter):
    def __init__(self, lm_client: BaseClient, client_method_name: str, include_neutral: bool = True, **kwargs):
        super().__init__(
            lm_client=lm_client,
            client_method_name=client_method_name,
            categories=["positive", "negative"] + (["neutral"] if include_neutral else []),
            lowercase=True,
            prompter=SentimentAnalysisPrompter(include_neutral=include_neutral, **kwargs),
        )
