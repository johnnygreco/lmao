from lmao.adapters import TextClassificationAdapter
from lmao.lm.clients.base import SUCCESS_STATUS_CODE
from lmao.tasks.base import TaskResponse, task_errors

__all__ = ["TextClassification"]


class TextClassification:
    def __init__(self, adapter: TextClassificationAdapter):
        self.categories = adapter.categories
        self.adapter: TextClassificationAdapter = adapter

    def predict(self, text: str, **kwargs) -> TaskResponse:
        success = True
        input_text = self.adapter.prompter.create_prompt(text)
        kwargs.update(self.adapter.to_endpoint_kwargs(input_text))
        response = getattr(self.adapter.client, self.adapter.endpoint_method_name)(**kwargs)
        if response.status_code == SUCCESS_STATUS_CODE:
            prediction = response.text.strip().lower() if self.adapter.lowercase else response.text.strip()
            prediction = prediction.replace(".", "")
            if prediction not in self.categories:
                prediction = task_errors.PREDICTION_ERROR
                success = False
        else:
            prediction = task_errors.CLIENT_ERROR
            success = False
        return TaskResponse(prediction=prediction, client_response=response, success=success)
