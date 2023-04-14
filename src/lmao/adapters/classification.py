from typing import List, Optional

from lmao.adapters.base import BaseTaskAdapter
from lmao.lm.clients.anthropic import AnthropicClient
from lmao.lm.clients.base import BaseClient, ClientResponse
from lmao.lm.clients.openai import OpenAIClient
from lmao.prompters.classification import ClassificationPrompter, SentimentAnalysisPrompter

__all__ = [
    "AnthropicSentimentAnalysisAdapter",
    "AnthropicTextClassificationAdapter",
    "OpenAISentimentAnalysisAdapter",
    "OpenAITextClassificationAdapter",
    "SentimentAnalysisAdapter",
    "TextClassificationAdapter",
]


class TextClassificationAdapter(BaseTaskAdapter):
    def __init__(
        self, client: BaseClient, endpoint_method_name: str, categories: List[str], lowercase: bool = True, **kwargs
    ):
        self.lowercase = lowercase
        self.categories = [c.lower() for c in categories] if lowercase else categories
        prompter = kwargs.pop("prompter", None) or ClassificationPrompter(categories=self.categories)
        super().__init__(client=client, endpoint_method_name=endpoint_method_name, prompter=prompter)


class SentimentAnalysisAdapter(TextClassificationAdapter):
    def __init__(self, client: BaseClient, endpoint_method_name: str, include_neutral: bool = True, **kwargs):
        super().__init__(
            client=client,
            endpoint_method_name=endpoint_method_name,
            categories=["positive", "negative"] + (["neutral"] if include_neutral else []),
            lowercase=True,
            prompter=kwargs.pop("prompter", None) or SentimentAnalysisPrompter(include_neutral=include_neutral),
            **kwargs,
        )


class AnthropicClassificationMixin:
    def postprocess_response(self, response: ClientResponse) -> ClientResponse:
        if response.text is not None:
            response.text = response.text.strip()
        return response

    def to_endpoint_kwargs(self, request) -> dict:
        return {"prompt": request, "stop_sequences": ["\n\n"]}


class AnthropicTextClassificationAdapter(AnthropicClassificationMixin, TextClassificationAdapter):
    def __init__(self, categories: List[str], lowercase: bool = True, api_key: Optional[str] = None, **kwargs):
        super().__init__(
            client=AnthropicClient(api_key),
            endpoint_method_name="complete",
            categories=categories,
            lowercase=lowercase,
            **kwargs,
        )


class AnthropicSentimentAnalysisAdapter(AnthropicClassificationMixin, SentimentAnalysisAdapter):
    def __init__(self, include_neutral: bool = True, api_key: Optional[str] = None, **kwargs):
        super().__init__(
            client=AnthropicClient(api_key),
            endpoint_method_name="complete",
            include_neutral=include_neutral,
            **kwargs,
        )


class OpenAIClassificationMixin:
    def postprocess_response(self, response: ClientResponse) -> ClientResponse:
        if response.text is not None:
            response.text = response.text.strip()
        return response

    def to_endpoint_kwargs(self, request) -> dict:
        return {"messages": [{"role": "user", "content": request}]}


class OpenAITextClassificationAdapter(OpenAIClassificationMixin, TextClassificationAdapter):
    def __init__(self, categories: List[str], lowercase: bool = True, api_key: Optional[str] = None, **kwargs):
        super().__init__(
            client=OpenAIClient(api_key),
            endpoint_method_name="chat",
            categories=categories,
            lowercase=lowercase,
            **kwargs,
        )


class OpenAISentimentAnalysisAdapter(OpenAIClassificationMixin, SentimentAnalysisAdapter):
    def __init__(self, include_neutral: bool = True, api_key: Optional[str] = None, **kwargs):
        super().__init__(
            client=OpenAIClient(api_key),
            endpoint_method_name="chat",
            include_neutral=include_neutral,
            **kwargs,
        )
