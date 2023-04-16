from typing import Optional

from lmao.adapters import BaseTaskAdapter
from lmao.lm.clients import AnthropicClient, BaseClient, ClientResponse
from lmao.lm.clients.cohere import CohereClient
from lmao.lm.clients.openai import OpenAIClient
from lmao.prompters import FermiProblemPrompter

__all__ = [
    "FermiProblemAdapter",
    "AnthropicFermiProblemAdapter",
    "CohereFermiProblemAdapter",
    "OpenAIFermiProblemAdapter",
]


class FermiProblemAdapter(BaseTaskAdapter):
    def __init__(self, client: BaseClient, endpoint_method_name: str):
        super().__init__(client, endpoint_method_name, prompter=FermiProblemPrompter())


class AnthropicFermiProblemAdapter(FermiProblemAdapter):
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(client=AnthropicClient(api_key), endpoint_method_name="complete")

    def postprocess_response(self, response: ClientResponse) -> ClientResponse:
        if response.text is not None:
            response.text = response.text.strip()
        return response

    def to_endpoint_kwargs(self, request) -> dict:
        return {"prompt": request}


class CohereFermiProblemAdapter(FermiProblemAdapter):
    def __init__(self, api_key: Optional[str] = None, api_version: str = "2022-12-06"):
        super().__init__(client=CohereClient(api_key, api_version=api_version), endpoint_method_name="complete")

    def postprocess_response(self, response: ClientResponse) -> ClientResponse:
        if response.text is not None:
            response.text = response.text.strip()
        return response

    def to_endpoint_kwargs(self, request) -> dict:
        return {"prompt": request}


class OpenAIFermiProblemAdapter(FermiProblemAdapter):
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(client=OpenAIClient(api_key), endpoint_method_name="chat")

    def postprocess_response(self, response: ClientResponse) -> ClientResponse:
        if response.text is not None:
            response.text = response.text.strip()
        return response

    def to_endpoint_kwargs(self, request) -> dict:
        return {"messages": [{"role": "user", "content": request}]}
