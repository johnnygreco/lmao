from lmao.adapters import FermiProblemAdapter
from lmao.lm.clients import ClientResponse

__all__ = ["FermiProblem"]


class FermiProblem:
    def __init__(self, adapter: FermiProblemAdapter):
        self.adapter = adapter

    def ask(self, question: str, **kwargs) -> ClientResponse:
        input_text = self.adapter.prompter.create_prompt(question)
        kwargs.update(self.adapter.to_endpoint_kwargs(input_text))
        response = getattr(self.adapter.client, self.adapter.endpoint_method_name)(**kwargs)
        response = self.adapter.postprocess_response(response)
        return response
