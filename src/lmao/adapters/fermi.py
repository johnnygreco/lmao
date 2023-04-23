from typing import Optional

from lmao.adapters import AnthropicAdapterMixin, BaseTaskAdapter, CohereAdapterMixin, OpenAIAdapterMixin
from lmao.prompters import FermiProblemPrompter

__all__ = [
    "FermiProblemAdapter",
    "AnthropicFermiProblemAdapter",
    "CohereFermiProblemAdapter",
    "OpenAIFermiProblemAdapter",
]


class FermiProblemAdapter(BaseTaskAdapter):
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(api_key=api_key, prompter=FermiProblemPrompter(), **kwargs)


class AnthropicFermiProblemAdapter(AnthropicAdapterMixin, FermiProblemAdapter):
    """Adapter for solving Fermi problems with Anthropic."""


class CohereFermiProblemAdapter(CohereAdapterMixin, FermiProblemAdapter):
    """Adapter for solving Fermi problems with Cohere."""


class OpenAIFermiProblemAdapter(OpenAIAdapterMixin, FermiProblemAdapter):
    """Adapter for solving Fermi problems with OpenAI."""
