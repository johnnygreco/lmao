from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import Extra, Field

from lmao.lm.schemas.base import API_DEFAULTS, BaseSchema

__all__ = [
    "OpenAIChatModels",
    "OpenAIGenerateModels",
    "OpenAIGenerateSchema",
]


class OpenAIChatModels(str, Enum):
    GPT_4 = "gpt-4"
    GPT_4_0314 = "gpt-4-0314"
    GPT_3p5_TURBO = "gpt-3.5-turbo"
    GPT_3p5_TURBO_0301 = "gpt-3.5-turbo-0301"


class OpenAIGenerateModels(str, Enum):
    TEXT_DAVINCI_003 = "text-davinci-003"


class OpenAIGenerateSchema(BaseSchema):
    """API Reference: https://platform.openai.com/docs/api-reference/completions/create"""

    class Config:
        extra = Extra.forbid

    model: str = OpenAIGenerateModels.TEXT_DAVINCI_003
    prompt: Optional[str] = Field(default=None)
    suffix: Optional[str] = Field(default=None)
    max_tokens: int = Field(default=API_DEFAULTS.generate_max_tokens, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    n: int = Field(default=1)
    stream: bool = Field(default=False)
    logprobs: Optional[int] = Field(default=None)
    echo: bool = Field(default=False)
    stop: Optional[Union[str, List[str]]] = Field(default=None)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    best_of: int = Field(default=1)
    logit_bias: Optional[Dict[str, int]] = Field(default=None, ge=-100, le=100)
    user: Optional[str] = Field(default=None)
