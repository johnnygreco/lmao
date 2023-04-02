import re
from typing import Dict, List, Optional, Union

from pydantic import Extra, Field, validator

from lmao.lm.schemas.base import API_DEFAULTS, BaseSchema

__all__ = ["OpenAIGenerateSchema"]


class OpenAIGenerateSchema(BaseSchema):
    """API Reference: https://platform.openai.com/docs/api-reference/completions/create"""

    class Config:
        extra = Extra.forbid

    model: str = Field(default="text-davinci-003", description="Must be of the form `text-davinci-[model_version]`.")
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

    @validator("model", pre=True, always=True)
    def validate_model(cls, v):
        if not re.search(r"text-davinci-\d\d\d", v):
            raise ValueError(f"{v} is not a valid model.")
        return v
