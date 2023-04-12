import inspect

import lmao.adapters as adapters
import lmao.orchestrators as orcs
from lmao.lm.clients import AnthropicClient, BaseClient, OpenAIClient

__all__ = ["load_orchestrator", "load_task_adapter"]


default_lm_method = {"openai": "chat"}

name_to_client = {"anthropic": AnthropicClient, "openai": OpenAIClient}

task_to_adapter = {
    "sentiment_analysis": adapters.SentimentAnalysisAdapter,
}

task_to_orchestrator = {
    "sentiment_analysis": orcs.SentimentAnalysisOrchestrator,
}


def _get_lm_kwargs(lm_provider: str, **kwargs):
    return {k: kwargs.pop(k) for k in inspect.signature(name_to_client[lm_provider]).parameters.keys() if k in kwargs}


def _validate_input(lm_client_name: str, task: str):
    lm_client_name = lm_client_name.lower()
    task = task.lower().replace(" ", "_")
    if lm_client_name not in default_lm_method or lm_client_name not in name_to_client:
        raise ValueError(f"LM provider {lm_client_name} not supported")
    if task not in task_to_orchestrator:
        raise ValueError(f"Task {task} not supported")
    return lm_client_name, task


def load_lm_client(lm_client_name: str, **kwargs) -> BaseClient:
    return name_to_client[lm_client_name](**kwargs)


def load_orchestrator(lm_client_name: str, task: str, **kwargs) -> orcs.BaseOrchestrator:
    lm_client_name, task = _validate_input(lm_client_name, task)
    return task_to_orchestrator[task](
        lm_client=name_to_client[lm_client_name](**_get_lm_kwargs(lm_client_name, **kwargs)),
        client_method_name=kwargs.pop("client_method_name", default_lm_method[lm_client_name]),
        **kwargs,
    )


def load_task_adapter(lm_client_name: str, task: str, **kwargs) -> adapters.TaskAdapter:
    lm_client_name, task = _validate_input(lm_client_name, task)
    return task_to_adapter[task](
        lm_client=name_to_client[lm_client_name](**_get_lm_kwargs(lm_client_name, **kwargs)),
        client_method_name=kwargs.pop("client_method_name", default_lm_method[lm_client_name]),
        **kwargs,
    )
