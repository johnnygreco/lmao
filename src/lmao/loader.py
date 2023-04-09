import inspect

import lmao.adapters as adapters
import lmao.orchestrators as orcs
from lmao.lm.clients.openai import OpenAI

__all__ = ["load_orchestrator", "load_task_adapter"]


default_lm_method = {"openai": "chat"}

provider_to_lm = {"openai": OpenAI}

task_to_adapter = {
    "sentiment_analysis": adapters.SentimentAnalysisAdapter,
}

task_to_orchestrator = {
    "sentiment_analysis": orcs.SentimentAnalysisOrchestrator,
}


def _validate_input(lm_provider: str, task: str):
    lm_provider = lm_provider.lower()
    task = task.lower().replace(" ", "_")
    if lm_provider not in default_lm_method or lm_provider not in provider_to_lm:
        raise ValueError(f"LM provider {lm_provider} not supported")
    if task not in task_to_orchestrator:
        raise ValueError(f"Task {task} not supported")
    return lm_provider, task


def _get_lm_kwargs(lm_provider: str, **kwargs):
    return {k: kwargs.pop(k) for k in inspect.signature(provider_to_lm[lm_provider]).parameters.keys() if k in kwargs}


def load_orchestrator(lm_provider: str, task: str, **kwargs) -> orcs.BaseOrchestrator:
    lm_provider, task = _validate_input(lm_provider, task)
    return task_to_orchestrator[task](
        lm=provider_to_lm[lm_provider](**_get_lm_kwargs(lm_provider, **kwargs)),
        lm_method_name=kwargs.pop("lm_method_name", default_lm_method[lm_provider]),
        **kwargs,
    )


def load_task_adapter(lm_provider: str, task: str, **kwargs) -> adapters.TaskAdapter:
    lm_provider, task = _validate_input(lm_provider, task)
    return task_to_adapter[task](
        lm=provider_to_lm[lm_provider](**_get_lm_kwargs(lm_provider, **kwargs)),
        lm_method_name=kwargs.pop("lm_method_name", default_lm_method[lm_provider]),
        **kwargs,
    )
