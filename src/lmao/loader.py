from lmao.lm.clients.openai import OpenAI
from lmao.orchestrators import SentimentAnalysisOrchestrator

__all__ = ["openai_sentiment_analysis_orchestrator"]


def openai_sentiment_analysis_orchestrator(include_neutral: bool = True, **kwargs):
    return SentimentAnalysisOrchestrator(lm=OpenAI(**kwargs), lm_method_name="chat", include_neutral=include_neutral)
