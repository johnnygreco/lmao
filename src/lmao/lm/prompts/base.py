from abc import ABC, abstractmethod
from typing import Optional

__all__ = ["default_templates", "Prompter"]

default_templates = {
    "classification": (
        "Classify the input text into one of the following {num_categories} categories: [{categories}]. "
        "Respond with the answer only, without punctuation.\n\n"
        "{examples}"
        "Input: {input_text}\nCategory:"
    ),
    "sentiment_analysis": (
        "Classify the sentiment of the input text into one of the following "
        "{num_categories} categories: [{categories}]. Respond with the answer only, without punctuation.\n\n"
        "{examples}"
        "Input: {input_text}\nSentiment:"
    ),
}


class Prompter(ABC):
    task: str = "none"

    def __init__(self, template: Optional[str] = None):
        if self.task not in default_templates:
            raise ValueError(f"There are no default prompts for task {self.task}.")
        self.template = template or default_templates[self.task]

    @abstractmethod
    def create_prompt(self, input_text: str) -> str:
        pass
