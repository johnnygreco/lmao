from typing import List, Optional

from .base import Prompter

__all__ = ["ClassificationPrompter"]


class ClassificationPrompter(Prompter):
    task = "classification"

    def __init__(
        self,
        categories: List[str],
        examples: Optional[List[str]] = None,
        template: Optional[str] = None,
    ):
        self.categories = categories
        self.examples = examples or []
        super().__init__(template=template)

    @staticmethod
    def create_example(input_text: str, category: str) -> str:
        return f"Input: {input_text}\nCategory: {category}\n\n"

    def add_example(self, input_text: str, category: str):
        if category not in self.categories:
            raise ValueError(f"Category {category} is not in the list of categories.")
        self.examples.append(self.create_example(input_text, category))

    def create_prompt(self, input_text: str) -> str:
        return self.template.format(
            num_categories=len(self.categories),
            categories=", ".join(self.categories),
            input_text=input_text,
            examples="".join(self.examples) if self.examples else "",
        )


class SentimentAnalysisPrompter(ClassificationPrompter):
    task = "sentiment_analysis"

    def __init__(
        self,
        examples: Optional[List[str]] = None,
        template: Optional[str] = None,
        include_neutral: bool = True,
    ):
        super().__init__(
            categories=["positive", "negative"] + (["neutral"] if include_neutral else []),
            examples=examples,
            template=template,
        )
