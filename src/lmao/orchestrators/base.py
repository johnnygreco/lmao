from abc import ABC, abstractmethod

__all__ = ["BaseOrchestrator"]


class BaseOrchestrator(ABC):
    @abstractmethod
    def run_pipeline(self, *args, **kwargs):
        pass
