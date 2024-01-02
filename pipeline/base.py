from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseRunner(ABC):
    @abstractmethod
    def run(self, *args, **kwargs):
        pass


class BasePipeline(ABC):
    @property
    @abstractmethod
    def default_parameters(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def preprocess(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass
