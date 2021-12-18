from abc import ABC, abstractmethod
from typing import List


class BaseAlgorithm(ABC):
    @abstractmethod
    def do_algorithm(self):
        pass


class Context:
    def __init__(self, strategy: BaseAlgorithm):
        self._strategy = strategy

    @property
    def strategy(self) -> BaseAlgorithm:
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: BaseAlgorithm):
        self._strategy = strategy

    def apply_algorithm(self):
        result = self._strategy.do_algorithm()
