import numpy as np
from swarm.base import ArgInfo, Problem, X
from abc import ABC, abstractmethod


class Evaluator(ABC):
    @staticmethod
    @abstractmethod
    def infer(x: X) -> float:
        ...

    @classmethod
    def getproblem(cls, nargs: int, **kwargs) -> Problem:
        return Problem([ArgInfo(**kwargs)] * nargs, cls.infer)


class Sphere(Evaluator):
    @staticmethod
    def infer(x: X):
        return np.sum(x ** 2).astype(float)


class Ackley(Evaluator):
    @staticmethod
    def infer(x: X):
        a = 20
        b = 0.2
        c = 2 * np.pi
        d = len(x)
        return -a * np.exp(-b * np.sqrt(np.sum(x ** 2) / d)) - np.exp(np.sum(np.cos(c * x)) / d) + a + np.e
