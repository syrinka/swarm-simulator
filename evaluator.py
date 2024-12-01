import numpy as np
from swarm.base import ArgInfo, Problem, Solution
from abc import ABC, abstractmethod


class Evaluator(ABC):
    @staticmethod
    @abstractmethod
    def infer(x: Solution) -> float:
        ...

    @classmethod
    def getproblem(cls, nargs: int) -> Problem:
        return Problem([ArgInfo()] * nargs, cls.infer)


class Sphere(Evaluator):
    @staticmethod
    def infer(x: Solution):
        return np.sum(x ** 2).astype(float)


class Ackley(Evaluator):
    @staticmethod
    def infer(x: Solution):
        a = 20
        b = 0.2
        c = 2 * np.pi
        d = len(x)
        return -a * np.exp(-b * np.sqrt(np.sum(x ** 2) / d)) - np.exp(np.sum(np.cos(c * x)) / d) + a + np.e
