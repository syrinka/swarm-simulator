from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeAlias, Any, Callable, Literal, NamedTuple

import numpy as np
import numpy.typing as npt


Solution: TypeAlias = npt.NDArray[np.float64]
TargetFunction: TypeAlias = Callable[[Solution], float]


OptimizeGoal: TypeAlias = Literal['maximum', 'minimum', 'zero']
BoundaryStrategy: TypeAlias = Literal['saturate', 'wrap']


@dataclass(kw_only=True, frozen=True)
class ArgInfo(object):
    min: float = -np.inf
    max: float = np.inf
    grain: float | None = None
    underflow: BoundaryStrategy = 'saturate'
    overflow: BoundaryStrategy = 'saturate'


    def __post_init__(self):
        if self.max == self.min:
            raise ValueError()
        if self.overflow == 'wrap' and self.min == -np.inf:
            raise ValueError()
        if self.underflow == 'wrap' and self.max == np.inf:
            raise ValueError()


    def isvalid(self, val: float) -> bool:
        return self.min < val < self.max and \
            (self.grain is not None and val % self.grain != 0)


    def constrain(self, val: float) -> float:
        if val > self.max:
            if self.overflow == 'saturate':
                val = self.max
            else:
                val = val % (self.max - self.min) + self.min
        if val < self.min:
            if self.underflow == 'saturate':
                val = self.min
            else:
                val = self.max - val % (self.max - self.min)
        if self.grain is not None and val % self.grain != 0:
            val = round(val / self.grain) * self.grain
        return val


@dataclass(frozen=True)
class Problem(object):
    args: list[ArgInfo]
    func: TargetFunction
    goal: OptimizeGoal = 'minimum'

    def init_solutions(self, num: int, method = None) -> list[Solution]:
        solutions = []
        for _ in range(num):
            solutions.append(np.random.rand(len(self.args)))
        return solutions


class Snapshot(NamedTuple):
    epoch: int
    best_fitness: float
    best_solution: Solution


class Swarm(ABC):
    pops: int
    problem: Problem
    solutions: list[Solution]
    record: bool
    records: list[Snapshot]
    metavar: dict[str, Any] = {}

    epoch = 0

    def __init__(self, population: int, problem: Problem, seed: int | None = None, record: bool = True, **metavar):
        self.pops = population
        self.problem = problem
        if seed is not None:
            np.random.seed(seed)
        self.solutions = problem.init_solutions(population)
        self.record = record
        self.records = []
        self.metavar.update(metavar)
        self.post_init()


    def evolve(self, epochs: int = 1):
        for i in range(epochs):
            self.epoch += 1
            fits = []
            for n, sol in enumerate(self.solutions):
                out = self.problem.func(sol)
                fit = out
                fits.append(fit)
            if self.record:
                bestv = min(fits)
                best = self.solutions[fits.index(bestv)].copy()
                self.records.append(Snapshot(
                    self.epoch, bestv, best
                ))
            if i != epochs - 1:
                # not last round
                self.solutions = self.update(self.solutions, fits)


    @property
    def nargs(self) -> int:
        return len(self.problem.args)


    def __repr__(self) -> str:
        return f'<{self.__class__.__name__} pop={self.pops} metavar={self.metavar}>'


    def summary(self) -> str:
        return '\n'.join([
            f'Instance: {self}',
            f'Records:',
        ] + [
            f'  {self.records[i]}' for i in range(len(self.records))
        ])


    def fitness_history(self) -> list[float]:
        return [i.best_fitness for i in self.records]


    @abstractmethod
    def update(self, sols: list[Solution], fits: list[float]) -> list[Solution]:
        raise NotImplementedError()


    def post_init(self):
        pass
