from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeAlias, Any, Callable, Literal, NamedTuple

import numpy as np
import numpy.typing as npt

from .utils import *


Solution: TypeAlias = npt.NDArray[np.float64]
TargetFunction: TypeAlias = Callable[[Solution], float]


OptimizeGoal: TypeAlias = Literal['maximum', 'minimum', 'zero']
BoundaryStrategy: TypeAlias = Literal['saturate', 'wrap']
InitializeStrategy: TypeAlias = Literal['random', 'lhs']


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

    def init_solutions(self, num: int, method: InitializeStrategy = 'random') -> list[Solution]:
        match method:
            case 'random':
                sols = []
                for _ in range(num):
                    tmp = []
                    for arg in self.args:
                        n = np.random.rand() * (arg.max - arg.min) + arg.min
                        tmp.append(n)
                    sols.append(np.array(tmp))
                return sols

            case 'lhs':
                # check if condition matches
                for arg in self.args:
                    if np.isinf(arg.max) or np.isinf(arg.min):
                        raise ValueError('LHS method needs a finite range.')
                x = []
                for arg in self.args:
                    samples = []
                    p = np.linspace(arg.min, arg.max, len(self.args))
                    for i in range(len(p) - 1):
                        samples.append(rand() * (p[i+1] - p[i]) + p[i])
                    x.append(samples)
                sols = []
                for _ in range(len(self.args)):
                    sol = []
                    for i in range(len(x)):
                        ri = randint(0, len(x[i]))
                        sol.append(x[i].pop(ri))
                    sols.append(np.array(sol))
                return sols

class Snapshot(NamedTuple):
    epoch: int
    best_fitness: float
    best_solution: Solution


class Swarm(ABC):
    pops: int
    problem: Problem
    solutions: list[Solution]
    metavar: dict[str, Any] = {}

    records: list[Snapshot]
    best: Snapshot

    epoch = 0
    max_epoch = 0

    def __init__(self, population: int, problem: Problem, seed: int | None = None, **metavar):
        self.pops = population
        self.problem = problem
        if seed is not None:
            np.random.seed(seed)
        self.solutions = problem.init_solutions(population)
        self.metavar.update(metavar)
        self.records = []
        self.post_init()


    def evolve(self, epochs: int = 1):
        self.max_epoch = epochs
        for i in range(epochs):
            self.epoch = i
            fits = []

            # evaluate
            for n, sol in enumerate(self.solutions):
                out = self.problem.func(sol)
                fit = out
                fits.append(fit)

            # record
            bestv = min(fits)
            best = self.solutions[fits.index(bestv)].copy()
            snap = Snapshot(self.epoch, bestv, best)
            self.records.append(snap)
            if snap.best_fitness > self.best.best_fitness:
                self.best = snap

            if i != epochs - 1:
                self.solutions = self.update(self.solutions, fits)
                # constrain args if needed
                for sol in self.solutions:
                    for n in range(self.nargs):
                        sol[n] = self.problem.args[n].constrain(sol[n])


    @property
    def nargs(self) -> int:
        return len(self.problem.args)


    @property
    def progress(self) -> float:
        return self.epoch / self.max_epoch


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
