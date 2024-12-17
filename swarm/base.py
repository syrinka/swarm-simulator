from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeAlias, Any, Callable, Literal, NamedTuple

import numpy as np
import numpy.typing as npt

from .utils import *


X: TypeAlias = npt.NDArray[np.float64]
"""Shape: (ndims, 1)"""
Xs: TypeAlias = npt.NDArray[np.float64]
"""Shape: (pops, ndims)"""
Y: TypeAlias = float
"""Type alias of float"""
Ys: TypeAlias = npt.NDArray[np.float64]
"""Shape: (pops, 1)"""
TargetFunction: TypeAlias = Callable[[X], Y]


OptimizeGoal: TypeAlias = Literal['maximum', 'minimum', 'zero']
BoundaryMethod: TypeAlias = Literal['saturate', 'wrap']
InitializeMethod: TypeAlias = Literal['random', 'lhs']


@dataclass(kw_only=True, frozen=True)
class ArgInfo(object):
    min: float = -np.inf
    max: float = np.inf
    grain: float | None = None
    underflow: BoundaryMethod = 'saturate'
    overflow: BoundaryMethod = 'saturate'


    def __post_init__(self):
        if self.max <= self.min:
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

    def initialize(self, num: int, method: InitializeMethod = 'random') -> Xs:
        match method:
            case 'random':
                sols = []
                for _ in range(num):
                    tmp = []
                    for arg in self.args:
                        n = np.random.rand() * (arg.max - arg.min) + arg.min
                        tmp.append(n)
                    sols.append(np.array(tmp))
                return np.array(sols)

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
                return np.array(sols)

class Record(NamedTuple):
    epoch: int
    best_output: float
    best_fitness: Y
    best_solution: X


class Swarm(ABC):
    pops: int
    problem: Problem
    solutions: Xs
    metavar: dict[str, Any] = {}

    pbestx: Xs
    pbesty: Ys
    gbestx: X
    gbesty: Y
    records: list[Record]

    epoch = 0
    max_epoch = 0

    def __init__(self, population: int, problem: Problem, seed: int | None = None, **metavar):
        self.pops = population
        self.problem = problem
        if seed is not None:
            np.random.seed(seed)
        self.solutions = problem.initialize(population)
        self.metavar.update(metavar)
        self.pbestx = np.zeros((self.pops, self.ndims))
        self.pbesty = np.zeros((self.pops, ))
        self.gbestx = np.zeros((self.ndims, ))
        self.gbesty = -np.inf
        self.records = []
        self.post_init()


    def evolve(self, epochs: int = 1):
        self.max_epoch = epochs
        for i in range(epochs):
            self.epoch = i
            fits = np.zeros((self.pops, ))
            outs = np.zeros((self.pops, ))

            # evaluate
            for n, sol in enumerate(self.solutions):
                out = self.problem.func(sol)
                match self.problem.goal:
                    case 'maximum':
                        fit = out
                    case 'minimum':
                        fit = -out
                    case 'zero':
                        fit = -abs(out)
                outs[n] = out
                fits[n] = fit

                if fit > self.pbesty[n]:
                    self.pbestx[n] = sol.copy()
                    self.pbesty[n] = fit
                if fit > self.gbesty:
                    self.gbestx = sol.copy()
                    self.gbesty = fit

            # record
            best_idx = fits.argmax()
            best_output = outs[best_idx]
            best_fitness = fits[best_idx]
            best_solution = self.solutions[best_idx].copy()
            rec = Record(self.epoch, best_output, best_fitness, best_solution)
            self.records.append(rec)

            if i != epochs - 1:
                new_solutions = self.update(self.solutions, fits)
                if isinstance(new_solutions, list):
                    new_solutions = np.array(new_solutions)
                self.solutions = new_solutions
                # constrain args if needed
                for sol in self.solutions:
                    for n in range(self.ndims):
                        sol[n] = self.problem.args[n].constrain(sol[n])


    @property
    def ndims(self) -> int:
        return len(self.problem.args)


    @property
    def progress(self) -> float:
        return self.epoch / self.max_epoch


    def __repr__(self) -> str:
        return f'<{self.__class__.__name__} pop={self.pops} metavar={self.metavar}>'


    def fitness_history(self) -> list[float]:
        return [i.best_fitness for i in self.records]


    def best_record(self) -> Record:
        return max(self.records, key=lambda i: i.best_fitness)


    @abstractmethod
    def update(self, sols: Xs, fits: Ys) -> Xs | list[X]:
        raise NotImplementedError()


    def post_init(self):
        pass
