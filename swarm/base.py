from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .types import *
from .utils import *


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

    def inbound(self, x: X) -> bool:
        """判断一个解是否符合约束条件"""
        for i, v in enumerate(x):
            info = self.args[i]
            if not info.min <= v <= info.max:
                return False
        return True


class Swarm(ABC):
    pops: int
    problem: Problem
    metavar: Metavar = {}

    xs: Xs

    pbestx: Xs
    """Personal best X(s)"""
    pbesty: Ys
    """Personal best Y(s)"""
    gbestx: X
    """Global best X"""
    gbesty: Y
    """Global best Y"""
    records: list[Record]

    # current/max epoch
    cur_epoch = 0
    max_epoch = 0

    def __init__(self, population: int, problem: Problem, seed: int | None = None, **metavar):
        self.pops = population
        self.problem = problem
        if seed is not None:
            np.random.seed(seed)
        self.xs = problem.initialize(population)
        self.metavar.update(metavar)
        self.pbestx = np.zeros((self.pops, self.ndims))
        self.pbesty = np.zeros((self.pops, ))
        self.gbestx = np.zeros((self.ndims, ))
        self.gbesty = np.inf
        self.records = []
        self.post_init()


    def evolve(self, epochs: int = 1):
        self.max_epoch = epochs
        for i in range(epochs):
            self.cur_epoch = i
            ys = np.zeros((self.pops, ))

            # evaluate
            for n, x in enumerate(self.xs):
                y = self.problem.func(x)
                ys[n] = y

                if y < self.pbesty[n]:
                    self.pbestx[n] = x.copy()
                    self.pbesty[n] = y
                if y < self.gbesty:
                    self.gbestx = x.copy()
                    self.gbesty = y

            # record
            best_idx = ys.argmax()
            besty = ys[best_idx]
            bestx = self.xs[best_idx]
            rec = Record(
                epoch = self.cur_epoch,
                besty = besty,
                bestx = bestx.copy(),
                gbesty = self.gbesty,
                gbestx = self.gbestx.copy()
            )
            self.records.append(rec)

            if i != epochs - 1:
                new_solutions = self.update(self.xs, ys)
                if isinstance(new_solutions, list):
                    new_solutions = np.array(new_solutions)
                self.xs = new_solutions
                # constrain args if needed
                for x in self.xs:
                    for n in range(self.ndims):
                        x[n] = self.problem.args[n].constrain(x[n])


    @property
    def ndims(self) -> int:
        return len(self.problem.args)


    @property
    def progress(self) -> float:
        return self.cur_epoch / self.max_epoch


    def __repr__(self) -> str:
        return f'<{self.__class__.__name__} pop={self.pops} metavar={self.metavar}>'


    def fitness_history(self) -> list[float]:
        return [i['besty'] for i in self.records]


    def last_record(self) -> Record:
        return self.records[-1]


    @abstractmethod
    def update(self, sols: Xs, fits: Ys) -> Xs | list[X]:
        raise NotImplementedError()


    def post_init(self):
        pass
