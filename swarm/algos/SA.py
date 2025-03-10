from ..base import Swarm
from ..utils import *
from ..types import *

import numpy as np


def neighbour(x: X, T: float) -> X:
    return x + np.random.normal(loc=0, scale=T, size=x.shape)


def distance(x1: X, x2: X) -> float:
    return np.sum(np.square(x1 - x2))


class SA(Swarm):
    T: float

    metavar = {
        'T0': 100,
        'k': 0.9
    }

    def post_init(self):
        self.T = self.metavar['T0']

    def request(self, x: X) -> Y:
        return self.problem.func(x)

    def update(self, sols, fits):
        out = []
        for x, y in zip(sols, fits):
            # 持续roll直到得到一个约束内的新位置
            newx = neighbour(x, self.T)
            while not self.problem.inbound(newx):
                newx = neighbour(x, self.T)
            newy = self.request(newx)

            delta = y - newy
            kT = self.metavar['k'] * self.T

            if newy < y:
                # 新位置比原位置优秀，接受
                out.append(newx)
            elif rand() < exp(delta / self.T):
                # 新位置比原位置更差，依概率接受
                out.append(newx)
            else:
                # just use the previous
                out.append(x)
        # 降温
        self.T *= self.metavar['k']
        return np.array(out)


class QSA(Swarm):
    T: float

    metavar = {
        'T0': 100,
        'k': 0.9
    }

    def post_init(self):
        self.T = self.metavar['T0']

    def request(self, x: X) -> Y:
        return self.problem.func(x)

    def update(self, sols, fits):
        out = []
        for x, y in zip(sols, fits):
            # 持续roll直到得到一个约束内的新位置
            newx = neighbour(x, self.T)
            while not self.problem.inbound(newx):
                newx = neighbour(x, self.T)
            newy = self.request(newx)

            delta = y - newy
            dist = distance(x, newx)
            kT = self.metavar['k'] * self.T

            if newy < y:
                # 新位置比原位置优秀，接受
                out.append(newx)
            elif rand() < exp(sqrt(delta)*dist / self.T):
                # 新位置比原位置更差，依概率接受
                out.append(newx)
            else:
                # just use the previous
                out.append(x)
        # 降温
        self.T *= self.metavar['k']
        return np.array(out)
