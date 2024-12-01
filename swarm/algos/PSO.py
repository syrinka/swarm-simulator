from ..base import Swarm, Solution

import numpy as np
from copy import deepcopy


class PSOSwarm(Swarm):
    """
    经典粒子群算法

    Metavar
    =======
    w: 惯性系数，范围 [0, 1]
    c1: 个体学习因子
    c2: 社会学习因子
    """
    metavar = {
        'w': 0.5,
        'c1': 2,
        'c2': 2,
    }

    gbest: Solution
    gbestv: float

    pbest: list[Solution]
    pbestv: list[float]

    v: np.ndarray

    def post_init(self):
        self.gbest = deepcopy(self.solutions[0])
        self.gbestv = np.inf
        self.pbest = deepcopy(self.solutions)
        self.pbestv = [np.inf] * self.pops
        self.v = np.zeros([self.pops, self.nargs])


    def update(self, sols, fits):
        meta = self.metavar

        # 更新 global best
        bestv = np.min(fits)
        if bestv < self.gbestv:
            self.gbestv = bestv
            self.gbest = sols[fits.index(self.gbestv)].copy()

        for n in range(self.pops):
            sol = sols[n]
            fit = fits[n]

            # 更新 personal best
            if self.pbest[n] is None or fit < self.pbestv[n]:
                self.pbest[n] = sol.copy()
                self.pbestv[n] = fit

            pbest = self.pbest[n]
            gbest = self.gbest

            for i in range(self.nargs):
                self.v[n][i] = meta['w'] * self.v[n][i] \
                    + meta['c1'] * np.random.rand() * (pbest[i] - sol[i]) \
                    + meta['c2'] * np.random.rand() * (gbest[i] - sol[i])
                sol[i] += self.v[n][i]

        return sols
