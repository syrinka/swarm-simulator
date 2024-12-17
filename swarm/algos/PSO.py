from ..base import Swarm

import numpy as np
from copy import deepcopy


class PSO(Swarm):
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

    v: np.ndarray

    def post_init(self):
        self.v = np.zeros([self.pops, self.ndims])


    def update(self, sols, fits):
        meta = self.metavar

        for n in range(self.pops):
            sol = sols[n]

            pbest = self.pbestx[n]
            gbest = self.gbestx

            for i in range(self.ndims):
                self.v[n][i] = meta['w'] * self.v[n][i] \
                    + meta['c1'] * np.random.rand() * (pbest[i] - sol[i]) \
                    + meta['c2'] * np.random.rand() * (gbest[i] - sol[i])
                sol[i] += self.v[n][i]

        return sols
