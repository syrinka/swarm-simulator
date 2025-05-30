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


class QPSO(Swarm):
    """
    量子粒子群算法

    Metavar
    =======
    beta (bmax, bmin): 收敛参数；若 bmax、bmin 被传入，
    type: 吸引子的计算方式
        1 - 所有个体最优解的平均
        2 - 个体与全局最优解的随机加权平均
        3 - 个体与全局最优解的 α 加权平均
    alpha: 当 type=3 时，控制计算吸引子时个体与全局最优解间的权重
    """

    metavar = {
        'beta': 0.75,
        'alpha': 0.5,
        'bmax': None,
        'bmin': None,
        'type': 1,
    }

    def update(self, sols, fits):
        meta = self.metavar

        for n in range(self.pops):
            sol = sols[n]

            pbest = self.pbestx[n]
            gbest = self.gbestx

            # 吸引子
            att = None
            match meta['type']:
                case 1:
                    att = np.average(self.pbestx)
                case 2:
                    rnd = np.random.rand()
                    att = rnd * pbest + (1-rnd) * gbest
                case 3:
                    att = meta['alpha'] * pbest + (1-meta['alpha']) * gbest
                case _:
                    raise ValueError()

            dist = abs(att - sol)

            bmax = meta['bmax']
            bmin = meta['bmin']
            # 收缩因子
            beta = None
            if bmax is not None and bmin is not None:
                beta = bmax - (bmax - bmin) * self.progress
            else:
                beta = meta['beta']

            pn = 1 if np.random.rand() > 0.5 else -1
            sols[n] = att + pn * beta * dist * np.log(1 / np.random.rand())

        return sols
