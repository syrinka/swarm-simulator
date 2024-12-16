from ..base import Swarm, Solution

import numpy as np
from copy import deepcopy


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

    gbest: Solution
    gbestv: float

    pbest: list[Solution]
    pbestv: list[float]

    def post_init(self):
        self.gbest = deepcopy(self.solutions[0])
        self.gbestv = np.inf
        self.pbest = deepcopy(self.solutions)
        self.pbestv = [np.inf] * self.pops


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

            # 吸引子
            att = None
            match meta['type']:
                case 1:
                    att = np.average(self.pbest)
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
