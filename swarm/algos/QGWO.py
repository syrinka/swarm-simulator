from ..base import Swarm
from ..utils import *

import numpy as np


class QGWO(Swarm):
    """
    量子灰狼算法
    """

    def update(self, sols, fits):
        # sort all solutions according to its fitness
        # pick top 3 as the leaders
        order = np.argsort(fits)
        alpha, beta, delta, *_ = sols[order]

        alpha = alpha.copy()
        beta = beta.copy()
        delta = delta.copy()

        newsols = []

        for sol in sols:
            avg = (alpha + beta + delta) / 3

            # >0 时探索，<0 时包围
            a = 0.5 - self.progress
            # [0, 2] 随机系数（每个头狼的权重）
            c1 = 2 * rand()
            c2 = 2 * rand()
            c3 = 2 * rand()
            # 吸引子
            g = avg + (1-c1) * alpha + (1-c2) * beta + (1-c3) * delta

            newsol = c1 * g + a * abs(sol - g) * np.log(1 / rand())
            newsols.append(newsol)

        return newsols
