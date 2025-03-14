from ..base import Swarm
from ..utils import *

import numpy as np


class GWO(Swarm):
    """
    灰狼算法

    Metavar
    =======
    a (float): 控制探索/包围阶段的比例
    """

    metavar = {
        'a': 2
    }

    def update(self, sols, fits):
        # sort all solutions according to its fitness
        # pick top 3 as the leaders
        order = np.argsort(fits)
        alpha, beta, delta, *_ = sols[order]

        alpha = alpha.copy()
        beta = beta.copy()
        delta = delta.copy()

        a = self.metavar['a'] * (1 - self.progress)

        newsols = []

        for sol in sols:
            # [-a, a] 步长向量
            a1 = a * (2 * np.random.rand() - 1)
            a2 = a * (2 * np.random.rand() - 1)
            a3 = a * (2 * np.random.rand() - 1)
            # [0, 2] 随机系数（每个头狼的权重）
            c1 = 2 * np.random.rand()
            c2 = 2 * np.random.rand()
            c3 = 2 * np.random.rand()
            # 与猎物的距离
            d1 = np.abs(c1 * alpha - sol)
            d2 = np.abs(c2 * beta - sol)
            d3 = np.abs(c3 * delta - sol)
            x1 = alpha - a1 * d1
            x2 = beta - a2 * d2
            x3 = delta - a3 * d3
            newsols.append((x1 + x2 + x3) / 3)

        return newsols


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
