from ..base import Swarm

import numpy as np


class GWO(Swarm):
    """
    灰狼算法
    """

    metavar = {
        'a': 2
    }

    def update(self, sols, fits):
        # sort all solutions according to its fitness
        # pick top 3 as the leaders
        order = np.argsort(fits)
        alpha, beta, delta, *_ = [x for _, x in sorted(zip(order, sols))]

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
