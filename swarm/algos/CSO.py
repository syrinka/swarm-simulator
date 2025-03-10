from ..base import Swarm
from ..utils import *

import numpy as np
from typing import Literal


ChickenType = Literal['rooster', 'hen', 'chick']

class CSO(Swarm):
    """
    鸡群优化算法

    Metavar
    =======
    rn (int/float): 公鸡的数量/比例
    cn (int/float)：小鸡的数量/比例
    g (int): 鸡群关系更新间隔
    fl (tuple[float]): 小鸡的探索参数
    """
    metavar = {
        'rn': .2,
        'cn': .2,
        'g': 10,
        'fl': (0.5, 0.9)
    }

    role: dict[int, ChickenType]
    leader: dict[int, int]


    def post_init(self):
        if self.metavar['rn'] < 1:
            self.metavar['rn'] = int(self.metavar['rn'] * self.pops)
        if self.metavar['cn'] < 1:
            self.metavar['cn'] = int(self.metavar['cn'] * self.pops)
        self.role = {}
        self.leader = {}


    def update(self, sols, fits):
        rn = self.metavar['rn'] # rooster num
        cn = self.metavar['cn'] # chick num
        hn = self.pops - rn - cn # hen num

        if self.epoch % self.metavar['g'] == 0:
            # 更新鸡群关系
            self.role.clear()
            self.leader.clear()
            order = np.argsort(fits)
            for i in range(0, rn):
                self.role[order[i]] = 'rooster'
                # self.leader[order[i]] = i
            for i in range(rn, rn+hn):
                self.role[order[i]] = 'hen'
                self.leader[order[i]] = randint(0, rn)
            for i in range(rn+hn, rn+hn+cn):
                self.role[order[i]] = 'chick'
                self.leader[order[i]] = randint(rn, rn+hn)

        newsols = []
        for i, sol in enumerate(sols):
            newsol = None
            match self.role[i]:
                case 'rooster':
                    k = randint(0, rn) # pick an opponent
                    scale = 0
                    if fits[i] < fits[k]:
                        scale = 1
                    else:
                        scale = exp((fits[k] - fits[i]) / (abs(fits[i]) + eps))
                    newsol = sol * (1 + normal(0, scale))
                case 'hen':
                    r1 = self.leader[i]
                    r2 = randint(rn, rn + hn)
                    s1 = exp((fits[i] - fits[r1]) / (abs(fits[i] + eps)))
                    s2 = exp(fits[r2] - fits[i])
                    newsol = sol + s1 * rand() * (sols[r1] - sol) + s2 * rand() * (sols[r2] - sol)
                case 'chick':
                    m = self.leader[i]
                    fl = randrange(*self.metavar['fl'])
                    newsol = sol + fl * (sols[m] - sol)
            newsols.append(newsol)
        return newsols


class QCSO(Swarm):
    """
    鸡群优化算法

    Metavar
    =======
    rn (int/float): 公鸡的数量/比例
    cn (int/float)：小鸡的数量/比例
    g (int): 鸡群关系更新间隔
    fl (tuple[float]): 小鸡的探索参数
    """

    metavar = {
        'rn': .2,
        'cn': .2,
        'g': 10,
        'fl': (0.5, 0.9),
        'beta': 0.75,
        'c1': 0.5
    }

    role: dict[int, ChickenType]
    leader: dict[int, int]


    def post_init(self):
        if self.metavar['rn'] < 1:
            self.metavar['rn'] = int(self.metavar['rn'] * self.pops)
        if self.metavar['cn'] < 1:
            self.metavar['cn'] = int(self.metavar['cn'] * self.pops)
        self.role = {}
        self.leader = {}


    @staticmethod
    def levy(beta, size=1):
        sigma = gamma(1+beta) * sin(pi*beta / 2) / (beta * gamma((1+beta) / 2) * 2**((beta-1)/2))
        u = normal(0, sigma, size)
        v = normal(0, 1, size)
        return u * sigma / abs(v) ** (1/beta)


    def update(self, sols, fits):
        rn = self.metavar['rn'] # rooster num
        cn = self.metavar['cn'] # chick num
        hn = self.pops - rn - cn # hen num

        beta = self.metavar['beta']

        if self.epoch % self.metavar['g'] == 0:
            # 更新鸡群关系
            self.role.clear()
            self.leader.clear()
            order = np.argsort(fits)[::-1]
            for i in range(0, rn):
                self.role[order[i]] = 'rooster'
                # self.leader[order[i]] = i
            for i in range(rn, rn+hn):
                self.role[order[i]] = 'hen'
                self.leader[order[i]] = randint(0, rn)
            for i in range(rn+hn, rn+hn+cn):
                self.role[order[i]] = 'chick'
                self.leader[order[i]] = randint(rn, rn+hn)

        newsols = []
        for i, sol in enumerate(sols):
            newsol = None
            match self.role[i]:
                case 'rooster':
                    k = randint(0, rn) # pick an opponent
                    scale = 0
                    if fits[i] > fits[k]:
                        scale = 1
                    else:
                        scale = exp((fits[k] - fits[i]) / (abs(fits[i]) + eps))
                    c1 = self.metavar['c1']
                    mbest = c1 * self.pbestx[i] + (1-c1) * self.gbestx
                    newsol = sol * (1 + normal(0, scale)) + pm() * beta * abs(mbest - sol) * np.log(1 / rand())
                case 'hen':
                    r1 = self.leader[i]
                    r2 = randint(rn, rn + hn)
                    s1 = exp((fits[i] - fits[r1]) / (abs(fits[i] + eps)))
                    s2 = exp(fits[r2] - fits[i])
                    newsol = sol + s1 * rand() * (sols[r1] - sol) * self.levy(beta) \
                                + s2 * rand() * (sols[r2] - sol) * self.levy(beta)
                case 'chick':
                    m = self.leader[i]
                    fl = randrange(*self.metavar['fl'])
                    newsol = sol + fl * (sols[m] - sol)
            newsols.append(newsol)
        return newsols
