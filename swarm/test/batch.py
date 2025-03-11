from ..base import *
from .evaluator import Evaluator

import numpy as np
import matplotlib.pyplot as plt


class SwarmConfig(NamedTuple):
    swarm: type[Swarm]
    pop: int
    epoch: int
    metavar: Metavar | None = None

    def __str__(self):
        return self.swarm.__name__


class BatchEvaluate(object):
    swarms: list[SwarmConfig]
    evaluators: list[type[Evaluator]]

    best_matrix: np.ndarray
    loss_matrix: np.ndarray

    def __init__(self, swarms: list[SwarmConfig], evaluators: list[type[Evaluator]]):
        self.swarms = swarms
        self.evaluators = evaluators

        slen = len(swarms)
        elen = len(evaluators)
        size = (slen, elen)
        self.best_matrix = np.zeros(size)
        self.loss_matrix = np.zeros(size)

    def swarm_names(self) -> list[str]:
        return [str(s) for s in self.swarms]

    def evaluator_names(self) -> list[str]:
        return [e.name for e in self.evaluators]

    def run(self):
        for i, cfg in enumerate(self.swarms):
            for j, ev in enumerate(self.evaluators):
                prob = ev.get_problem()
                swarm = cfg.swarm(cfg.pop, prob, **cfg.metavar or dict())
                swarm.evolve(cfg.epoch)
                record = swarm.last_record()
                self.best_matrix[i,j] = record['gbesty']
                self.loss_matrix[i,j] = record['gbesty'] - ev.minimum

    def visualize(self, mark_prevail: bool = True, round_decimal: int = 2):
        """
        将批量测试的结果以热力图形式可视化出来

        Parameters:
        mark_prevail (bool): 是否在可视图中标注每个评估函数的最优结果
        round_decimal (int): 适应度舍入精度
        """
        # V=evaluator H=swarm for better layout
        result = self.loss_matrix.transpose()
        result += np.finfo(np.float64).eps # prevent divide by zero
        result = np.log(result)

        slen = len(self.swarms)
        elen = len(self.evaluators)

        for i in range(elen):
            prevail = np.argmin(result[i,:])
            for j in range(slen):
                text = np.round(result[i, j], round_decimal)
                if mark_prevail and j == prevail:
                    text = f'[ {text} ]'
                plt.text(j, i, text, ha='center', va='center', color='w')

        plt.xticks(range(slen), labels=self.swarm_names())
        plt.yticks(np.arange(elen), labels=self.evaluator_names())

        plt.imshow(result)
        plt.colorbar()
        return plt.gcf()
