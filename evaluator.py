"""
This file contains all the standard test function.

Data from https://www.sfu.ca/~ssurjano/index.html
"""
import numpy as np
from swarm.base import Swarm, ArgInfo, Problem, X
from swarm.utils import *
from swarm.types import *
from abc import ABC, abstractmethod

from typing import Self, Literal


class Evaluator(ABC):
    all_evaluators: list[type[Self]] = []

    def __init_subclass__(cls, name: str | None = None):
        if name is not None:
            cls.name = name
        else:
            cls.name = cls.__name__
        if isinstance(cls.domains[0], list):
            cls._domain_type = 'each'
        else:
            cls._domain_type = 'same'
        cls.all_evaluators.append(cls)


    name: str
    """full name of the function"""
    dimensions: int = 0
    """suggested dimensions"""
    domains: list[list[float]] | list[float]
    """suggested input domains"""
    minimum: float = 0
    """global minimum (besty)"""

    _domain_type: Literal['each', 'same']

    @staticmethod
    @abstractmethod
    def infer(x: X) -> float:
        ...

    @classmethod
    def get_problem(cls, nargs: int, **kwargs) -> Problem:
        if cls.dimensions != 0:
            nargs = cls.dimensions
        infos = []
        for i in range(nargs):
            domain: list
            if cls._domain_type == 'each':
                domain = cls.domains[i] # type: ignore
            elif cls._domain_type == 'same':
                domain = cls.domains
            else:
                raise ValueError
            infos.append(ArgInfo(min=domain[0], max=domain[1]))
        return Problem(infos, cls.infer)


class BatchEvaluateInput(NamedTuple):
    # Swarm related
    swarm: type[Swarm]
    pop: int
    metavar: Metavar
    epoch: int

    evaluators: list[type[Evaluator]]

class BatchEvaluateResult(NamedTuple):
    besty: Ys
    loss: Ys


def batch_evaluate(dim: int, input: BatchEvaluateInput) -> BatchEvaluateResult:
    best = []
    loss = []
    for ev in input.evaluators:
        prob = ev.get_problem(dim)
        swarm = input.swarm(input.pop, prob, **input.metavar)
        swarm.evolve(input.epoch)
        record = swarm.last_record()
        best.append(record['gbesty'])
        loss.append(record['gbesty'] - ev.minimum)

    return BatchEvaluateResult(
        np.array(best),
        np.array(loss)
    )


## --------------------------- ##

# Many Local Minima

class Ackley(Evaluator):
    @staticmethod
    def infer(x: X):
        a = 20
        b = 0.2
        c = 2 * pi
        d = len(x)
        return -a * exp(-b * sqrt(sum(x ** 2) / d)) - exp(sum(cos(c * x)) / d) + a + e
    
    domains = [-32.768, 32.768]


class Bukin_N6(Evaluator, name='Bukin N.6'):
    @staticmethod
    def infer(x: X):
        x1, x2 = x[:]
        return 100 * sqrt(abs(x2 - 0.01*x1**2)) + 0.01 * abs(x1 + 10)
    
    dimensions = 2
    domains = [[-15, 5], [-3, 3]]


class Cross_In_Tray(Evaluator, name='Cross-in-Tray'):
    @staticmethod
    def infer(x: X):
        x1, x2 = x[:]
        a = abs(100 - sqrt(x1*x1 + x2*x2)/pi)
        return -0.0001*(abs(sin(x1) * sin(x2) * exp(a)) + 1)**0.1
    
    dimensions = 2
    domains = [-10, 10]
    minimum = -2.062611870822739


class Drop_Wave(Evaluator, name='Drop-Wave'):
    @staticmethod
    def infer(x: X):
        x1, x2 = x[:]
        sq = x1*x1 + x2*x2
        return - (1 + cos(12*sqrt(sq))) / (0.5*sq + 2)
    
    dimensions = 2
    domains = [-5.12, 5.12]
    minimum = -1


class Eggholder(Evaluator):
    @staticmethod
    def infer(x: X):
        x1, x2 = x[:]
        return -(x2 + 47) * sin(sqrt(abs(x2 + x1/2 + 47))) - x1*sin(sqrt(abs(x1-x2-47)))
    
    dimensions = 2
    domains = [-512, 512]
    minimum = -959.640662720850742


class Gramacy_n_Lee(Evaluator, name='Gramacy & Lee (2012)'):
    @staticmethod
    def infer(x: X):
        x1 = x[0]
        return sin(10*pi*x1) / (2*x1) + (x1-1)**4
    
    dimensions = 1
    domains = [0.5, 2.5]
    minimum = -0.869011134989500


class Griewank(Evaluator):
    @staticmethod
    def infer(x: X):
        dim = x.shape[0]
        l = 0
        r = 1
        for i in range(dim):
            l += x[i]**2 / 4000
            r *= cos(x[i] / sqrt(i+1))
        return l - r + 1
    
    domains = [-600, 600]


class Holder_Table(Evaluator, name='Holder Table'):
    @staticmethod
    def infer(x: X):
        x1, x2 = x[:]
        sq = x1*x1 + x2*x2
        return -abs(sin(x1)*cos(x2)*exp(abs(1-sqrt(sq)/pi)))
    
    dimensions = 2
    domains = [-10, 10]
    minimum = -19.20850256788675


class Langermann(Evaluator):
    @staticmethod
    def infer(x: X):
        A = np.array([3,5,5,2,2,1,1,4,7,9]).reshape([5, 2])
        c = [1,2,5,2,3]
        m = 5
        result = 0
        for i in range(m):
            fact = 0
            for j in range(2):
                fact += (x[j] - A[i,j])**2
            result += - c[i] * exp(-1/pi * fact) * cos(pi * fact)
        return result

    dimensions = 2
    domains = [0, 10]
    minimum = -5.1621259


class Levy(Evaluator):
    @staticmethod
    def infer(x: X):
        dim = x.shape[0]
        w1 = 1 + (x[0] - 1)/4
        wd = 1 + (x[-1] - 1)/4

        result = 0
        for i in range(dim-1):
            wi = 1 + (x[i] - 1)/4
            result = (wi-1)**2 * (1 + 10*sin(pi*wi+1)**2)
        
        return result + sin(pi*w1)**2 + (wd-1)**2 * (1 + sin(2*pi*wd)**2)
    
    domains = [-10, 10]


class Levy_N13(Evaluator, name='Levy N.13'):
    @staticmethod
    def infer(x: X):
        x1, x2 = x[:]
        return sin(3*pi*x1)**2 + (x1-1)**2*(1+sin(3*pi*x2)**2) + (x2-1)**2*(1+sin(2*pi*x2)**2)
    
    dimensions = 2
    domains = [-10, 10]


class Rastrigin(Evaluator):
    @staticmethod
    def infer(x: X):
        d = x.shape[0]
        y = 10 * d
        for i in x:
            y += i*i - 10 * cos(2*pi*i)
        return y

    domains = [-5.12, 5.12]


class Schaffer_N2(Evaluator, name='Schaffer N.2'):
    @staticmethod
    def infer(x: X):
        x1, x2 = x[:]
        fact1 = sin(x1*x1-x2*x2)**2 - 0.5
        fact2 = (1+0.001*(x1*x1+x2*x2))**2
        return 0.5 + fact1/fact2
    
    dimensions = 2
    domains = [-100, 100]


class Schaffer_N4(Evaluator, name='Schaffer N.4'):
    @staticmethod
    def infer(x: X):
        x1, x2 = x[:]
        fact1 = cos(sin(abs(x1*x1-x2*x2)))**2 - 0.5
        fact2 = (1+0.001*(x1*x1+x2*x2))**2
        return 0.5 + fact1/fact2
    
    dimensions = 2
    domains = [-100, 100]
    minimum = 0.292579


class Schwefel(Evaluator):
    @staticmethod
    def infer(x: X):
        dim = x.shape[0]
        return 418.9829*dim - sum(x[i] * sin(sqrt(abs(x[i]))) for i in range(dim))
    
    domains = [-500, 500]

# Bowl-Shaped

class Sphere(Evaluator):
    @staticmethod
    def infer(x: X):
        return float(sum(x ** 2))
    
    domains = [-5.12, 5.12]

# Plate-Shaped

class Zakharov(Evaluator):
    @staticmethod
    def infer(x: X):
        fact1 = fact2 = 0
        dim = x.shape[0]
        for i in range(dim):
            idx = i+1
            fact1 += x[i]*x[i]
            fact2 += 0.5*idx*x[i]
        return fact1 + fact2**2 + fact2**4
    
    domains = [-5, 10]

# Vally-Shaped

class Rosenbrock(Evaluator):
    @staticmethod
    def infer(x: X):
        dim = x.shape[0]
        s = 0
        for i in range(dim-1):
            s += 100*(x[i+1]-x[i]**2)**2 + (x[i]-1)**2
        return s

    domains = [-5, 10]

# Hybrid & Composition

class Bent_Cigar(Evaluator):
    @staticmethod
    def infer(x: X):
        return x[0]**2 + 10e6 * sum(x[1:] ** 2)
    
    domains = [-100, 100]

# class HGBat(Evaluator):
#     pass

# class High_Conditioned_Elliptic(Evaluator):
#     pass

# class Katsuura(Evaluator):
#     pass

# class Happycat(Evaluator):
#     pass

# class Expanded_Rosenbrock_n_Griewangk(Evaluator):
#     pass
