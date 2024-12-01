from .base import OptimizeGoal, Solution
import numpy as np


def bestof(x: Solution, goal: OptimizeGoal):
    match goal:
        case 'maximum':
            return np.max(x)
        case 'minimum':
            return np.min(x)
        case 'zero':
            ind = np.argmin(np.abs(x))
            return x[ind]
