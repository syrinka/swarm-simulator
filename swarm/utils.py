import numpy as np


abs = np.abs
exp = np.exp
rand = np.random.rand
randint = np.random.randint
normal = np.random.normal

def randrange(a, b):
    return rand() * (b - a) + a

eps = np.finfo(np.float64).eps
