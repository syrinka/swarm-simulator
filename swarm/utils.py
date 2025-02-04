import numpy as np


abs = np.abs
exp = np.exp
sqrt = np.sqrt
rand = np.random.rand
randint = np.random.randint
normal = np.random.normal
gamma = np.random.gamma
sin = np.sin
cos = np.cos
pi = np.pi
e = np.e

def randrange(a, b):
    return rand() * (b - a) + a

def pm(r=0.5):
    return 1 if rand() > r else -1

eps = np.finfo(np.float64).eps
