from typing import TypeAlias, Any, Callable, Literal, NamedTuple
import numpy as np
import numpy.typing as npt


X: TypeAlias = npt.NDArray[np.float64]
"""Shape: (ndims, 1)"""
Xs: TypeAlias = npt.NDArray[np.float64]
"""Shape: (pops, ndims)"""
Y: TypeAlias = float
"""Type alias of float"""
Ys: TypeAlias = npt.NDArray[np.float64]
"""Shape: (pops, 1)"""
TargetFunction: TypeAlias = Callable[[X], Y]
"""Metavar"""
Metavar: TypeAlias = dict[str, Any]


BoundaryMethod: TypeAlias = Literal['saturate', 'wrap']
InitializeMethod: TypeAlias = Literal['random', 'lhs']


class Record(NamedTuple):
    epoch: int
    best_fitness: Y
    best_solution: X
