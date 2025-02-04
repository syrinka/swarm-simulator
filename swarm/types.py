from typing import TypeAlias, Any, Callable, Literal, TypedDict, NamedTuple
import numpy as np
import numpy.typing as npt


X: TypeAlias = npt.NDArray[np.float64]
"""Type X, Shape: (ndims, 1)"""
Xs: TypeAlias = npt.NDArray[np.float64]
"""Type Xs, Shape: (pops, ndims)"""
Y: TypeAlias = float
"""Type Y, Type alias of float"""
Ys: TypeAlias = npt.NDArray[np.float64]
"""Type Ys,Shape: (pops, 1)"""
TargetFunction: TypeAlias = Callable[[X], Y]
"""Metavar"""
Metavar: TypeAlias = dict[str, Any]


BoundaryMethod: TypeAlias = Literal['saturate', 'wrap']
InitializeMethod: TypeAlias = Literal['random', 'lhs']


class Record(TypedDict):
    epoch: int
    besty: Y
    """Best Y in this epoch"""
    bestx: X
    """Best X in this epoch"""
    gbesty: Y
    """Best Y till this epoch"""
    gbestx: X
    """Best X till this epoch"""
