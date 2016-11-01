"""
Functions that performes arithmetics on polynomial.
"""

import numpy as np

import chaospy.poly.fraction
import chaospy.poly.dimension
import chaospy.poly.typing

from chaospy.poly.base import Poly


def add(*args):
    """Polynomial addition."""
    if len(args) > 2:
        return add(args[0], add(args[1], args[1:]))

    if len(args) == 1:
        return args[0]

    part1, part2 = args

    if isinstance(part2, Poly):

        if part2.dim > part1.dim:
            part1 = chaospy.dimension.setdim(part1, part2.dim)
        elif part2.dim < part1.dim:
            part2 = chaospy.dimension.setdim(part2, part1.dim)

        dtype = chaospy.poly.typing.dtyping(part1.dtype, part2.dtype)

        core1 = part1.A.copy()
        core2 = part2.A.copy()

        if np.prod(part2.shape) > np.prod(part1.shape):
            shape = part2.shape
            ones = np.ones(shape, dtype=int)
            for key in core1:
                core1[key] = core1[key]*ones
        else:
            shape = part1.shape
            ones = np.ones(shape, dtype=int)
            for key in core2:
                core2[key] = core2[key]*ones

        if part1.dtype == chaospy.poly.fraction.frac:
            for idx in core2:
                if core1.has_key(idx):
                    core1[idx] = core1[idx] + core2[idx]
                else:
                    core1[idx] = core2[idx]
            out = core1
        else:
            for idx in core1:
                if idx in core2:
                    core2[idx] = core2[idx] + core1[idx]
                else:
                    core2[idx] = core1[idx]
            out = core2

        return Poly(out, part1.dim, shape, dtype)

    if isinstance(part2, (float, list, tuple, int)):
        part2 = np.array(part2)

    if isinstance(part2, (np.ndarray, chaospy.poly.fraction.frac)):

        core = part1.A.copy()
        dtype = chaospy.poly.typing.dtyping(part1.dtype, part2.dtype)

        zero = (0,)*part1.dim

        if zero not in core:
            core[zero] = np.zeros(part1.shape, dtype=int)

        core[zero] = core[zero] + part2

        if np.prod(part2.shape) > np.prod(part1.shape):

            ones = np.ones(part2.shape, dtype=dtype)
            for key in core:
                core[key] = core[key]*ones

        return Poly(core, part1.dim, None, dtype)

    return NotImplemented


def mul(*args):
    """Polynomial multiplication."""
    if len(args) > 2:
        return add(args[0], add(args[1], args[1:]))

    if len(args) == 1:
        return args[0]

    part1, part2 = args

    if not isinstance(part2, Poly):

        if isinstance(part2, (float, int)):
            part2 = np.array(part2)

        if not part2.shape:
            core = part1.A.copy()
            for key in part1.keys:
                core[key] = core[key]*part2
            return Poly(core, part1.dim, part1.shape, part1.dtype)

        part2 = Poly(part2)

    if part2.dim > part1.dim:
        part1 = chaospy.dimension.setdim(part1, part2.dim)

    elif part2.dim < part1.dim:
        part2 = chaospy.dimension.setdim(part2, part1.dim)

    if np.prod(part1.shape) >= np.prod(part2.shape):
        shape = part1.shape
    else:
        shape = part2.shape

    dtype = chaospy.poly.typing.dtyping(part1.dtype, part2.dtype)
    if part1.dtype != part2.dtype:

        if part1.dtype == dtype:
            if dtype == chaospy.poly.fraction.frac:
                part2 = chaospy.poly.typing.asfrac(part2)
            else:
                part2 = chaospy.poly.typing.asfloat(part2)

        else:
            if dtype == chaospy.poly.fraction.frac:
                part1 = chaospy.poly.typing.asfrac(part1)
            else:
                part1 = chaospy.poly.typing.asfloat(part1)

    core = {}
    for idx1 in part2.A:
        for idx2 in part1.A:
            key = tuple(np.array(idx1) + np.array(idx2))
            core[key] = core.get(key, 0) + part2.A[idx1]*part1.A[idx2]

    for key, value in core.items():
        if np.all(value == 0):
            del core[key]

    out = Poly(core, part1.dim, shape, dtype)
    return out


if __name__ == '__main__':
    import doctest
    doctest.testmod()
