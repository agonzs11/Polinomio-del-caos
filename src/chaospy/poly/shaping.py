"""
Function for changing polynomial array's shape.
"""
import numpy as np

import chaospy.poly.fraction
from chaospy.poly.base import Poly


def flatten(vari):
    """
    Flatten a shapeable quantity.

    Args:
        vari (Poly, frac, array_like) : Shapeable input quantity.

    Returns:
        (Poly, frac, array_like) : Same type as `vari` with `len(Q.shape)==1`.

    Examples:
        >>> P = cp.reshape(cp.prange(4), (2,2))
        >>> print(P)
        [[1, q0], [q0^2, q0^3]]
        >>> print(cp.flatten(P))
        [1, q0, q0^2, q0^3]
    """
    if isinstance(vari, (np.ndarray, tuple, list, int, float)):
        return np.array(vari).flatten()

    elif isinstance(vari, (Poly, chaospy.poly.fraction.frac)):
        shape = int(np.prod(vari.shape))
        return reshape(vari, (shape,))

    raise ValueError("input not recognised %r" % str(vari))


def reshape(vari, shape):
    """
    Reshape the shape of a shapeable quantity.

    Args:
        vari (Poly, frac, array_like) : Shapeable input quantity
        shape (tuple) : The polynomials new shape. Must be compatible with the
                number of elements in `vari`.

    Returns:
        (Poly, frac, array_like) : Same type as `vari`.

    Examples:
        >>> poly = cp.prange(6)
        >>> print(poly)
        [1, q0, q0^2, q0^3, q0^4, q0^5]
        >>> print(cp.reshape(poly, (2,3)))
        [[1, q0, q0^2], [q0^3, q0^4, q0^5]]
    """
    if isinstance(vari, (np.ndarray, list, tuple, int, float)):
        return np.array(vari).reshape(*shape)

    elif isinstance(vari, chaospy.poly.fraction.frac):
        return chaospy.poly.fraction.frac(
            vari.a.reshape(shape),
            vari.b.reshape(shape),
        )

    elif isinstance(vari, Poly):
        core = vari.A.copy()
        for key in vari.keys:
            core[key] = reshape(core[key], shape)
        out = Poly(core, vari.dim, shape, vari.dtype)
        return out

    raise ValueError("input not recognised %s" % str(vari))


def rollaxis(vari, axis, start=0):
    """
    Roll the specified axis backwards, until it lies in a given position.

    Args:
        vari (Poly, array_like) : Input array or polynomial.
        axis (int) : The axis to roll backwards. The positions of the
            other axes do not change realtie to one another.
        start (int, optional) : The axis is rolled until it lies before
            thes position.
    """
    if isinstance(vari, (int, float, list, tuple, np.ndarray)):
        return np.rollaxis(vari, axis, start)

    elif isinstance(vari, chaospy.poly.fraction.frac):
        denom = np.rollaxis(vari.a, axis, start)
        nume = np.rollaxis(vari.b, axis, start)
        return chaospy.poly.fraction.frac(denom, nume)

    elif isinstance(vari, Poly):
        core_old = vari.A.copy()
        core_new = {}
        for key in vari.keys:
            core_new[key] = rollaxis(core_old[key], axis, start)
        return Poly(core_new, vari.dim, None, vari.dtype)

    raise ValueError("input not recognised %s" % str(vari))


def swapaxes(vari, ax1, ax2):
    """Interchange two axes of a polynomial."""
    if isinstance(vari, (int, float, list, tuple, np.ndarray)):
        return np.swapaxes(vari, ax1, ax2)

    elif isinstance(vari, chaospy.poly.fraction.frac):
        return chaospy.poly.fraction.frac(
            np.swapaxes(vari.a, ax1, ax2),
            np.swapaxes(vari.b, ax1, ax2),
        )

    elif isinstance(vari, Poly):
        core = vari.A.copy()
        for key in vari.keys:
            core[key] = swapaxes(core[key], ax1, ax2)

        return Poly(core, vari.dim, None, vari.dtype)

    raise NotImplementedError


def roll(vari, shift, axis=None):
    """Roll array elements along a given axis."""
    if isinstance(vari, (int, float, np.ndarray)):
        return np.roll(vari, shift, axis)

    elif isinstance(vari, chaospy.poly.fraction.frac):
        return chaospy.poly.fraction.frac(
            np.roll(vari.a, shift, axis),
            np.roll(vari.b, shift, axis),
        )

    elif isinstance(vari, Poly):
        core = vari.A.copy()
        for key in vari.keys:
            core[key] = roll(core[key], shift, axis)
        return Poly(core, vari.dim, None, vari.dtype)

    raise NotImplementedError


def transpose(vari):
    """
    Transpose a shapeable quantety.

    Args:
        vari (Poly, frac, array_like) : Quantety of interest.

    Returns:
        Q (Poly, frac, array_like) : Same type as `vari`.

    Examples:
        >>> P = cp.reshape(cp.prange(4), (2,2))
        >>> print(P)
        [[1, q0], [q0^2, q0^3]]

        >>> print(cp.transpose(P))
        [[1, q0^2], [q0, q0^3]]
    """
    if isinstance(vari, (int, float, np.ndarray)):
        return np.transpose(vari)

    elif isinstance(vari, chaospy.poly.fraction.frac):
        return chaospy.poly.fraction.frac(vari.a.T, vari.b.T)

    elif isinstance(vari, Poly):
        core = vari.A.copy()
        for key in vari.keys:
            core[key] = transpose(core[key])
        return Poly(core, vari.dim, vari.shape[::-1], vari.dtype)

    raise NotImplementedError


if __name__ == '__main__':
    import chaospy as cp
    import doctest
    doctest.testmod()
