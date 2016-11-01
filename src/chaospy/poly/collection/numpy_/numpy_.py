"""
Function that overlaps with the numpy library.
"""
import numpy as np

import chaospy as cp
import chaospy.poly
from chaospy.poly.base import Poly


def sum(vari, axis=None): # pylint: disable=redefined-builtin
    """
    Sum the components of a shapeable quantity along a given axis.

    Args:
        vari (Poly, frac, array_like) : Input data.
        axis (int, optional) : Axis over which the sum is taken. By default
                `axis` is None, and all elements are summed.

    Returns:
        (Poly, frac, array_like) : Polynomial array with same shape as `vari`,
                with the specified axis removed. If `vari` is an 0-d array, or
                `axis` is None, a (non-iterable) component is returned.

    Examples:
        >>> vari = cp.prange(3)
        >>> print(vari)
        [1, q0, q0^2]
        >>> print(cp.sum(vari))
        q0^2+q0+1
    """
    if isinstance(vari, (np.ndarray, float, list, tuple, int)):
        return np.sum(vari, axis)

    elif isinstance(vari, chaospy.poly.fraction.frac):
        if not vari.shape:
            return vari
        if axis is None:
            denom = chaospy.poly.shaping.flatten(vari.a)
            nume = chaospy.poly.shaping.flatten(vari.b)
        else:
            denom = chaospy.poly.shaping.rollaxis(vari.a, axis)
            nume = chaospy.poly.shaping.rollaxis(vari.b, axis)

        denom = np.sum(
            [
                np.prod(nume[:idx], 0) * np.prod(nume[idx+1:], 0) * denom[idx]
                for idx in range(len(nume))
            ],
            0,
        )
        nume = np.prod(nume, 0)
        return chaospy.poly.fraction.frac(denom, nume)

    elif isinstance(vari, Poly):

        core = vari.A.copy()
        for key in vari.keys:
            core[key] = sum(core[key], axis)

        return Poly(core, vari.dim, None, vari.dtype)

    raise NotImplementedError


def cumsum(vari, axis=None):
    """
    Cumulative sum the components of a shapeable quantity along a given axis.

    Args:
        vari (Poly, frac, array_like) : Input data.
        axis (int, optional) : Axis over which the sum is taken. By default
                `axis` is None, and all elements are summed.

    Returns:
        (Poly, frac, array_like) : Polynomial array with same shape as `vari`.

    Examples:
        >>> poly = cp.prange(3)
        >>> print(poly)
        [1, q0, q0^2]
        >>> print(cp.cumsum(poly))
        [1, q0+1, q0^2+q0+1]
    """
    if isinstance(vari, (list, tuple, int, float, np.ndarray)):
        return np.cumsum(vari, axis)

    elif isinstance(vari, chaospy.poly.fraction.frac):
        if not vari.shape:
            return vari

        if axis is None:
            denom = vari.a.flatten()
            nume = vari.b.flatten()
            axis = 0
        else:
            denom = np.rollaxis(vari.a, axis)
            nume = np.rollaxis(vari.b, axis)

        denom = [np.prod(nume[:i], 0)*np.prod(nume[i+1:], 0)*denom[i] \
            for i in range(len(nume))]
        denom = np.rollaxis(np.cumsum(denom, 0), axis)
        nume = np.rollaxis(np.cumprod(nume, 0), axis)
        return chaospy.poly.fraction.frac(denom, nume)

    elif isinstance(vari, Poly):
        core = vari.A.copy()
        for key, val in core.items():
            core[key] = cumsum(val, axis)
        return Poly(core, vari.dim, None, vari.dtype)

    raise NotImplementedError


def prod(vari, axis=None):
    """
    Product of the components of a shapeable quantity along a given axis.

    Args:
        vari (Poly, frac, array_like) : Input data.
        axis (int, optional) : Axis over which the sum is taken. By default
                `axis` is None, and all elements are summed.

    Returns:
        (Poly, frac, array_like) : Polynomial array with same shape as `vari`,
                with the specified axis removed. If `vari` is an 0-d array, or
                `axis` is None, a (non-iterable) component is returned.

    Examples:
        >>> vari = cp.prange(3)
        >>> print(vari)
        [1, q0, q0^2]
        >>> print(cp.prod(vari))
        q0^3
    """
    if isinstance(vari, (np.ndarray, float, list, tuple, int)):
        return np.prod(vari, axis)

    elif isinstance(vari, chaospy.poly.fraction.frac):
        chaospy.poly.fraction.frac(
            prod(vari.a, axis),
            prod(vari.b, axis),
        )

    elif isinstance(vari, Poly):
        if axis is None:
            vari = chaospy.poly.shaping.flatten(vari)
            axis = 0

        vari = chaospy.poly.shaping.rollaxis(vari, axis)
        out = vari[0]
        for poly in vari[1:]:
            out = out*poly
        return out

    raise NotImplementedError


def cumprod(vari, axis=None):
    """
    Perform the cumulative product of a shapeable quantity over a given axis.

    Args:
        vari (Poly, frac, array_like) : Input data.
        axis (int, optional) : Axis over which the product is taken.  By
                default, the product of all elements is calculated.

    Returns:
        (Poly) : An array shaped as `vari` but with the specified axis removed.

    Examples:
        >>> vari = cp.prange(4)
        >>> print(vari)
        [1, q0, q0^2, q0^3]
        >>> print(cp.cumprod(vari))
        [1, q0, q0^3, q0^6]
    """
    if isinstance(vari, (list, tuple, int, float, np.ndarray)):
        return np.cumprod(vari, axis)

    elif isinstance(vari, chaospy.poly.fraction.frac):
        chaospy.poly.fraction.frac(
            cumprod(vari.a, axis),
            cumprod(vari.b, axis),
        )

    elif isinstance(vari, Poly):
        if np.prod(vari.shape) == 1:
            return vari.copy()
        if axis is None:
            vari = chaospy.poly.shaping.flatten(vari)
            axis = 0

        vari = chaospy.poly.shaping.rollaxis(vari, axis)
        out = [vari[0]]

        for poly in vari[1:]:
            out.append(out[-1]*poly)
        return Poly(out, vari.dim, vari.shape, vari.dtype)

    raise NotImplementedError


if __name__ == '__main__':
    import doctest
    doctest.testmod()
