"""
Functions related to the type of the coefficients.
"""
import numpy

from chaospy.poly.base import Poly
import chaospy.poly.fraction


def dtyping(*args):
    """
    Find least common denominator dtype.

    Examples:
        >>> print(dtyping(int, float))
        <type 'float'>
        >>> print(dtyping(int, Poly))
        <class 'chaospy.poly.base.Poly'>
    """
    args = list(args)

    for idx, arg in enumerate(args):

        if isinstance(arg, numpy.ndarray):
            args[idx] = arg.dtype

        elif isinstance(arg, (float, chaospy.poly.fraction.frac, int)):
            args[idx] = type(arg)

    for type_ in [
            Poly,
            float, numpy.float64,
            object, chaospy.poly.fraction.frac,
            int, numpy.int64,
    ]:
        if type_ in args:
            return type_
    raise ValueError(
        "dtypes not recognised " + str(args))


def asfloat(vari, limit=10**300):
    """
    Convert dtype of polynomial coefficients to float.

    Example:
        >>> poly = 2*cp.variable()+1
        >>> print(poly)
        2q0+1
        >>> print(cp.asfloat(poly))
        2.0q0+1.0
    """
    if isinstance(vari, (numpy.ndarray, float, int)):
        return numpy.asfarray(vari)

    elif isinstance(vari, chaospy.poly.fraction.frac):

        denom = vari.a
        denom = denom.flatten()

        enum = vari.b
        enum = enum.flatten()

        shape = denom.shape

        limit = 10**300
        while numpy.any(enum > limit):

            limit = limit/10**10
            if numpy.any(enum > limit):
                over = enum > limit
                denom[over], enum[over] = \
                    chaospy.poly.fraction.limit_denominator(denom[over],
                                                            enum[over], limit)

        denom, enum = denom.reshape(shape), enum.reshape(shape)
        return numpy.asfarray(denom*1./enum)

    elif isinstance(vari, Poly):
        core = vari.A.copy()
        for key in vari.keys:
            core[key] = core[key]*1.
        return Poly(core, vari.dim, vari.shape, float)

    raise NotImplementedError


def asfrac(vari, limit=None):
    """
    Convert dtype of coefficients to fraction.

    Args:
        poly (Poly) : polynoomial to convert.
        limit (int) : fraction size limitation.

    Returns:
        (Poly) : polynomial with fraction coefficients.

    Example:
        >>> poly = .5*cp.variable()+.25
        >>> print(poly)
        0.5q0+0.25
        >>> print(cp.asfrac(poly))
        1/2q0+1/4
    """
    if isinstance(vari, (numpy.ndarray, float, int)):
        return chaospy.poly.fraction.frac(vari, 1, limit)

    elif isinstance(vari, fracion.frac):
        return chaospy.poly.fraction.frac(vari.a, vari.b, limit)

    elif isinstance(vari, p.Poly):
        core = poly.A.copy()
        for key in poly.keys:
            core[key] = chaospy.poly.fraction.frac(core[key], 1, limit)

        out = Poly(core, poly.dim, poly.shape, chaospy.poly.fraction.frac)
        return out

    raise NotImplementedError


def asint(vari):
    """
    Convert dtype of polynomial coefficients to float.

    Example:
        >>> poly = 1.5*cp.variable()+2.25
        >>> print(poly)
        1.5q0+2.25
        >>> print(cp.asint(poly))
        q0+2
    """
    if isinstance(vari, (numpy.ndarray, float, int)):
        return numpy.array(vari, dtype=int)

    elif isinstance(vari, f.frac):
        return vari.a//vari.b

    elif isinstance(vari, p.Poly):

        core = vari.A.copy()
        if vari.dtype == chaospy.poly.fraction.frac:
            for key in vari.keys:
                core[key] = core[key].a//core[key].b
        else:
            for key in vari.keys:
                core[key] = numpy.array(core[key], dtype=int)

        return Poly(core, vari.dim, vari.shape, int)

    raise NotImplementedError


def tolist(poly):
    """
    Convert polynomial array into a list of polynomials.

    Examples:
        >>> poly = cp.prange(3)
        >>> print(poly)
        [1, q0, q0^2]
        >>> array = cp.tolist(poly)
        >>> print(isinstance(array, list))
        True
        >>> print(array[1])
        q0
    """
    return toarray(poly).tolist()


def toarray(A):
    """
    Convert polynomial array into a numpy array of polynomials.

    Args:
        A (p.Poly, f.frac, array_like) : Input data.

    Returns:
        Q (ndarray) : A numpy.ndarray with `Q.shape==A.shape`.

    Examples:
        >>> poly = cp.prange(3)
        >>> print(poly)
        [1, q0, q0^2]
        >>> array = cp.toarray(poly)
        >>> print(isinstance(array, numpy.ndarray))
        True
        >>> print(array[1])
        q0
    """
    if isinstance(vari, Poly):
        shape = vari.shape
        out = numpy.array(
            [{} for _ in range(numpy.prod(shape))],
            dtype=object
        )
        core = vari.A.copy()
        for key in core.keys():

            core[key] = core[key].flatten()

            for i in range(numpy.prod(shape)):

                if not numpy.all(core[key][i] == 0):
                    out[i][key] = core[key][i]

        for i in range(numpy.prod(shape)):
            out[i] = Poly(out[i], vari.dim, (), vari.dtype)

        out = out.reshape(shape)
        return out

    elif isinstance(vari, (int, float, list, tuple, numpy.ndarray)):
        return numpy.array(vari)

    elif isinstance(A, f.frac):
        return f.toarray(A)

    raise ValueError("variable type not recognised ({})".format(vari))



if __name__ == '__main__':
    import chaospy as cp # pylint: disable=unused-import
    import doctest
    doctest.testmod()
