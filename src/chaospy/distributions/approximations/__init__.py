"""
Collection of approximation methods

Global methods are used on a distribution as a wrapper. Local
function are used by the graph-module as part of calculations.

These include:

``pdf``
    Probability density function (local)
``pdf_full``
    Probability density function (global)
``ppf``
    Inverse CDF (local)
``inv``
    Inverse CDF (global)
``mom``
    Raw statistical moments (global)
``find_interior_point``
    Find an interior point (global)
"""
from .pdf import pdf, pdf_full
from .inv import inv
from .ppf import ppf




import numpy

import chaospy.quad

from ..baseclass import Dist


def mom(dist, K, retall=False, control_var=None,
        **kws):
    """
Approxmethod for estimation of raw statistical moments.

Parameters
----------
dist : Dist
    Distribution domain with dim=len(dist)
K : numpy.ndarray
    The exponents of the moments of interest with shape (dim,K).

Optional keywords

control_var : Dist
    If provided will be used as a control variable to try to reduce
    the error.
acc : int, optional
    The order of quadrature/MCI
sparse : bool
    If True used Smolyak's sparse grid instead of normal tensor
    product grid in numerical integration.
rule : str
    Quadrature rule
    Key     Description
    ----    -----------
    "G"     Optiomal Gaussian quadrature from Golub-Welsch
            Slow for high order and composit is ignored.
    "E"     Gauss-Legendre quadrature
    "C"     Clenshaw-Curtis quadrature. Exponential growth rule is
            used when sparse is True to make the rule nested.

    Monte Carlo Integration
    Key     Description
    ----    -----------
    "H"     Halton sequence
    "K"     Korobov set
    "L"     Latin hypercube sampling
    "M"     Hammersley sequence
    "R"     (Pseudo-)Random sampling
    "S"     Sobol sequence

composit : int, array_like optional
    If provided, composit quadrature will be used.
    Ignored in the case if gaussian=True.

    If int provided, determines number of even domain splits
    If array of ints, determines number of even domain splits along
        each axis
    If array of arrays/floats, determines location of splits

antithetic : array_like, optional
    List of bool. Represents the axes to mirror using antithetic
    variable during MCI.
    """

    dim = len(dist)
    shape = K.shape
    size = int(K.size/dim)
    K = K.reshape(dim,size)

    if dim>1:
        shape = shape[1:]

    order = kws.pop("order", 40)
    X,W = chaospy.quad.generate_quadrature(order, dist, **kws)


    grid = numpy.mgrid[:len(X[0]),:size]
    X = X.T[grid[0]].T
    K = K.T[grid[1]].T
    out = numpy.prod(X**K, 0)*W

    if not (control_var is None):

        Y = control_var.ppf(dist.fwd(X))
        mu = control_var.mom(numpy.eye(len(control_var)))

        if mu.size==1 and dim>1:
            mu = mu.repeat(dim)

        for d in range(dim):
            alpha = numpy.cov(out, Y[d])[0,1]/numpy.var(Y[d])
            out -= alpha*(Y[d]-mu)

    out = numpy.sum(out, -1)

    return out


def find_interior_point(dist):
    """
Find interior point using the range-function

Parameters
----------
dist : Dist
    Distribution to find interior on.

Returns
-------
interior_point : numpy.ndarray
    shape=(len(dist),)
    """
    try:
        x = dist.inv([.5]*len(dist))
        return x
    except:
        pass

    bnd = dist.range(numpy.zeros(len(dist)))
    x = .5*(bnd[1]-bnd[0])

    for i in range(10):
        bnd = dist.range(x)
        x_ = .5*(bnd[1]-bnd[0])
        if numpy.allclose(x, x_):
            break
        x = x_

    return x

# TODO: integrate these two functions.
def ttr(order, domain, **kws):

    prm = kws
    prm["accuracy"] = order
    prm["retall"] = True

    def _three_terms_recursion(self, keys, **kws):
        _, _, coeffs1, coeffs2 = chaospy.quad.generate_stieltjes(
            domain, numpy.max(keys)+1, **self1.prm)
        out = numpy.ones((2,) + keys.shape)
        idx = 0
        for idzs in keys.T:
            idy = 0
            for idz in idzs:
                if idz:
                    out[:, idy, idx] = coeffs1[idy, idz], coeffs2[idy, idz]
                idy += 1
            idx += 1

    return _three_terms_recursion

def moment_generator(order, domain, accuracy=100, sparse=False, rule="C",
                     composite=1, part=None, trans=lambda x:x, **kws):
    """Moment generator."""
    if isinstance(domain, Dist):
        dim = len(domain)
    else:
        dim = numpy.array(domain[0]).size

    if not numpy.array(trans(numpy.zeros(dim))).shape:
        func = trans
        trans = lambda x: [func(x)]

    if part is None:

        abscissas, weights = chaospy.quad.generate_quadrature(
            order, domain=domain, accuracy=accuracy, sparse=sparse,
            rule=rule, composite=composite, part=part, **kws)
        values = numpy.transpose(trans(abscissas))

        def moment_function(keys):
            """Raw statistical moment function."""
            return numpy.sum(numpy.prod(values**keys, -1)*weights, 0)
    else:

        isdist = isinstance(domain, Dist)
        if isdist:
            lower, upper = domain.range()
        else:
            lower, upper = numpy.array(domain)

        abscissas = []
        weights = []
        values = []
        for idx in numpy.ndindex(*part):
            abscissa, weight = chaospy.quad.collection.clenshaw_curtis(
                order, lower, upper, part=(idx, part))
            value = numpy.array(trans(abscissa))

            if isdist:
                weight *= domain.pdf(abscissa).flatten()

            if numpy.any(weight):
                abscissas.append(abscissa)
                weights.append(weight)
                values.append(value)

        def moment_function(keys):
            """Raw statistical moment function."""
            out = 0.
            for idx in range(len(abscissas)):
                out += numpy.sum(
                    numpy.prod(values[idx].T**keys, -1)*weights[idx], 0)
            return out

    def mom(keys, **kws):
        """Statistical moment function."""
        return numpy.array([moment_function(key) for key in keys.T])

    return mom

