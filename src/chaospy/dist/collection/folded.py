import numpy
import scipy.misc
import scipy.special

from chaospy.dist.baseclass import Dist


def Foldcauchy(shape=0, scale=1, shift=0):
    """
    Folded Cauchy distribution.

    Args:
        shape (float, Dist) : Shape parameter
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    class foldcauchy(Dist):

        def __init__(self, c=0):
            Dist.__init__(self, c=c)
        def _pdf(self, x, c):
            return 1.0/numpy.pi*(1.0/(1+(x-c)**2) + 1.0/(1+(x+c)**2))
        def _cdf(self, x, c):
            return 1.0/numpy.pi*(numpy.arctan(x-c) + numpy.arctan(x+c))
        def _bnd(self, c):
            return 0, 10**10
    dist = foldcauchy(shape)*scale + shift
    dist.addattr(str="Foldcauchy(%s,%s,%s)"%(shape, scale, shift))
    return dist


def Foldnormal(mu=0, sigma=1, loc=0):
    """
    Folded normal distribution.

    Args:
        mu (float, Dist) : Location parameter in normal distribution
        sigma (float, Dist) : Scaling parameter (in both normal and fold)
        loc (float, Dist) : Location of fold
    """
    class foldnorm(Dist):

        def __init__(self, c=1):
            Dist.__init__(self, c=c)
        def _pdf(self, x, c):
            return numpy.sqrt(2.0/numpy.pi)*numpy.cosh(c*x)*numpy.exp(-(x*x+c*c)/2.0)
        def _cdf(self, x, c):
            return scipy.special.ndtr(x-c) + scipy.special.ndtr(x+c) - 1.0
        def _bnd(self, c):
            return 0, 7.5+c
    dist = foldnorm(mu-loc)*sigma + loc
    dist.addattr(str="Foldnorm(%s,%s,%s)"%(mu, sigma, loc))
    return dist
