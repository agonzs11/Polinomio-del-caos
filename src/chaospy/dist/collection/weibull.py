import numpy
import scipy.misc
import scipy.special

from chaospy.dist.baseclass import Dist


def Weibull(shape=1, scale=1, shift=0):
    """
    Weibull Distribution.

    Args:
        shape (float, Dist) : Shape parameter.
        scale (float, Dist) : Scale parameter.
        shift (float, Dist) : Location of lower bound.

    Examples:
        >>> cp.seed(1000)
        >>> f = cp.Weibull(2)
        >>> q = numpy.linspace(0,1,6)[1:-1]
        >>> print(f.inv(q))
        [ 0.47238073  0.71472066  0.95723076  1.26863624]
        >>> print(f.fwd(f.inv(q)))
        [ 0.2  0.4  0.6  0.8]
        >>> print(f.sample(4))
        [ 1.02962665  0.34953609  1.73245653  0.8112642 ]
        >>> print(f.mom(1))
        0.886226925453
    """
    class weibull(Dist):

        def __init__(self, a=1):
            Dist.__init__(self, a=a)
        def _pdf(self, x, a):
            return a*x**(a-1)*numpy.e**(-x**a)
        def _cdf(self, x, a):
            return (1-numpy.e**(-x**a))
        def _ppf(self, q, a):
            return (-numpy.log(1-q+1*(q==1)))**(1./a)*(q!=1) +\
                30.**(1./a)*(q==1)
        def _mom(self, k, a):
            return scipy.special.gamma(1.+k*1./a)
        def _bnd(self, a):
            return 0, 30.**(1./a)
        def _str(self, a):
            return "wei(%s)" % a
    dist = weibull(shape)*scale + shift
    dist.addattr(str="Weibull(%s,%s,%s)" % (shape, scale, shift))
    return dist


def Dbl_weibull(shape=1, scale=1, shift=0):
    """
    Double weibull distribution.

    Args:
        shape (float, Dist) : Shape parameter
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    class dbl_weibull(Dist):

        def __init__(self, c):
            Dist.__init__(self, c=c)
        def _pdf(self, x, c):
            ax = numpy.abs(x)
            Px = c/2.0*ax**(c-1.0)*numpy.exp(-ax**c)
            return Px
        def _cdf(self, x, c):
            Cx1 = 0.5*numpy.exp(-abs(x)**c)
            return numpy.where(x > 0, 1-Cx1, Cx1)
        def _ppf(self, q, c):
            q_ = numpy.where(q>.5, 1-q, q)
            Cq1 = (-numpy.log(2*q_))**(1./c)
            return numpy.where(q>.5, Cq1, -Cq1)
        def _bnd(self, c):
            return self._ppf(1e-10, c), self._ppf(1-1e-10, c)
    dist = dbl_weibull(shape)*scale + shift
    dist.addattr(str="Dbl_weibull(%s,%s,%s)"%(shape, scale, shift))
    return dist


def Exponweibull(a=1, b=1, scale=1, shift=0):
    """
    Expontiated Weibull distribution.

    Args:
        a (float, Dist) : First shape parameter
        b (float, Dist) : Second shape parameter
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    class exponweibull(Dist):

        def __init__(self, a=1, c=1):
            Dist.__init__(self, a=a, c=c)
        def _pdf(self, x, a, c):
            exc = numpy.exp(-x**c)
            return a*c*(1-exc)**(a-1) * exc * x**(c-1)
        def _cdf(self, x, a, c):
            exm1c = -numpy.expm1(-x**c)
            return (exm1c)**a
        def _ppf(self, q, a, c):
            return (-numpy.log1p(-q**(1.0/a)))**(1.0/c)
        def _bnd(self, c):
            return 0, self._ppf(1-1e-10, c)
    dist = exponweibull(a, b)*scale + shift
    dist.addattr(str="Exponweibull(%s,%s,%s,%s)"%(a, b, scale,shift))
    return dist

