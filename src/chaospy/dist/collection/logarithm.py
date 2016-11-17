import numpy
import scipy.misc
import scipy.special

from chaospy.dist.baseclass import Dist


def Loguniform(lo=0, up=1, scale=1, shift=0):
    """
    Log-uniform distribution

    Args:
        lo (float, Dist) : Location of lower threshold
        up (float, Dist) : Location of upper threshold
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    class loguniform(Dist):
        def __init__(self, lo=0, up=1):
            Dist.__init__(self, lo=lo, up=up)
        def _pdf(self, x, lo, up):
            return 1./(x*(up-lo))
        def _cdf(self, x, lo, up):
            return (numpy.log(x)-lo)/(up-lo)
        def _ppf(self, q, lo, up):
            return numpy.e**(q*(up-lo) + lo)
        def _bnd(self, lo, up):
            return numpy.e**lo, numpy.e**up
        def _mom(self, k, lo, up):
            return ((numpy.e**(up*k)-numpy.e**(lo*k))/((up-lo)*(k+(k==0))))**(k!=0)
        def _str(self, lo, up):
            return "loguni(%s,%s)" % (lo, up)
    dist = loguniform(lo, up)*scale + shift
    dist.addattr(str="Loguniform(%s,%s,%s,%s)" % (lo,up,scale,shift))
    return dist

def Lognormal(mu=0, sigma=1, shift=0, scale=1):
    R"""
    Log-normal distribution

    Args:
        mu (float, Dist) : Mean in the normal distribution.  Overlaps with
                scale by mu=log(scale)
        sigma (float, Dist) : Standard deviation of the normal distribution.
        shift (float, Dist) : Location of the lower bound.
        scale (float, Dist) : Scale parameter. Overlaps with mu by scale=e**mu

    Examples:
        >>> cp.seed(1000)
        >>> f = cp.Lognormal(0, 1)
        >>> q = numpy.linspace(0,1,6)[1:-1]
        >>> print(f.inv(q))
        [ 0.43101119  0.77619841  1.28833038  2.32012539]
        >>> print(f.fwd(f.inv(q)))
        [ 0.2  0.4  0.6  0.8]
        >>> print(f.sample(4))
        [ 1.48442856  0.30109692  5.19451094  0.95632796]
        >>> print(f.mom(1))
        1.6487212707
    """
    class lognormal(Dist):
        def __init__(self, a=1):
            Dist.__init__(self, a=a)
        def _pdf(self, x, a):
            out = numpy.e**(-numpy.log(x+(1-x)*(x<=0))**2/(2*a*a)) / \
                ((x+(1-x)*(x<=0))*a*numpy.sqrt(2*numpy.pi))*(x>0)
            return out
        def _cdf(self, x, a):
            return scipy.special.ndtr(numpy.log(x+(1-x)*(x<=0))/a)*(x>0)
        def _ppf(self, x, a):
            return numpy.e**(a*scipy.special.ndtri(x))
        def _mom(self, k, a):
            return numpy.e**(.5*a*a*k*k)
        def _ttr(self, n, a):
            return \
        (numpy.e**(n*a*a)*(numpy.e**(a*a)+1)-1)*numpy.e**(.5*(2*n-1)*a*a), \
                    (numpy.e**(n*a*a)-1)*numpy.e**((3*n-2)*a*a)
        def _bnd(self, a):
            return 0, self._ppf(1-1e-10, a)
        def _str(self, a):
            return "lognor(%s)" % a
    dist = lognormal(sigma)*scale*numpy.e**mu + shift
    dist.addattr(str="Lognormal(%s,%s,%s,%s)"%(mu,sigma,shift,scale))
    return dist


def Fisk(shape=1, scale=1, shift=0):
    """
    Fisk or Log-logistic distribution.

    Args:
        shape (float, Dist) : Shape parameter
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    class fisk(Dist):

        def __init__(self, c=1.):
            Dist.__init__(self, c=c)
        def _pdf(self, x, c):
            return c*(x**(-c-1.0))*((1+x**(-c*1.0))**(-1.0))
        def _cdf(self, x, c):
            return (1+x**(-c*1.0))**(-1.0)
        def _ppf(self, q, c):
            return (q**(-1.0)-1)**(-1.0/c)
        def _bnd(self, c):
            return 0, self._ppf(1-1e-10, c)
    dist = fisk(c=shape)*scale + shift
    dist.addattr(str="Fisk(%s,%s,%s)"%(shape, scale, shift))
    return dist


def Logweibul(scale=1, loc=0):
    """
    Gumbel or Log-Weibull distribution.

    Args:
        scale (float, Dist) : Scaling parameter
        loc (float, Dist) : Location parameter
    """
    class gumbel(Dist):
        def __init__(self):
            Dist.__init__(self)
        def _pdf(self, x):
            ex = numpy.exp(-x)
            return ex*numpy.exp(-ex)
        def _cdf(self, x):
            return numpy.exp(-numpy.exp(-x))
        def _ppf(self, q):
            return -numpy.log(-numpy.log(q))
        def _bnd(self):
            return self._ppf(1e-10), self._ppf(1-1e-10)
    dist = gumbel()*scale + loc
    dist.addattr(str="Gumbel(%s,%s)"%(scale, loc))
    return dist


def Loggamma(shape=1, scale=1, shift=0):
    """
    Log-gamma distribution

    Args:
        shape (float, Dist) : Shape parameter
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    class loggamma(Dist):

        def __init__(self, c):
            Dist.__init__(self, c=c)
        def _pdf(self, x, c):
            return numpy.exp(c*x-numpy.exp(x)-scipy.special.gammaln(c))
        def _cdf(self, x, c):
            return scipy.special.gammainc(c, numpy.exp(x))
        def _ppf(self, q, c):
            return numpy.log(scipy.special.gammaincinv(c,q))
        def _bnd(self, c):
            return self._ppf(1e-10, c), self._ppf(1-1e-10, c)
    dist = loggamma(shape)*scale + shift
    dist.addattr(str="Loggamma(%s,%s,%s)"%(shape, scale, shift))
    return dist


def Loglaplace(shape=1, scale=1, shift=0):
    """
    Log-laplace distribution

    Args:
        shape (float, Dist) : Shape parameter
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    class loglaplace(Dist):

        def __init__(self, c):
            Dist.__init__(self, c=c)
        def _pdf(self, x, c):
            cd2 = c/2.0
            c = numpy.where(x < 1, c, -c)
            return cd2*x**(c-1)
        def _cdf(self, x, c):
            return numpy.where(x < 1, 0.5*x**c, 1-0.5*x**(-c))
        def _ppf(self, q, c):
            return numpy.where(q < 0.5, (2.0*q)**(1.0/c), (2*(1.0-q))**(-1.0/c))
        def _bnd(self, c):
            return 0.0, self._ppf(1-1e-10, c)
    dist = loglaplace(shape)*scale + shift
    dist.addattr(str="Loglaplace(%s,%s,%s)"%(shape, scale, shift))
    return dist


def Gilbrat(scale=1, shift=0):
    """
    Gilbrat distribution.

    Standard log-normal distribution

    Args:
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    dist = lognormal(1)*scale + shift
    dist.addattr(str="Gilbrat(%s,%s)"%(scale, shift))
    return dist
