import numpy
import scipy.misc
import scipy.special

from chaospy.dist.baseclass import Dist

def Genexpon(a=1, b=1, c=1, scale=1, shift=0):
    """
    Generalized exponential distribution.

    Args:
        a (float, Dist) : First shape parameter
        b (float, Dist) : Second shape parameter
        c (float, Dist) : Third shape parameter
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter

    Note:
        "An Extension of Marshall and Olkin's Bivariate Exponential Distribution",
        H.K. Ryu, Journal of the American Statistical Association, 1993.

        "The Exponential Distribution: Theory, Methods and Applications",
        N. Balakrishnan, Asit P. Basu.
    """
    class genexpon(Dist):
        def __init__(self, a=1, b=1, c=1):
            Dist.__init__(self, a=a, b=b, c=c)
        def _pdf(self, x, a, b, c):
            return (a+b*(-numpy.expm1(-c*x)))*numpy.exp((-a-b)*x+b*(-numpy.expm1(-c*x))/c)
        def _cdf(self, x, a, b, c):
            return -numpy.expm1((-a-b)*x + b*(-numpy.expm1(-c*x))/c)
        def _bnd(self, a, b, c):
            return 0, 10**10
    dist = genexpon(a=1, b=1, c=1)*scale + shift
    dist.addattr(str="Genexpon(%s,%s,%s)"%(a, b, c))
    return dist


def Genextreme(shape=0, scale=1, loc=0):
    """
    Generalized extreme value distribution
    Fisher-Tippett distribution

    Args:
        shape (float, Dist) : Shape parameter
        scale (float, Dist) : Scaling parameter
        loc (float, Dist) : Location parameter
    """
    class genextreme(Dist):
        def __init__(self, c=1):
            Dist.__init__(self, c=c)
        def _pdf(self, x, c):
            cx = c*x
            logex2 = numpy.where((c==0)*(x==x),0.0,numpy.log1p(-cx))
            logpex2 = numpy.where((c==0)*(x==x),-x,logex2/c)
            pex2 = numpy.exp(logpex2)
            logpdf = numpy.where((cx==1) | (cx==-numpy.inf),-numpy.inf,-pex2+logpex2-logex2)
            numpy.putmask(logpdf,(c==1) & (x==1),0.0)
            return numpy.exp(logpdf)

        def _cdf(self, x, c):
            loglogcdf = numpy.where((c==0)*(x==x),-x,numpy.log1p(-c*x)/c)
            return numpy.exp(-numpy.exp(loglogcdf))

        def _ppf(self, q, c):
            x = -numpy.log(-numpy.log(q))
            return numpy.where((c==0)*(x==x),x,-numpy.expm1(-c*x)/c)
        def _bnd(self, c):
            return self._ppf(1e-10, c), self._ppf(1-1e-10, c)
    dist = genextreme(shape)*scale + loc
    dist.addattr(str="Genextreme(%s,%s,%s)"%(shape, scale, loc))
    return dist


def Gengamma(shape1, shape2, scale, shift):
    """
    Generalized gamma distribution

    Args:
        shape1 (float, Dist) : Shape parameter 1
        shape2 (float, Dist) : Shape parameter 2
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    class gengamma(Dist):

        def __init__(self, x, a, c):
            Dist.__init__(self, a=a, c=c)
        def _pdf(self, x, a, c):
            return abs(c)* numpy.exp((c*a-1)*numpy.log(x)-x**c- scipy.special.gammaln(a))
        def _cdf(self, x, a, c):
            val = scipy.special.gammainc(a,x**c)
            cond = c + 0*val
            return numpy.where(cond>0,val,1-val)
        def _ppf(self, q, a, c):
            val1 = scipy.special.gammaincinv(a,q)
            val2 = scipy.special.gammaincinv(a,1.0-q)
            ic = 1.0/c
            cond = c+0*val1
            return numpy.where(cond > 0,val1**ic,val2**ic)
        def _mom(self, k, a, c):
            return scipy.special.gamma((c+k)*1./a)/scipy.special.gamma(c*1./a)
        def _bnd(self, a, c):
            return 0.0, self._ppf(1-1e-10, a, c)
    dist = gengamma(shape1, shape2)*scale + shift
    dist.addattr(
        str="Gengamma(%s,%s,%s,%s)"%(shape1,shape2,scale,shift))
    return dist


def Genhalflogistic(shape, scale, shift):
    """
    Generalized half-logistic distribution

    Args:
        shape (float, Dist) : Shape parameter
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    class genhalflogistic(Dist):
        def __init__(self, c=1):
            Dist.__init__(self, c=c)
        def _pdf(self, x, c):
            limit = 1.0/c
            tmp = (1-c*x)
            tmp0 = tmp**(limit-1)
            tmp2 = tmp0*tmp
            return 2*tmp0 / (1+tmp2)**2
        def _cdf(self, x, c):
            limit = 1.0/c
            tmp = (1-c*x)
            tmp2 = tmp**(limit)
            return (1.0-tmp2) / (1+tmp2)
        def _ppf(self, q, c):
            return 1.0/c*(1-((1.0-q)/(1.0+q))**c)
        def _bnd(self, c):
            return 0.0, 1/numpy.where(c<10**-10, 10**-10, c)
    dist = genhalflogistic(shape)*scale + shift
    dist.addattr(str="Genhalflogistic(%s,%s,%s)"%(shape, scale, shift))
    return dist
