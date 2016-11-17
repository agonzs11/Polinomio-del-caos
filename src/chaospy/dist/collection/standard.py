import numpy
import scipy.misc
import scipy.special

from chaospy.dist.baseclass import Dist
import chaospy


def Uniform(lo=0, up=1):
    r"""
    Uniform distribution

    Args:
        lo (float, Dist) : Lower threshold of distribution. Must be smaller than up.
        up (float, Dist) : Upper threshold of distribution.

    Examples:
        >>> cp.seed(1000)
        >>> f = cp.Uniform(2, 4)
        >>> q = numpy.linspace(0,1,5)
        >>> print(f.inv(q))
        [ 2.   2.5  3.   3.5  4. ]
        >>> print(f.fwd(f.inv(q)))
        [ 0.    0.25  0.5   0.75  1.  ]
        >>> print(f.sample(4))
        [ 3.30717917  2.23001389  3.90056573  2.9643828 ]
        >>> print(f.mom(1))
        3.0
    """
    uniform = chaospy.dist.construct(
        cdf=lambda x: .5*x+.5,
        bnd=lambda: (-1, 1),
        pdf=lambda x: .5,
        ppf=lambda q: 2*q-1,
        mom=lambda k: 1./(k+1)*(k%2 == 0),
        ttr=lambda n: (0., n*n/(4.*n*n-1)),
    )
    dist = uniform()*((up-lo)*.5)+((up+lo)*.5)
    dist.addattr(str="Uniform(%s,%s)"%(lo,up))
    return dist


def Normal(mu=0, sigma=1):
    R"""
    Normal (Gaussian) distribution

    Args:
        mu (float, Dist) : Mean of the distribution.
        sigma (float, Dist) : Standard deviation.  sigma > 0

    Examples:
        >>> cp.seed(1000)
        >>> f = cp.Normal(2, 2)
        >>> q = numpy.linspace(0,1,6)[1:-1]
        >>> print(f.inv(q))
        [ 0.31675753  1.49330579  2.50669421  3.68324247]
        >>> print(f.fwd(f.inv(q)))
        [ 0.2  0.4  0.6  0.8]
        >>> print(f.sample(4))
        [ 2.79005978 -0.40064618  5.29520496  1.91069125]
        >>> print(f.mom(1))
        2.0
    """
    normal = chaospy.dist.construct(
        cdf=lambda x: scipy.special.ndtr(x),
        bnd=lambda: (-7.5, 7.5),
        pdf=lambda x: (2*numpy.pi)**(-.5)*numpy.e**(-x**2/2.),
        ttr=lambda n: (0., 1.*n),
        mom=lambda k: .5*scipy.misc.factorial2(k-1)*(1+(-1)**k),
        ppf=lambda x: scipy.special.ndtri(x),
    )
    dist = normal()*sigma + mu
    dist.addattr(str="Normal(%s,%s)"%(mu, sigma))
    return dist


def Exponential(scale=1, shift=0):
    R"""
    Exponential Probability Distribution

    Args:
        scale (float, Dist) : Scale parameter. scale!=0
        shift (float, Dist) : Location of the lower bound.

    Examples;:
        >>> cp.seed(1000)
        >>> f = cp.Exponential(1)
        >>> q = numpy.linspace(0,1,6)[1:-1]
        >>> print(f.inv(q))
        [ 0.22314355  0.51082562  0.91629073  1.60943791]
        >>> print(f.fwd(f.inv(q)))
        [ 0.2  0.4  0.6  0.8]
        >>> print(f.sample(4))
        [ 1.06013104  0.12217548  3.00140562  0.65814961]
        >>> print(f.mom(1))
        1.0
    """
    expon = chaospy.dist.construct(
        cdf=lambda x: 1-numpy.e**-x,
        bnd=lambda: (0, 42),
        pdf=lambda x: numpy.e**-x,
        ppf=lambda q: -numpy.log(1-q),
        mom=lambda k: scipy.misc.factorial(k),
        ttr=lambda n: (2*n+1, n*n),
    )
    dist = expon()*scale + shift
    dist.addattr(str="Expon(%s,%s)" % (scale, shift))
    return dist


def Gamma(shape=1, scale=1, shift=0):
    """
    Gamma distribution.

    Also an Erlang distribution when shape=k and scale=1./lamb.

    Args:
        shape (float, Dist) : Shape parameter. a>0
        scale () : Scale parameter. scale!=0
        shift (float, Dist) : Location of the lower bound.

    Examples:
        >>> cp.seed(1000)
        >>> f = cp.Gamma(1, 1)
        >>> q = numpy.linspace(0,1,6)[1:-1]
        >>> print(f.inv(q))
        [ 0.22314355  0.51082562  0.91629073  1.60943791]
        >>> print(f.fwd(f.inv(q)))
        [ 0.2  0.4  0.6  0.8]
        >>> print(f.sample(4))
        [ 1.06013104  0.12217548  3.00140562  0.65814961]
        >>> print(f.mom(1))
        1.0
    """
    gamma = chaospy.dist.construct(
        pdf=lambda x, a: x**(a-1)*numpy.e**(-x)/scipy.special.gamma(a),
        cdf=lambda x, a: scipy.special.gammainc(a, x),
        ppf=lambda q, a: scipy.special.gammaincinv(a, q),
        mom=lambda k, a: scipy.special.gamma(a+k)/scipy.special.gamma(a),
        ttr=lambda n, a: (2*n+a, n*n+n*(a-1)),
        bnd=lambda a: (0, 40+2*a),
    )
    dist = gamma(a=shape)*scale + shift
    dist.addattr(str="Gamma(%s,%s,%s)"%(shape, scale, shift))
    return dist


def _beta_ttr(n, a, b):
    nab = 2*n+a+b
    A = ((a-1)**2-(b-1)**2)*.5/\
            (nab*(nab-2) + (nab==0) + (nab==2)) + .5
    B1 = a*b*1./((a+b+1)*(a+b)**2)
    B2 = (n+a-1)*(n+b-1)*n*(n+a+b-2.)/\
        ((nab-1)*(nab-3)*(nab-2)**2+2.*((n==0)+(n==1)))
    B = numpy.where((n==0)+(n==1), B1, B2)
    return A, B

def Beta(a, b, lo=0, up=1):
    """
    Beta probability distribution.

    Args:
        a (float, Dist) : First shape parameter, a > 0
        b (float, Dist) : Second shape parameter, b > 0
        lo (float, Dist) : Lower threshold
        up (float, Dist) : Upper threshold

    Examples:
        >>> cp.seed(1000)
        >>> f = cp.Beta(2, 2, 2, 3)
        >>> q = numpy.linspace(0,1,6)[1:-1]
        >>> print(f.inv(q))
        [ 2.28714073  2.43293108  2.56706892  2.71285927]
        >>> print(f.fwd(f.inv(q)))
        [ 0.2  0.4  0.6  0.8]
        >>> print(f.sample(4))
        [ 2.60388804  2.21123197  2.86505298  2.48812537]
        >>> print(f.mom(1))
        2.5
    """
    beta = chaospy.dist.construct(
        pdf=lambda x, a, b: (x**(a-1)*(1-x)**(b-1)/
                             scipy.special.beta(a, b)),
        cdf=lambda x, a, b: scipy.special.btdtr(a, b, x),
        ppf=lambda q, a, b: scipy.special.btdtri(a, b, q),
        mom=lambda k, a, b: scipy.special.beta(a+k,b)/scipy.special.beta(a,b),
        ttr=_beta_ttr,
        bnd=lambda a, b: (0., 1.),
    )
    dist = beta(a, b)*(up-lo) + lo
    dist.addattr(str="Beta(%s,%s,%s,%s)" % (a,b,lo,up))
    return dist
