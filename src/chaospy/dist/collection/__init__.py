"""
Frontend for the collection distributions.

This modules provides a wrapper with documentation for the dist.cores module.
"""
from .standard import (
    Uniform,
    Normal,
    Exponential,
    Gamma,
    Beta,
)
from .logarithm import (
    Loguniform,
    Lognormal,
    Fisk,
    Logweibul,
    Loggamma,
    Loglaplace,
    Gilbrat,
)
from .weibull import (
    Weibull,
    Dbl_weibull,
    Exponweibull,
)
from .folded import (
    Foldcauchy,
    Foldnormal,
)
from .generalisations import (
    Genexpon,
    Genextreme,
    Gengamma,
    Genhalflogistic,
)
from .multivariate import (
    MvNormal,
    MvLognormal,
    MvStudent_t,
)


import numpy
from scipy.stats import gaussian_kde

from chaospy.dist.baseclass import Dist

def Alpha(shape=1, scale=1, shift=0):
    """
    Alpha distribution.

    Args:
        shape (float, Dist) : Shape parameter
        scale (float, Dist) : Scale Parameter
        shift (float, Dist) : Location of lower threshold
    """
    alpha = chaospy.dist.construct(
        cdf=lambda x, a: scipy.special.ndtr(a-1./x) / scipy.special.ndtr(a),
        ppf=lambda q, a: 1.0/(a-scipy.special.ndtri(q*scipy.special.ndtr(a))),
        pdf=lambda x, a: (1.0/(x**2)/scipy.special.ndtr(a)*
                          numpy.e**(.5*(a-1.0/x)**2)/numpy.sqrt(2*numpy.pi)),
        bnd=lambda a: (0,10000.),
    )
    dist = alpha(shape)*scale + shift
    dist.addattr(str="Alpha(%s,%s,%s)" % (shape, scale, shift))
    return dist


def Anglit(loc=0, scale=1):
    """
    Anglit distribution.

    Args:
        loc (float, Dist) : Location parameter
        scale (float, Dist) : Scaling parameter
    """
    anglit = chaospy.dist.construct(
        pdf=lambda x: numpy.cos(2*x),
        cdf=lambda x: numpy.sin(x+numpy.pi/4)**2.0,
        ppf=lambda q: (numpy.arcsin(numpy.sqrt(q))-numpy.pi/4),
        bnd=lambda: (-numpy.pi/4, numpy.pi/4),
    )
    dist = anglit()*scale + loc
    dist.addattr(str="Anglit(%s,%s)"(loc, scale))
    return dist


def Arcsinus(shape=0.5, lo=0, up=1):
    """
    Generalized Arc-sinus distribution

    shape : float, Dist
        Shape parameter where 0.5 is the default non-generalized case.
    lo : float, Dist
        Lower threshold
    up : float, Dist
        Upper threshold
    """
    dist = Beta(shape, 1-shape)*(up-lo) + lo
    dist.addattr(str="Arcsinus(%s,%s,%s)" % (shape, lo, up))
    return dist


def Bradford(shape=1, lo=0, up=1):
    """
    Bradford distribution.

    Args:
        shape (float, Dist) : Shape parameter
        lo (float, Dist) : Location of lower threshold
        up (float, Dist) : Location of upper threshold
    """
    bradford = chaospy.dist.construct(
        pdf=lambda x, c:  c / (c*x + 1.0) / numpy.log(1.0+c),
        cdf=lambda x, c: numpy.log(1.0+c*x) / numpy.log(c+1.0),
        ppf=lambda q, c: ((1.0+c)**q-1)/c,
        bnd=lambda c: (0, 1),
    )
    dist = bradford(c=shape)*(up-lo) + lo
    dist.addattr(str="Bradford(%s,%s,%s)"%(shape, lo, up))
    return dist


def Burr(c=1, d=1, loc=0, scale=1):
    """
    Burr Type XII or Singh-Maddala distribution.

    Args:
        c (float, Dist) : Shape parameter
        d (float, Dist) : Shape parameter
        loc (float, Dist) : Location parameter
        scale (float, Dist) : Scaling parameter
    """
    burr = chaospy.dist.construct(
        pdf=lambda x, c, d: c*d*(x**(-c-1.0))*((1+x**(-c*1.0))**(-d-1.0)),
        cdf=lambda x, c, d: (1+x**(-c*1.0))**(-d**1.0),
        ppf=lambda q, c, d: (q**(-1.0/d)-1)**(-1.0/c),
        bnd=lambda c, d: (0, (.9999999**(-1./d)-1)**(-1./c)),
    )
    dist = burr(c=1., d=1.)*scale + loc
    dist.addattr(str="Burr(%s,%s,%s,%s)"%(c, d, loc, scale))
    return dist


def Cauchy(loc=0, scale=1):
    """
    Cauchy distribution.

    Args:
        loc (float, Dist) : Location parameter
        scale (float, Dist) : Scaling parameter
    """
    cauchy = chaospy.dist.construct(
        pdf=lambda x: 1.0/numpy.pi/(1.0+x*x),
        cdf=lambda x: 0.5 + 1.0/numpy.pi*numpy.arctan(x),
        ppf=lambda q: numpy.tan(numpy.pi*q-numpy.pi/2.0),
        bnd=lambda: (-1000, 1000),
    )
    dist = cauchy()*scale + loc
    dist.addattr(str="Cauchy(%s,%s)"%(loc,scale))
    return dist


def Chi(df=1, scale=1, shift=0):
    """
    Chi distribution.

    Args:
        df (float, Dist) : Degrees of freedom
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    chi = chaospy.dist.construct(
        pdf=lambda x, df: (x**(df-1.)*numpy.exp(-x*x*0.5)/(2.0)**(df*0.5-1)
                           /scipy.special.gamma(df*0.5)),
        cdf=lambda x, df: scipy.special.gammainc(df*0.5,0.5*x*x),
        ppf=lambda q, df: numpy.sqrt(2*scipy.special.gammaincinv(df*0.5,q)),
        bnd=lambda df: (0, 10000.),
        mom=lambda k, df: (2**(.5*k)*scipy.special.gamma(.5*(df+k))
                           /scipy.special.gamma(.5*df)),
    )
    dist = chi(df)*scale + shift
    dist.addattr(str="Chi(%s,%s,%s)"%(df, scale, shift))
    return dist


def Chisquard(df=1, scale=1, shift=0, nc=0):
    """
    (Non-central) Chi-squared distribution.

    Args:
        df (float, Dist) : Degrees of freedom
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
        nc (float, Dist) : Non-centrality parameter
    """
    def _chisquared_pdf(x, df, nc):
        a = df/2.0
        fac = (-nc-x)/2.0 + (a-1)*numpy.log(x)-a*numpy.log(2)-scipy.special.gammaln(a)
        fac += numpy.nan_to_num(numpy.log(scipy.special.hyp0f1(a, nc * x/4.0)))
        return numpy.numpy.exp(fac)
    chisquared = chaospy.dist.construct(
        pdf=_chisquared_pdf,
        cdf=lambda x, df, nc: scipy.special.chndtr(x,df,nc),
        ppf=lambda q, df, nc: scipy.special.chndtrix(q,df,nc),
        bnd=lambda df, nc: (0.0, 10000.),
    )
    dist = chisquared(df, nc)*scale + shift
    dist.addattr(str="Chisquared(%s,%s,%s,%s)"%(df, nc,scale,shift))
    return dist


def Dbl_gamma(shape=1, scale=1, shift=0):
    """
    Double gamma distribution.

    Args:
        shape (float, Dist) : Shape parameter
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    def _dblgamma_cdf(x, a):
        fac = 0.5*scipy.special.gammainc(a,abs(x))
        return numpy.where(x>0,0.5+fac,0.5-fac)
    dbl_gamma = chaospy.dist.construct(
        pdf= lambda x, a: (1.0/(2*scipy.special.gamma(a))*
                           abs(x)**(a-1.0) * numpy.exp(-abs(x))),
        cdf=_dblgamma_cdf,
        ppf=lambda q, a: (scipy.special.gammainccinv(a,1-abs(2*q-1))*
                          numpy.where(q>0.5, 1, -1)),
        bnd=lambda a: (-1000, 1000),
    )
    dist = dbl_gamma(shape)*scale + shift
    dist.addattr(str="Dbl_gamma(%s,%s,%s)"%(shape, scale, shift))
    return dist


def Exponpow(shape=0, scale=1, shift=0):
    """
    Expontial power distribution.

    Also known as Generalized error distribution and Generalized normal
    distribution version 1.

    Args:
        shape (float, Dist) : Shape parameter
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    exponpow = chaospy.dist.construct(
        pdf=lambda x, b: (numpy.exp(1)*b*x**(b-1)*
                          numpy.exp(x**b - numpy.exp(x**b))),
        cdf=lambda x, b: -numpy.expm1(-numpy.expm1(x**b)),
        ppf=lambda q, b: numpy.pow(numpy.log1p(-numpy.log1p(-q)), 1.0/b),
        bnd=lambda b: (0, 100000.),
    )
    dist = exponpow(shape)*scale + shift
    dist.addattr(str="Exponpow(%s,%s,%s)"%(shape, scale, shift))
    return dist


def F(n=1, m=1, scale=1, shift=0, nc=0):
    """
    (Non-central) F or Fisher-Snedecor distribution.

    Args:
        n (float, Dist) : Degres of freedom for numerator
        m (float, Dist) : Degres of freedom for denominator
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
        nc (float, Dist) : Non-centrality parameter
    """
    f = chaospy.dist.construct(
        cdf=lambda x, dfn, dfd, nc: scipy.special.ncfdtr(dfn,dfd,nc,x),
        ppf=lambda q, dfn, dfd, nc: scipy.special.ncfdtri(dfn, dfd, nc, q),
        bnd=lambda dfn, dfd, nc: (0.0, 10000.),
    )
    dist = f(n, m, nc)*scale + shift
    dist.addattr(str="F(%s,%s,%s,%s,%s)"%(n, m, scale, shift, nc))
    return dist


def Fatiguelife(shape=1, scale=1, shift=0):
    """
    Fatigue-Life or Birmbaum-Sanders distribution

    Args:
        shape (float, Dist) : Shape parameter
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    fatiguelife = chaospy.dist.construct(
        pdf=lambda x, c: ((x+1)/(2*c*numpy.sqrt(2*numpy.pi*x**3))*
                          numpy.exp(-(x-1)**2/(2.0*x*c**2))),
        cdf=lambda x, c: (scipy.special.ndtr(
            1.0/c*(numpy.sqrt(x)-1.0/numpy.sqrt(x)))),
        ppf=lambda q, c: 0.25*(c*scipy.special.ndtri(q)+
                               numpy.sqrt(c*scipy.special.ndtri(q)**2 + 4))**2,
        bnd=lambda c: (0, 10000.),
    )
    dist = fatiguelife(shape)*scale + shift
    dist.addattr(str="Fatiguelife(%s,%s,%s)"%(shape, scale, shift))
    return dist


def Frechet(shape=1, scale=1, shift=0):
    """
    Frechet or Extreme value distribution type 2.

    Args:
        shape (float, Dist) : Shape parameter
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    frechet = chaospy.dist.construct(
        pdf=lambda x, c: c*pow(x,c-1)*numpy.exp(-pow(x,c)),
        cdf=lambda x, c: -numpy.expm1(-pow(x,c)),
        ppf=lambda q, c: pow(-numpy.log1p(-q),1.0/c),
        mom=lambda k, c: scipy.special.gamma(1-k*1./c),
        bnd=lambda c: (0, 10000.),
    )
    dist = frechet(shape)*scale + shift
    dist.addattr(str="Frechet(%s,%s,%s)"%(shape, scale, shift))
    return dist


def Gompertz(shape, scale, shift):
    """
    Gompertz distribution

    Args:
        shape (float, Dist) : Shape parameter
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    gompertz = chaospy.dist.construct(

        pdf=lambda x, c: c*numpy.exp(x)*numpy.exp(-c*(numpy.exp(x)-1)),
        cdf=lambda x, c: 1.0-numpy.exp(-c*(numpy.exp(x)-1)),
        ppf=lambda q, c: numpy.log(1-1.0/c*numpy.log(1-q)),
        bnd=lambda c: (0.0, 1000.),
    )
    dist = gompertz(shape)*scale + shift
    dist.addattr(str="Gompertz(%s,%s,%s)"%(shape, scale, shift))
    return dist


def Hypgeosec(loc=0, scale=1):
    """
    hyperbolic secant distribution

    Args:
        loc (float, Dist) : Location parameter
        scale (float, Dist) : Scale parameter
    """
    hypgeosec = chaospy.dist.construct(
        cdf=lambda x: 2/numpy.pi*numpy.arctan(numpy.e**(numpy.pi*x/2.)),
        bnd=lambda: (-20, 20),
        pdf=lambda x: .5*numpy.cosh(numpy.pi*x/2.)**-1,
        ppf=lambda q: 2/numpy.pi*numpy.log(numpy.tan(numpy.pi*q/2.)),
        mom=lambda k: numpy.abs(scipy.special.euler(k))[-1],
    )
    dist = hypgeosec()*scale + loc
    dist.addattr(str="Hypgeosec(%s,%s)"%(loc, scale))
    return dist


def Kumaraswamy(a, b, lo=0, up=1):
    """
    Kumaraswswamy's double bounded distribution

    Args:
        a (float, Dist) : First shape parameter
        b (float, Dist) : Second shape parameter
        lo (float, Dist) : Lower threshold
        up (float, Dist) : Upper threshold
    """
    kumaraswamy = chaospy.dist.construct(
        pdf=lambda x, a, b: a*b*x**(a-1)*(1-x**a)**(b-1),
        cdf=lambda x, a, b: 1-(1-x**a)**b,
        ppf=lambda q, a, b: (1-(1-q)**(1./b))**(1./a),
        mom=lambda k, a, b: (b*scipy.special.gamma(1+k*1./a)*
                             scipy.special.gamma(b)/
                             scipy.special.gamma(1+b+k*1./a)),
        str=lambda a, b: "kum(%s,%s)" % (a,b),
        bnd=lambda a, b: (0, 1),
    )
    assert numpy.all(a>0) and numpy.all(b>0)
    dist = kumaraswamy(a,b)*(up-lo) + lo
    dist.addattr(str="Kumaraswamy(%s,%s,%s,%s)"%(a,b,lo,up))
    return dist


def Laplace(mu=0, scale=1):
    R"""
    Laplace Probability Distribution

    Args:
        mu (float, Dist) : Mean of the distribution.
        scale (float, Dist) : Scaleing parameter.
            scale > 0

    Examples:
        >>> cp.seed(1000)
        >>> f = cp.Laplace(2, 2)
        >>> q = numpy.linspace(0,1,6)[1:-1]
        >>> print(f.inv(q))
        [ 0.16741854  1.5537129   2.4462871   3.83258146]
        >>> print(f.fwd(f.inv(q)))
        [ 0.2  0.4  0.6  0.8]
        >>> print(f.sample(4))
        [ 2.73396771 -0.93923119  6.61651689  1.92746607]
        >>> print(f.mom(1))
        2.0
    """
    laplace = chaospy.dist.construct(
        pdf=lambda x: numpy.e**-numpy.abs(x)/2,
        cdf=lambda x: (1+numpy.sign(x)*(1-numpy.e**-abs(x)))/2,
        mom=lambda k: .5*scipy.misc.factorial(k)*(1+(-1)**k),
        ppf=lambda x: numpy.where(x>.5, -numpy.log(2*(1-x)), numpy.log(2*x)),
        bnd=lambda: (-32., 32.),
    )
    dist = laplace()*scale + mu
    dist.addattr(str="Laplace(%s,%s)"%(mu,scale))
    return dist


def Levy(loc=0, scale=1):
    """
    Levy distribution

    Args:
        loc (float, Dist) : Location parameter
        scale (float, Dist) : Scaling parameter
    """
    levy = chaospy.dist.construct(
        pdf=lambda x: 1/numpy.sqrt(2*numpy.pi*x)/x*numpy.exp(-1/(2*x)),
        cdf=lambda x: 2*(1-Normal()._cdf(1/numpy.sqrt(x))),
        ppf=lambda q: 1./Normal()._ppf(1-q/2.0)**2,
        bnd=lambda: (0.0, 10000),
    )
    dist = levy()*scale+loc
    dist.addattr(str="Levy(%s,%s)"%(loc, scale))
    return dist


def Logistic(loc=0, scale=1, skew=1):
    """
    Generalized logistic type 1 distribution
    Sech squared distribution

    loc (float, Dist) : Location parameter
    scale (float, Dist) : Scale parameter
    skew (float, Dist) : Shape parameter
    """
    logistic = chaospy.dist.construct(
        pdf=lambda x, c: numpy.e**-x/(1+numpy.e**-x)**(c+1),
        cdf=lambda x, c: (1+numpy.e**-x)**-c,
        ppf=lambda q, c: -numpy.log(q**(-1/c)-1),
        bnd=lambda c: (-10000, 10000),
    )
    dist = logistic()*scale + loc
    dist.addattr(str="Logistic(%s,%s)"%(loc, scale))
    return dist


def Maxwell(scale=1, shift=0):
    """
    Maxwell-Boltzmann distribution
    Chi distribution with 3 degrees of freedom

    Args:
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    dist = Chi(3, scale=scale, shift=shift)
    dist.addattr(str="Maxwell(%s,%s)"%(scale, shift))
    return dist


def Mielke(kappa=1, expo=1, scale=1, shift=0):
    """
    Mielke's beta-kappa distribution

    Args:
        kappa (float, Dist) : First shape parameter
        expo (float, Dist) : Second shape parameter
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    mielke = chaospy.dist.construct(
        pdf=lambda x, k, s: k*x**(k-1.0) / (1.0+x**s)**(1.0+k*1.0/s),
        cdf=lambda x, k, s: x**k / (1.0+x**s)**(k*1.0/s),
        ppf=lambda q, k, s: numpy.pow(q**(s*1./k)/(1.0-q**(s*1./k)), 1.0/s),
        bnd=lambda k, s: (0.0, 10000.),
    )
    dist = mielke(kappa, expo)*scale + shift
    dist.addattr(str="Mielke(%s,%s,%s,%s)"%(kappa,expo,scale,shift))
    return dist




def Nakagami(shape=1, scale=1, shift=0):
    """
    Nakagami-m distribution.

    Args:
        shape (float, Dist) : Shape parameter
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    nakagami = chaospy.dist.construct(
        pdf=lambda x, nu: 2*nu**nu/scipy.special.gamma(nu)*(x**(2*nu-1.0))*numpy.exp(-nu*x*x),
        cdf=lambda x, nu: scipy.special.gammainc(nu,nu*x*x),
        ppf=lambda q, nu: numpy.sqrt(1.0/nu*scipy.special.gammaincinv(nu,q)),
        bnd=lambda nu: (0.0, 10000.),
    )
    dist = nakagami(shape)*scale + shift
    dist.addattr(str="Nakagami(%s,%s,%s)"%(shape,scale,shift))
    return dist



def Pareto1(shape=1, scale=1, loc=0):
    """
    Pareto type 1 distribution.

    Lower threshold at scale+loc and survival: x^-shape

    Args:
        shape (float, Dist) : Tail index parameter
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    pareto1 = chaospy.dist.construct(
        pdf=lambda x, b: b * x**(-b-1),
        cdf=lambda x, b: 1 -  x**(-b),
        ppf=lambda q, b: pow(1-q, -1.0/b),
        bnd=lambda b: (1.0, 10000.),
    )
    dist = pareto1(shape)*scale + loc
    dist.addattr(str="Pareto(%s,%s,%s)" % (shape, scale, loc))
    return dist


def Pareto2(shape=1, scale=1, loc=0):
    """
    Pareto type 2 distribution.

    Also known as Lomax distribution (for loc=0).

    Lower threshold at loc and survival: (1+x)^-shape.

    Args:
        shape (float, Dist) : Shape parameter
        scale (float, Dist) : Scaling parameter
        loc (float, Dist) : Location parameter
    """
    pareto2 = chaospy.dist.construct(
        pdf=lambda x, c: c*1.0/(1.0+x)**(c+1.0),
        cdf=lambda x, c: 1.0-1.0/(1.0+x)**c,
        ppf=lambda q, c: pow(1.0-q,-1.0/c)-1,
        bnd=lambda c: (0.0, 10000.),
    )
    dist = pareto(shape)*scale + loc
    dist.addattr(str="Pareto(%s,%s,%s)"%(shape, scale, loc))
    return dist


def Powerlaw(shape=1, lo=0, up=1):
    """
    Powerlaw distribution

    Args:
        shape (float, Dist) : Shape parameter
        lo (float, Dist) : Location of lower threshold
        up (float, Dist) : Location of upper threshold
    """
    dist = Beta(shape, 1, lo, up)
    dist.addattr(str="Powerlaw(%s,%s,%s)"%(shape, lo, up))
    return dist


def Powerlognormal(shape=1, mu=0, sigma=1, shift=0, scale=1):
    """
    Power log-normal distribution

    Args:
        shape (float, Dist) : Shape parameter
        mu (float, Dist) : Mean in the normal distribution.  Overlaps with
                scale by mu=log(scale)
        sigma (float, Dist) : Standard deviation of the normal distribution.
        shift (float, Dist) : Location parameter
        scale (float, Dist) : Scaling parameter. Overlap with mu in scale=e**mu
    """
    powerlognorm = chaospy.dist.construct(
        pdf=lambda x, c, s: c/(x*s)*Normal().pdf(
                    numpy.log(x)/s)*pow(Normal().fwd(
                        -numpy.log(x)/s),c*1.0-1.0),
        cdf=lambda x, c, s: 1.0 - pow(Normal().fwd(-numpy.log(x)/s),c*1.0),
        ppf=lambda q, c, s: numpy.exp(-s*Normal().inv(pow(1.0-q,1.0/c))),
        bnd=lambda c, s: (0.0, 10000.),
    )
    dist = powerlognorm(shape, sigma)*scale*numpy.e**mu + shift
    dist.addattr(str="Powerlognorm(%s,%s,%s,%s,%s)"%\
            (shape, mu, sigma, shift, scale))
    return dist


def Powernorm(shape=1, mu=0, scale=1):
    """
    Power normal or Box-Cox distribution.

    Args:
        shape (float, Dist) : Shape parameter
        mu (float, Dist) : Mean of the normal distribution
        scale (float, Dist) : Standard deviation of the normal distribution
    """
    powernorm = chaospy.dist.construct(

        pdf=lambda x, c: c*normal._pdf(x)*(Normal().fwd(-x)**(c-1.0)),
        cdf=lambda x, c: 1.0-normal._cdf(-x)**(c*1.0),
        ppf=lambda q, c: -normal._ppf(pow(1.0-q,1.0/c)),
        bnd=lambda c: (-10000., 10000.),
    )
    dist = powernorm(shape)*scale + mu
    dist.addattr(str="Powernorm(%s,%s,%s)"%(shape, mu, scale))
    return dist


def Raised_cosine(loc=0, scale=1):
    """
    Raised cosine distribution

    Args:
        loc (float, Dist) : Location parameter
        scale (float, Dist) : Scale parameter
    """
    raised_cosine = chaospy.dist.construct(
        pdf=lambda x: .5 + .5*numpy.cos(numpy.pi*x),
        cdf=lambda x: .5 + .5*x + numpy.sin(numpy.pi*x)/(2*numpy.pi),
        bnd=lambda: (-1,1),
        mom=lambda k: (numpy.where(k%2, 0, 2/(k+2) + 1/(k+1)*
                       scipy.special.hyp1f2(
                           (k+1)/2.), .5, (k+3)/2., -numpy.pi**2/4)),
    )
    dist = raised_cosine()*scale + loc
    dist.addattr(str="Raised_cosine(%s,%s)"%(loc,scale))
    return dist


def Rayleigh(scale=1, shift=0):
    """
    Rayleigh distribution

    Args:
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    dist = Chi(2, scale=scale, shift=shift)
    dist.addattr(str="Rayleigh(%s,%s)"%(scale, shift))
    return dist


def Reciprocal(lo=1, up=2):
    """
    Reciprocal distribution

    Args:
        lo (float, Dist) : Location of lower threshold
        up (float, Dist) : Location of upper threshold
    """
    reciprocal = chaospy.dist.construct(
        pdf=lambda x, lo, up: 1./(x*numpy.log(up/lo)),
        cdf=lambda x, lo, up: numpy.log(x/lo)/numpy.log(up/lo),
        ppf=lambda q, lo, up: numpy.e**(q*numpy.log(up/lo) + numpy.log(lo)),
        bnd=lambda lo, up: (lo, up),
        mom=lambda k, lo, up: ((up*numpy.e**k-lo*numpy.e**k)/(numpy.log(up/lo)*(k+(k==0))))**(k!=0),
    )
    dist = reciprocal(lo,up)
    dist.addattr(str="Reciprocal(%s,%s)"%(lo,up))
    return dist


def Student_t(df, loc=0, scale=1, nc=0):
    """
    (Non-central) Student-t distribution

    Args:
        df (float, Dist) : Degrees of freedom
        loc (float, Dist) : Location parameter
        scale (float, Dist) : Scale parameter
        nc (flat, Dist) : Non-centrality parameter
    """
    student_t = chaospy.dist.construct(
        pdf=lambda x, a: (scipy.special.gamma(.5*a+.5)*(1+x*x/a)**(-.5*a-.5)/
                          (numpy.sqrt(a*numpy.pi)*scipy.special.gamma(.5*a))),
        cdf=lambda x, a: scipy.special.stdtr(a, x),
        ppf=lambda q, a: scipy.special.stdtrit(a, q),
        bnd=lambda a: (-10000., 10000.),
        mom=lambda k, a: numpy.where(
            k%2 == 0,
            scipy.special.gamma(.5*k+.5)
            *scipy.special.gamma(.5*a-.5*k)*a**(.5*k)
            /(numpy.pi**.5*scipy.special.gamma(.5*a)),
            0
        ),
        ttr=lambda k, a: (0., k*a*(a-k+1.)/ ((a-2*k)*(a-2*k+2))),
        str=lambda a: "stt(%s)" % a,
    )
    dist = student_t(df)*scale + loc
    dist.addattr(str="Student_t(%s,%s,%s)" % (df, loc, scale))
    return dist


def Triangle(lo, mid, up):
    """
    Triangle Distribution.

    Must have lo <= mid <= up.

    Args:
        lo (float, Dist) : Lower bound
        mid (float, Dist) : Location of the top
        up (float, Dist) : Upper bound

    Examples:
        >>> cp.seed(1000)
        >>> f = cp.Triangle(2, 3, 4)
        >>> q = numpy.linspace(0,1,6)[1:-1]
        >>> print(f.inv(q))
        [ 2.63245553  2.89442719  3.10557281  3.36754447]
        >>> print(f.fwd(f.inv(q)))
        [ 0.2  0.4  0.6  0.8]
        >>> print(f.sample(4))
        [ 3.16764141  2.47959763  3.684668    2.98202994]
        >>> print(f.mom(1))
        3.0
    """
    def tri_ttr(k, a):
        from chaospy.quadrature import clenshaw_curtis
        q1,w1 = clenshaw_curtis(int(10**3*a), 0, a)
        q2,w2 = clenshaw_curtis(int(10**3*(1-a)), a, 1)
        q = numpy.concatenate([q1,q2], 1)
        w = numpy.concatenate([w1,w2])
        w = w*numpy.where(q<a, 2*q/a, 2*(1-q)/(1-a))

        from chaospy.poly import variable
        x = variable()

        orth = [x*0, x**0]
        inner = numpy.sum(q*w, -1)
        norms = [1., 1.]
        A,B = [],[]

        for n in range(k):
            A.append(inner/norms[-1])
            B.append(norms[-1]/norms[-2])
            orth.append((x-A[-1])*orth[-1]-orth[-2]*B[-1])

            y = orth[-1](*q)**2*w
            inner = numpy.sum(q*y, -1)
            norms.append(numpy.sum(y, -1))

        A, B = numpy.array(A).T[0], numpy.array(B).T
        A = numpy.array([[A[_] for _ in k[0]]])
        B = numpy.array([[B[_] for _ in k[0]]])
        return A, B

    triangle = chaospy.dist.construct(
        pdf=lambda D, a: numpy.where(D<a, 2*D/a, 2*(1-D)/(1-a)),
        cdf=lambda D, a: numpy.where(D<a, D**2/(a + (a==0)),
                    (2*D-D*D-a)/(1-a+(a==1))),
        ppf=lambda q, a: numpy.where(q<a, numpy.sqrt(q*a), 1-numpy.sqrt(1-a-q*(1-a))),
        mom=lambda k, a: numpy.where(
            a == 1,
            2./(k+2),
            2*(1.-a**(k+1))/((k+1)*(k+2)*(1-a+(a == 1))),
        ),
        bnd=lambda a: (0., 1.),
        ttr=lambda k, a: tri_ttr(k, a),
    )
    a = (mid-lo)*1./(up-lo)
    assert numpy.all(a>=0) and numpy.all(a<=1)
    dist = triangle(a=a)*(up-lo) + lo
    dist.addattr(str="Triangle(%s,%s,%s)" % (lo, mid, up))
    return dist


def Truncexpon(up=1, scale=1, shift=0):
    """
    Truncated exponential distribution.

    Args:
        up (float, Dist) : Location of upper threshold
        scale (float, Dist) : Scaling parameter in the exponential distribution
        shift (float, Dist) : Location parameter
    """
    truncexpon = chaospy.dist.construct(
        pdf=lambda x, b: numpy.exp(-x)/(1-numpy.exp(-b)),
        cdf=lambda x, b: (1.0-numpy.exp(-x))/(1-numpy.exp(-b)),
        ppf=lambda q, b: -numpy.log(1-q+q*numpy.exp(-b)),
        bnd=lambda b: (0.0, b),
    )
    dist = truncexpon((up-shift)/scale)*scale + shift
    dist.addattr(str="Truncexpon(%s,%s,%s)"%(up, scale, shift))
    return dist


def Truncnorm(lo=-1, up=1, mu=0, sigma=1):
    """
    Truncated normal distribution

    Args:
        lo (float, Dist) : Location of lower threshold
        up (float, Dist) : Location of upper threshold
        mu (float, Dist) : Mean of normal distribution
        sigma (float, Dist) : Standard deviation of normal distribution
    """
    class truncnorm(Dist):
        def __init__(self, a, b, mu, sigma):
            Dist.__init__(self, a=a, b=b)
            self.norm = normal()*sigma+mu
            self.fa = self.norm.fwd(a)
            self.fb = self.norm.fwd(b)
        def _pdf(self, x, a, b):
            return self.norm.pdf(x) / (self.fb-self.fa)
        def _cdf(self, x, a, b):
            return (self.norm.fwd(x) - self.fa) / (self.fb-self.fa)
        def _ppf(self, q, a, b):
            return self.norm.inv(q*(self.fb-self.fa) + self.fa)
        def _bnd(self, a, b):
            return a, b
    dist = truncnorm(lo, up, mu, sigma)
    dist.addattr(str="Truncnorm(%s,%s,%s,%s)"%(lo,up,mu,sigma))
    return dist


def Tukeylambda(shape=0, scale=1, shift=0):
    """
    Tukey-lambda distribution,

    Args:
        lam (float, Dist) : Shape parameter
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    class tukeylambda(Dist):
        def __init__(self, lam):
            Dist.__init__(self, lam=lam)
        def _pdf(self, x, lam):
            Fx = (special.tklmbda(x,lam))
            Px = Fx**(lam-1.0) + ((1-Fx))**(lam-1.0)
            Px = 1.0/(Px)
            return np.where((lam <= 0) | (abs(x) < 1.0/(lam)), Px, 0.0)
        def _cdf(self, x, lam):
            return special.tklmbda(x, lam)
        def _ppf(self, q, lam):
            q = q*1.0
            vals1 = (q**lam - (1-q)**lam)/lam
            vals2 = np.log(q/(1-q))
            return np.where((lam==0)&(q==q), vals2, vals1)
        def _bnd(self, lam):
            return self._ppf(1e-10, lam), self._ppf(1-1e-10, lam)
    dist = tukeylambda(shape)*scale + shift
    dist.addattr(str="Tukeylambda(%s,%s,%s)"%(shape, scale, shift))
    return dist


def Wald(mu=0, scale=1, shift=0):
    """
    Wald distribution
    Reciprocal inverse Gaussian distribution

    Args:
        mu (float, Dist) : Mean of the normal distribution
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    wald = chaospy.dist.construct(
        pdf=lambda x, mu: 1.0/numpy.sqrt(2*numpy.pi*x)*numpy.exp(-(1-mu*x)**2.0 / (2*x*mu**2.0)),
        cdf=lambda x, mu: (1.0-Normal().fwd(
            (1./mu-x)*x**-.5)-numpy.exp(2.0/mu)*Normal().fwd(
                -isqx*(1./mu+1)*x**-.5)),
        bnd=lambda mu: (0.0, 10**10),
    )
    dist = wald(mu)*scale + shift
    dist.addattr(str="Wald(%s,%s,%s)"%(mu, scale, shift))
    return dist


def Wigner(radius=1, shift=0):
    """
    Wigner (semi-circle) distribution

    Args:
        radius (float, Dist) : radius of the semi-circle (scale)
        shift (float, Dist) : location of the origen (location)
    """
    dist = radius*(2*Beta(1.5, 1.5)-1) + shift
    dist.addattr(str="Wigner(%s,%s)" % (radius, shift))
    return dist


def Wrapcauchy(shape=0.5, scale=1, shift=0):
    """
    Wraped Cauchy distribution

    Args:
        shape (float, Dist) : Shape parameter
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
    """
    class wrapcauchy(Dist):

        def __init__(self, c):
            Dist.__init__(self, c=c)
        def _pdf(self, x, c):
            return (1.0-c*c)/(2*np.pi*(1+c*c-2*c*np.cos(x)))
        def _cdf(self, x, c):
            output = 0.0*x
            val = (1.0+c)/(1.0-c)
            c1 = x<np.pi
            c2 = 1-c1

            xn = np.extract(c2,x)
            if (any(xn)):
                valn = np.extract(c2, np.ones_like(x)*val)
                xn = 2*np.pi - xn
                yn = np.tan(xn/2.0)
                on = 1.0-1.0/np.pi*np.arctan(valn*yn)
                np.place(output, c2, on)

            xp = np.extract(c1,x)
            if (any(xp)):
                valp = np.extract(c1, np.ones_like(x)*val)
                yp = np.tan(xp/2.0)
                op = 1.0/np.pi*np.arctan(valp*yp)
                np.place(output, c1, op)

            return output

        def _ppf(self, q, c):
            val = (1.0-c)/(1.0+c)
            rcq = 2*np.arctan(val*np.tan(np.pi*q))
            rcmq = 2*np.pi-2*np.arctan(val*np.tan(np.pi*(1-q)))
            return np.where(q < 1.0/2, rcq, rcmq)
        def _bnd(self, c):
            return 0.0, 2*np.pi
    dist = wrapcauchy(shape)*scale + shift
    dist.addattr(str="Wrapcauchy(%s,%s,%s)"%(shape, scale, shift))
    return dist


def SampleDist(samples, lo=None, up=None):
    """
    Distribution based on samples.

    Estimates a distribution from the given samples by constructing a kernel
    density estimator (KDE).

    Args:
        samples:
            Sample values to construction of the KDE
        lo (float) : Location of lower threshold
        up (float) : Location of upper threshold
    """
    class kdedist(Dist):
        """
    A distribution that is based on a kernel density estimator (KDE).
        """
        def __init__(self, kernel, lo, up):
            self.kernel = kernel
            super(kdedist, self).__init__(lo=lo, up=up)

        def _cdf(self, x, lo, up):
            cdf_vals = numpy.zeros(x.shape)
            for i in range(0, len(x)):
                cdf_vals[i] = [self.kernel.integrate_box_1d(0, x_i) for x_i in x[i]]
            return cdf_vals

        def _pdf(self, x, lo, up):
            return self.kernel(x)

        def _bnd(self, lo, up):
            return (lo, up)
        def sample(self, size=(), rule="R", antithetic=None,
                verbose=False, **kws):
            """
                Overwrite sample() function, because the constructed Dist that is
                based on the KDE is only working with the random sampling that is 
                given by the KDE itself.
            """
            size_ = numpy.prod(size, dtype=int)
            dim = len(self)
            if dim>1:
                if isinstance(size, (tuple,list,numpy.ndarray)):
                    shape = (dim,) + tuple(size)
                else:
                    shape = (dim, size)
            else:
                shape = size

            out = self.kernel.resample(size_)[0]
            try:
                out = out.reshape(shape)
            except:
                if len(self)==1:
                    out = out.flatten()
                else:
                    out = out.reshape(dim, out.size/dim)

            return out

    if lo is None:
        lo = samples.min()
    if up is None:
        up = samples.max()

    try:
        #construct the kernel density estimator
        kernel = gaussian_kde(samples, bw_method="scott")
        dist = kdedist(kernel, lo, up)
        dist.addattr(str="SampleDist(%s,%s)" % (lo, up))

    #raised by gaussian_kde if dataset is singular matrix
    except numpy.linalg.LinAlgError:
        dist = Uniform(lo=-numpy.inf, up=numpy.inf)

    return dist
