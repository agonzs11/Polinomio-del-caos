"""
Multivariate distributions.
"""
import numpy

import scipy.misc
import scipy.special

from chaospy.dist.baseclass import Dist
import chaospy.dist

def MvLognormal(loc=[0,0], scale=[[1,.5],[.5,1]]):
    """
    Multivariate Log-Normal Distribution.

    Args:
        loc (float, Dist) : Mean vector
        scale (float, Dist) : Covariance matrix or variance vector if scale is
                a 1-d vector.
    """
    class mvlognormal(Dist):
        def __init__(self, loc=[0,0], scale=[[1,.5],[.5,1]]):

            loc, scale = numpy.asfarray(loc), numpy.asfarray(scale)
            assert len(loc)==len(scale)

            dist = chaospy.dist.joint.Iid(
                chaospy.dist.collection.Normal(), len(loc))
            C = numpy.linalg.cholesky(scale)
            Ci = numpy.linalg.inv(C)
            Dist.__init__(self, dist=dist, loc=loc, C=C, Ci=Ci,
                    scale=scale, _length=len(scale), _advance=True)

        def _cdf(self, x, graph):

            y = numpy.log(numpy.abs(x) + 1.*(x<=0))
            out = graph(numpy.dot(graph.keys["Ci"], (y.T-graph.keys["loc"].T).T),
                    graph.dists["dist"])
            return numpy.where(x<=0, 0., out)

        def _ppf(self, q, graph):
            return numpy.e**(numpy.dot(graph.keys["C"], \
                    graph(q, graph.dists["dist"])).T+graph.keys["loc"].T).T

        def _mom(self, k, graph):
            scale, loc = graph.keys["scale"], graph.keys["loc"]
            return numpy.e**(numpy.dot(k.T, loc).T+ \
                .5*numpy.diag(numpy.dot(k.T, numpy.dot(scale, k))))

        def _bnd(self, x, graph):
            loc, scale = graph.keys["loc"], graph.keys["scale"]
            up = (7.1*numpy.sqrt(numpy.diag(scale))*x.T**0 + loc.T).T
            return 0*up, numpy.e**up

        def _val(self, graph):
            if "dist" in graph.keys:
                return (numpy.dot(graph.keys["dist"].T, graph.keys["C"].T)+graph.keys["loc"].T).T
            return self

        def _dep(self, graph):

            dist = graph.dists["dist"]
            S = graph(dist)
            out = [set([]) for _ in range(len(self))]
            C = graph.keys["C"]

            for i in range(len(self)):
                for j in range(len(self)):
                    if C[i,j]:
                        out[i].update(S[j])
            return out

        def _str(self, loc, C, **prm):
            print("mvlognor(%s,%s)" % (loc, C))
    dist = mvlognormal(loc, scale)
    dist.addattr(str="MvLognormal(%s,%s)" % (loc, scale))
    return dist


def MvNormal(loc=[0,0], scale=[[1,.5],[.5,1]]):
    """
    Multivariate Normal Distribution

    Args:
        loc (float, Dist) : Mean vector
        scale (float, Dist) : Covariance matrix or variance vector if scale is a 1-d vector.

    Examples:
        >>> cp.seed(1000)
        >>> f = cp.MvNormal([0,0], [[1,.5],[.5,1]])
        >>> q = [[.4,.5,.6],[.4,.5,.6]]
        >>> print(f.inv(q))
        [[-0.2533471   0.          0.2533471 ]
        [-0.34607858  0.          0.34607858]]
        >>> print(f.fwd(f.inv(q)))
        [[ 0.4  0.5  0.6]
        [ 0.4  0.5  0.6]]
        >>> print(f.sample(3))
        [[ 0.39502989 -1.20032309  1.64760248]
        [ 0.15884312  0.38551963  0.1324068 ]]
        >>> print(f.mom((1,1)))
        0.5
    """
    class mvnormal(Dist):
        def __init__(self, loc=[0,0], scale=[[1,.5],[.5,1]]):
            loc, scale = numpy.asfarray(loc), numpy.asfarray(scale)
            C = numpy.linalg.cholesky(scale)
            Ci = numpy.linalg.inv(C)
            Dist.__init__(self, C=C, Ci=Ci, loc=loc,
                    _advance=True, _length=len(C))

        def _cdf(self, x, graph):
            Ci, loc = graph.keys["Ci"], graph.keys["loc"]
            return scipy.special.ndtr(numpy.dot(Ci, (x.T-loc.T).T))

        def _ppf(self, q, graph):
            return (numpy.dot(graph.keys["C"], scipy.special.ndtri(q)).T+graph.keys["loc"].T).T

        def _pdf(self, x, graph):

            loc, C, Ci = graph.keys["loc"], graph.keys["C"], graph.keys["Ci"]
            det = numpy.linalg.det(numpy.dot(C,C.T))

            x_ = numpy.dot(Ci.T, (x.T-loc.T).T)
            out = numpy.ones(x.shape)
            out[0] =  numpy.e**(-.5*numpy.sum(x_*x_, 0))/numpy.sqrt((2*numpy.pi)**len(Ci)*det)
            return out

        def _bnd(self, x, graph):

            C, loc = graph.keys["C"], graph.keys["loc"]
            scale = numpy.sqrt(numpy.diag(numpy.dot(C,C.T)))
            lo,up = numpy.zeros((2,)+x.shape)
            lo.T[:] = (-7.5*scale+loc)
            up.T[:] = (7.5*scale+loc)
            return lo,up

        def _mom(self, k, graph):

            C, loc = graph.keys["C"], graph.keys["loc"]
            scale = numpy.dot(C, C.T)

            def mom(k):

                zeros = (numpy.sum(k,0)%2==1)+numpy.any(numpy.array(k)<0, 0)
                if numpy.all(zeros, 0):
                    return 0.

                dim, K = k.shape
                ra = numpy.arange(dim).repeat(K).reshape(dim,K)

                i = numpy.argmax(k!=0, 0)

                out = numpy.zeros(k.shape[1:])
                out[:] = numpy.where(numpy.choose(i,k),
                        (numpy.choose(i,k)-1)*scale[i,i]*mom(k-2*(ra==i)), 1)
                for x in range(1, dim):
                    out += \
                    (numpy.choose(i,k)!=0)*(x>i)*k[x]*scale[i,x]*mom(k-(ra==i)-(ra==x))

                return out

            dim = len(loc)
            K = numpy.mgrid[[slice(0,_+1,1) for _ in numpy.max(k, 1)]]
            K = K.reshape(dim, K.size/dim)
            M = mom(K)

            out = numpy.zeros(k.shape[1])
            for i in range(len(M)):
                coef = numpy.prod(scipy.misc.comb(k.T, K[:,i]).T, 0)
                diff = k.T - K[:,i]
                pos = diff>=0
                diff = diff*pos
                pos = numpy.all(pos, 1)
                loc_ = numpy.prod(loc**diff, 1)
                out += pos*coef*loc_*M[i]

            return out

        def _dep(self, graph):
            n = chaospy.dist.collection.Normal()
            out = [set([n]) for _ in range(len(self))]
            return out

        def _str(self, C, loc, **prm):
            return "mvnor(%s,%s)" % (loc, C)

    if numpy.all((numpy.diag(numpy.diag(scale))-scale)==0):
        out = chaospy.dist.joint.J(
            *[chaospy.dist.collection.Normal(
                loc[i], scale[i,i]) for i in range(len(scale))])
    else:
        out = mvnormal(loc, scale)
    out.addattr(str="MvNormal(%s,%s)" % (loc, scale))
    return out


def MvStudent_t(df=1, loc=[0,0], scale=[[1,.5],[.5,1]]):
    """
    Args:
        df (float, Dist) : Degree of freedom
        loc (array_like, Dist) : Location parameter
        scale (array_like) : Covariance matrix
    """
    class mvstudentt(Dist):

        def __init__(self, a=1, loc=[0,0], scale=[[1,.5],[.5,1]]):
            loc, scale = numpy.asfarray(loc), numpy.asfarray(scale)
            C = numpy.linalg.cholesky(scale)
            Ci = numpy.linalg.inv(C)
            Dist.__init__(self, a=a, C=C, Ci=Ci, loc=loc, _length=len(C))

        def _cdf(self, x, a, C, Ci, loc):
            x = numpy.dot(Ci, (x.T-loc.T).T)
            return scipy.special.stdtr(a, x)

        def _ppf(self, q, a, C, Ci, loc):
            z = scipy.special.stdtrit(a, q)
            out = (numpy.dot(C, z).T + loc.T).T
            return out

        def _pdf(self, x, a, C, Ci, loc):

            det = numpy.linalg.det(numpy.dot(C,C.T))
            k = len(C)

            x_ = numpy.dot(Ci.T, (x.T-loc.T).T)
            out = numpy.ones(x.shape)
            out[0] = scipy.special.gamma(.5*(a+k))/(scipy.special.gamma(.5*a)* \
                    a**(.5*k)*numpy.pi**(.5*k)*det**.5*\
                    (1+numpy.sum(x_*x_,0)/a))
            return out

        def _bnd(self, a, C, Ci, loc):

            scale = numpy.sqrt(numpy.diag(numpy.dot(C,C.T)))
            lo,up = numpy.zeros((2,len(self)))
            lo.T[:] = (-10**5*scale+loc)
            up.T[:] = (10**5*scale+loc)
            return lo,up

        def _dep(self, graph):
            n = student_t()
            out = [set([n]) for _ in range(len(self))]
            return out

        def _str(self, a, loc, C, **prm):
            return "mvstt(%s,%s,%s)" % (a,loc,C)

    out = mvstudentt(df, loc, scale)
    out.addattr(str="MvStudent_t(%s,%s,%s)" % (df, loc, scale))
    return out


#  class Dirichlet(be.Dist):
#      """
#  Dirichlet \sim Dir(alpha)
#
#  Parameters
#  ----------
#  alpha : array_like
#      Shape parameters.
#      len(alpha)>1
#      numpy.all(alpha>0)
#
#  Examples
#  --------
#  >>> cp.seed(1000)
#  >>> f = cp.Dirichlet([1,2,3])
#  >>> q = [[.3,.3,.7,.7],[.3,.7,.3,.7]]
#  >>> print(f.inv(q))
#  [[ 0.06885008  0.06885008  0.21399691  0.21399691]
#   [ 0.25363028  0.47340104  0.21409462  0.39960771]]
#  >>> print(f.fwd(f.inv(q)))
#  [[ 0.3  0.3  0.7  0.7]
#   [ 0.3  0.7  0.3  0.7]]
#  >>> print(f.sample(4))
#  [[ 0.12507651  0.00904026  0.06508353  0.07888277]
#   [ 0.29474152  0.26985323  0.69375006  0.30848838]]
#  >>> print(f.mom((1,1)))
#  0.047619047619
#      """
#
#      def __init__(self, alpha=[1,1,1]):
#
#          dists = [co.beta() for _ in range(len(alpha)-1)]
#          ba.Dist.__init__(self, _dists=dists, alpha=alpha, _name="D")
#
#      def _upd(self, alpha, **prm):
#
#          alpha = alpha.flatten()
#          dim = len(alpha)-1
#          out = [None]*dim
#          _dists = prm.pop("_" + self.name)
#          cum = _dists[0]
#
#          _dists[0].upd(a=alpha[0], b=numpy.sum(alpha[1:], 0))
#          out[0] = _dists[0]
#          for i in range(1, dim):
#              _dists[i].upd(a=alpha[i], b=numpy.sum(alpha[i+1:], 0))
#              out[i] = _dists[i]*(1-cum)
#              cum = cum+out[i]
#
#          prm = dict(alpha=alpha)
#          prm["_" + self.name] = out
#          return prm
#
#      def _mom(self, k, alpha, **prm):
#
#          out = numpy.empty(k.shape[1:])
#          out[:] = scipy.special.gamma(numpy.sum(alpha, 0))
#          out /= scipy.special.gamma(numpy.sum(alpha, 0)+numpy.sum(k, 0))
#          out *= numpy.prod(scipy.special.gamma(alpha[:-1]+k.T).T, 0)
#          out /= numpy.prod(scipy.special.gamma(alpha[:-1]), 0)
#          return out
