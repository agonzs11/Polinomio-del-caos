"""Forward and inverse Rosenblatt transformation wrapper."""
import numpy


def fwd(dist, x):
    """Forward Rosenblatt transformation."""
    dim = len(dist)
    x = numpy.asfarray(x)
    shape = x.shape
    size = int(x.size/dim)
    x = x.reshape(dim, size)

    bnd = dist.graph.run(x, "range")
    x_ = numpy.where(x < bnd[0], bnd[0], x)
    x_ = numpy.where(x_ > bnd[1], bnd[1], x_)
    out = dist.graph.run(x_, "fwd")
    out = numpy.where(x < bnd[0], 0, out)
    out = numpy.where(x > bnd[1], 1, out)

    out = out.reshape(shape)
    return out


def inv(dist, q, maxiter=100, tol=1e-5, **kws):
    """Inverse Rosenblatt transformation."""
    q = numpy.array(q)
    assert numpy.all(q>=0) and numpy.all(q<=1), q

    dim = len(dist)
    shape = q.shape
    size = int(q.size/dim)
    q = q.reshape(dim, size)

    try:
        out = dist.graph.run(q, "inv", maxiter=maxiter, tol=tol)

    except NotImplementedError:
        from . import approximations
        out, N, q_ = approximations.inv(dist, q, maxiter=maxiter, tol=tol, retall=True)
        diff = numpy.max(numpy.abs(q-q_))
        print("approximate %s.inv w/%d calls and eps=%g" % (dist, N, diff))

    lo, up = dist.graph.run(out, "range")
    out = numpy.where(out.T>up.T, up.T, out.T).T
    out = numpy.where(out.T<lo.T, lo.T, out.T).T
    out = out.reshape(shape)

    return out
