import numpy as np
from copy import copy


def gcd(a, b):
    """
    Greates common denominator.

    Args:
        a (array_like) : Fraction denominator. Values must be either int or
                long.
        b (int, np.ndarray) : Fraction numerator. Values must be either int or
                long.  Also `a` and `b` must have compatible shapes.

    Returns:
        (np.ndarray) : With `c.shape=a.shape`

    Examples:
        >>> print(gcd(120, 6))
        6
        >>> print(gcd([0,1,2,3], 2))
        [2 1 2 1]
    """
    if isinstance(a, (int)) and \
        isinstance(b, (int)):
        while b:
            a, b = b, a%b
        return a

    a, b = np.array(a), np.array(b)
    if np.prod(a.shape+b.shape)==1:
        a, b = a.item(), b.item()
        return gcd(a,b)

    if not np.any(a):
        return copy(b)

    az = a==0
    bz = copy(b)
    while np.any(b):
        a_ = np.where(b, a, 1)
        b_ = np.where(b, b, 1)
        a_, b_ = b_, a_%b_
        a = np.where(b==0, a, a_)
        b = np.where(b==0, b, b_)
    a = np.where(az, bz, a)
    return a


def limit_denominator(a, b, max_b=10**6, index=None):

    p0, q0, p1, q1, p2, q2 = 0, 1, 1, 0, 1, 0
    A, B = a, b

    if not (index is None):

        a[index], b[index] = limit_denominator(a[index], b[index],
            max_b=max_b)
        return a, b

    if not a.shape:

        a,b = a.item(), b.item()

        if not a:
            return 0,1

        while True:

            _ = a/b
            q2 = q0+_*q1
            if q2 > max_b:
                break
            p0, q0, p1, q1 = p1, q1, p0+_*p1, q2
            a, b = b, a-_*b

        k = (max_b-q0)//q1
        bound1 = frac(p0+k*p1, q0+k*q1)
        bound2 = frac(p1, q1)
        if abs(bound2-frac(a,b))<=abs(bound1-frac(a,b)):
            return bound1.a, bound1.b
        else:
            return bound2.a, bound2.b

    i = 0
    default = b<max_b
    while True:

        _ = a / b
        q2 = np.where(p2, q0+_*q1, q2)
        p2 = (q2<=max_b)-default
        if not np.any(p2):
            break
        p0, p0_ = np.where(p2, p1, p0), p0
        q0 = np.where(p2, q1, q0)
        p1 = np.where(p2, p0_+_*p1, p1)
        q1 = np.where(p2, q2, q1)
        a, a_ = np.where(p2, b, a), a
        b = np.where(p2, a_-_*b, b)

        i += 1
        if i==10:
            break

    q1 = q1 + default
    k = (max_b-q0)//q1
    alt1 = abs(A*(q0+k*q1)*q1-q1*B*(p0+k*p1))
    alt2 = abs(A*(q0+k*q1)*q1-p1*B*(q0+k*q1))
    a = np.where(alt1>alt2, p1, p0+k*p1)
    a = np.where(default, A, a)
    b = np.where(alt1>alt2, q1, q0+k*q1)
    b = np.where(default, B, b)
    return a, b


class frac(object):
    """General fraction class."""

    def __init__(self, a, b=1, limit=None):
        """
        Args:
            a (array_like) : denomiator.
            b (array_like) : nominator. Dimensions must match a.
        """
        a = np.array(a)

        # Numerator
        if a.dtype==float:

            shape = a.shape
            if shape:
                a, b_ = zip(*[_.as_integer_ratio() \
                    for _ in a.flatten()])
                a = np.array(a, dtype=object).reshape(shape)
                b_ = np.array(b_, dtype=object).reshape(shape)
                b *= b_
            else:
                a, b_ = a.item().as_integer_ratio()
                a = np.array(a, dtype=object)
                b = b*b_

        if a.shape and type(a.flatten()[0])==frac:

            shape1 = a.shape
            a = a.flatten()
            shape2 = a[0].shape
            shape = shape1+shape2

            a_,b_ = [], []
            for i in range(len(a)):
                a_.append( a[i].a )
                b_.append( a[i].b )
            a_, b_ = np.array(a_), np.array(b_)
            a_, b_ = a_.reshape(shape), b_.reshape(shape)
            a = a_
            b = b*b_

        elif not a.shape and type(a.item())==frac:
            a = a.item()
            a, b_ = a.a, a.b
            b = b*b_

        a = np.array(a, dtype=object)

        # Denominator
        b = np.array(b)
        if b.dtype==float:

            shape = b.shape
            if shape:
                a_, b = zip(*[_.as_integer_ratio()
                    for _ in b.flatten()])
                a_ = np.array(a_, dtype=object).reshape(shape)
                b = np.array(b, dtype=object).reshape(shape)
                a = a*a_
            else:
                b, a_ = b.item().as_integer_ratio()
                b = np.array(b, dtype=object)
                a = a*a_

        b = np.array(b, dtype=object)

        if np.any(b==0):
            raise ValueError("frac devision by zero\n%s,%s" % (a, b))

        # Sign
        if np.any(np.sign(b)==-1):
            a, b = np.array(a*np.sign(b), dtype=object), \
                np.array(np.abs(b), dtype=object)

        if a.shape!=b.shape:

            b = b*a**0

        shape = a.shape
        a, b = a.flatten(), b.flatten()

        dom = gcd(a, b)
        a, b = a/dom, b/dom
        if not (limit is None) and np.any(b>limit):
            a, b = limit_denominator(a, b, limit)

        a, b = a.reshape(shape), b.reshape(shape)

        self.a, self.b = a, b
        self.dtype = frac
        self.shape = np.array(a).shape

    def __add__(self, x):

        if isinstance(x, float) or \
            isinstance(x, np.ndarray) and x.dtype==float:
                return np.array(asfloat(self)+x)

        if isinstance(x, (int, np.ndarray, frac)):
            if not isinstance(x, frac):
                x = frac(x)
            out = frac(self.a*x.b + self.b*x.a, self.b*x.b)
            return out

        return NotImplemented

    def __radd__(self, x):
        return self+x

    def __mul__(self, x):

        if isinstance(x, float) or \
            isinstance(x, np.ndarray) and x.dtype==float:
                return np.array(asfloat(self)*x)

        if isinstance(x, (int, np.ndarray, frac)):
            if not isinstance(x, frac):
                x = frac(x)
            out = frac(self.a*x.a, self.b*x.b)
            return out

        return NotImplemented

    def __rmul__(self, x):
        return self*x

    def __sub__(self, x):

        if isinstance(x, float) or \
            isinstance(x, np.ndarray) and x.dtype==float:
                return np.array(asfloat(self) - x)

        if isinstance(x, (int, np.ndarray, frac)):
            if not isinstance(x, frac):
                x = frac(x)
            return frac(self.a*x.b - self.b*x.a, self.b*x.b)

        return NotImplemented

    def __rsub__(self, x):
        return -self + x

    def __div__(self, x):

        if isinstance(x, float) or \
            isinstance(x, np.ndarray) and x.dtype==float:
                return np.array(asfloat(self)/x)

        if isinstance(x, (int, np.ndarray, frac)):
            if not isinstance(x, frac):
                x = frac(x)
            out = frac(self.a*x.b, self.b*x.a)
            return out

        return NotImplemented

    def __rdiv__(self, x):
        return frac(self.b, self.a)*x

    def __pow__(self, x):

        if isinstance(x, float) or \
            isinstance(x, np.ndarray) and x.dtype==float:
                return np.array(asfloat(self)**x)

        if isinstance(x, (int, np.ndarray, frac)):
            if not isinstance(x, frac):
                x = frac(x)

            if np.any(x.b!=1):
                raise ValueError("fraction exponent not supported")

            out = frac(self.a**x.a, self.b**x.a)
            return out

        return NotImplemented

    def __rpow__(self, x):

        if isinstance(x, float) or \
            isinstance(x, np.ndarray) and x.dtype==float:
                return np.array(x**(self.b*1./self.a))

        if not isinstance(x, frac):
            x = frac(x)

        if np.any(self.b!=1):
            raise ValueError("fraction exponent not supported")

        return frac(x.a**self.a, x.b**self.a)

    def __pos__(self):
        return frac(self.a, self.b)

    def __neg__(self):
        return frac(-self.a, self.b)

    def __abs__(self):
        return frac(abs(self.a), self.b)

    def __eq__(self, x):

        if isinstance(x, (int, np.ndarray, frac)):
            if not isinstance(x, frac):
                x = frac(x)
            return (self.a==x.a)*(self.b==x.b)
        return asfloat(self)==x

    def __ne__(self, x):

        if isinstance(x, (int, np.ndarray, frac)):
            if not isinstance(x, frac):
                x = frac(x)
            return (self.a!=x.a)+(self.b!=x.b)
        return asfloat(self)!=x

    def __lt__(self, x):

        if not isinstance(x, frac):
            x = frac(x)
        return self.a*x.b<x.a*self.b

    def __gt__(self, x):

        if not isinstance(x, frac):
            x = frac(x)
        return self.a*x.b>x.a*self.b

    def __le__(self, x):

        if not isinstance(x, frac):
            x = frac(x)
        return self.a*x.b<=x.a*self.b

    def __ge__(self, x):

        if not isinstance(x, frac):
            x = frac(x)
        return self.a*x.b>=x.a*self.b

    def __repr__(self):
        a,b = self.a, self.b

        out = "frac("
        if not a.shape:
            out += repr(a.item())
            if b!=1:
                out += ", " + repr(b.item())
        else:
            out += repr(a.tolist())
            if np.any(b!=1):
                out += ", "
                out += repr(b.tolist())
        out += ")"
        return out

    def __str__(self):

        if not self.shape:
            if self.b==1:
                return str(self.a.item())
            return str(self.a.item()) + "/" + str(self.b.item())

        out = "["
        out += ", ".join([str(_) for _ in self])
        out += "]"
        return out


    def __getitem__(self, i):
        return frac(self.a[i], self.b[i])

    def __setitem__(self, i, y):

        if not isinstance(y, frac):
            y = frac(y)
        self.a[i] = y.a
        self.b[i] = y.b

    def copy(self):
        return frac(self.a, self.b)


if __name__=='__main__':
    import __init__ as cp
    import doctest
    doctest.testmod()
