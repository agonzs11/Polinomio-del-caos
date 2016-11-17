r"""
As noted throughout the documentation, known distributions are created easily
by calling their name of the function of interest. For example to create
a Gaussian random variable::

   >>> distribution = chaospy.Normal(0,1)

To construct simple multivariate random variables with stochastically
independent components, either all the same using :func:`~Iid`::

   >>> distribution = chaospy.Iid(chaospy.Normal(0, 1), 2)

Or with more detailed control through :func:`~J`::

   >>> distribution = chaospy.J(chaospy.Normal(0, 1), chaospy.Normal(0, 1))

The functionality of the distributions are covered in various other sections:

* To generate random samples, see :ref:`montecarlo`.
* To create transformations, see :ref:`rosenblatt`.
* To generate raw statistical moments, see :ref:`moments`.
* To generate three terms recurrence coefficients, see :ref:`orthogonality`.
* To analyse statistical properies, see :ref:`descriptives`.
"""
from .baseclass import Dist
from .construction import construct

from . import rosenblatt

from .graph import Graph
from .sampler import *
from .approx import *
from .joint import *
from .copulas import *
from .collection import *
from .operators import *

from numpy.random import seed


if __name__ == "__main__":
    seed(1000)
    import doctest
    import chaospy
    doctest.testmod()
