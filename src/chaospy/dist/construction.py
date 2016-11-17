"""
Construct functions.
"""
from chaospy.dist.baseclass import Dist

def construct(
        cdf,
        bnd,
        parent=None,
        pdf=None,
        ppf=None,
        mom=None,
        ttr=None,
        val=None,
        doc=None,
        str=None,
        dep=None,
        defaults=None,
        advance=False,
        length=1,
):
    """
    Random variable constructor.

    Args:
        cdf (callable) : Cumulative distribution function. Optional if parent
                is used.
        bnd (callable) : Boundary interval. Optional if parent is used.
        parent (Dist) : Distribution used as basis for new distribution. Any
                other argument that is omitted will instead take is function
                from parent.
        doc (str, optional) : Documentation for the distribution.
        str (str, callable, optional) : Pretty print of the variable.
        pdf (callable, optional) : Probability density function.
        ppf (callable, optional) : Point percentile function.
        mom (callable, optional) : Raw moment generator.
        ttr (callable, optional) : Three terms recursion coefficient generator
        val (callable, optional) : Value function for transferable
                distributions.
        dep (callable, optional) : Dependency structure.
        advance (bool) : If True, advance mode is used. See dist.graph for
                details.
        length (int) : If constructing an multivariate random variable, this
                sets the assumed length. Defaults to 1.
        init (callable, optional) : Custom constructor method.

    Returns:
        dist (Dist) : New custom distribution.
    """
    if not (parent is None):
        if hasattr(parent, "_cdf"):
            cdf = cdf or parent._cdf
        if hasattr(parent, "_bnd"):
            bnd = bnd or parent._bnd
        if hasattr(parent, "_pdf"):
            pdf = pdf or parent._pdf
        if hasattr(parent, "_ppf"):
            ppf = ppf or parent._ppf
        if hasattr(parent, "_mom"):
            mom = mom or parent._mom
        if hasattr(parent, "_ttr"):
            ttr = ttr or parent._ttr
        if hasattr(parent, "_str"):
            str = str or parent._str
        if hasattr(parent, "_dep"):
            dep = dep or parent._dep
        val = val or parent._val
        doc = doc or parent.__doc__

    def crash_func(*a, **kw):
        raise NotImplementedError
    if advance:
        ppf = ppf or crash_func
        pdf = pdf or crash_func
        mom = mom or crash_func
        ttr = ttr or crash_func

    def custom(**kws):

        if defaults is not None:
            keys = defaults.keys()
            assert all([key in keys for key in kws.keys()])
            prm = defaults.copy()
        else:
            prm = {}
        prm.update(kws)
        _length = prm.pop("_length", length)
        _advance = prm.pop("_advance", advance)

        dist = Dist(_advance=_advance, _length=_length, **prm)

        dist.addattr(cdf=cdf)
        dist.addattr(bnd=bnd)

        if not (pdf is None):
            dist.addattr(pdf=pdf)
        if not (ppf is None):
            dist.addattr(ppf=ppf)
        if not (mom is None):
            dist.addattr(mom=mom)
        if not (ttr is None):
            dist.addattr(ttr=ttr)
        if not (val is None):
            dist.addattr(val=val)
        if not (str is None):
            dist.addattr(str=str)
        if not (dep is None):
            dist.addattr(dep=dep)

        return dist

    if not (doc is None):
        doc = """
Custom random variable
        """
    setattr(custom, "__doc__", doc)

    return custom
