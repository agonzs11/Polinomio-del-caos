# pylint: disable=protected-access
"""Backend for random number generator."""
import numpy

def rnd_call(self, dist):
    "Sample generator call backend wrapper"

    self.counting(dist, "rnd")
    graph = self.graph
    self.dist, dist_ = dist, self.dist

    for key, value in dist.prm.items():
        if not isinstance(value, numpy.ndarray) and\
                not "key" in graph.node[value]:
            rnd_call(self, value)

    if dist.advance:
        key = self.run(dist, "val")
    else:
        rnd = numpy.random.random((len(dist), self.size))
        from .inv import inv_call
        key = inv_call(self, rnd, dist)

    assert isinstance(key, numpy.ndarray)
    graph.add_node(dist, key=key)

    self.dist = dist_

    if dist is self.root:
        out = graph.node[dist]["key"]
        return out
