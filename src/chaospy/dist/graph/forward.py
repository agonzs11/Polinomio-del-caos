import numpy as np

def forward_call(self, x, dist):
    self.counting(dist, "fwd")
    assert x.shape==(len(dist), self.size)
    self.dist, dist_ = dist, self.dist

    graph = self.graph
    graph.add_node(dist, key=x)
    out = np.empty(x.shape)

    prm = self.dists.build()
    prm.update(self.keys.build())
    for k,v in prm.items():
        if not isinstance(v, np.ndarray):
            v_ = self.run(v, "val")[0]
            if isinstance(v_, np.ndarray):
                prm[k] = v_
                graph.add_node(v, key=v_)

    if dist.advance:
        out[:] = dist._cdf(x, self)
    else:
        out[:] = dist._cdf(x, **prm)

    graph.add_node(dist, val=out)

    self.dist = dist_
    return np.array(out)
