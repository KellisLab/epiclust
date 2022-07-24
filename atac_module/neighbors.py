
from .distance import distance
from .spline import build_spline, build_ridge

def neighbors(adata, n_neighbors=100, use_rep="scm",
              transpose=True, batch_key=None, min_std=0.001, **neighbors_kwargs):
    import scanpy as sc
    from functools import partial
    import numpy as np
    mean_spl = build_ridge(adata, key=use_rep, spline="mean")
    std_spl = build_ridge(adata, key=use_rep, spline="std")
    if transpose:
        tdata = adata.T
    else:
        tdata = adata
    if batch_key is not None and batch_key in tdata.obs.columns:
        func = partial(sc.external.pp.bbknn, adata=tdata, batch_key=batch_key)
    else:
        func = partial(sc.pp.neighbors, adata=tdata)
    func(n_neighbors=n_neighbors,
                    metric=distance,
                    metric_kwds={"mean_weights": mean_spl.astype(np.float64),
                                 "std_weights": std_spl.astype(np.float64),
                                 "min_std": min_std},
                    use_rep=adata.uns[use_rep]["rep"],
                    **neighbors_kwargs)
    if transpose:
        adata.varp = tdata.obsp
    if "key_added" in neighbors_kwargs.keys():
        adata.uns[neighbors_kwargs["key_added"]]["params"]["metric"] = use_rep
    else:
        adata.uns["neighbors"]["params"]["metric"] = use_rep  ### custom func is not saveable
