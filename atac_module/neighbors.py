
from .distance import distance
from .spline import build_spline

def neighbors(adata, n_neighbors=100, use_rep="scm", spline_k=2,
              transpose=True, batch_key=None, min_std=0.001, **neighbors_kwargs):
    import scanpy as sc
    from functools import partial
    mean_spl = build_spline(adata, key=use_rep, spline="mean", k=spline_k)
    std_spl = build_spline(adata, key=use_rep, spline="std", k=spline_k)
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
                    metric_kwds={"mean_knots_x": mean_spl.get_knots()[0],
                                 "mean_knots_y": mean_spl.get_knots()[1],
                                 "mean_coeffs": mean_spl.get_coeffs(),
                                 "std_knots_x": std_spl.get_knots()[0],
                                 "std_knots_y": std_spl.get_knots()[1],
                                 "std_coeffs": std_spl.get_coeffs(),
                                 "min_std": min_std,
                                 "k": spline_k},
                    use_rep=adata.uns[use_rep]["rep"],
                    **neighbors_kwargs)
    if transpose:
        adata.varp = tdata.obsp
    if "key_added" in neighbors_kwargs.keys():
        adata.uns[neighbors_kwargs["key_added"]]["params"]["metric"] = use_rep
    else:
        adata.uns["neighbors"]["params"]["metric"] = use_rep  ### custom func is not saveable
    return adata
