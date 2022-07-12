
from .distance import distance
from .spline import build_spline
from scipy.interpolate.dfitpack import bispeu

def neighbors(adata, n_neighbors=100, use_rep="scm", spline_k=2,
              transpose=True, min_std=0.001, **neighbors_kwargs):
    import scanpy as sc
    mean_spl = build_spline(adata, key=use_rep, spline="mean", k=spline_k)
    std_spl = build_spline(adata, key=use_rep, spline="std", k=spline_k)
    if transpose:
        tdata = adata.T
    sc.pp.neighbors(tdata,
                    n_neighbors=n_neighbors,
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
    adata.varp = tdata.obsp
    adata.uns["neighbors"]["params"]["metric"] = "custom"
    return adata
