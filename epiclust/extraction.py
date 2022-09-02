
import numpy as np


def extract_pca(adata, n_pcs=None):
    """we know that the S component is always positive
    so it can be recontructed from s^2/(DoF)"""
    Us = adata.obsm["X_pca"]
    s = adata.uns["pca"]["variance"]
    s = np.sqrt(s * (adata.shape[0] - 1))
    U = Us @ np.diag(1 / s)
    VT = adata.varm["PCs"].T
    if n_pcs is not None and n_pcs <= len(s):
        U = U[:, range(n_pcs)]
        s = s[range(n_pcs)]
        VT = VT[range(n_pcs), :]
    return U, s, VT


def extract_rep(adata, power=0.0, margin="log1p_total_counts",
                key_added="epiclust", n_pcs=None, zero_center=True):
    if margin not in adata.var.columns:
        print("Margin", margin, "not in adata.var")
        return -1
    if "PCs" not in adata.varm or "X_pca" not in adata.obsm or "pca" not in adata.uns:
        print("Must run sc.pp.pca first")
        return -1
    U, s, VT = extract_pca(adata, n_pcs=n_pcs)
    X_adj = VT.T.astype(np.float64) @ np.diag(s**power)
    if zero_center:
        X_adj = X_adj - X_adj.mean(1)[:, None]
    norm = np.linalg.norm(X_adj, axis=1, ord=2)[:, None]
    X_adj = X_adj / norm.clip(1e-100, np.inf)
    rep = "X_%s" % key_added
    M = adata.var[margin].values[:, None]
    M = (M - np.min(M)) / (np.max(M) - np.min(M))  # min-max scale
    adata.varm[rep] = np.hstack((2 * M - 1,  # -1 to 1 scale
                                 X_adj)).astype(np.float32)
    adata.uns[key_added] = {"rep": rep,
                            "margin": margin,
                            "power": power,
                            "n_pcs": len(s),
                            "zero_center": zero_center}
    return 0
