
def extract_pca(adata, n_pcs=None, extract_u=True):
    """we know that the S component is always positive
    so it can be recontructed from s^2/(DoF)"""
    import numpy as np
    Us = adata.obsm["X_pca"]
    s = adata.uns["pca"]["variance"].astype(np.float64)
    s = np.sqrt(s * (adata.shape[0] - 1))
    if extract_u:
        U = Us @ np.diag(1 / s)
    VT = adata.varm["PCs"].T
    if n_pcs is not None and n_pcs <= len(s):
        if extract_u:
            U = U[:, range(n_pcs)]
        s = s[range(n_pcs)]
        VT = VT[range(n_pcs), :]
    if not extract_u:
        U = None
    return U, s, VT


def extract_rep(adata, power=0.0, margin="log1p_total_counts",
                key_added="epiclust", n_pcs=None, zero_center=True):
    import numpy as np
    if margin not in adata.var.columns:
        print("Margin", margin, "not in adata.var")
        return -1
    if "PCs" not in adata.varm or "X_pca" not in adata.obsm or "pca" not in adata.uns:
        print("Must run sc.pp.pca first")
        return -1
    _, s, VT = extract_pca(adata, n_pcs=n_pcs, extract_u=False)
    s = s.astype(np.float64) ** power
    s = s / np.linalg.norm(s, ord=2).clip(1e-100, np.inf)
    X_adj = VT.T.astype(np.float64) @ np.diag(s)
    del VT
    if zero_center:
        X_adj = X_adj - X_adj.mean(1)[:, None]
    norm = np.linalg.norm(X_adj, axis=1, ord=2)[:, None]
    X_adj = X_adj / norm.clip(1e-100, np.inf)
    rep = "X_%s" % key_added
    M = adata.var[margin].values[:, None]
    M = (M - np.min(M)) / (np.max(M) - np.min(M))  # min-max scale
    adata.varm[rep] = np.hstack((2 * M - 1,  # -1 to 1 scale
                                 X_adj)).astype(np.float32)
    if key_added in adata.uns.keys() and "graphs" in adata.uns[key_added].keys():
        graphs = adata.uns[key_added]["graphs"]
    else:
        graphs = []
    adata.uns[key_added] = {"rep": rep,
                            "margin": margin,
                            "power": power,
                            "graphs": graphs,
                            "n_pcs": len(s),
                            "zero_center": zero_center}
    return 0
