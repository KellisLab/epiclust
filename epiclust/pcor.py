
import numba
import numpy as np
from .extraction import extract_pca

@numba.njit
def pcor_adjust(cor, row_varm, col_varm, inv):
    res = np.empty(cor.shape[0], dtype=cor.dtype)
    for i in range(len(cor)):
        sub = row_varm[i, :].dot(inv).dot(col_varm[i, :])
        denom_L = row_varm[i, :].dot(inv).dot(row_varm[i, :].T)
        denom_R = col_varm[i, :].dot(inv).dot(col_varm[i, :].T)
        denom_square = (1 - denom_L) * (1 - denom_R)
        if denom_square < 1e-20:
            denom_square = 1e-20
        one_over_det = denom_square - (cor[i] - sub)**2
        res[i] = np.sign(one_over_det) * (cor[i] - sub) / np.sqrt(denom_square)
    return res


def adjust_covariates(adata, covariates=None,
                      min_variance=1e-20, batch_size=1000, key="epiclust"):
    import numpy as np
    import pandas as pd
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import scipy.sparse
    from tqdm.auto import tqdm
    if not isinstance(covariates, list):
        if covariates is None:
            return 0
        else:
            covariates = [covariates]
    if len(covariates) == 0:
        return 0
    br = pd.get_dummies(adata.obs[covariates]).values
    # remove zero variance cols e.g. pd.Categorical not present in data
    br = br[:, np.std(br, axis=0) > 0]
    # then Z-scale for vars. like age
    br = StandardScaler().fit_transform(br)
    # first compute full rank PCA:
    pca = PCA(n_components=br.shape[1]).fit(br)
    n_comp = np.sum(pca.explained_variance_ratio_ > min_variance)
    # then trim to maximize variance
    br = PCA(n_components=n_comp).fit_transform(br)  # zero centered
    br = br / np.linalg.norm(br, axis=0, ord=2)[None, :]
    RR = br.T.dot(br)  # corr coef
    PR = np.zeros((adata.shape[1], br.shape[1]))
    br = br - br.mean(0)[None, :]
    U, s, VT = extract_pca(adata, n_pcs=adata.uns[key]["n_pcs"])
    power = adata.uns[key]["power"]
    Us = U.dot(np.diag(s**power)).astype(np.float64)
    del U, s
    for left in tqdm(np.arange(0, adata.shape[1], batch_size)):
        right = min(left + batch_size, adata.shape[1])
        X = np.asarray(Us @ VT[:, left:right], dtype=np.float64)
        X = X - X.mean(0)[None, :]
        X = X / np.linalg.norm(X, ord=2, axis=0)[None, :].clip(1e-50, np.inf)
        PR[left:right, :] = X.T.dot(br)
    ### Now to append to rep
    vkey = adata.uns[key]["rep"]
    if len(np.shape(PR)) == 1:
        adata.varm[vkey] = np.hstack((adata.varm[vkey], PR[:, None]))
    else:
        adata.varm[vkey] = np.hstack((adata.varm[vkey], PR))
    adata.uns[key]["adjust"] = {
        "inv": np.linalg.pinv(RR),
        "covariates": covariates
    }
    return adata


def extract_pcor_info(adata, key="epiclust"):
    if (key not in adata.uns) or ("adjust" not in adata.uns[key]):
        return {}
    return {
        "pcor_inv": adata.uns[key]["adjust"]["inv"]
    }
