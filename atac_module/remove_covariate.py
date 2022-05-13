
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from .utils import outer_correlation, outer_correlation_svd
from .module import extract_pca
from .pcor import PartialCor

def get_covariate_transformer(adata, batch_remove=[], batch_keep=[], batch_size=5000, min_variance=1e-20):
        if not isinstance(batch_remove, list):
                if batch_remove is None:
                        batch_remove = []
                else:
                        batch_remove = [batch_remove]
        if not isinstance(batch_keep, list):
                if batch_keep is None:
                        batch_keep = []
                else:
                        batch_keep = [batch_keep]
        if len(batch_remove) == 0:
                raise ValueError("Must have batch to remove if using partial correlations")
        br = pd.get_dummies(adata.obs[batch_remove]).values
        br = br[:, np.std(br, axis=0) > 0] ### remove zero variance cols e.g. pd.Categorical not present in data
        ### first compute full rank PCA:
        pca = PCA(n_components=br.shape[1]).fit(br)
        n_comp = np.sum(pca.explained_variance_ratio_ > min_variance)
        ### then trim to maximize variance
        br = PCA(n_components=n_comp).fit_transform(br)
        RR = outer_correlation(br.T, br.T, batch_size=batch_size)
        U, s, VT = extract_pca(adata, transpose=True)
        print("Correlating removal-batches with data")
        PR = outer_correlation_svd(U, s, VT, br.T, batch_size=batch_size)
        if len(batch_keep) > 0:
                #### remove "keep" covariates from "remove" covariates by doing partial correlations
                #### on the two matrices required by the "remove" covariance matrix
                print(batch_keep)
                bk = pd.get_dummies(adata.obs[batch_keep]).values
                bk = bk[:, np.std(bk, axis=0) > 0]
                pca = PCA(n_components=bk.shape[1]).fit(bk)
                print("Batch-Keep Variance ratio:", pca.explained_variance_ratio_)
                n_comp = np.sum(pca.explained_variance_ratio_ > min_variance)
                bk = PCA(n_components=n_comp).fit_transform(bk)
                RK = outer_correlation(br.T, bk.T, batch_size=batch_size)
                KK = outer_correlation(bk.T, bk.T, batch_size=batch_size)
                pc = PartialCor(RK, KK, RK.T, batch_size=batch_size)
                RR = pc(RR)
                print("Correlating keep-batches with data")
                PK = outer_correlation_svd(U, s, VT, bk.T, batch_size=batch_size)
                pc = PartialCor(PK, KK, RK.T, batch_size=batch_size)
                PR = pc(PR)
        return PartialCor(PR, RR, PR.T, batch_size=batch_size)
