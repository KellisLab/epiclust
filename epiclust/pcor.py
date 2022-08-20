
class PartialCor:
        def __init__(self, feat_cor_mat_L, self_cor_mat, feat_cor_mat_R, batch_size=5000):
                import numpy as np
                if len(np.shape(feat_cor_mat_L)) == 1:
                        self.lfcor = feat_cor_mat_L[:, None]
                else:
                        self.lfcor = feat_cor_mat_L
                if len(np.shape(feat_cor_mat_R)) == 1:
                        self.rfcor = feat_cor_mat_R[None, :]
                else:
                        self.rfcor = feat_cor_mat_R
                if len(np.shape(self_cor_mat)) != 2:
                        self_cor_mat = np.array(self_cor_mat).reshape(1, 1)
                assert self.lfcor.shape[1] == self_cor_mat.shape[0]
                assert self.rfcor.shape[0] == self_cor_mat.shape[1]
                ### TODO: check nearPD
                # if not isPD(self_cor_mat):
                #         self_cor_mat = nearPD(self_cor_mat)
                self.ainv = np.linalg.inv(self_cor_mat)
                self.batch_size = batch_size
        def __call__(self, cor, row, col, eps=1e-16):
                """partial cor formula from stored correlations:
                (PP - RP_i^T (RR^{-1}) RP_j) / sqrt((1 - RP_i^T RR^{-1} RP_i) * (1 - RP_i^T RR^{-1} RP_i))
                but RP is lfcor and rfcor in case of asymmetry
                """
                import numpy as np
                cor = np.ravel(cor).astype(np.float64)
                assert len(cor) == len(row)
                assert len(cor) == len(col)
                out = np.zeros_like(cor)
                for left in np.arange(0, len(cor), self.batch_size):
                        right = min(left + self.batch_size, len(cor))
                        IL = row[left:right]
                        IR = col[left:right]
                        sub = np.einsum("ij,jk,ki->i", self.lfcor[IL, :], inv, self.rfcor[:, IR])
                        denom_L = np.einsum("ij,jk,ki->i", self.lfcor[IL, :], inv, self.lfcor[IL, :].T)
                        denom_R = np.einsum("ij,jk,ki->i", self.rfcor[:, IR].T, inv, self.rfcor[:, IR])
                        denom_square = np.clip((1 - denom_L) * (1 - denom_R), a_min=1e-20, a_max=np.inf)
                        one_over_det = denom_square - (cor[left:right] - sub)**2
                        out[left:right] = (cor[left:right] - sub)/np.sqrt(denom_square)
                        out[left:right] *= np.sign(one_over_det)
                return out

def adjust_covariates(adata, covariates, min_variance=1e-20, batch_size=10000):
        import numpy as np
        import pandas as pd
        from sklearn.decomposition import PCA
        br = pd.get_dummies(adata.obs[covariates]).values
        br = br[:, np.std(br, axis=0) > 0] ### remove zero variance cols e.g. pd.Categorical not present in data
        ### first compute full rank PCA:
        pca = PCA(n_components=br.shape[1]).fit(br)
        n_comp = np.sum(pca.explained_variance_ratio_ > min_variance)
        ### then trim to maximize variance
        br = PCA(n_components=n_comp).fit_transform(br)### zero centered
        br = br / np.linalg.norm(br, axis=0, ord=2)[None, :]
        RR = br.T.dot(br) ### corr coef
        PR = np.zeros((adata.shape[1], br.shape[1]))
        br = br - br.mean(0)[None, :]
        for left in np.arange(0, adata.shape[1], batch_size):
                right = min(left + batch_size, adata.shape[1])
                if scipy.sparse.issparse(adata.X):
                        X = np.asarray(adata.X[:, left:right].todense(), dtype=np.float64)
                else:
                        X = np.asarray(adata.X[:, left:right], dtype=np.float64)
                X = X - X.mean(0)[None, :]
                X = X / np.linalg.norm(X, ord=2, axis=0)[None, :].clip(1e-50, np.inf)
                PR[left:right, :] = X.T.dot(br)
        return PartialCor(PR, RR, PR.T, batch_size=batch_size)
