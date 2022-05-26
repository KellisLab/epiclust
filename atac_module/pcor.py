
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import scipy.sparse
from .utils import nearPD, isPD
def adjust_partial_cor(mat, L, R, inv):
        """formula: sub = L INV R^T
        f"""
        sub = L @ inv @ R
        denom_row = np.einsum("ij,jk,ki->i", L, inv, L.T)
        denom_col = np.einsum("ij,jk,ki->i", R.T, inv, R)
        denom_square = np.multiply.outer(1-denom_row, 1-denom_col)
        denom_square = np.clip(denom_square, a_min=1e-20, a_max=np.inf)
        one_over_det = denom_square - np.multiply(mat - sub, mat - sub)
        return np.sign(one_over_det) * (mat - sub)/np.sqrt(denom_square)

def full_partial_correct(zmat, row, col, epsilon=1e-50):
        cor = np.tanh(zmat)
        if not isPD(cor):
                cor = nearPD(cor)
        icor = np.linalg.inv(cor)
        D = np.diag(icor) + 0
        D[np.abs(D) < epsilon] = epsilon
        neg_out = icor * np.outer(D, D)**(-0.5)
        print(neg_out.shape)
        return np.arctanh(np.clip(-neg_out, -1+1e-16, 1-1e-16))

class PartialCor:
        def __init__(self, feat_cor_mat_L, self_cor_mat, feat_cor_mat_R, batch_size=5000):
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
                if not isPD(self_cor_mat):
                        self_cor_mat = nearPD(self_cor_mat)
                self.ainv = np.linalg.inv(self_cor_mat)
                self.batch_size = batch_size
        def __call__(self, mat, row=None, col=None, eps=1e-16):
                if row is None:
                        row = np.arange(mat.shape[0])
                if col is None:
                        col = np.arange(mat.shape[1])
                assert len(row) == mat.shape[0]
                assert len(col) == mat.shape[1]
                out = np.zeros_like(mat)
                for rbegin in range(0, len(row), self.batch_size):
                        rend = min(rbegin + self.batch_size, len(row))
                        MR = mat[rbegin:rend, :]
                        L = self.lfcor[row[rbegin:rend], :]
                        for cbegin in range(0, len(col), self.batch_size):
                                cend = min(cbegin + self.batch_size, len(col))
                                MRC = MR[:, cbegin:cend]
                                if scipy.sparse.issparse(MRC):
                                        MRC = MRC.todense()
                                R = self.rfcor[:, col[cbegin:cend]]
                                out[rbegin:rend, cbegin:cend] = adjust_partial_cor(np.tanh(np.asarray(MRC)),
                                                                                   L=L, R=R,
                                                                                   inv=self.ainv)
                out = np.arctanh(np.clip(out, a_min=-1+eps, a_max=1-eps))
                return (out + mat)/2 ## average
