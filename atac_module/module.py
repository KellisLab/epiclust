from .perbin import create_bins_quantile, calc_perbin_stats
from .spline import Spline
from .zmatrix import fill_matrix, fill_matrix_parallel
import numpy as np
import os
def extract_pca(adata, transpose=False, npc=0):
        """we know that the S component is always positive
        so it can be recontructed from s^2/(DoF)"""
        Us = adata.obsm["X_pca"].astype(np.float64)
        s = adata.uns["pca"]["variance"].astype(np.float64)
        s = np.sqrt(s * (adata.shape[0] - 1))
        U = Us @ np.diag(1/s)
        VT = adata.varm["PCs"].T.astype(np.float64)
        if npc > 0 and npc <= len(s):
                U = U[:, range(npc)]
                s = s[range(npc)]
                VT = VT[range(npc), :]
        if transpose:
                return VT.T, s, U.T
        else:
                return U, s, VT

class ModuleMatrix:
        def __init__(self, adata, nbins=50, margin="n_cells_by_counts", npc=0):
                self.U, self.s, self.VT = extract_pca(adata, npc=npc)
                if margin not in adata.var.columns:
                        raise ValueError("Margin %s not in adata.var" % margin)
                self.margin = adata.var[margin].values
                self.varnames = adata.var.index.values
                self.bin_assign, self.bin_edges = create_bins_quantile(self.margin, nbins=nbins)
        def _build_splines(self, X_adj, min_std, k, **kwargs):
                print("Building splines")
                cps = calc_perbin_stats(X_adj, self.bin_assign, **kwargs)
                S = {}
                S["std"] = Spline(self.bin_assign, self.bin_edges,
                                  cps["std"], cps["counts"], k=k, a_min=min_std)
                S["mean"] = Spline(self.bin_assign, self.bin_edges,
                                   cps["mean"], cps["counts"], k=k)
                return S
        def build(self, power=0, correct=None, cutoff_z=4, sample_z=2,
                  margin_of_error=0.05, n_bins_sample=2, k=2, min_std=0.001,
                  nproc=os.cpu_count(), output="output.h5"):
                X_adj = self.VT.T @ np.diag(self.s**power)
                X_adj = X_adj / np.linalg.norm(X_adj, axis=1, ord=2)[:, None]
                S = self._build_splines(X_adj, min_std=min_std, k=k, z=sample_z,
                                        margin_of_error=margin_of_error,
                                        n_bins_sample=n_bins_sample)
                writer = {"output": output, "names": self.varnames}
                return fill_matrix(margin=self.margin, X_adj=X_adj,
                                            bin_assign=self.bin_assign,
                                            spline_table=S, z=cutoff_z,
                                            nproc=nproc,
                                            writer=writer, correct=correct)
