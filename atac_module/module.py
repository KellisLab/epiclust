from .perbin import create_bins_quantile, calc_perbin_stats
from .spline import build_spline
from .zmatrix import fill_matrix #, fill_corrected_matrix
class ModuleMatrix:
        def __init__(self, adata, nbins=50, margin="n_cells_by_counts", npc=0):
                self.U, self.s, self.VT = extract_pca(adata, npc=npc)
                if margin not in adata.var.columns:
                        raise ValueError("Margin %s not in adata.var" % margin)
                self.margin = adata.var[margin].values
                self.bin_assign, self.bin_edges = create_bins_quantile(self.margin, nbins=nbins)
        def _build_splines(self, X_adj, min_std, k, **kwargs):
                cps = calc_perbin_stats(X_adj, self.bin_assign, **kwargs)
                S = {}
                S["std"] = build_spline(self.bin_assign, self.bin_edges, cps["std"], cps["counts"], k=k, a_min=min_std)
                S["mean"] = build_spline(self.bin_assign, self.bin_edges, cps["mean"], cps["counts"], k=k)
                return S
        def build(self, power=0, correct=None, cutoff_z=4, sample_z=2, margin_of_error=0.05, n_bins_sample=2, k=2, min_std=0.001):
                X_adj = self.VT.T.astype(np.double) @ np.diag(self.s.astype(double)**power)
                S = self._build_splines(X_adj, min_std=min_std, k=k, z=sample_z, margin_of_error=margin_of_error, n_bins_sample=n_bins_sample)
                if correct is None:
                        return fill_matrix(margin=self.margin, X_adj=X_adj, bin_assign=self.bin_assign, spline_table=S, z=cutoff_z)
                else:
                        return fill_matrix(margin=self.margin, X_adj=X_adj, bin_assign=self.bin_assign, spline_table=S, z=cutoff_z, correct=correct)
