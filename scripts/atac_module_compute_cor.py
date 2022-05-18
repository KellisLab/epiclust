#!/usr/bin/env python3
import os
import argparse
import dask
dask.config.set(scheduler='processes')  # overwrite default with multiprocessing scheduler

def run(input=None, output=None,
        nbin=50, npc=0, margin="n_cells_by_counts",
        batch_keep=None, batch_remove=None, power=0,
        z=4., margin_of_error=0.05, nbin_sample=2,
        spline_k=2, min_std=0.001, nproc=os.cpu_count()):
        import anndata
        import atac_module as am
        adata = anndata.read(input, backed="r")
        if batch_remove is not None:
                correct = am.get_covariate_transformer(adata,
                                                       batch_remove=batch_remove,
                                                       batch_keep=batch_keep)
        else:
                correct = None
        mm = am.ModuleMatrix(adata, nbins=nbin,
                             margin=margin, npc=npc)
        del adata
        return mm.build(power, correct=correct, cutoff_z=z, sample_z=2,
                        margin_of_error=margin_of_error,
                        n_bins_sample=nbin_sample, k=spline_k,
                        min_std=min_std, nproc=nproc, output=output)

if __name__ == "__main__":
        ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        ap.add_argument("-i", "--input", required=True,
                        help="AnnData H5AD file with pre-computed margin (--margin) and PCA")
        ap.add_argument("-o", "--output", required=True, help="H5 file output")
        ap.add_argument("-p", "--power", default=0, type=float,
                        help="SVD power of eigenvalues used for covariance matrix calculation.")
        ap.add_argument("-z", default=4., type=float, help="Cutoff used for output matrix")
        ap.add_argument("--nbin", default=50, type=int,
                        help="Number of bins that features are placed into according to splitting of margin variable")
        ap.add_argument("--nbin-sample", type=int, default=2,
                        help="Number of bins (one-sided, inclusive) used for mean & std calculation for one bin")
        ap.add_argument("--batch-remove", nargs="+",
                        help="Batch-level effects (in AnnData .obs) that confound correlation")
        ap.add_argument("--batch-keep", nargs="+",
                        help="Batch-level effects (in AnnData .obs) that you want to keep for correlation that would be removed if --batch-remove options are removed")
        ap.add_argument("--nproc", default=os.cpu_count(), type=int,
                        help="Number of worker pool processes to use. Note that a writer process is separately launched outside of this number")
        ap.add_argument("--margin-of-error", type=float, default=0.01,
                        help="Margin of error used to calculate number of samples necessary for mean & std calculation")
        ap.add_argument("--npc", default=0, type=int,
                        help="Number of components to use")
        ap.add_argument("--spline-k", default=2, type=int,
                        help="Degree of bivariate spline used for mean & std interpolation. 2 seems to work")
        ap.add_argument("--margin", default="n_cells_by_counts",
                        help="Variable (in AnnData .var) that is confounding correlation matrix")
        ap.add_argument("--min-std", default=0.001, type=float,
                        help="Minimum standard deviation such that higher correlations can be found")
        args = vars(ap.parse_args())
        run(**args)
