
import numpy as np
import numba
from .bispeu import bispeu_wrapped as bispeu
import numba
import logging

@numba.njit
def distance(x, y, mean_knots_x=None, mean_knots_y=None, mean_coeffs=None,
             std_knots_x=None, std_knots_y=None, std_coeffs=None,
             min_std=0.001, k=2):
    """kwargs is .uns[key]
-logcdf ensures self distance is zero
TODO: change .copy() for knots into readonly bispeu numba signature
TODO: exp(-x) to more cdf like
"""
    # logger = logging.getLogger("distance")
    # logger.setLevel(logging.DEBUG)
    # logger.debug(" ".join(["%s" % k for k in kwargs.keys()]))
    dim = x.shape[0]
    result = np.zeros(1)
    for i in range(1, dim):
        dot = x[i] * y[i]
        result += dot
    mean = bispeu(mean_knots_x.copy(), mean_knots_y.copy(), mean_coeffs.copy(),
                  k, k, x[:1].astype(np.float32), y[:1].astype(np.float32))
    std = bispeu(std_knots_x.copy(), std_knots_y.copy(), std_coeffs.copy(),
                 k, k, x[:1].astype(np.float32), y[:1].astype(np.float32))
    std[std < min_std] = min_std
    result[result < -1+1e-16] = -1+1e-16
    result[result > 1-1e-16] = 1-1e-16
    return np.exp(-1 * (np.arctanh(result) - mean) / std)[0]
