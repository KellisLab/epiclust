
import numpy as np
import numba
from .bispeu import bispeu_wrapped as bispeu
import numba
import logging
from numba import float64, types

@numba.njit
def distance(x, y, min_std=0.001, mean_weights=np.ones(6), std_weights=np.ones(6)):
    """kwargs is .uns[key]"""
    dim = x.shape[0]
    margin = np.ones(6)
    margin[1] = x[0]
    margin[2] = x[1]
    margin[3] = x[0] * x[0]
    margin[4] = x[0] * x[1]
    margin[5] = x[1] * x[1]
    # margin[6] = x[0] * x[0] * x[0]
    # margin[7] = x[0] * x[0] * x[1]
    # margin[8] = x[0] * x[1] * x[1]
    # margin[9] = x[1] * x[1] * x[1]
    result = np.zeros(1)
    mean = np.zeros(1)
    std = np.zeros(1)
    for i in range(1, dim):
        dot = x[i] * y[i]
        result += dot
    mean += margin.dot(mean_weights)
    std += margin.dot(std_weights)
    std[std < min_std] = min_std
    result[result < -1+1e-16] = -1+1e-16
    result[result > 1-1e-16] = 1-1e-16
    out = np.exp(-1 * (np.arctanh(result) - mean) / std)
    return(out[0])
