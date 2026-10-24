"""Spearman rank correlation with ties at twelve decimals, and the mean of a top-k selection."""
import numpy as np

from scipy.stats import spearmanr

DECIMALS = 12

def tied(values):
    """Number of values that share their rounded value with another."""
    v = np.round(np.asarray(values, float), DECIMALS)
    _, counts = np.unique(v, return_counts=True)
    return int(counts[counts > 1].sum())

def spearman(a, b):
    """Spearman's rho with ties at the resolution of the fit resolved as ties."""
    return float(spearmanr(np.round(np.asarray(a, float), DECIMALS),
                           np.round(np.asarray(b, float), DECIMALS)).statistic)

def top_k_mean(rank_by, values, k):
    """Mean of `values` over the k molecules with the largest `rank_by`.

    Where a tie straddles the cut, `numpy.argsort` takes whichever of the tied molecules it
    happens to order first, so the answer depends on the arithmetic again. A chemist choosing
    ten building blocks from a list with three tied at tenth place picks one of the three; the
    value of that choice, averaged over which one, is what this returns — the tied group shares
    the remaining slots equally.
    """
    v = np.round(np.asarray(rank_by, float), DECIMALS)
    values = np.asarray(values, float)
    weights = np.zeros(len(v))
    left = k
    for level in np.unique(v)[::-1]:
        at = np.flatnonzero(v == level)
        take = min(left, len(at))
        weights[at] = take / len(at)
        left -= take
        if left == 0:
            break
    return float(np.sum(weights * values) / k)
