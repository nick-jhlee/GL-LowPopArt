import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import multiprocessing as mp
import warnings
import functools

# Suppress RuntimeWarning
warnings.filterwarnings('ignore', category=RuntimeWarning)

def sigmoid(x):
    x = np.minimum(np.maximum(x, -14), 14)
    y = 1 / (1 + np.exp(- x))
    return y

def dsigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Catoni-style truncation of the spectrum
def psi(x):
    if x >= 0:
        return np.log(1 + x + x**2/2)
    else:
        return -np.log(1 - x + x**2/2)

def dilation(A):
    d1, d2 = A.shape
    return np.bmat([[np.zeros((d1, d1)), A], [A.T, np.zeros((d2, d2))]])

def psi_nu(A, nu):
    d1, d2 = A.shape
    A_dilation = dilation(A)
    # eigenvalue decomposition
    D, V = np.linalg.eigh(A_dilation)
    D = np.diag([psi(nu * d) for d in D])
    tmp = (1 / nu) * V @ D @ V.T
    return tmp[:d1, d1:d1+d2]

def compute_boot_stat(args):
    """Compute (boot_mean, boot_std) using double bootstrap."""
    i, seed_outer, seed_inner, data, n_boot2 = args
    n = len(data)

    rng_outer = np.random.default_rng(seed_outer)
    boot_sample = rng_outer.choice(data, size=n, replace=True)
    boot_mean = np.mean(boot_sample)

    rng_inner = np.random.default_rng(seed_inner)
    boot2_means = np.array([
        np.mean(rng_inner.choice(boot_sample, size=n, replace=True))
        for _ in range(n_boot2)
    ])
    boot_std = np.std(boot2_means, ddof=1)  # SE estimate of boot_mean

    return boot_mean, boot_std

def studentized_double_bootstrap(data, n_boot=1000, n_boot2=500, alpha=0.05):
    """
    Bias-corrected studentized double bootstrap with fallback to percentile bootstrap.

    Args:
        data: 1D numpy array of observations
        n_boot: First-level bootstrap samples
        n_boot2: Second-level bootstrap samples per outer sample
        alpha: Significance level (e.g., 0.05 for 95% CI)

    Returns:
        (ci_lower, ci_upper): Confidence interval
    """
    n = len(data)
    original_mean = np.mean(data)

    # Generate independent RNG seeds for outer and inner bootstrap levels
    rng = np.random.default_rng()
    seeds_outer = rng.integers(0, 2**32, size=n_boot)
    seeds_inner = rng.integers(0, 2**32, size=n_boot)

    # Prepare parallel arguments
    args = [(i, seeds_outer[i], seeds_inner[i], data, n_boot2) for i in range(n_boot)]

    # Parallel computation of bootstrap statistics
    with mp.Pool(processes=mp.cpu_count()) as pool:
        boot_stats = list(pool.imap(compute_boot_stat, args))

    boot_means, boot_stds = zip(*boot_stats)
    boot_means = np.array(boot_means)
    boot_stds = np.array(boot_stds)

    # Bias correction: 2 * original - mean of boot_means
    bias_corrected_mean = 2 * original_mean - np.mean(boot_means)

    # Compute studentized t-statistics
    with np.errstate(divide='ignore', invalid='ignore'):
        studentized_stats = (boot_means - bias_corrected_mean) / boot_stds

    # Compute standard error of original mean via second-level bootstrap
    se_rng = np.random.default_rng()
    original_boot2_means = np.array([
        np.mean(se_rng.choice(data, size=n, replace=True))
        for _ in range(n_boot2)
    ])
    se_hat = np.std(original_boot2_means, ddof=1)

    # Studentized CI using percentiles of t-statistics
    t_lower = np.percentile(studentized_stats, 100 * (alpha / 2))
    t_upper = np.percentile(studentized_stats, 100 * (1 - alpha / 2))

    ci_lower = original_mean - t_upper * se_hat
    ci_upper = original_mean - t_lower * se_hat

    # Fallback to percentile CI if NaNs appear or ci_lower < 0
    if np.isnan(ci_lower) or np.isnan(ci_upper) or ci_lower < 0:
        ci_lower = np.percentile(boot_means, 100 * (alpha / 2))
        ci_upper = np.percentile(boot_means, 100 * (1 - alpha / 2))

    return ci_lower, ci_upper