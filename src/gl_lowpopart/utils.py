"""Utility helpers for GL-LowPopArt."""

import multiprocessing as mp
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)


def sigmoid(x):
    x = np.minimum(np.maximum(x, -14), 14)
    return 1 / (1 + np.exp(-x))


def dsigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def psi(x):
    if x >= 0:
        return np.log(1 + x + x**2 / 2)
    return -np.log(1 - x + x**2 / 2)


def dilation(a):
    d1, d2 = a.shape
    return np.bmat([[np.zeros((d1, d1)), a], [a.T, np.zeros((d2, d2))]])


def psi_nu(a, nu):
    d1, d2 = a.shape
    a_dilation = dilation(a)
    d, v = np.linalg.eigh(a_dilation)
    d = np.diag([psi(nu * x) for x in d])
    tmp = (1 / nu) * v @ d @ v.T
    return tmp[:d1, d1 : d1 + d2]


def compute_boot_stat(args):
    _, seed_outer, seed_inner, data, n_boot2 = args
    n = len(data)

    rng_outer = np.random.default_rng(seed_outer)
    boot_sample = rng_outer.choice(data, size=n, replace=True)
    boot_mean = np.mean(boot_sample)

    rng_inner = np.random.default_rng(seed_inner)
    boot2_means = np.array(
        [np.mean(rng_inner.choice(boot_sample, size=n, replace=True)) for _ in range(n_boot2)]
    )
    boot_std = np.std(boot2_means, ddof=1)
    return boot_mean, boot_std


def studentized_double_bootstrap(data, n_boot=1000, n_boot2=500, alpha=0.05):
    original_mean = np.mean(data)
    rng = np.random.default_rng()
    seeds_outer = rng.integers(0, 2**32, size=n_boot)
    seeds_inner = rng.integers(0, 2**32, size=n_boot)
    args = [(i, seeds_outer[i], seeds_inner[i], data, n_boot2) for i in range(n_boot)]

    with mp.Pool(processes=mp.cpu_count()) as pool:
        boot_stats = list(pool.imap(compute_boot_stat, args))

    boot_means, boot_stds = zip(*boot_stats)
    boot_means = np.array(boot_means)
    boot_stds = np.array(boot_stds)
    bias_corrected_mean = 2 * original_mean - np.mean(boot_means)

    with np.errstate(divide="ignore", invalid="ignore"):
        studentized_stats = (boot_means - bias_corrected_mean) / boot_stds

    se_rng = np.random.default_rng()
    original_boot2_means = np.array(
        [np.mean(se_rng.choice(data, size=len(data), replace=True)) for _ in range(n_boot2)]
    )
    se_hat = np.std(original_boot2_means, ddof=1)

    t_lower = np.percentile(studentized_stats, 100 * (alpha / 2))
    t_upper = np.percentile(studentized_stats, 100 * (1 - alpha / 2))

    ci_lower = original_mean - t_upper * se_hat
    ci_upper = original_mean - t_lower * se_hat

    if np.isnan(ci_lower) or np.isnan(ci_upper) or ci_lower < 0:
        ci_lower = np.percentile(boot_means, 100 * (alpha / 2))
        ci_upper = np.percentile(boot_means, 100 * (1 - alpha / 2))

    return ci_lower, ci_upper

