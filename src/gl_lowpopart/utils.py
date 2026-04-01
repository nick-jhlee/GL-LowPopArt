"""Utility helpers for GL-LowPopArt."""

import warnings

import numpy as np

try:
    from scipy.stats import t as student_t
except ImportError:
    student_t = None

warnings.filterwarnings("ignore", category=RuntimeWarning)


def sigmoid(x):
    x = np.minimum(np.maximum(x, -14), 14)
    return 1 / (1 + np.exp(-x))


def dsigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def psi(x):
    x_arr = np.asarray(x)
    out = np.where(
        x_arr >= 0,
        np.log1p(x_arr + 0.5 * x_arr**2),
        -np.log1p(-x_arr + 0.5 * x_arr**2),
    )
    if np.ndim(out) == 0:
        return float(out)
    return out


def dilation(a):
    d1, d2 = a.shape
    return np.bmat([[np.zeros((d1, d1)), a], [a.T, np.zeros((d2, d2))]])

def psi_nu(A_batch, nu):
    """
    Computes psi_nu for a batch of matrices simultaneously.
    A_batch: shape (N, d1, d2)
    """
    N, d1, d2 = A_batch.shape
    d_sum = d1 + d2
    
    # 1. Batched Dilation
    # Create an array of zeros of shape (N, d1+d2, d1+d2)
    A_dil = np.zeros((N, d_sum, d_sum))
    # Place A_batch in the upper right
    A_dil[:, :d1, d1:] = A_batch
    # Place transposed A_batch in the lower left
    A_dil[:, d1:, :d1] = A_batch.transpose(0, 2, 1) 
    
    # 2. Batched Eigendecomposition
    # np.linalg.eigh accepts (N, M, M) and returns w: (N, M), v: (N, M, M)
    w, v = np.linalg.eigh(A_dil)
    
    # 3. Apply psi function to all eigenvalues
    # Assuming psi_func can handle numpy arrays. If not, use np.vectorize(psi)(nu * w)
    w_psi = psi(nu * w) # Shape: (N, d_sum)
    
    # 4. Batched Matrix Reconstruction: v @ diag(w_psi) @ v.T
    # We use numpy broadcasting (w_psi[:, np.newaxis, :]) to multiply each column 
    # of v by its corresponding eigenvalue before doing the batched matrix multiplication.
    v_scaled = v * w_psi[:, np.newaxis, :] 
    tmp_batch = (1 / nu) * np.matmul(v_scaled, v.transpose(0, 2, 1))
    
    # 5. Extract the upper right block for all N matrices
    return tmp_batch[:, :d1, d1:]

def mean_t_ci(data, alpha=0.05):
    """Two-sided t-based confidence interval for the sample mean."""
    arr = np.asarray(data, dtype=float)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return np.nan, np.nan

    mean = float(np.mean(arr))
    if arr.size == 1:
        return mean, mean

    se = float(np.std(arr, ddof=1) / np.sqrt(arr.size))
    if se == 0.0:
        return mean, mean

    if student_t is not None:
        t_crit = float(student_t.ppf(1.0 - alpha / 2.0, arr.size - 1))
    else:
        # Fallback to normal approximation if scipy is unavailable.
        t_crit = 1.959963984540054

    half_width = t_crit * se
    return mean - half_width, mean + half_width

