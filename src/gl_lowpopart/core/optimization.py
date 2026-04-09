"""Stage-I and Stage-II optimization routines."""

import os
import warnings

import numpy as np

from gl_lowpopart.utils import psi_nu
import cvxpy as cp
import mosek

warnings.filterwarnings("ignore", category=UserWarning, module="mosek")
warnings.filterwarnings("ignore", category=UserWarning, module="cvxpy")

mosek_env = mosek.Env()
# Prefer user-level license outside repository.
_default_license = os.path.expanduser("~/.mosek/mosek.lic")
_license_path = os.environ.get("MOSEKLM_LICENSE_FILE", _default_license)
if os.path.exists(_license_path):
    mosek_env.putlicensepath(_license_path)


def E_optimal_design(env):
    X_arms = env.X_arms
    K = env.K
    pi_E = cp.Variable(K, nonneg=True)
    constraints = [cp.sum(pi_E) == 1]
    V_pi = X_arms.T @ cp.diag(pi_E) @ X_arms
    objective_E = cp.Maximize(cp.lambda_min(V_pi))
    prob_E = cp.Problem(objective_E, constraints)
    try:
        prob_E.solve(solver=cp.MOSEK)
    except Exception:
        prob_E.solve(solver=cp.MOSEK, verbose=True)
    pi_E = np.abs(np.array(pi_E.value))
    pi_E /= np.sum(pi_E)
    return pi_E


def _svt(matrix, threshold):
    """Singular Value Thresholding (prox of nuclear norm)."""
    U, s, Vt = np.linalg.svd(matrix, full_matrices=False)
    s_shrunk = np.maximum(s - threshold, 0.0)
    return U @ (s_shrunk[:, None] * Vt)


def _neg_loglik_and_grad(theta_mat, X, y, model):
    """Average negative log-likelihood and gradient wrt theta matrix."""
    n_samples = X.shape[0]
    theta = theta_mat.flatten("F")
    eta = X @ theta

    if model == "bernoulli":
        eta_clip = np.clip(eta, -30.0, 30.0)
        mu = 1.0 / (1.0 + np.exp(-eta_clip))
        loss = np.mean(np.log1p(np.exp(eta_clip)) - y * eta_clip)
        grad_vec = (X.T @ (mu - y)) / n_samples
    elif model == "poisson":
        eta_clip = np.clip(eta, -20.0, 20.0)
        mu = np.exp(eta_clip)
        loss = np.mean(mu - y * eta_clip)
        grad_vec = (X.T @ (mu - y)) / n_samples
    else:
        raise ValueError(f"Unsupported model: {model}")

    grad = grad_vec.reshape(theta_mat.shape, order="F")
    return float(loss), grad


def _fista_nuclear_mle(X, y, d1, d2, nuc_coef, model, max_iter=500, tol=1e-6):
    """Solve min f(Theta) + nuc_coef * ||Theta||_* via restarted FISTA + backtracking."""
    theta = np.zeros((d1, d2), dtype=np.float64)
    yk = theta.copy()
    tk = 1.0
    lipschitz = 1.0

    for _ in range(max_iter):
        f_yk, grad_yk = _neg_loglik_and_grad(yk, X, y, model)

        while True:
            step = 1.0 / lipschitz
            candidate = _svt(yk - step * grad_yk, nuc_coef * step)
            f_candidate, _ = _neg_loglik_and_grad(candidate, X, y, model)
            diff = candidate - yk
            quad_upper = f_yk + np.sum(grad_yk * diff) + 0.5 * lipschitz * np.sum(diff * diff)
            if f_candidate <= quad_upper + 1e-12:
                break
            lipschitz *= 2.0

        t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * tk * tk))
        momentum = ((tk - 1.0) / t_next) * (candidate - theta)
        y_next = candidate + momentum

        # Adaptive restart (O'Donoghue-Candes): drop momentum if it is misaligned.
        if np.sum((candidate - theta) * (yk - candidate)) > 0.0:
            t_next = 1.0
            y_next = candidate

        denom = max(1.0, np.linalg.norm(theta, ord="fro"))
        rel_change = np.linalg.norm(candidate - theta, ord="fro") / denom

        theta = candidate
        yk = y_next
        tk = t_next
        lipschitz = max(lipschitz * 0.9, 1e-6)

        if rel_change < tol:
            break

    return theta


def _cvxpy_nuclear_mle(X, y, d1, d2, nuc_coef, model):
    Theta = cp.Variable((d1, d2))
    theta = cp.vec(Theta, order="F")
    linear_term = X @ theta

    if model == "bernoulli":
        log_likelihood = cp.sum(cp.multiply(y, linear_term) - cp.logistic(linear_term)) / X.shape[0]
    elif model == "poisson":
        log_likelihood = cp.sum(cp.multiply(y, linear_term) - cp.exp(linear_term)) / X.shape[0]
    else:
        raise ValueError(f"Unsupported model: {model}")

    prob = cp.Problem(cp.Maximize(log_likelihood - nuc_coef * cp.normNuc(Theta)))
    try:
        prob.solve(solver=cp.MOSEK)
    except Exception:
        prob.solve(solver=cp.SCS, verbose=False)

    if Theta.value is None:
        raise RuntimeError("cvxpy Stage-I solver failed to produce a solution.")
    return np.array(Theta.value)


def nuc_norm_MLE(env, N1, d1, d2, nuc_coef, E_optimal=True, stage1_solver="fista"):
    K = env.K
    arm_set = env.arm_set
    pi_E = E_optimal_design(env) if E_optimal else np.ones(K) / K

    X1, y1 = np.zeros((N1, d1 * d2)), np.zeros(N1)
    idx_1 = np.random.choice(K, N1, p=pi_E)
    for i, idx in enumerate(idx_1):
        arm = arm_set[idx]
        X1[i] = arm.flatten("F")
        y1[i] = env.get_reward(arm)

    if stage1_solver == "fista":
        Theta = _fista_nuclear_mle(X1, y1, d1, d2, nuc_coef, env.model)
    elif stage1_solver == "cvxpy":
        Theta = _cvxpy_nuclear_mle(X1, y1, d1, d2, nuc_coef, env.model)
    else:
        raise ValueError(f"Unknown stage1_solver: {stage1_solver}")
    return Theta, X1, y1


def GL_LowPopArt(env, N2, d1, d2, delta, Theta0, GL_optimal=True):
    X_arms = env.X_arms
    K = env.K
    arm_set = env.arm_set
    d_ = d1 * d2
    theta0 = Theta0.flatten("F")

    mu_diags = np.diag([env.dmean_from_eta(tmp) for tmp in X_arms @ theta0])
    mu_diags = np.ascontiguousarray(mu_diags, dtype=np.float64)

    if GL_optimal:
        pi = cp.Variable(K, nonneg=True)
        mu_diags_cp = cp.Constant(mu_diags)
        H_pi = X_arms.T @ cp.diag(pi) @ mu_diags_cp @ X_arms
        H_inv = cp.Variable((d_, d_), PSD=True)

        # GL-design objective indexing (Fortran vec-order consistent).
        # - D_col accumulates blocks indexed by I_m^(row): fixed row m, varying columns -> size d2 x d2
        # - D_row accumulates blocks indexed by I_m^(col): fixed column m, varying rows -> size d1 x d1
        D_col = cp.Constant(np.zeros((d2, d2)))
        for m in range(d1):
            idx_set = [m + l * d1 for l in range(d2)]
            D_col = D_col + H_inv[np.ix_(idx_set, idx_set)]

        D_row = cp.Constant(np.zeros((d1, d1)))
        for m in range(d2):
            idx_set = [m * d1 + i for i in range(d1)]
            D_row = D_row + H_inv[np.ix_(idx_set, idx_set)]

        objective = cp.Minimize(cp.maximum(cp.lambda_max(D_col), cp.lambda_max(D_row)))
        prob = cp.Problem(
            objective,
            [cp.sum(pi) == 1, cp.bmat([[H_pi, np.eye(d_)], [np.eye(d_), H_inv]]) >> 0],
        )
        try:
            prob.solve(solver=cp.MOSEK)
        except Exception:
            prob.solve(solver=cp.MOSEK, verbose=True)
        pi_optimal = np.abs(np.array(pi.value))
        pi_optimal /= np.sum(pi_optimal)
        design_value = prob.value
    else:
        pi_optimal = np.ones(K) / K
        H_pi = X_arms.T @ np.diag(pi_optimal) @ mu_diags @ X_arms
        H_inv = np.linalg.inv(H_pi)
        # GL-design objective indexing (Fortran vec-order consistent).
        D_col = np.zeros((d2, d2))
        for m in range(d1):
            idx_set = [m + l * d1 for l in range(d2)]
            D_col = D_col + H_inv[np.ix_(idx_set, idx_set)]

        D_row = np.zeros((d1, d1))
        for m in range(d2):
            idx_set = [m * d1 + i for i in range(d1)]
            D_row = D_row + H_inv[np.ix_(idx_set, idx_set)]
        design_value = max(np.linalg.eigvals(D_col).real.max(), np.linalg.eigvals(D_row).real.max())

    H_inv_optimal = np.linalg.inv(X_arms.T @ np.diag(pi_optimal) @ mu_diags @ X_arms)

    X2, y2 = np.zeros((N2, d_)), np.zeros(N2)
    idx_2 = np.random.choice(K, N2, p=pi_optimal)
    for i, idx in enumerate(idx_2):
        arm = arm_set[idx]
        X2[i] = arm.flatten("F")
        y2[i] = env.get_reward(arm)
    
    nu = np.sqrt(2 * np.log(4*(d1 + d2) / delta) / (5 * design_value * N2))
    
    # 1. Compute all residuals at once
    residuals = y2 - env.mean_from_eta(X2 @ theta0) # Shape: (N2,)

    # 2. Compute the weighted vectors for all N2 samples
    # X2 is (N2, d_), H_inv_optimal is (d_, d_)
    vectors_batch = residuals[:, None] * (X2 @ H_inv_optimal) # Shape: (N2, d_)

    # 3. Reshape all vectors into matrices simultaneously 
    matrices_batch = vectors_batch.reshape(N2, d1, d2, order="F")

    # 4. Apply psi_nu and sum over the sample dimension
    Theta_Catoni = np.sum(psi_nu(matrices_batch, nu), axis=0) / N2
    Theta_Catoni += Theta0

    U, S, Vt = np.linalg.svd(Theta_Catoni)
    tau = np.sqrt(40 * design_value) * nu
    S_truncated = S * (S > tau).astype(int)
    Theta_final = U @ np.diag(S_truncated) @ Vt
    return Theta_final

