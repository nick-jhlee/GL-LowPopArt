"""Stage-I and Stage-II optimization routines."""

import warnings

import cvxpy as cp
import mosek
import numpy as np

from gl_lowpopart.utils import psi_nu

warnings.filterwarnings("ignore", category=UserWarning, module="mosek")
warnings.filterwarnings("ignore", category=UserWarning, module="cvxpy")

mosek_env = mosek.Env()
mosek_env.putlicensepath("../mosek/mosek.lic")


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


def nuc_norm_MLE(env, N1, d1, d2, nuc_coef, E_optimal=True):
    K = env.K
    arm_set = env.arm_set
    pi_E = E_optimal_design(env) if E_optimal else np.ones(K) / K

    X1, y1 = np.zeros((N1, d1 * d2)), np.zeros(N1)
    idx_1 = np.random.choice(K, N1, p=pi_E)
    for i, idx in enumerate(idx_1):
        arm = arm_set[idx]
        X1[i] = arm.flatten("F")
        y1[i] = env.get_reward(arm)

    Theta = cp.Variable((d1, d2))
    theta = cp.vec(Theta, order="F")
    linear_term = X1 @ theta
    if env.model == "bernoulli":
        log_likelihood = cp.sum(cp.multiply(y1, linear_term) - cp.logistic(linear_term)) / N1
    elif env.model == "poisson":
        log_likelihood = cp.sum(cp.multiply(y1, linear_term) - cp.exp(linear_term)) / N1
    else:
        raise ValueError(f"Unsupported model: {env.model}")

    objective = cp.Maximize(log_likelihood - nuc_coef * cp.normNuc(Theta))
    prob = cp.Problem(objective)
    try:
        prob.solve(solver=cp.MOSEK)
    except Exception:
        prob.solve(solver=cp.MOSEK, verbose=True)

    return np.array(Theta.value), X1, y1


def GL_LowPopArt(env, N2, d1, d2, delta, Theta0, c_nu=1, GL_optimal=True):
    X_arms = env.X_arms
    K = env.K
    arm_set = env.arm_set
    d_ = d1 * d2
    theta0 = Theta0.flatten("F")

    log_factor = np.log(4 * (d1 + d2) / delta)
    nu_factor = c_nu * np.sqrt(log_factor)

    mu_diags = np.diag([env.dmean_from_eta(tmp) for tmp in X_arms @ theta0])
    mu_diags = np.ascontiguousarray(mu_diags, dtype=np.float64)

    if GL_optimal:
        pi = cp.Variable(K, nonneg=True)
        mu_diags_cp = cp.Constant(mu_diags)
        H_pi = X_arms.T @ cp.diag(pi) @ mu_diags_cp @ X_arms
        H_inv = cp.Variable((d_, d_), PSD=True)

        D_col = cp.Constant(np.zeros((d2, d2)))
        for m in range(d1):
            idx_set = [m * d1 + i for i in range(d1)]
            D_col = D_col + H_inv[np.ix_(idx_set, idx_set)]

        D_row = cp.Constant(np.zeros((d1, d1)))
        for m in range(d2):
            idx_set = [m + l * d1 for l in range(d2)]
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
        D_col = np.zeros((d2, d2))
        for m in range(d1):
            idx_set = [m * d1 + i for i in range(d1)]
            D_col = D_col + H_inv[np.ix_(idx_set, idx_set)]
        D_row = np.zeros((d1, d1))
        for m in range(d2):
            idx_set = [m + l * d1 for l in range(d2)]
            D_row = D_row + H_inv[np.ix_(idx_set, idx_set)]
        design_value = max(np.linalg.eigvals(D_col).real.max(), np.linalg.eigvals(D_row).real.max())

    H_inv_optimal = np.linalg.inv(X_arms.T @ np.diag(pi_optimal) @ mu_diags @ X_arms)

    X2, y2 = np.zeros((N2, d_)), np.zeros(N2)
    idx_2 = np.random.choice(K, N2, p=pi_optimal)
    for i, idx in enumerate(idx_2):
        arm = arm_set[idx]
        X2[i] = arm.flatten("F")
        y2[i] = env.get_reward(arm)

    nu = nu_factor / np.sqrt(design_value * N2)
    Theta_Catoni = np.zeros((d1, d2))
    for t, y in enumerate(y2):
        x = X2[t].reshape((d_, 1))
        vector_one_sample = (y - env.mean_from_eta(np.dot(theta0, x))) * H_inv_optimal @ x
        matrix_one_sample = np.reshape(vector_one_sample, (d1, d2), "F")
        Theta_Catoni += psi_nu(matrix_one_sample, nu)

    Theta_Catoni /= N2
    Theta_Catoni += Theta0

    U, S, Vt = np.linalg.svd(Theta_Catoni)
    tau = np.sqrt(16 * design_value * np.log(4 * (d1 + d2) / delta) / (c_nu * N2))
    S_truncated = S * (S > tau).astype(int)
    return U @ np.diag(S_truncated) @ Vt

