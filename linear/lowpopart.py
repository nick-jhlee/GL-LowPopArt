from utils import *
import matplotlib.pyplot as plt

class LinearCompletion:
    def __init__(self, arm_set, Theta_star, sigma=1):
        self.arm_set = arm_set
        self.Theta_star = Theta_star
        self.d1 = Theta_star.shape[0]
        self.d2 = Theta_star.shape[1]
        self.sigma = sigma

        self.K = len(arm_set)
        self.X_arms = np.ascontiguousarray(np.concatenate([arm.flatten('F').reshape(1,-1) for arm in arm_set], axis=0), dtype=np.float64)
    
    def get_reward(self, arm):
        """
        Simulate the reward for a given arm.
        The reward is generated based on the inner product of the arm and the true parameter matrix.
        """
        eps = np.random.randn() * self.sigma
        return np.vdot(arm, self.Theta_star) + eps



def nuc_norm_MLE_linear(env, N1, d1, d2, nuc_coef):
    """
    Nuclear norm MLE for the linear matrix recovery, with E-optimal design
    """
    K = env.K
    arm_set = env.arm_set

    # Stage I. Nuclear norm MLE
    ## E-Optimal Design
    pi_E = E_optimal_design(env)

    ## Sample from pi_E
    X1, y1 = np.zeros((N1, d1*d2)), np.zeros(N1)
    idx_1 = np.random.choice(K, N1, p=pi_E)
    for i, idx in enumerate(idx_1):
        arm = arm_set[idx]
        X1[i] = arm.flatten('F')
        y1[i] = env.get_reward(arm)
    
    # Nuclear norm regularized MLE
    Theta = cp.Variable((d1, d2))
    theta = cp.vec(Theta, 'F')  # column major vectoriation

    l2_loss = cp.sum_squares(y1 - X1 @ theta) / N1
    objective = cp.Minimize(l2_loss + nuc_coef * cp.normNuc(Theta))
    prob = cp.Problem(objective)
    try:
        prob.solve(solver=cp.MOSEK)
    except:
        print("Solver status for MLE:", prob.status)
        prob.solve(solver=cp.MOSEK, verbose=True)

    Theta0 = np.array(Theta.value)

    return Theta0


def LowPopArt(env, N2, d1, d2, delta, Theta0):
    """
    Jang et al. (ICML 2024)
    """
    X_arms = env.X_arms
    K = env.K
    arm_set = env.arm_set
    
    theta0 = Theta0.flatten('F')
    # theta_star = env.Theta_star.flatten('F')

    # Stage II. Catoni Style
    ## Experimental Design
    pi = cp.Variable(K, nonneg=True)
    d_ = d1 * d2

    V_pi = X_arms.T @ cp.diag(pi) @ X_arms
    V_inv = cp.Variable((d_, d_), PSD=True)  # for Schur complement & epigraph formulation
    
    # objective function
    D_col = cp.Constant(np.zeros((d2, d2)))
    for m in range(d1):
        idx_set = [m*d1 + i for i in range(d1)]
        D_col = D_col + V_inv[np.ix_(idx_set, idx_set)]

    D_row = cp.Constant(np.zeros((d1, d1)))
    for m in range(d2):
        idx_set = [m + l*d1 for l in range(d2)]
        D_row = D_row + V_inv[np.ix_(idx_set, idx_set)]
    
    objective = cp.Minimize(cp.maximum(cp.lambda_max(D_col), cp.lambda_max(D_row)))
    prob = cp.Problem(objective, [cp.sum(pi) == 1, cp.bmat([[V_pi, np.eye(d_)], [np.eye(d_), V_inv]]) >> 0])
    try:
        prob.solve(solver=cp.MOSEK)
    except:
        print(f"Solver status for GL-LowPopArt:", prob.status)
        prob.solve(solver=cp.MOSEK, verbose=True)
        # if prob.status != cp.OPTIMAL:
        #     print(f"Solver status for {type}-design (d={d}):", prob.status)
            # return None
    # print(f"Solver status for {type}-design (d={d}):", prob.status)
    # print(f"Optimal value for {type}-design:", prob.value)
    # print(f"Solver tolerance for {type}-design:", prob.solver_stats.solve_time)
    pi_optimal = np.abs(np.array(pi.value))
    pi_optimal /= np.sum(pi_optimal)
    design_value = prob.value
    V_inv_optimal = np.linalg.inv(X_arms.T @ np.diag(pi_optimal) @ X_arms)

    ## Sample from pi_optimal
    X2, y2 = np.zeros((N2, d_)), np.zeros(N2)
    idx_2 = np.random.choice(K, N2, p=pi_optimal)
    for i, idx in enumerate(idx_2):
        arm = arm_set[idx]
        X2[i] = arm.flatten('F')
        y2[i] = env.get_reward(arm)
    
    ## matrix Catoni
    R0 = 2
    d = max(d1, d2)
    # print("Design value:", design_value)
    nu = (1 / (R0 + env.sigma)) * np.sqrt(2 * np.log(2*d / delta) / (design_value * N2))
    # print("nu:", nu)

    Theta_Catoni = np.zeros((d1, d2))
    for t, y in enumerate(y2):
        x = X2[t].reshape((d_, 1))

        # one-sample estimator
        vector_one_sample = (y - np.dot(theta0, x)) * V_inv_optimal @ x
        matrix_one_sample = np.reshape(vector_one_sample, (d1, d2), 'F')

        # matrix Catoni estimator
        Theta_Catoni += psi_nu(matrix_one_sample, nu)

    Theta_Catoni /= N2
    Theta_Catoni += Theta0

    # Truncation
    U, S, Vt = np.linalg.svd(Theta_Catoni)
    tau = 2 * (R0 + env.sigma) * np.sqrt(design_value * np.log(2*d / delta) / N2)
    S_truncated = S * (S > tau).astype(int)
    # print(S_truncated)
    return U @ np.diag(S_truncated) @ Vt
