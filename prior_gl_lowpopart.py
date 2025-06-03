from utils import *

class OneBitCompletion:
    def __init__(self, arm_set, Theta_star):
        self.arm_set = arm_set
        self.Theta_star = Theta_star
        self.d1 = Theta_star.shape[0]
        self.d2 = Theta_star.shape[1]

        self.K = len(arm_set)
        self.X_arms = np.ascontiguousarray(np.concatenate([arm.flatten('F').reshape(1,-1) for arm in arm_set], axis=0), dtype=np.float64)
    
    def get_reward(self, arm):
        """
        Simulate the reward for a given arm.
        The reward is generated based on the inner product of the arm and the true parameter matrix.
        """
        return np.random.binomial(1, sigmoid(np.vdot(arm, self.Theta_star)))




def nuc_norm_MLE(env, N1, d1, d2, nuc_coef):
    """
    Nuclear norm MLE for the GL-LowPopArt algorithm, with E-optimal design
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
    theta = cp.vec(Theta)

    log_likelihood = cp.sum(cp.multiply(y1, X1 @ theta) - cp.logistic(X1 @ theta)) / N1
    objective = cp.Maximize(log_likelihood - nuc_coef * cp.normNuc(Theta))
    prob = cp.Problem(objective)
    try:
        prob.solve(solver=cp.MOSEK)
    except:
        print("Solver status for MLE:", prob.status)
        prob.solve(solver=cp.MOSEK, verbose=True)

    Theta0 = np.array(Theta.value)

    return Theta0, X1, y1


def GL_LowPopArt(env, N2, d1, d2, delta, Theta0, c_nu=1):
    X_arms = env.X_arms
    K = env.K
    arm_set = env.arm_set
    
    theta0 = Theta0.flatten('F')

    mu_diags = np.diag([dsigmoid(tmp) for tmp in X_arms @ theta0])
    mu_diags = np.ascontiguousarray(mu_diags, dtype=np.float64)

    # Stage II. Catoni Style
    ## Experimental Design
    pi = cp.Variable(K, nonneg=True)
    d_ = d1 * d2

    mu_diags_cp = cp.Constant(mu_diags)
    H_pi = X_arms.T @ cp.diag(pi) @ mu_diags_cp @ X_arms
    H_inv = cp.Variable((d_, d_), PSD=True)  # for Schur complement & epigraph formulation
    
    # objective function
    D_col = cp.Constant(np.zeros((d2, d2)))
    for m in range(d1):
        idx_set = [m*d1 + i for i in range(d1)]
        D_col = D_col + H_inv[np.ix_(idx_set, idx_set)]

    D_row = cp.Constant(np.zeros((d1, d1)))
    for m in range(d2):
        idx_set = [m + l*d1 for l in range(d2)]
        D_row = D_row + H_inv[np.ix_(idx_set, idx_set)]
    
    objective = cp.Minimize(cp.maximum(cp.lambda_max(D_col), cp.lambda_max(D_row)))
    prob = cp.Problem(objective, [cp.sum(pi) == 1, cp.bmat([[H_pi, np.eye(d_)], [np.eye(d_), H_inv]]) >> 0])
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
    H_inv_optimal = np.linalg.inv(X_arms.T @ np.diag(pi_optimal) @ mu_diags @ X_arms)

    ## Sample from pi_optimal
    X2, y2 = np.zeros((N2, d_)), np.zeros(N2)
    idx_2 = np.random.choice(K, N2, p=pi_optimal)
    for i, idx in enumerate(idx_2):
        arm = arm_set[idx]
        X2[i] = arm.flatten('F')
        y2[i] = env.get_reward(arm)
    
    ## matrix Catoni
    # print("Design value:", design_value)
    nu = c_nu * 2 * np.sqrt(np.log(4*(d1 + d2) / delta) / (design_value * N2))

    Theta_Catoni = np.zeros((d1, d2))
    for t, y in enumerate(y2):
        x = X2[t].reshape((d_, 1))

        # one-sample estimator
        # vector_one_sample = (y - sigmoid(np.sum(theta0 * x))) * H_inv_optimal @ x
        vector_one_sample = (y - sigmoid(np.dot(theta0, x))) * H_inv_optimal @ x
        matrix_one_sample = np.reshape(vector_one_sample, (d1, d2), 'F')

        # matrix Catoni estimator
        Theta_Catoni += psi_nu(matrix_one_sample, nu)

    Theta_Catoni /= N2
    Theta_Catoni += Theta0

    # Truncation
    U, S, Vt = np.linalg.svd(Theta_Catoni)
    tau = np.sqrt(16 * design_value * np.log(4*(d1 + d2) / delta) / (c_nu * N2))
    S_truncated = S * (S > tau).astype(int)
    # print(S_truncated)
    return U @ np.diag(S_truncated) @ Vt