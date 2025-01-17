import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def antisymmetric_projection(d):
    I_d = np.eye(d)
    P = (1 / np.sqrt(2)) * np.vstack([np.kron(I_d[i], I_d[j]) - np.kron(I_d[j], I_d[i]) for i in range(d) for j in range(i + 1, d)])
    return P.T

def optimal_design(phis, type='S'):
    d = phis[0].shape[0]
    K = len(phis)
    P = antisymmetric_projection(d)

    S_design = type == 'S'

    # cvxpy Variables
    if S_design:
        d_ = d*(d-1)//2
        pi = cp.Variable(K*(K-1)//2, nonneg=True)
    else:
        d_ = d**2
        pi = cp.Variable(K**2, nonneg=True)
    Y = cp.Variable((d_, d_), symmetric=True)  # for Schur complement

    # stack Kroneckers
    if S_design:
        X = np.vstack([np.kron(phis[j], phis[i]) for i in range(K) for j in range(i+1,K)])
    else:
        X = np.vstack([np.kron(phis[j], phis[i]) for i in range(K) for j in range(K)])

    V_pi = X.T @ cp.diag(pi) @ X
    if S_design:
        V_pi = P.T @ V_pi @ P
        V_inv = P @ Y @ P.T
    else:
        V_inv = Y


    # objective function
    D_col = cp.Constant(np.zeros((d, d)))
    for m in range(d):
        idx_set = [m*d + i for i in range(d)]
        D_col = D_col + V_inv[np.ix_(idx_set, idx_set)]

    if S_design:
        objective = cp.Minimize(cp.lambda_max(D_col))
    else:
        D_row = cp.Constant(np.zeros((d, d)))
        for m in range(d):
            idx_set = [m + i*d for i in range(d)]
            D_row = D_row + V_inv[np.ix_(idx_set, idx_set)]
        objective = cp.Minimize(cp.maximum(cp.lambda_max(D_col), cp.lambda_max(D_row)))

    prob = cp.Problem(objective, [cp.sum(pi) == 1, cp.bmat([[V_pi, np.eye(d_)], [np.eye(d_), Y]]) >> 0])
    try:
        prob.solve(solver=cp.MOSEK)
    except:
        print(f"Solver status for {type}-design (d={d}):", prob.status)
        prob.solve(solver=cp.MOSEK, verbose=True)
        # if prob.status != cp.OPTIMAL:
        #     print(f"Solver status for {type}-design (d={d}):", prob.status)
            # return None
    # print(f"Solver status for {type}-design (d={d}):", prob.status)
    # print(f"Optimal value for {type}-design:", prob.value)
    # print(f"Solver tolerance for {type}-design:", prob.solver_stats.solve_time)
    pi_optimal = pi.value
    pi_optimal /= np.sum(pi_optimal)

    V_pi = X.T @ np.diag(pi_optimal) @ X
    if S_design:
        V_inv = P @ np.linalg.inv(P.T @ V_pi @ P) @ P.T
    else:
        V_inv = np.linalg.inv(V_pi)

    # objective function
    D_col = np.zeros((d, d))
    for m in range(d):
        idx_set = [m * d + i for i in range(d)]
        D_col = D_col + V_inv[np.ix_(idx_set, idx_set)]
    eig_max1 = np.max(np.linalg.eigvals(D_col))

    if S_design:
        return pi_optimal, eig_max1
    else:
        D_row = np.zeros((d, d))
        for m in range(d):
            idx_set = [m + i * d for i in range(d)]
            D_row = D_row + V_inv[np.ix_(idx_set, idx_set)]
        eig_max2 = np.max(np.linalg.eigvals(D_row))
        # print(eig_max1, eig_max2)
        return pi_optimal, max(eig_max1, eig_max2)


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

def LowPopArt(design_value, V_pi_inv, pilot_estimator, arms_left, arms_right, rewards, delta=0.05):
    d = arms_left[0].shape[0]
    N = len(rewards)
    mu = lambda z : (1 + z) / 2

    c = 1 + 2 + 1
    nu = np.sqrt(np.log(4*d / delta) / (c * N * design_value))

    Theta_Catoni = np.zeros((d, d))
    for t, reward in enumerate(rewards):
        arm_left, arm_right = arms_left[t], arms_right[t]

        # one-sample estimator
        vector_one_sample = (reward - mu(arm_left.T @ pilot_estimator @ arm_right)) * V_pi_inv @ np.kron(arms_right[t], arms_left[t])
        matrix_one_sample = np.reshape(vector_one_sample, (d, d))

        # matrix Catoni estimator
        Theta_Catoni += psi_nu(matrix_one_sample, nu)

    Theta_Catoni /= N
    Theta_Catoni += pilot_estimator

    Theta_Catoni = 0.5 * (Theta_Catoni - Theta_Catoni.T)

    # Truncation
    U, S, Vt = np.linalg.svd(Theta_Catoni)
    S_truncated = S
    S_truncated[2:] = 0
    # tau = np.sqrt(design_value * np.log(4*d / delta) / (c * N))
    # S_truncated = np.maximum(S - tau, 0)
    return U @ np.diag(S_truncated) @ Vt


if __name__ == '__main__':
    # ## Trace Regression
    # d = 4
    # P = antisymmetric_projection(d)
    # I_d = np.eye(d)
    # normalize = lambda x: x / np.sqrt(np.sum(x ** 2))
    # mu = lambda z: (1 + z) / 2
    #
    # # create arms
    # phis = [np.array([int(x) for x in f"{i:0{d}b}"]) for i in range(2 ** d)]
    # phis = [normalize(phi) for phi in phis if np.sum(phi) > 0]
    # # phis = [I_d[i] for i in range(d)]
    # # phis = [I_d[0], normalize(I_d[0] + (1 / np.sqrt(d)) * I_d[1])] + [I_d[i] for i in range(2, d)]
    # # phis = [I_d[0], normalize(I_d[0] + I_d[1])] + [I_d[i] for i in range(2, d)]
    # K = len(phis)
    #
    # ## Unknown, low-rank skew-symmetric matrix
    # r = 2
    # Theta_star = np.random.randn(d, d)
    # Theta_star = Theta_star - Theta_star.T
    # u, s, vt = np.linalg.svd(Theta_star)
    # s[r:] = 0
    # Theta_star = u @ np.diag(s) @ vt
    # Theta_star /= np.linalg.norm(Theta_star, 2)
    # print("Rank of the unknown matrix:", np.linalg.matrix_rank(Theta_star))
    # # print(min([np.abs(phis[i].T @ Theta_star @ phis[j]) for i in range(K) for j in range(i+1, K)]))
    #
    # arm_indices_S = [(i, j) for i in range(K) for j in range(i + 1, K)]
    # pi_star_S, S_star = optimal_design(phis, 'S')
    # X_S = np.vstack([np.kron(phis[j], phis[i]) for i in range(K) for j in range(i+1, K)])
    # V_pi_inv_S = P @ np.linalg.inv(P.T @ X_S.T @ np.diag(pi_star_S) @ X_S @ P) @ P.T
    #
    # arm_indices_B = [(i, j) for i in range(K) for j in range(K)]
    # pi_star_B, B_star = optimal_design(phis, 'B')
    # X_B = np.vstack([np.kron(phis[j], phis[i]) for i in range(K) for j in range(K)])
    # V_pi_inv_B = np.linalg.inv(X_B.T @ np.diag(pi_star_B) @ X_B)
    #
    # # print(arm_indices_S)
    # # print("pi_star_S:", pi_star_S)
    # # print([mu(phis[i].T @ Theta_star @ phis[j]) for i in range(K) for j in range(i+1, K)])
    # # print(arm_indices_B)
    # # print("pi_star_B:", pi_star_B)
    # # print([mu(phis[i].T @ Theta_star @ phis[j]) for i in range(K) for j in range(K)])
    # # # print(pi_star_B)
    # # print([pi_star_B[i + K*j] + pi_star_B[K*i + j] for i in range(K) for j in range(i+1, K)])
    # # print([pi_star_B[K*i] for i in range(K)])
    #
    # errors_S, errors_B = [], []
    # T_list = [int(1e3), int(2e3), int(3e3), int(4e3), int(5e3), int(1e4), int(2e4), int(3e4)]
    # for T in tqdm(T_list):
    #     ## S-optimal design
    #     pilot_estimator = np.zeros((d, d))
    #     arms_left, arms_right, rewards = [], [], []
    #     for t in range(T):
    #         # sample (i, j) according to pi_star
    #         pair_idx = np.random.choice(range(K*(K-1)//2), size=1, p=pi_star_S)
    #         i, j = arm_indices_S[pair_idx[0]]
    #         # sample Bernoulli reward
    #         reward = np.random.binomial(1, mu(phis[i].T @ Theta_star @ phis[j]))
    #         arms_left.append(phis[i])
    #         arms_right.append(phis[j])
    #         rewards.append(reward)
    #
    #     Theta_hat_S = LowPopArt(S_star, V_pi_inv_S, pilot_estimator, arms_left, arms_right, rewards)
    #     errors_S.append(np.linalg.norm(Theta_hat_S - Theta_star, 2))
    #
    #
    #     ## B-optimal design
    #     pilot_estimator = np.zeros((d, d))
    #     arms_left, arms_right, rewards = [], [], []
    #     for t in range(T):
    #         # sample (i, j) according to pi_star
    #         pair_idx = np.random.choice(range(K ** 2), size=1, p=pi_star_B)
    #         i, j = arm_indices_B[pair_idx[0]]
    #         # sample Bernoulli reward
    #         reward = np.random.binomial(1, mu(phis[i].T @ Theta_star @ phis[j]))
    #         arms_left.append(phis[i])
    #         arms_right.append(phis[j])
    #         rewards.append(reward)
    #
    #     Theta_hat_B = LowPopArt(B_star, V_pi_inv_B, pilot_estimator, arms_left, arms_right, rewards)
    #     errors_B.append(np.linalg.norm(Theta_hat_B - Theta_star, 2))
    #
    #
    # plt.figure(1)
    # plt.plot(T_list, errors_S, marker='o', linestyle='--', color='r')
    # plt.plot(T_list, errors_B, marker='o', linestyle='--', color='b')
    # plt.legend(['S-optimal', 'B-optimal'])
    # plt.xlabel('Number of samples')
    # plt.ylabel('Error in Operator Norm')
    # plt.show()



    # d_list = np.array([4, 6, 8, 10])
    # d_list = [3,4,5,6]
    d_list = [3, 4, 5, 6, 7, 8, 9, 10, 11]
    normalize = lambda x: x / np.sqrt(np.sum(x ** 2))

    ## Hard instance #1.
    B_min_list_1 = []
    S_min_list_1 = []
    for d in tqdm(d_list):
        I_d = np.eye(d)
        # d_ = d // 2
        phis = [I_d[0], normalize(I_d[0] + (1 / np.sqrt(d))*I_d[1])] + [I_d[i] for i in range(2, d)]
        # binary codes
        # phis = [np.array([int(x) for x in f"{i:0{d}b}"]) for i in range(2**d)]
        # phis = [normalize(phi) for phi in phis if np.sum(phi) > 0]
        # alpha = 2  # Scaling factor
        # phis = [normalize(I_d[i] + alpha * I_d[(i + 1) % d]) for i in range(d)]
        # phis = [normalize(I_d[2*i-1] + I_d[2*i]) for i in range(d_)] + [I_d[2*i-1] for i in range(d_)]
        # phis = [normalize(np.ones(d) - I_d[i]) for i in range(d)]
        # phis = [I_d[0]] + [I_d[0] + (1/np.sqrt(d))*I_d[i] for i in range(1, d)]
        # phis = [I_d[2*i-1] for i in range(d_)] + [I_d[2*i-1] + 1 / np.sqrt(d) * I_d[2*i] for i in range(d_)]
        # phis = [I_d[2*i-1] for i in range(d_)] + [normalize(I_d[2*i-1] + 1 / np.sqrt(d) * I_d[2*i]) for i in range(d_)]
        # rank_X, rank_P = 0, 0
        # while rank_X != d**2 and rank_P != d*(d-1)//2:
        #     phis = [normalize(np.random.randn(d)) for _ in range(d)]
        #     X = np.vstack([np.kron(phis[j], phis[i]) for i in range(d) for j in range(d)])
        #     rank_X = np.linalg.matrix_rank(X)
        #     P = antisymmetric_projection(d)
        #     rank_P = np.linalg.matrix_rank(P @ P.T @ X)
        #     print(rank_X, rank_P)

        _, S_min = optimal_design(phis, 'S')
        _, B_min = optimal_design(phis, 'B')
        B_min_list_1.append(B_min)
        S_min_list_1.append(S_min)
    # d = 10
    # I_d = np.eye(d)
    # d_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # for alpha in tqdm(d_list):
    #     phis = [I_d[0]] + [normalize(alpha * I_d[0] + I_d[i]) for i in range(1, d)]
    #     # phis = [normalize(np.ones(d) - I_d[i]) for i in range(d)]
    #     # phis = [I_d[0]] + [I_d[0] + I_d[i] for i in range(1, d)]
    #     # phis = [I_d[2*i-1] for i in range(d_)] + [I_d[2*i-1] + 1 / np.sqrt(d) * I_d[2*i] for i in range(d_)]
    #     # phis = [I_d[2*i-1] for i in range(d_)] + [normalize(I_d[2*i-1] + 1 / np.sqrt(d) * I_d[2*i]) for i in range(d_)]
    #     # rank_X, rank_P = 0, 0
    #     # while rank_X != d**2 and rank_P != d*(d-1)//2:
    #     #     phis = [normalize(np.random.randn(d)) for _ in range(d)]
    #     #     X = np.vstack([np.kron(phis[j], phis[i]) for i in range(d) for j in range(d)])
    #     #     rank_X = np.linalg.matrix_rank(X)
    #     #     P = antisymmetric_projection(d)
    #     #     rank_P = np.linalg.matrix_rank(P @ P.T @ X)
    #     #     print(rank_X, rank_P)
    #
    #     S_min = optimal_design(phis, 'S')
    #     B_min = optimal_design(phis, 'B')
    #     B_min_list_1.append(B_min)
    #     S_min_list_1.append(S_min)
    # plt.figure(1)
    # plt.plot(d_list, np.log(B_min_list_1), marker='o', linestyle='--', color='r')
    # plt.plot(d_list, np.log(S_min_list_1), marker='o', linestyle='--', color='b')
    # plt.legend(['B_min', 'S_min'])
    # plt.xlabel('log d')
    # plt.ylabel('log of Optimal values')
    # plt.title('Hard instance #1')
    #
    # # find the slope and intercept of loglog-plot
    # slope_1, intercept_1 = np.polyfit(d_list, np.log(B_min_list_1), 1)
    # slope_2, intercept_2 = np.polyfit(d_list, np.log(S_min_list_1), 1)
    # print("Slopes:", slope_1, slope_2)
    # print("Intercepts:", intercept_1, intercept_2)

    plt.figure(1)
    plt.plot(d_list, B_min_list_1, marker='o', linestyle='--', color='r')
    plt.plot(d_list, S_min_list_1, marker='o', linestyle='--', color='b')
    plt.legend(['B_min', 'S_min'])
    plt.xlabel('d')
    plt.ylabel('Optimal values')
    plt.title('Hard instance #1')

    plt.figure(2)
    plt.loglog(d_list, B_min_list_1, marker='o', linestyle='--', color='r')
    plt.loglog(d_list, S_min_list_1, marker='o', linestyle='--', color='b')
    plt.legend(['B_min', 'S_min'])
    plt.xlabel('d')
    plt.ylabel('Optimal values')
    plt.title('Hard instance #1')

    plt.figure(3)
    plt.plot(d_list, np.log(B_min_list_1), marker='o', linestyle='--', color='r')
    plt.plot(d_list, np.log(S_min_list_1), marker='o', linestyle='--', color='b')
    plt.legend(['B_min', 'S_min'])
    plt.xlabel('d')
    plt.ylabel('Log of Optimal values')
    plt.title('Hard instance #1')

    # plot the ratio of B_min and S_min
    plt.figure(4)
    plt.plot(d_list, np.array(B_min_list_1) - np.array(S_min_list_1), marker='o', linestyle='--', color='g')
    plt.xlabel('d')
    plt.ylabel('B_min - S_min')
    plt.title('Hard instance #1')

    coefs = np.polyfit(d_list, np.array(B_min_list_1) - np.array(S_min_list_1), 2)
    print(f"B_min - S_min = {coefs[0]}d^2 + {coefs[1]}d + {coefs[2]}")
    # # find the slope and intercept of loglog-plot
    # slope_1, intercept_1 = np.polyfit(np.log(np.pow(d_list, 2)), np.log(B_min_list_1), 1)
    # slope_2, intercept_2 = np.polyfit(np.log((np.pow(d_list, 2) - d_list)/2), np.log(S_min_list_1), 1)
    # print("Slopes:", slope_1, slope_2)
    # print("Intercepts:", intercept_1, intercept_2)

    # ## Hard instance #2.
    # B_min_list_2 = []
    # S_min_list_2 = []
    # for d in d_list:
    #     I_d = np.eye(d)
    #     phis = [1 / np.sqrt(d) * I_d[0]] + [I_d[i] for i in range(1, d)]
    #     B_min = optimal_design(phis, 'B')
    #     S_min = optimal_design(phis, 'S')
    #     B_min_list_2.append(B_min)
    #     S_min_list_2.append(S_min)
    #
    #
    # plt.figure(2)
    # plt.loglog(d_list, B_min_list_2, marker='o', linestyle='--', color='r')
    # plt.loglog(d_list, S_min_list_2, marker='o', linestyle='--', color='b')
    # plt.legend(['B_min', 'S_min'])
    # plt.xlabel('log d')
    # plt.ylabel('log of Optimal values')
    # plt.title('Hard instance #2')
    #
    # # find the slope of loglog-plot
    # slope_1 = np.polyfit(np.log(d_list), np.log(B_min_list_2), 1)[0]
    # slope_2 = np.polyfit(np.log(d_list), np.log(S_min_list_2), 1)[0]
    # print("Slopes:", slope_1, slope_2)
    #
    # # save datas
    # np.savez('gls2-lowpopart.npz', d_list=d_list, B_min_list_1=B_min_list_1, S_min_list_1=S_min_list_1,
    #          B_min_list_2=B_min_list_2, S_min_list_2=S_min_list_2)

    plt.show()