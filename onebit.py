from gl_lowpopart import *
from tqdm import tqdm
from math import floor


def create_random_env(Theta_star, K):
    """
    Create a random environment for the OneBitCompletion problem.
    """
    arm_set = []
    for k in range(K):
        arm = np.random.randn(d, d)
        arm /= np.linalg.norm(arm)
        arm_set.append(arm)

    return OneBitCompletion(arm_set, Theta_star)

if __name__ == '__main__':
    ## Symmetric 5x5 matrix completion of rank at most 2
    ## The learner doesn't know that the matrix is symmetric
    d, r = 6, 3

    ## randomly sample from the unit sphere
    vecs = [np.random.randn(d) for _ in range(r)]
    # vecs_normalized = [vec / np.linalg.norm(vec) for vec in vecs]
    vecs_outers = [np.outer(vec, vec) for vec in vecs]
    Theta_star = np.sum(vecs_outers, axis=0)
    Theta_star = Theta_star / np.linalg.norm(Theta_star)
    print("rank of Theta_star: ", np.linalg.matrix_rank(Theta_star))

    U_star, S_star, V_star = np.linalg.svd(Theta_star)
    print("singular values of Theta_star: ", S_star)
    # print("nuclear norm of Theta_star: ", np.linalg.norm(Theta_star, ord='nuc'))
    
    # Theta_star = np.eye(d)
    # Theta_star[0, 0] = 0
    # Theta_star[1, 1] = 0
    # Theta_star[2, 2] = 0

    ## Matrix completion basis
    # arm_set = []
    # for i in range(d):
    #     basis_i = np.zeros(d)
    #     basis_i[i] = 1
    #     for j in range(d):
    #         basis_j = np.zeros(d)
    #         basis_j[j] = 1
    #         arm_set.append(np.outer(basis_i, basis_j))
    # ## randomly sample from Frobenius norm ball
    # K = 100
    # env = create_random_env(Theta_star, K)

    ## Hard instance
    I_d = np.eye(d * d)
    arm_set = [I_d[0].reshape(d, d) / np.sqrt(d)]
    for i in range(1, d**2):
        arm = I_d[0] + I_d[i]/np.sqrt(d)
        arm_set.append(arm.reshape(d, d))
    env = OneBitCompletion(arm_set, Theta_star)

    num_repeats = 5
    delta = 0.001
    # Ns = [100, 1000, 10000, 100000]
    Ns = [1000, 5000, 10000, 50000, 100000, 500000]
    c_lambda = 0.001
    c_nu = 1e2 # scaling for nu in Stage II Catoni
    
    errors0_all = []
    errors_all = []
    for N in tqdm(Ns):
        errors0_reps = []
        errors_reps = []
        for _ in range(num_repeats):
            # Stage I. Nuclear norm regularized MLE
            nuc_coef = np.sqrt(np.log((d + d) / delta) / N) * c_lambda
            Theta0, _, _ = nuc_norm_MLE(env, N, d, d, nuc_coef)
            errors0_reps.append(np.linalg.norm(Theta0 - Theta_star))

            # Stage II. matrix Catoni
            # for fair comparison, we use N/2 samples for Stage I and II
            N1 = 2 * floor(np.sqrt(N))
            N2 = N - N1
            nuc_coef = np.sqrt(np.log((d + d) / delta) / N1) * c_lambda
            Theta0, X1, y1 = nuc_norm_MLE(env, N1, d, d, nuc_coef)
            Theta = GL_LowPopArt(env, N2, d, d, delta, Theta0, X1, y1, c_nu)
            errors_reps.append(np.linalg.norm(Theta - Theta_star))
        errors0_all.append(errors0_reps)
        errors_all.append(errors_reps)

    # Compute mean and standard deviation
    errors0_mean = np.mean(errors0_all, axis=1)
    errors0_std = np.std(errors0_all, axis=1)
    errors_mean = np.mean(errors_all, axis=1)
    errors_std = np.std(errors_all, axis=1)

    # Plot with error bars
    plt.figure(1)
    plt.errorbar(Ns, errors0_mean, yerr=errors0_std, fmt='o-', label='Stage I', color='blue', capsize=5)
    plt.errorbar(Ns, errors_mean, yerr=errors_std, fmt='o-', label='Stage I + II', color='orange', capsize=5)
    # plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel('N')
    plt.ylabel('Spectral norm error')
    plt.legend()
    plt.title('Error vs. Sample Size with Error Bars')
    plt.show()