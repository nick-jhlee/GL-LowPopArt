from gl_lowpopart import *
from burer_monteiro import *
from tqdm import tqdm
from math import floor


def create_random_env(Theta_star, K, d1, d2):
    """
    Create a random environment for the OneBitCompletion problem.
    """
    arm_set = []
    for _ in range(K):
        arm = np.random.randn(d1, d2)
        arm /= np.linalg.norm(arm)
        arm_set.append(arm)

    return OneBitCompletion(arm_set, Theta_star)

if __name__ == '__main__':
    ## 3x3 matrix completion of rank 1
    d1, d2, r = 3, 3, 1

    def generate_Theta_star():
        ## randomly sample from the unit sphere
        vecs_1 = [np.random.randn(d1) for _ in range(r)]
        vecs_1 = [vec / np.linalg.norm(vec) for vec in vecs_1]
        vecs_2 = [np.random.randn(d2) for _ in range(r)]
        vecs_2 = [vec / np.linalg.norm(vec) for vec in vecs_2]
        # vecs_normalized = [vec / np.linalg.norm(vec) for vec in vecs]
        # vecs_outers = [np.outer(vec1, vec2) for (vec1, vec2) in zip(vecs_1, vecs_2)]
        vecs_outers = [np.outer(vec1, vec1) for vec1 in vecs_1]
        return np.sum(vecs_outers, axis=0)
        # # Theta_star = Theta_star / np.linalg.norm(Theta_star)
        # print("rank of Theta_star: ", np.linalg.matrix_rank(Theta_star))

        # U_star, S_star, V_star = np.linalg.svd(Theta_star)
        # print("singular values of Theta_star: ", S_star)
    
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

    # ## Hard instance
    # I_d = np.eye(d * d)
    # arm_set = [I_d[0].reshape(d, d) / np.sqrt(d)]
    # for i in range(1, d**2):
    #     arm = I_d[0] + I_d[i]/np.sqrt(d)
    #     arm_set.append(arm.reshape(d, d))

    ## Randomly sample from Frobenius norm ball
    K = 150
    arm_set = []
    for i in range(K):
        arm = np.random.randn(d1, d2)
        arm /= np.linalg.norm(arm)
        arm_set.append(arm)

    num_repeats = 10
    delta = 0.001
    # Ns = [1000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
    Ns = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
    c_lambda = 1    # scaling for lambda in Stage I nuc-regularized MLE
    c_nu = 1       # scaling for nu in Stage II Catoni
    Rmax = 1/4

    ## Experiment 1. MLE vs BMF vs MLE + Catoni
    errors0_all, errors_all, errors_BMF_all = [], [], []
    for N in tqdm(Ns):
        errors0_reps = []
        errors_reps = []
        errors_bmf_reps = []
        for _ in range(num_repeats):
            # random problem instance
            Theta_star = generate_Theta_star()
            env = OneBitCompletion(arm_set, Theta_star)
            kappa_star = np.min([dsigmoid(tmp) for tmp in env.X_arms @ Theta_star.flatten('F')])
            # print("kappa_star: ", kappa_star)
            # true_dmus = mu_diags = np.diag([dsigmoid(tmp) for tmp in env.X_arms @ Theta_star.flatten('F')])
            # true_Hessian = env.X_arms.T @ true_dmus @ env.X_arms
            # print("spectrum of Hessian: ", np.linalg.svd(true_Hessian, compute_uv=False))


            ## Algorithm 1. Nuclear norm regularized MLE
            nuc_coef = c_lambda * np.sqrt(8 * Rmax * np.log((d1 + d2) / delta) / N)
            Theta0, X1, y1 = nuc_norm_MLE(env, N, d1, d2, nuc_coef)
            errors0_reps.append(np.linalg.norm(Theta0 - Theta_star, 'nuc'))

            ## Algorithm 2. Burer-Monteiro Factorization
            X_bmf, y_bmf = X1, y1
            # X_bmf, y_bmf = np.zeros((N, d1*d2)), np.zeros(N)
            # idx_bmf = np.random.choice(K, N)
            # for i, idx in enumerate(idx_bmf):
            #     arm = arm_set[idx]
            #     X_bmf[i] = arm.flatten('F')
            #     y_bmf[i] = env.get_reward(arm)
            Theta_BMF = Burer_Monteiro(d1, r, X_bmf, y_bmf)
            errors_bmf_reps.append(np.linalg.norm(Theta_BMF - Theta_star, 'nuc'))

            ## Algorithm 3. GL-LowPopArt
            ## for fair comparison, we use total of N samples for both stages combined
            N1 = 2 * floor(np.sqrt(N))
            N2 = N - N1
            ### Stage I. Nuclear norm MLE
            nuc_coef = c_lambda * np.sqrt(8 * Rmax * np.log((d1 + d2) / delta) / N1)
            Theta0, _, _ = nuc_norm_MLE(env, N1, d1, d2, nuc_coef)
            ### Stage II. matrix Catoni
            Theta = GL_LowPopArt(env, N2, d1, d2, delta, Theta0, c_nu)
            errors_reps.append(np.linalg.norm(Theta - Theta_star, 'nuc'))
        
        errors0_all.append(errors0_reps)
        errors_all.append(errors_reps)
        errors_BMF_all.append(errors_bmf_reps)

    # Compute mean and standard deviation
    errors0_mean = np.mean(errors0_all, axis=1)
    errors0_std = np.std(errors0_all, axis=1)
    errors_mean = np.mean(errors_all, axis=1)
    errors_std = np.std(errors_all, axis=1)
    errors_bmf_mean = np.mean(errors_BMF_all, axis=1)
    errors_bmf_std = np.std(errors_BMF_all, axis=1)
    # Save results
    np.savez('Fig1.npz', errors0_mean=errors0_mean, errors0_std=errors0_std, errors_mean=errors_mean, errors_std=errors_std, errors_bmf_mean=errors_bmf_mean, errors_bmf_std=errors_bmf_std)

    # Plot with error bars
    plt.figure(1)
    plt.errorbar(Ns, errors0_mean, yerr=errors0_std, fmt='o-', label='Stage I', color='blue', capsize=5)
    plt.errorbar(Ns, errors_mean, yerr=errors_std, fmt='o-', label='Stage I + II', color='orange', capsize=5)
    plt.errorbar(Ns, errors_bmf_mean, yerr=errors_bmf_std, fmt='o-', label='BMF', color='green', capsize=5)
    print("Stage I mean error: ", errors0_mean)
    print("Stage I + II mean error: ", errors_mean)
    print("BMF mean error: ", errors_bmf_mean)
    # plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel('N')
    plt.ylabel('Nuclear norm error')
    plt.legend()
    plt.title('1-Bit Recovery of Symmetric Rank-1 Matrix')
    plt.savefig('Fig1.png', dpi=300)
    plt.show()


    ## Experiment 2. Ablation on the importance of initialization for matrix Catoni
    errors1_all, errors2_all, errors3_all = [], [], []
    # Fix an instance
    Theta_star = 5 * generate_Theta_star()
    env = OneBitCompletion(arm_set, Theta_star)
    kappa_star = np.min([dsigmoid(tmp) for tmp in env.X_arms @ Theta_star.flatten('F')])
    Theta_0_zero = np.zeros((d1, d2))
    Theta_0_random = np.random.randn(d1, d2)
    Theta_0_random /= np.linalg.norm(Theta_0_random)
    N_MLE = int(3e4)
    nuc_coef = c_lambda * np.sqrt(8 * Rmax * np.log((d1 + d2) / delta) / N_MLE)
    Theta_0_MLE, _, _ = nuc_norm_MLE(env, N_MLE, d1, d2, nuc_coef)
    Ns = [1000, 10000, 20000, 30000, 40000, 50000]
    for N in tqdm(Ns):
        errors1_reps = []
        errors2_reps = []
        errors3_reps = []
        for _ in range(num_repeats):
            Theta = GL_LowPopArt(env, N, d1, d2, delta, Theta_0_zero, c_nu)
            errors1_reps.append(np.linalg.norm(Theta - Theta_star, 'nuc'))

            Theta = GL_LowPopArt(env, N, d1, d2, delta, Theta_0_random, c_nu)
            errors2_reps.append(np.linalg.norm(Theta - Theta_star, 'nuc'))
            
            Theta = GL_LowPopArt(env, N, d1, d2, delta, Theta_0_MLE, c_nu)
            errors3_reps.append(np.linalg.norm(Theta - Theta_star, 'nuc'))
        
        errors1_all.append(errors1_reps)
        errors2_all.append(errors2_reps)
        errors3_all.append(errors3_reps)

    # Compute mean and standard deviation
    errors1_mean = np.mean(errors1_all, axis=1)
    errors1_std = np.std(errors1_all, axis=1)
    errors2_mean = np.mean(errors2_all, axis=1)
    errors2_std = np.std(errors2_all, axis=1)
    errors3_mean = np.mean(errors3_all, axis=1)
    errors3_std = np.std(errors3_all, axis=1)
    np.savez('Fig2.npz', errors1_mean=errors1_mean, errors1_std=errors1_std, errors2_mean=errors2_mean, errors2_std=errors2_std, errors3_mean=errors3_mean, errors3_std=errors3_std)

    # Plot with error bars
    plt.figure(2)
    plt.errorbar(Ns, errors1_mean, yerr=errors1_std, fmt='o-', label='Zero Initialization', color='blue', capsize=5)
    plt.errorbar(Ns, errors2_mean, yerr=errors2_std, fmt='o-', label='Random Initialization', color='orange', capsize=5)
    plt.errorbar(Ns, errors3_mean, yerr=errors3_std, fmt='o-', label='MLE Initialization', color='green', capsize=5)
    print("Zero Initialization mean error: ", errors1_mean)
    print("Random Initialization mean error: ", errors2_mean)
    print("MLE Initialization mean error: ", errors3_mean)
    # plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel('N (matrix Catoni)')
    plt.ylabel('Nuclear norm error')
    plt.legend()
    plt.title('Error vs. Sample Size for matrix Catoni')
    plt.savefig('Fig2.png', dpi=300)
    plt.show()
