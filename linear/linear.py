from lowpopart import *
from tqdm import tqdm
from math import floor

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
        vecs_outers = [np.outer(vec1, vec2) for (vec1, vec2) in zip(vecs_1, vecs_2)]
        return np.sum(vecs_outers, axis=0)
        # # Theta_star = Theta_star / np.linalg.norm(Theta_star)
        # print("rank of Theta_star: ", np.linalg.matrix_rank(Theta_star))

        # U_star, S_star, V_star = np.linalg.svd(Theta_star)
        # print("singular values of Theta_star: ", S_star)

    ## Hard instance
    # I_d = np.eye(d1 * d2)
    # arm_set = [I_d[0].reshape(d1, d2) / np.sqrt(d)]
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
    # Ns = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    # Ns = [1000, 2000, 4000, 6000, 8000, 10000, 15000, 20000]
    Ns = [1000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]

    errors0_all = []
    errors_all = []
    Theta0 = np.zeros((d1, d2))
    for N in tqdm(Ns):
        errors0_reps = []
        errors_reps = []
        for _ in range(num_repeats):
            # random problem instance
            Theta_star = generate_Theta_star()
            env = LinearCompletion(arm_set, Theta_star)

            # nuclear norm regularized MLE
            nuc_coef = (1 + np.max([np.abs(tmp) for tmp in env.X_arms @ Theta_star.flatten('F')]))*np.sqrt(np.log(d1+d2) / N)
            Theta0 = nuc_norm_MLE_linear(env, N, d1, d2, nuc_coef)
            errors0_reps.append(np.linalg.norm(Theta0 - Theta_star, 'nuc'))

            # LowPopArt
            Theta = LowPopArt(env, N, d1, d2, delta, np.zeros((d1, d2)))
            errors_reps.append(np.linalg.norm(Theta - Theta_star, 'nuc'))
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
    print("Stage I mean error: ", errors0_mean)
    print("Stage I + II mean error: ", errors_mean)
    # plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel('N')
    plt.ylabel('Spectral norm error')
    plt.legend()
    plt.title('Error vs. Sample Size with Error Bars')
    plt.show()