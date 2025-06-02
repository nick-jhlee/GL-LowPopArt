from utils import *
import numpy as np
import functools

class OneBitCompletion:
    def __init__(self, arm_set, Theta_star):
        self.arm_set = arm_set
        self.Theta_star = Theta_star
        self.r = np.linalg.matrix_rank(Theta_star)
        self.d1 = Theta_star.shape[0]
        self.d2 = Theta_star.shape[1]

        self.K = len(arm_set)
        self.X_arms = np.array([arm.flatten('F') for arm in arm_set])
    
    def get_reward(self, arm):
        """
        Simulate the reward for a given arm.
        The reward is generated based on the inner product of the arm and the true parameter matrix.
        """
        x = arm.flatten('F')
        theta = self.Theta_star.flatten('F')
        p = 1 / (1 + np.exp(-np.dot(x, theta)))
        return np.random.binomial(1, p)

@functools.lru_cache(maxsize=32)
def generate_arm_set(d1, d2, K, mode='completion'):
    """
    Generate arm set for 1-bit matrix completion/recovery
    """
    if mode == 'completion':
        # For completion, generate random arms
        arm_set = []
        for _ in range(K):
            arm = np.zeros((d1, d2))
            # Randomly select one entry to be 1
            i = np.random.randint(0, d1)
            j = np.random.randint(0, d2)
            arm[i, j] = 1
            arm_set.append(arm)
    else:  # recovery mode
        # For recovery, generate arms with specific patterns
        arm_set = []
        for _ in range(K):
            arm = np.zeros((d1, d2))
            # Generate a random pattern
            pattern = np.random.randn(d1, d2)
            pattern = pattern / np.linalg.norm(pattern)
            arm_set.append(pattern)
    
    return arm_set

def generate_Theta_star(d1, d2, r, symmetric=True):
    """
    Generate true parameter matrix Theta_star
    """
    # Generate random orthogonal matrices
    U = np.random.randn(d1, r)
    # QR decomposition to get orthogonal matrices
    U, _ = np.linalg.qr(U)
    # Generate singular values
    S = np.random.rand(r)
    S = S / np.max(S)  # Normalize
    # Construct Theta_star
    if symmetric:
        Theta_star = U @ np.diag(S) @ U.T
    else:
        V = np.random.randn(d2, r)
        V, _ = np.linalg.qr(V)
        Theta_star = U @ np.diag(S) @ V.T
    return Theta_star