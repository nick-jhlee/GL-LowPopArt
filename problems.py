from utils import *
import numpy as np
import functools
import os
from datetime import datetime
import h5py
import itertools
from config import PROBLEM_INSTANCES_DIR

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

def save_problem_instance(arm_set, Theta_star, mode, run_idx, save_dir=PROBLEM_INSTANCES_DIR):
    """
    Save problem instance (arm set and Theta_star) to HDF5 file.
    
    Args:
        arm_set: List of arm matrices
        Theta_star: Target matrix
        mode: Experiment mode ('completion' or 'recovery')
        run_idx: Run index
        save_dir: Directory to save the instance
    """
    try:
        os.makedirs(save_dir, exist_ok=True)
        filename = f'{save_dir}/{mode}_run{run_idx}_instance.h5'
        with h5py.File(filename, 'w') as f:
            # Save arm set
            arm_set_array = np.array([arm.flatten('F') for arm in arm_set])
            f.create_dataset('arm_set', data=arm_set_array)
            # Save Theta_star
            f.create_dataset('Theta_star', data=Theta_star)
            # Save metadata
            f.attrs['mode'] = mode
            f.attrs['run_idx'] = run_idx
            f.attrs['timestamp'] = datetime.now().isoformat()
    except Exception as e:
        raise RuntimeError(f"Failed to save problem instance: {str(e)}")

def load_problem_instance(mode, run_idx, save_dir=PROBLEM_INSTANCES_DIR):
    """
    Load problem instance from HDF5 file.
    
    Args:
        mode: Experiment mode ('completion' or 'recovery')
        run_idx: Run index
        save_dir: Directory containing the instance file
        
    Returns:
        Tuple of (arm_set, Theta_star)
    """
    try:
        filename = f'{save_dir}/{mode}_run{run_idx}_instance.h5'
        with h5py.File(filename, 'r') as f:
            arm_set_array = f['arm_set'][:]
            Theta_star = f['Theta_star'][:]
            # Reshape arm set back to original format
            arm_set = [arm.reshape(Theta_star.shape, order='F') for arm in arm_set_array]
        return arm_set, Theta_star
    except Exception as e:
        raise RuntimeError(f"Failed to load problem instance: {str(e)}")

@functools.lru_cache(maxsize=32)
def generate_arm_set(d1, d2, K, mode='completion', rng=None):
    """
    Generate arm set for 1-bit matrix completion/recovery
    
    Args:
        d1: First dimension
        d2: Second dimension
        K: Number of arms
        mode: Experiment mode ('completion', 'recovery', or 'hard')
            - 'completion': Uses standard basis matrices (single entry = 1)
            - 'recovery': Uniformly samples arms from unit Frobenius sphere
            - 'hard': Uses the hard instance from Jang et al. (2024)
        rng: Random number generator (optional)
    """
    if rng is None:
        rng = np.random.RandomState()
    
    arm_set = []
    if mode == 'completion':
        # Matrix completion basis
        arm_set = []
        for i, j in itertools.product(range(d1), range(d2)):
            arm = np.zeros((d1, d2))
            arm[i, j] = 1
            arm_set.append(arm)
    elif mode == 'recovery':
        # Uniformly sample arms from the unit Frobenius sphere
        arm_set = []
        for _ in range(K):
            # Generate a random pattern
            arm = rng.randn(d1, d2)  # Fixed: removed extra parentheses
            arm = arm / np.linalg.norm(arm)  # Normalize to unit Frobenius norm
            arm_set.append(arm)
    elif mode == 'hard':
        # Hard instance from Jang et al. (2024)
        e1 = np.zeros(d1*d2)
        e1[0] = 1
        vec_arm = (1 / np.sqrt(d1)) * e1
        arm_set = [vec_arm.reshape((d1, d2))]
        for i in range(1, d1*d2):
            ei = np.zeros(d1*d2)
            ei[i] = 1
            arm = e1 + (1 / np.sqrt(d1)) * ei
            arm_set.append(arm.reshape((d1, d2)))
    else:
        raise ValueError(f"Invalid mode: {mode}")
    return arm_set

def generate_Theta_star(d1, d2, r, rng=None, symmetric=True):
    """
    Generate true parameter matrix Theta_star
    
    Args:
        d1: First dimension
        d2: Second dimension
        r: Rank
        rng: Random number generator (optional)
        symmetric: Whether to generate a symmetric matrix (optional)
    """
    if rng is None:
        rng = np.random.RandomState()
    
    # Generate random orthogonal matrices
    U = rng.randn(d1, r)
    # QR decomposition to get orthogonal matrices
    U, _ = np.linalg.qr(U)
    # Generate singular values
    S = rng.rand(r)
    S = S / np.max(S)  # Normalize
    # Construct Theta_star
    if symmetric:
        Theta_star = U @ np.diag(S) @ U.T
    else:
        V = rng.randn(d2, r)
        V, _ = np.linalg.qr(V)
        Theta_star = U @ np.diag(S) @ V.T
    return Theta_star