"""Problem generation and observation models."""

import functools
import itertools
import os
from datetime import datetime

import h5py
import numpy as np

from gl_lowpopart.config import PROBLEM_INSTANCES_DIR
from gl_lowpopart.utils import dsigmoid, sigmoid


class MatrixCompletion:
    def __init__(self, arm_set, Theta_star, model="bernoulli", poisson_clip=10.0):
        self.arm_set = arm_set
        self.Theta_star = Theta_star
        self.model = model
        self.poisson_clip = poisson_clip
        self.r = np.linalg.matrix_rank(Theta_star)
        self.d1 = Theta_star.shape[0]
        self.d2 = Theta_star.shape[1]
        self.K = len(arm_set)
        self.X_arms = np.ascontiguousarray(
            np.concatenate([arm.flatten("F").reshape(1, -1) for arm in arm_set], axis=0),
            dtype=np.float64,
        )

    def mean_from_eta(self, eta):
        if self.model == "bernoulli":
            return sigmoid(eta)
        if self.model == "poisson":
            return np.exp(np.clip(eta, -self.poisson_clip, self.poisson_clip))
        raise ValueError(f"Unsupported model: {self.model}")

    def dmean_from_eta(self, eta):
        if self.model == "bernoulli":
            return dsigmoid(eta)
        if self.model == "poisson":
            return np.exp(np.clip(eta, -self.poisson_clip, self.poisson_clip))
        raise ValueError(f"Unsupported model: {self.model}")

    def get_reward(self, arm):
        x = arm.flatten("F")
        theta = self.Theta_star.flatten("F")
        eta = np.dot(x, theta)
        if self.model == "bernoulli":
            return np.random.binomial(1, self.mean_from_eta(eta))
        if self.model == "poisson":
            return np.random.poisson(self.mean_from_eta(eta))
        raise ValueError(f"Unsupported model: {self.model}")


def save_problem_instance(arm_set, Theta_star, mode, run_idx, save_dir=PROBLEM_INSTANCES_DIR, filename=None):
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename) if filename else os.path.join(save_dir, f"{mode}_run{run_idx}_instance.h5")
    with h5py.File(filepath, "w") as f:
        arm_set_array = np.array([arm.flatten("F") for arm in arm_set])
        f.create_dataset("arm_set", data=arm_set_array)
        f.create_dataset("Theta_star", data=Theta_star)
        f.attrs["mode"] = mode
        f.attrs["run_idx"] = run_idx
        f.attrs["timestamp"] = datetime.now().isoformat()


def load_problem_instance(mode, run_idx, save_dir=PROBLEM_INSTANCES_DIR, filename=None):
    filepath = os.path.join(save_dir, filename) if filename else os.path.join(save_dir, f"{mode}_run{run_idx}_instance.h5")
    with h5py.File(filepath, "r") as f:
        arm_set_array = f["arm_set"][:]
        Theta_star = f["Theta_star"][:]
        arm_set = [arm.reshape(Theta_star.shape, order="F") for arm in arm_set_array]
    return arm_set, Theta_star


@functools.lru_cache(maxsize=32)
def generate_arm_set(d1, d2, K, mode="completion", rng=None):
    if rng is None:
        rng = np.random.RandomState()

    if mode == "completion":
        arm_set = []
        for i, j in itertools.product(range(d1), range(d2)):
            arm = np.zeros((d1, d2))
            arm[i, j] = 1
            arm_set.append(arm)
        return arm_set

    if mode == "recovery":
        arm_set = []
        for _ in range(K):
            arm = rng.randn(d1, d2)
            arm = arm / np.linalg.norm(arm)
            arm_set.append(arm)
        return arm_set

    if mode == "hard":
        e1 = np.zeros(d1 * d2)
        e1[0] = 1
        vec_arm = (1 / np.sqrt(d1)) * e1
        arm_set = [vec_arm.reshape((d1, d2))]
        for i in range(1, d1 * d2):
            ei = np.zeros(d1 * d2)
            ei[i] = 1
            arm = e1 + (1 / np.sqrt(d1)) * ei
            arm_set.append(arm.reshape((d1, d2)))
        return arm_set

    raise ValueError(f"Invalid mode: {mode}")


def generate_Theta_star(d1, d2, r, rng=None, symmetric=True):
    if rng is None:
        rng = np.random.RandomState()
    U = rng.randn(d1, r)
    U, _ = np.linalg.qr(U)
    if symmetric:
        Theta_star = U @ U.T
    else:
        V = rng.randn(d2, r)
        V, _ = np.linalg.qr(V)
        Theta_star = U @ V.T
    return 2.0*Theta_star

