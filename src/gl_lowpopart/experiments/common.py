"""Shared helpers for experiment runners."""

import os
from typing import Any, List, Tuple

import numpy as np

from gl_lowpopart.config import PROBLEM_INSTANCES_DIR
from gl_lowpopart.core.burer_monteiro import Burer_Monteiro
from gl_lowpopart.core.optimization import GL_LowPopArt, nuc_norm_MLE
from gl_lowpopart.core.problems import (
    MatrixCompletion,
    generate_Theta_star,
    generate_arm_set,
    load_problem_instance,
    save_problem_instance,
)


def run_bmf(env: Any, d1: int, r: int, X1: np.ndarray, y1: np.ndarray) -> float:
    Theta_BMF = Burer_Monteiro(d1, r, X1, y1)
    return np.linalg.norm(Theta_BMF - env.Theta_star, "nuc")


def run_stage1(env: Any, N: int, d1: int, d2: int, nuc_coef: float, e_optimal: bool, stage1_solver: str = "fista"):
    Theta0, X1, y1 = nuc_norm_MLE(env, N, d1, d2, nuc_coef, E_optimal=e_optimal, stage1_solver=stage1_solver)
    return np.linalg.norm(Theta0 - env.Theta_star, "nuc"), X1, y1


def run_stage1_2(
    env: Any,
    N1: int,
    N2: int,
    d1: int,
    d2: int,
    nuc_coef: float,
    c_nu: float,
    delta: float,
    e_optimal: bool,
    gl_optimal: bool,
    stage1_solver: str = "fista",
):
    Theta0, _, _ = nuc_norm_MLE(env, N1, d1, d2, nuc_coef, E_optimal=e_optimal, stage1_solver=stage1_solver)
    Theta = GL_LowPopArt(env, N2, d1, d2, delta, Theta0, c_nu, GL_optimal=gl_optimal)
    return np.linalg.norm(Theta - env.Theta_star, "nuc")


def build_problem_instances(
    mode: str, model: str, d1: int, d2: int, r: int, K: int, num_repeats: int, seed: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    problem_instances = []
    rng = np.random.RandomState(seed)

    arm_file = f"{model}_{mode}_arm_set.h5"
    arm_path = os.path.join(PROBLEM_INSTANCES_DIR, arm_file)
    if os.path.exists(arm_path):
        arm_set, _ = load_problem_instance(mode, 0, PROBLEM_INSTANCES_DIR, filename=arm_file)
    else:
        arm_set = generate_arm_set(d1, d2, K, mode=mode, rng=rng)
        save_problem_instance(arm_set, np.zeros((d1, d2)), mode, 0, PROBLEM_INSTANCES_DIR, filename=arm_file)

    for run_idx in range(num_repeats):
        theta_file = f"{model}_{mode}_run{run_idx}_theta.h5"
        theta_path = os.path.join(PROBLEM_INSTANCES_DIR, theta_file)
        if os.path.exists(theta_path):
            _, Theta_star = load_problem_instance(mode, run_idx, PROBLEM_INSTANCES_DIR, filename=theta_file)
        else:
            Theta_star = generate_Theta_star(d1, d2, r, rng=rng)
            save_problem_instance([], Theta_star, mode, run_idx, PROBLEM_INSTANCES_DIR, filename=theta_file)
        problem_instances.append((arm_set, Theta_star))

    return problem_instances


def build_env(arm_set, Theta_star, model: str):
    env = MatrixCompletion(arm_set, Theta_star, model=model)
    env.Theta_star = Theta_star
    return env

