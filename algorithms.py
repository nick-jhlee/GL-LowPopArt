"""Algorithm implementations for matrix completion experiments."""

from math import floor
from typing import Any, Tuple

import numpy as np

from burer_monteiro import Burer_Monteiro
from gl_lowpopart import GL_LowPopArt
from problems import OneBitCompletion

def run_bmf(env: Any, d1: int, r: int, X1: np.ndarray, y1: np.ndarray) -> float:
    """
    Run BMF evaluation.
    
    Args:
        env: Environment instance
        d1: First dimension
        r: Rank
        X1: Feature matrix
        y1: Response vector
        
    Returns:
        Nuclear norm error
    """
    X_bmf, y_bmf = X1, y1
    Theta_BMF = Burer_Monteiro(d1, r, X_bmf, y_bmf)
    return np.linalg.norm(Theta_BMF - env.Theta_star, 'nuc')

def run_stage1(env: Any, N: int, d1: int, d2: int, nuc_coef: float, 
               e_optimal: bool) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Run Stage I (Nuclear norm MLE) evaluation.
    
    Args:
        env: Environment instance
        N: Sample size
        d1: First dimension
        d2: Second dimension
        nuc_coef: Nuclear norm coefficient
        e_optimal: Whether to use E-optimal design
        
    Returns:
        Tuple of (nuclear norm error, feature matrix, response vector)
    """
    Theta0, X1, y1 = nuc_norm_MLE(env, N, d1, d2, nuc_coef, E_optimal=e_optimal)
    return np.linalg.norm(Theta0 - env.Theta_star, 'nuc'), X1, y1

def run_stage1_2(env: Any, N1: int, N2: int, d1: int, d2: int, nuc_coef: float,
                 c_nu: float, delta: float, e_optimal: bool, gl_optimal: bool) -> float:
    """
    Run Stage I + II evaluation.
    
    Args:
        env: Environment instance
        N1: First stage sample size
        N2: Second stage sample size
        d1: First dimension
        d2: Second dimension
        nuc_coef: Nuclear norm coefficient
        c_nu: Nu coefficient
        delta: Confidence parameter
        e_optimal: Whether to use E-optimal design
        gl_optimal: Whether to use GL-optimal design
        
    Returns:
        Nuclear norm error
    """
    # Stage I
    Theta0, _, _ = nuc_norm_MLE(env, N1, d1, d2, nuc_coef, E_optimal=e_optimal)
    # Stage II
    Theta = GL_LowPopArt(env, N2, d1, d2, delta, Theta0, c_nu, GL_optimal=gl_optimal)
    return np.linalg.norm(Theta - env.Theta_star, 'nuc') 