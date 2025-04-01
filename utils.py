import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


def sigmoid(x):
  x = np.minimum(np.maximum(x, -14), 14)
  y = 1 / (1 + np.exp(- x))
  return y

def dsigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

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


def E_optimal_design(env):
    """
    E-optimal design
    """
    X_arms = env.X_arms
    K = env.K

    # E-optimal design
    pi_E = cp.Variable(K, nonneg=True)
    constraints = [cp.sum(pi_E) == 1]
    V_pi = X_arms.T @ cp.diag(pi_E) @ X_arms
    objective_E = cp.Maximize(cp.lambda_min(V_pi))
    prob_E = cp.Problem(objective_E, constraints)
    try:
        prob_E.solve(solver=cp.MOSEK)
    except:
        print("Solver status for E-optimal design:", prob_E.status)
        prob_E.solve(solver=cp.MOSEK, verbose=True)
    pi_E = np.abs(np.array(pi_E.value))
    pi_E /= np.sum(pi_E)

    return pi_E