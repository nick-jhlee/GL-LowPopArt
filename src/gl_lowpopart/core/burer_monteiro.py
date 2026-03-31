"""Burer-Monteiro baseline for Bernoulli model."""

import logging

import numpy as np

from gl_lowpopart.utils import sigmoid


def loss(U, X1, y1):
    n1 = X1.shape[0]
    theta = (U @ U.T).flatten("F")
    logits = X1 @ theta
    return np.sum(y1 * logits - np.log(1 + np.exp(logits))) / (2 * n1)


def grad(U, X1, y1):
    n1 = X1.shape[0]
    d = U.shape[0]
    Theta = U @ U.T
    logits = X1 @ Theta.flatten("F")
    residuals = y1 - sigmoid(logits)
    grad_matrix = np.reshape(X1.T @ residuals, (d, d), "F")
    return (grad_matrix @ U) / (2 * n1)


def grad_norm(U, X1, y1):
    return np.linalg.norm(grad(U, X1, y1))


def Burer_Monteiro(d, r, X1, y1, lr=1e-2, grad_tol=1e-6, max_iter=1e4):
    U = 1e-4 * np.random.randn(d, r)
    it = 0
    while grad_norm(U, X1, y1) > grad_tol:
        U -= grad(U, X1, y1) * lr
        it += 1
        if it > max_iter:
            logging.warning("Burer-Monteiro did not converge within max_iter")
            break
    return U @ U.T

