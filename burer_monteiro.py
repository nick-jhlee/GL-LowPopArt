from utils import *
import numpy as np
import logging

def loss(U, X1, y1):
    """
    Compute the loss for the Burer-Monteiro factorization.
    """
    N1 = X1.shape[0]
    theta = (U @ U.T).flatten('F')
    logits = X1 @ theta
    return np.sum(y1 * logits - np.log(1 + np.exp(logits))) / (2 * N1)

def grad(U, X1, y1):
    """
    Compute the gradient of the loss with respect to U.
    """
    N1 = X1.shape[0]
    d = U.shape[0]
    Theta = U @ U.T
    logits = X1 @ Theta.flatten('F')
    residuals = y1 - sigmoid(logits)  # Vector of residuals
    grad_matrix = np.reshape(X1.T @ residuals, (d, d), 'F')  # Reshape to matrix
    return (grad_matrix @ U) / (2 * N1)

def grad_norm(U, X1, y1):
    """
    Compute the norm of the gradient.
    """
    return np.linalg.norm(grad(U, X1, y1))

def Burer_Monteiro(d, r, X1, y1, lr=1e-2, grad_tol=1e-6, max_iter=1e4):
    """
    Perform Burer-Monteiro factorization to solve the low-rank matrix optimization problem.
    """
    # Small initialization for U (see Stoger & Soltanolkotabi (2021) and Chung & Kim (2023), although their theory is for linear matrix recovery)
    U = 1e-4 * np.random.randn(d, r)
    iter = 0
    grad_norms = [grad_norm(U, X1, y1)]
    
    while grad_norm(U, X1, y1) > grad_tol:
        # Gradient descent update
        U -= grad(U, X1, y1) * lr
        iter += 1

        # Compute the gradient norm
        current_grad_norm = grad_norm(U, X1, y1)
        grad_norms.append(current_grad_norm)

        if iter > max_iter:
            logging.warning("Burer-Monteiro did not converge within max_iter")
            break
    
    return U @ U.T

def sigmoid(x):
    return 1 / (1 + np.exp(-x))