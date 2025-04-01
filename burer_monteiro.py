from utils import *

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

def Burer_Monteiro(d, r, X1, y1, lr=3e-1, grad_tol=1e-6, max_iter=1e4):
    """
    Perform Burer-Monteiro factorization to solve the low-rank matrix optimization problem.
    """
    # Small initialization for U (see Stoger & Soltanolkotabi (2021) and Chung & Kim (2023), although their theory is for linear matrix recovery)
    U = 1e-3 * np.random.randn(d, r)
    iter = 0
    # prev_loss = loss(U, X1, y1)
    grad_norms = [grad_norm(U, X1, y1)]
    while grad_norm(U, X1, y1) > grad_tol:
        # Gradient descent update
        U -= grad(U, X1, y1) * lr
        iter += 1

        # Compute the loss and gradient norm
        current_loss = loss(U, X1, y1)
        current_grad_norm = grad_norm(U, X1, y1)
        grad_norms.append(current_grad_norm)
        # # Check for convergence or stagnation
        # current_loss = loss(U, X1, y1)
        # if abs(prev_loss - current_loss) < 1e-8:  # Stagnation threshold
        #     print(f"Converged: Loss change below threshold at iteration {iter}.")
        #     break
        # prev_loss = current_loss

        if iter > max_iter:
            print("Burer-Monteiro did not converge within max_iter.")
            break
    
    # print(f"Burer-Monteiro converged in {iter} iterations.", grad_norm(U, X1, y1))

    # plt.plot(grad_norms)
    # plt.show()

    return U @ U.T