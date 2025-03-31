import numpy as np


def sigmoid(x):
  x = np.minimum(np.maximum(x, -14), 14)
  y = 1 / (1 + np.exp(- x))
  return y

def dmu(x):
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