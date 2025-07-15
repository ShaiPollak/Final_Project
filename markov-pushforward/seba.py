import scipy.linalg as sla
import numpy as np

def seba(V, R0=None):
    """
    :param V: PxR matrix (R vectors of length P as columns, assumed orthonormal)
    :return: matrix with columns approximately spanning the column space of V
    """
    # Enforce orthonormality
    V, _ = sla.qr(V, mode='economic')  
    p, r = V.shape
    mu = 0.99/np.sqrt(p)

    # Perturb near-constant vectors
    for j in range(0, r):
        if (np.max(V[:, j]) - np.min(V[:, j])) < 1e-14:
            V[:, j] += (np.random.rand(p)-1/2)*1e-12

    # rotation vector
    if R0 is None:
        Rn = np.eye(r)
    else:
        P, _, Q = sla.svd(R0, full_matrices=False)
        R0 = P @ Q.T # Ensure orthonormality of R0
        Rn = R0

    R = 0
    S = np.zeros_like(V)
    maxiter = 5000
    
    iter = 0
    while sla.norm(Rn-R, ord=2) > 1e-14 and iter < maxiter:
        iter += 1
        R = np.copy(Rn)
        Z = V @ R.T

        #Threshold to solve sparse approximation problem
        for i in range(0, r):
            S[:, i] = np.sign(Z[:, i]) * np.maximum(np.abs(Z[:, i]) - mu, 0)
            S[:, i] /= sla.norm(S[:, i])  # 2-norm for vectors

        # Polar decomposition to solve Procrustes problem
        P, _, Q = sla.svd(S.T @ V, full_matrices=False)
        Rn = P @ Q

    # Modify the sign of the vectors and scale to 1
    for i in range(0, r):
        S[:,i] *= np.sign(sum(S[:,i]))
        S[:,i] /= np.max(S[:,i])

    # Sort so that most reliable vectors appear first
    I = np.argsort(np.min(S,0))[::-1]
    return S[:, I]


def subpartition_unity(S):
    """
    Subpartition unity apply hard thresholding to obtain a sub-partition of unity from the columns of S. Also create a maximum-likelihood sub-partition.
    :param S: output of SEBA algorithm
    :return: S, A, taupu
    """

    # Take non-negative part
    S = np.maximum(S,0)            
    
    # Sort each row in descending order
    S_descend = np.fliplr(np.sort(S, 1))
    
    # Must be <=1 for partition of unity
    S_sum = np.cumsum(S_descend,1)
    
    # Largest element where row sum > 1
    if len(S_descend[S_sum>1]):          
        taupu = np.max(S_descend[S_sum>1])
    else:
        taupu = 0
    
    # Apply hard thresholding
    S[S <= taupu] = 0                    

    # find largest by row
    A = np.argmax(S, 1)  # index of column
    M = np.max(S,1)      # value

    # remove lines where the maximum
    # is zeros and assign the clusters -1
    A[M <= 1e-10] = -1

    # clusters value from [0, N]
    A += 1
    
    return S, A, taupu


def max_likelihood(S):
    """
    Apply maximum likelihood to obtain disjoint supports from the columns of S, resolving ties arbitrarily. Also create the corresponding sub-partition. 
    :param S: output of SEBA algorithm
    :return: S, A
    """
    
    A = np.argmax(S, 1)  # column with largest element by row
    M = np.max(S,1)      # largest element by row

    # remove lines where there is no maximum where
    # all columns are 0 and assign the clusters -1
    A[M <= 1e-10] = -1
    
    S[:] = 0
    r = S.shape[1]
    for i in range(0, r):
        S[A==i, i] = M[A == i]
    
    # clusters value from [0, N]
    A += 1
    
    return S, A

def disjoint_support(S):
    """
    Apply hard thresholding to obtain disjoint supports for the columns of S.
    :param S: output of SEBA algorithm
    :return: S, A, taupu
    """

    # Take non-negative part
    S = np.maximum(S,0)  
    
    # Sort each row in descending order
    S_descend = np.fliplr(np.sort(S, 1))
    
    # Sort each row in descending order
    S_descend = np.fliplr(np.sort(S, 1))
    
    # Take largest non-leading element
    taupu = np.max(S_descend[:, 1])
    
    # Apply hard thresholding
    S[S <= taupu] = 0
    
    # find largest by row
    A = np.argmax(S, 1)  # index of column
    M = np.max(S,1)      # value

    # remove lines where the maximum
    # is zeros and assign the clusters -1
    A[M <= 1e-10] = -1

    # clusters value from [0, N]
    A += 1
    
    return S, A, taupu
