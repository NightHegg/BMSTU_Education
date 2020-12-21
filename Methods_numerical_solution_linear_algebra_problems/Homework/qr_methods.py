import numpy as np
from math import sqrt, hypot

def house(x):
    u_tilde = np.copy(x)
    u_tilde[0] += np.sign(x[0]) * np.linalg.norm(x)
    return u_tilde / np.linalg.norm(u_tilde)

def qr_householder(A):
    m, n = A.shape
    Q = np.identity(m)
    R = np.copy(A)
    for i in range(n):
        P_i = np.identity(m)
        x = R[i:, i]
        u = house(x)
        P_streak = np.identity(m - i) - 2 * np.outer(u, u)
        P_i[i:, i:] = np.copy(P_streak)
        R = P_i @ R
        Q = Q @ P_i 
    return Q[:, :n], R[:n, :] 

def qr_householder_effective(A):
    curr_A = np.copy(A)
    list_first_elements = []
    m, n = A.shape
    for i in range(n):
        x = curr_A[i:, i]
        u = house(x) 
        curr_A[i:m, i:n] -= 2 * np.outer(u, np.matmul(u, curr_A[i:m, i:n]))
        curr_A[i + 1:m, i] = u[1:]
        list_first_elements.append(u[0])
    return curr_A, list_first_elements

def qr_givens(A):
    m, n = A.shape
    Q = np.identity(m)
    R = np.copy(A)
    for j in range(n):
        for i in range(j + 1, m):
            E = np.identity(m)
            c = R[j, j] / hypot(R[j, j], R[i, j])
            s = -R[i, j] / hypot(R[j, j], R[i, j])
            E[j, j] = E[i, i] = c
            E[i, j], E[j, i] = s, -s
            R = E @ R
            Q = Q @ E.T
    return Q[:, :n], R[:n, :]

def implicit_givens(A):
    m, n = A.shape
    curr_A = np.copy(A)
    for i in range(n - 1):
        sigma = curr_A[-1, -1]
        Q = np.identity(m)
        b = hypot(curr_A[i, i] - sigma, curr_A[i + 1, i])
        c = (curr_A[i, i] - sigma) / b
        s = curr_A[i + 1, i] / b
        Q[i, i] = Q[i + 1, i + 1] = c
        Q[i + 1, i], Q[i, i + 1] = s, -s
        curr_A = Q.T @ curr_A @ Q
    return curr_A

def implicit_householder(A):
    m, n = A.shape
    curr_A = np.copy(A)
    for i in range(n - 2):
        Q = np.identity(m) ## 1
        x = curr_A[i + 1:i + 3, i] ## 2
        u = house(x) ## 3
        P = np.identity(2) - 2 * np.outer(u, u) ## 4
        Q[i + 1: i + 3, i + 1: i + 3] = np.copy(P) ## 5
        curr_A = np.dot(np.dot(Q, curr_A), np.transpose(Q))
    return curr_A

def hessenberg_householder(A):
    m, n = A.shape
    curr_A = np.copy(A)
    for i in range(n - 2):
        Q = np.identity(m)
        x = curr_A[i + 1:, i]
        u = house(x)
        P = np.identity(m - i - 1) - 2 * np.outer(u, u)
        Q[i + 1:, i + 1:] = np.copy(P)
        curr_A = Q @ curr_A @ Q.T
    return curr_A

def hessenberg_givens(A):
    m, n = A.shape
    curr_A = np.copy(A)
    for i in range(n - 2):
        for j in range(i + 2, m):
            Q = np.identity(m)
            c = curr_A[i + 1, i] / hypot(curr_A[i + 1, i], curr_A[j, i])
            s = -curr_A[j, i] / hypot(curr_A[i + 1, i], curr_A[j, i])
            Q[i + 1, i + 1] = Q[j, j] = c
            Q[j, i + 1], Q[i + 1, j] = s, -s
            curr_A = Q @ curr_A @ Q.T
    return curr_A    