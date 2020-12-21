import numpy as np
import pandas as pd
from scipy.linalg import hessenberg as hb
from math import sqrt
from qr_methods import implicit_householder, hessenberg_givens, implicit_givens, hessenberg_householder, qr_givens, qr_householder, qr_householder_effective

def construct_delta_A(N_var, eps):
    par = 10
    delta_A = []
    c = N_var * eps / (N_var + 1)
    for i in range(par):
        temp = []
        for j in range(par):
            temp.append(0 if i == j else c / (i + j))
        delta_A.append(temp)
    return np.array(delta_A)

def basic_qr(A):
    curr_A = np.copy(A)
    amnt_iters = 0
    while True:
        amnt_iters += 1
        Q, R = qr_householder(curr_A)
        curr_A = np.dot(R, Q)
        if np.allclose(curr_A, np.triu(curr_A)) or amnt_iters > 400:
            break
    return np.diag(curr_A).sort(), amnt_iters

def shifting_qr(A, use_hessenberg = True):
    m, n = A.shape
    curr_A = np.copy(A)
    list_eigenval = []
    if use_hessenberg:
        curr_A = hessenberg_householder(curr_A)
    amnt_iters = 0
    for i in range(n, 0, -1):
        lc_it = 0
        if curr_A.size == 1:
            list_eigenval.append(curr_A[0, 0])
            break
        while True:
            sigma = curr_A[-1, -1]
            Q, R = qr_householder(curr_A - sigma * np.identity(i))
            curr_A = np.dot(R, Q) + sigma * np.identity(i)
            lc_it += 1
            cond = lambda elem: np.allclose(elem, np.zeros(i - 1))
            if cond(curr_A[-1, :-1]) and cond(curr_A[:-1, -1]) or lc_it > 50:
                amnt_iters += lc_it
                list_eigenval.append(curr_A[-1, -1])
                curr_A = np.copy(curr_A[:-1, :-1])
                break
    return np.array(sorted(list_eigenval)), amnt_iters

def implicit_shifting_qr(A):
    m, n = A.shape
    curr_A = np.copy(A)
    list_eigenval = []
    curr_A = hessenberg_householder(curr_A)
    amnt_iters = 0
    for i in range(n, 0, -1):
        lc_it = 0
        if curr_A.size == 1:
            list_eigenval.append(curr_A[0, 0])
            break
        while True:
            curr_A = implicit_givens(curr_A)
            lc_it += 1
            cond = lambda elem: np.allclose(elem, np.zeros(i - 1))
            if cond(curr_A[-1, :-1]) and cond(curr_A[:-1, -1]) or lc_it > 50:
                amnt_iters += lc_it
                list_eigenval.append(curr_A[-1, -1])
                curr_A = np.copy(curr_A[:-1, :-1])
                break
    return np.array(sorted(list_eigenval)), amnt_iters


def task_01(A_0, N_var, eps):
    delta_A = construct_delta_A(N_var, eps)
    A = A_0 + delta_A

    amnt_deletable_cols = 1
    list_deletable_cols = [9 - i for i in range(amnt_deletable_cols)]
    A_hat = np.delete(A, list_deletable_cols, axis = 1)
    x_0 = np.random.rand(10 - amnt_deletable_cols) 
    b = A_hat @ x_0

    Q, R = qr_householder(A_hat)

    assert ((Q @ R - A_hat) < 1e-6).all(), "QR decomposition is wrong"
    x = np.linalg.inv(R) @ Q.T @ b
    value_estimation = np.linalg.norm(x - x_0) / np.linalg.norm(x_0)

    b_effective = np.copy(b)
    A_effective, u_first_elements = qr_householder_effective(A_hat)
    for i in range(10 - amnt_deletable_cols):
        u = np.hstack((u_first_elements[i], A_effective[i + 1:, i]))
        gamma = -2 * np.inner(u, b_effective[i:])
        b_effective[i:] += gamma * u
    x_effective = np.dot(np.linalg.inv(np.triu(A_effective[:-amnt_deletable_cols])), b_effective[:-amnt_deletable_cols])

    msg_result = (f"Решение было получено. \n"
            f"x_0                       = {[round(i, 4) for i in x_0]} \n"
            f"Найдённое решение x       = {[round(i, 4) for i in x]} \n"
            f"Эффективное решение x     = {[round(i, 4) for i in x_effective]} \n"
            f"Относительная погрешность = {value_estimation:.4e}")

    return msg_result

def task_02(A_0, N_var, eps):
    m, n = A_0.shape
    eigenvalue_0 = np.array([2 * (1 - np.cos(np.pi * j / (n + 1))) for j in range(1, n + 1)])
    eigenvector_0 = lambda j: np.array([sqrt(2 / (n + 1)) * np.sin(np.pi * j * k / (n + 1)) for k in range(1, n + 1)])
    eigenval_results = []
    eigenvec_results = []
    
    for curr_eps in eps:
        temp_eigenvec = []
        A = A_0 + construct_delta_A(N_var, curr_eps)
        curr_eigenvalues, iters = implicit_shifting_qr(A)
        print(iters)
        diff_eigenvalues = abs(curr_eigenvalues - eigenvalue_0)
        eigenval_results.append([f"{i:.3e}" for i in diff_eigenvalues])
        for idx, eigenval in enumerate(sorted(curr_eigenvalues)):
            y = np.linalg.solve(A - eigenval * np.identity(10), np.ones(10))
            x = y / np.linalg.norm(y)
            if np.sign(x[0]) != np.sign(eigenvector_0(idx + 1)[0]):
                x *= -1
            temp_eigenvec.append(f"{np.linalg.norm(x - eigenvector_0(idx + 1)):.3e}")
        eigenvec_results.append(temp_eigenvec)

    df_eigenval = pd.DataFrame(eigenval_results, index = [f"{curr_eps:.0e}" for curr_eps in eps]).T
    df_eigenvec = pd.DataFrame(eigenvec_results, index = [f"{curr_eps:.0e}" for curr_eps in eps]).T
    print("Таблица погрешностей собственных значений:")
    print(df_eigenval.to_latex())
    print("\n Таблица погрешностей собственных векторов:")
    print(df_eigenvec.to_latex())

if __name__ == "__main__":
    A_0 = np.identity(10) * 2 + np.eye(10, 10, k = -1) * -1 + np.eye(10, 10, k = 1) * -1
    N_var = 3
    
    msg_result = task_01(A_0, N_var, 1e-1)
    print(msg_result)

    task_02(A_0, N_var, [1e-1, 1e-3, 1e-6])