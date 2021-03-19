import numpy as np

def explicit_euler(y, F, list_t):
    for t_num in range(1, list_t.size):
        y_test = y[-1] + (list_t[t_num] - list_t[t_num - 1]) * F(y[-1])
        y = np.append(y, [y_test], axis = 0)
    return y

def implicit_euler(y, F, deriv_F, list_t):
    for t_num in range(1, list_t.size):
        func = lambda x: x - y[-1] - (list_t[t_num] - list_t[t_num - 1]) * F(x)
        deriv_func = lambda x: 1 - (list_t[t_num] - list_t[t_num - 1]) * deriv_F(x)
        x_0 = np.array([2, 2])
        it=0
        while True:
            x = x_0 - func(x_0) * (1 / deriv_func(x_0))
            it+= 1
            if np.allclose(x, x_0) or it > 100:
                result = x
                break
            x_0 = x
        y = np.append(y, [result], axis = 0)
    return y