from math import exp, sin, cos
import numpy as np

def func(x, y):
    result = np.array([2*x, -2*x]) * y * np.log(np.amax(np.vstack((np.flip(y, 0), np.array([1e-3, 1e-3]))), 0))
    print(result)

def runge_kutta(task, method, tau):
    t = task['interval']
    f = task['func']
    a = method['a']
    b = method['b']
    c = method['c']

    x = t[0]
    y = np.array([task['init_cond']])
    amnt_iterations = int((t[1] - t[0]) / tau)

    for k in range(amnt_iterations):
        k = []
        for i in range(method['s']):
            k.append(f(x + c[i] * tau, y[-1] + tau * np.dot(np.array(k[:i]), a[i, :i])))
        y = np.append(y, y[-1] + tau * np.array(k) @ b)
        x += tau
    print(y[-1], exp(5))
    print(np.isclose(exp(5), y[-1]))

def runge_kutta_template(function, tau, x_0, y_0, a, c):
        k = []
        for i in range(c.size):
            k.append(
                        function(
                            x_0 + c[i] * tau, 
                            y_0 + tau * np.dot(np.array(k[:i]), a[i, :i])
                        )
                    )
        return np.array(k)

def runge_kutta(f, tau, x_0, y_0, a, c):
    x = t[0]
    y = np.array([task['init_cond']])
    tau = 0.05
    x, y = [x_0], [y_0]
    amnt_small_steps = 2
    amnt_iters = 0
    while True:
        y_small_steps = [y_0]
        x = x_0
        for i in range(amnt_small_steps):
            k = runge_kutta_template(f, tau, x, y_small_steps[-1], a, c)
            y = np.append(y_small_steps, y_small_steps[-1] + tau * k @ b)
            x += tau
        
        k = runge_kutta_template(f, amnt_small_steps * tau, x, y_0, a, c)
        y_big_step = y_0 + amnt_small_steps * tau * k @ b

        




class task_01():
    def __init__(self, t):
        self.t = t
        self.y_current = np.array([exp(sin(self.t[0] ** 2)), exp(cos(self.t[0] ** 2))])
        func(self.t[0], self.y_current)
 
if __name__ == "__main__":
    N_var = 2
    t_1 = N_var * 0.1

    task = {
        'primary': {
            'func':      lambda x, y: np.array([2*x, -2*x]) * y * np.log(np.amax(np.vstack((np.flip(y, 0), np.array([1e-3, 1e-3]))), 0)),
            'init_cond': np.array([exp(sin(t_1 ** 2)), exp(cos(t_1** 2))]),
            'interval':  np.array([t_1, t_1 + 4])
            },
        'test': {
            'func': lambda x, y: y,
            'init_cond': 1,
            'interval': np.array([0, 5])
            }
        }

    methods = {
        'classis_RK': {
            's': 4,
            'a': np.array([[0, 0, 0, 0], [1/2, 0, 0, 0], [0, 1/2, 0, 0], [0, 0, 1, 0]]),
            'b': np.array([1/6, 2/6, 2/6, 1/6]),
            'c': np.array([0, 1/2, 1/2, 1])
        },
        '3/8_RK': {
            'a': np.array([[0, 0, 0, 0], [1/3, 0, 0, 0], [-1/3, 1, 0, 0], [1, -1, 1, 0]]),
            'b': np.array([1/8, 3/8, 3/8, 1/8]),
            'c': np.array([0, 1/3, 2/3, 1])
        },
        'Dormand-Prince': {
            'a':  np.array([[0, 0, 0, 0, 0, 0, 0], [1/5, 0, 0, 0, 0, 0, 0], [3, 1, 0, 0], [1, -1, 1, 0]]),
            'b':  np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]),
            'c':  np.array([0, 1/3, 2/3, 1]),
            'b"': np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])
        },
        'simple_Runge_2': {
            's': 2,
            'a': np.array([[0, 0], [1/2, 0]]),
            'b': np.array([0, 1]),
            'c': np.array([0, 1/2])
        }
        
    }
    runge_kutta(task['test'], methods['classis_RK'], 0.05)