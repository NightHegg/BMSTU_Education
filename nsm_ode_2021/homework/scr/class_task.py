from math import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

class task:
    def __init__(self, N_var):
        self.t_1 = N_var * 0.1
        self.t_2 = self.t_1 + 4
        self.function = lambda t, y: np.array([2*t, -2*t]) * y * np.log(np.amax(np.vstack((np.flip(y, 0), np.array([1e-3, 1e-3]))), 0))
        self.exact_solution = lambda t: np.exp([sin(t ** 2), cos(t ** 2)])
        self.methods = {
            'classic_RK': {
                'a': np.array([
                    [0, 0, 0, 0], 
                    [1/2, 0, 0, 0], 
                    [0, 1/2, 0, 0], 
                    [0, 0, 1, 0]]),
                'c': np.array([0, 1/2, 1/2, 1]),
                'b': np.array([1/6, 2/6, 2/6, 1/6])
            },
            '3_8_RK': {
                'a': np.array([
                    [0, 0, 0, 0], 
                    [1/3, 0, 0, 0], 
                    [-1/3, 1, 0, 0], 
                    [1, -1, 1, 0]]),
                'c': np.array([0, 1/3, 2/3, 1]),
                'b': np.array([1/8, 3/8, 3/8, 1/8])
            },
            'DP': {
                'a': np.array([
                    [0, 0, 0, 0, 0, 0, 0], 
                    [1/5, 0, 0, 0, 0, 0, 0], 
                    [3/40, 9/40, 0, 0, 0, 0, 0], 
                    [44/45, -56/15, 32/9, 0, 0, 0, 0],
                    [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0, 0],
                    [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0, 0],
                    [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]]),
                'c': np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1]),
                'b': np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]),
                'b_add': np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])
            },
        }
    

    def runge_kutta_template(self, t, y, step, a, c, b):
        k = np.zeros((c.size, y.size))
        for i in range(c.size):
            k[i] = self.function(t + step * c[i], y + step * a[i] @ k)
        return y + step * b @ k


    def get_dict_results(self, type_task, tol, step, fac, facmin, facmax, absolute_error):
        solution = []
        list_steps = []
        amnt_iters = 0
        amnt_rejected_steps = 0

        y = np.array([self.exact_solution(self.t_1)])
        t = self.t_1

        init_time = time.time()
        while True:
            cur_y = np.array([y[-1]])
            cur_t = np.array([t])
            if type_task == 'DP':
                p = 5
                sample_1 = self.methods[type_task]['a'], self.methods[type_task]['c'], self.methods[type_task]['b']
                sample_2 = self.methods[type_task]['a'], self.methods[type_task]['c'], self.methods[type_task]['b_add']

                y_2 = self.runge_kutta_template(cur_t[0], cur_y[0], step, *sample_1)
                y_1 = self.runge_kutta_template(cur_t[0], cur_y[0], step, *sample_2)

                cur_t = np.append(cur_t, cur_t[-1] + step)

            else:
                p = 4
                method = 'classic_RK'
                for i in range(2):
                    cur_y = np.append(cur_y, [self.runge_kutta_template(cur_t[-1], cur_y[-1], step, **self.methods[method])], axis = 0)
                    cur_t = np.append(cur_t, cur_t[-1] + step)

                y_few_steps = cur_y[-1]
                omega = self.runge_kutta_template(cur_t[0], cur_y[0], step * 2, **self.methods[method])

                y_1 = y_few_steps
                y_2 = omega
            
            d_i = np.linalg.norm(y_1 + ((y_1 - y_2) / (2**p - 1))) if absolute_error else 1
            err = (1 / (2**p - 1)) * np.linalg.norm(y_1 - y_2) / d_i
            step_new = step * min([facmax, max([facmin, fac * (tol / err) ** (1 /(p+1))])])
            amnt_iters += 1
            if err <= tol:
                y = np.append(y, [y_1], axis = 0)
                t = cur_t[-1]
                step = step_new
            else:
                amnt_rejected_steps += 1
                step = step_new
                facmax = 1

            if t >= self.t_2:
                dict_results = {
                    'amnt_iters': amnt_iters,
                    'amnt_rejected_steps': amnt_rejected_steps,
                    'total_time': time.time() - init_time,
                    'global_error': np.linalg.norm(y - self.exact_solution(self.t_2)) / np.linalg.norm(self.exact_solution(self.t_2))
                    }
                return dict_results

    def test_task(self, type_task):
        p = 5 if type_task == 'DP' else 4

        parameters = {
                'type_task': type_task,
                'tol': 1e-4,
                'step': 0.1,
                'fac': 0.25 ** (1 / (p + 1)),
                'facmin': 0.6,
                'facmax': 4,
                'absolute_error': False
                }
        self.get_dict_results(**parameters)


    def task_01(self, type_task):
        p = 5 if type_task == 'DP' else 4

        step = np.array([1e-3, 1e-2, 1e-1, 1])
        fac = [0.8, 0.9, 0.25 ** (1 / (p + 1)), 0.38 ** (1 / (p + 1))]
        facmin = np.linspace(0.2, 0.66, num = 10)
        facmax = np.linspace(1.5, 5, num = 10)

        dict_review = {}
        array_for_tests = facmax
        for cur_param in array_for_tests:
            parameters = {
                'type_task': type_task,
                'tol': 1e-4,
                'step': step[-2],
                'fac': fac[-2],
                'facmin': 0.6,
                'facmax': cur_param,
                'absolute_error': False
                }
            dict_review[f'{cur_param:.3g}'] = self.get_dict_results(**parameters)
        
        plt.plot(dict_review.keys(), list(map(lambda x: dict_review[x]['amnt_iters'], dict_review.keys())))
        plt.title('Количество итераций')
        plt.show()

        plt.plot(dict_review.keys(), list(map(lambda x: dict_review[x]['amnt_rejected_steps'], dict_review.keys())))
        plt.title('Количество отброшенных шагов')
        plt.show()

        plt.plot(dict_review.keys(), list(map(lambda x: dict_review[x]['total_time'], dict_review.keys())))
        plt.title('Общее время')
        plt.show()

        plt.plot(dict_review.keys(), list(map(lambda x: dict_review[x]['global_error'], dict_review.keys())))
        plt.title('Глобальная погрешность')
        plt.show()