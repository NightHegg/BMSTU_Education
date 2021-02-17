import matplotlib.pyplot as plt
import numpy as np
from scr.euler import explicit_euler, implicit_euler

class shooting_method():
    def __init__(self, task, data):
        self.task = task
        self.func = data["function"]
        self.title = data["title"]
        self.list_time = data["list_time"]
        self.list_degree = data["list_degree"]
        self.init_cond = data["init_cond"]

    def get_solution(self):
        fig, ax = plt.subplots()
        fig.suptitle(self.title, fontsize=30)
        plt.xlabel("t", fontsize=20)
        plt.ylabel("y", fontsize=20)
        ax.grid()
        fig.set_figheight(8)
        fig.set_figwidth(15)
        for cur_degree in self.list_degree:
            y = explicit_euler(np.array([[self.init_cond[0], cur_degree]]), self.func, self.list_time)[:, 0]
            ax.plot(self.list_time, y)
    
    def save_solution(self):
        plt.savefig(f"img/{self.task}.png")


class euler_method():
    def __init__(self, task, data):
        self.method = data["method"]
        self.task = task
        self.func = data["function"]
        self.deriv_func = data["deriv_function"]
        self.title = data["title"]
        self.list_time = data["list_time"]
        self.init_cond = data["init_cond"]

    def get_solution(self):
        fig, ax = plt.subplots()
        fig.suptitle(self.title, fontsize=30)
        plt.xlabel("t", fontsize=20)
        plt.ylabel("y", fontsize=20)
        ax.grid()
        fig.set_figheight(8)
        fig.set_figwidth(15)
        if self.method == "explicit":
            y = explicit_euler(np.array([self.init_cond]), self.func, self.list_time)
        elif self.method == "implicit":
            y = implicit_euler(np.array([self.init_cond]), self.func, self.deriv_func, self.list_time)
        else:
            pass
        ax.plot(y[:, 0], y[:, 1])
        ax.scatter(self.init_cond[0], self.init_cond[1], c = "deeppink")

    def save_solution(self):
        plt.savefig(f"img/{self.task}.png")

if __name__ == "__main__":
    list_degree = np.linspace(-5, 2, 10)
    list_t = np.linspace(0, 1, 20)
    tasks_shooting_method = {
        "ex_01_01" : 
                {"function": lambda x: np.array([x[1], np.exp(x[0])]),
                 "title": f"Задача для случая exp(y)",
                 "list_time": np.linspace(0, 1, 20),
                 "list_degree": np.linspace(-5, 1.5, 10),
                 "init_cond": np.array([1, 1])},

        "ex_01_02" : 
                {"function": lambda x: np.array([x[1], -np.exp(x[0])]),
                 "title": f"Задача для случая exp(y)",
                 "list_time": np.linspace(0, 1, 20),
                 "list_degree": np.linspace(-5, 5, 20),
                 "init_cond": np.array([1, 1])}
            }
    tasks_euler_method = {
        "ex_02_01": 
                {"method": "explicit",
                 "function": lambda x: np.array([x[1], -10 * x[0] / 0.05]),
                 "deriv_function": lambda x: np.array([1, -10 / 0.05]),
                 "title": f"Явный метод Эйлера",
                 "list_time": np.linspace(0, 1, 50),
                 "init_cond": np.array([2, 0])},

        "ex_02_02": 
                {"method": "implicit",
                 "function": lambda x: np.array([x[1], -10 * x[0] / 0.05]),
                 "deriv_function": lambda x: np.array([1, -10 / 0.05]),
                 "title": f"Неявный метод Эйлера",
                 "list_time": np.linspace(0, 1, 50),
                 "init_cond": np.array([2, 0])},
            }

    for task, data in tasks_euler_method.items():
        obj = euler_method(task, data)
        obj.get_solution()
        obj.save_solution()