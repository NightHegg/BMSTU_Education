from modules.neural_network import NeuronNetwork
import matplotlib.pyplot as plt
import numpy as np

def logistic_map(par_lambda, x):
    return par_lambda * x * (1 - x)

def get_lyapunov_exponent(x):
    r = 4
    result = []
    for cur_element in x:
        result.append(np.log(abs(r - 2*r*cur_element)))
    return np.mean(result)

def get_logistic_map_bar(x_start, par_lambda, amnt_iterations):
    par_lambda = 4
    x = x_start
    bar = [x_start]
    for _ in range(amnt_iterations):
        x = logistic_map(par_lambda, x)
        bar.append(x)
    return np.array(bar)


def create_data(bar_test, size_first_layer, amnt_train_tests):
    train_data = []
    train_results = []
    test_data = []
    test_results = []

    for i in range(amnt_train_tests):
        train_data.append(bar_test[i * size_first_layer:i * size_first_layer + size_first_layer])
        train_results.append(bar_test[i * size_first_layer + size_first_layer])
    
    for i in range(10):
        test_data.append(bar_test[200 + i:200 + i + size_first_layer])
        test_results.append(bar_test[200 + i + size_first_layer])

    return np.array(train_data), np.array(train_results), np.array(test_data), np.array(test_results)


bar_plot = get_logistic_map_bar(0.01, 4, 200)
bar_test = get_logistic_map_bar(0.11, 4, 8000)

#plt.plot(np.arange(200 + 1), bar_plot)
#plt.title("Первые 200 шагов")
#plt.show()

amnt_iters = 40
amnt_train_tests = 15
size_first_layer = 12
info_hidden_layers = dict(zip(range(2), [10, 10]))

network = NeuronNetwork(size_first_layer, info_hidden_layers)

train_data, train_results, test_data, test_results = create_data(bar_test, size_first_layer, amnt_train_tests)

network.train(train_data, train_results, amnt_epochs = 200)

init_data = np.ravel(train_data[0])
for i in range(amnt_iters):
    init_data = np.append(init_data, network.feedforward(init_data[i:i + size_first_layer]))

print(f"Calculated lyapunov exponent = {get_lyapunov_exponent(init_data)}")

plt.plot(network.epochs, network.mse_errors)
plt.savefig('img/mse_errors.png')
plt.show()


plt.plot(np.arange(init_data[:size_first_layer].size), init_data[:size_first_layer],"r")
plt.plot(np.arange(size_first_layer, init_data.size), init_data[size_first_layer:],"b")
plt.title("Первые 200 шагов")
plt.show()