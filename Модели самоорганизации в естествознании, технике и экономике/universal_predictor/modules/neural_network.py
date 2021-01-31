import numpy as np

def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean() 

def sigmoid(x, deriv = False):
    return 1 / (1 + np.exp(-x))

def deriv(x):
    return x * (1 - x)

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
        
    def feedforward(self, inputs):
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)
    
    def inspect(self):
        print(f"Weights = {self.weights}, bias = {self.bias}, shift = {self.shift}")

class NeuronLayer:
    def __init__(self, amnt_neurons, input_size):
        weights = np.ones(input_size) * 0.5
        self.list_neurons = [Neuron(weights, 0.5) for _ in range(amnt_neurons)]
        
    def feedforward(self, inputs):
        self.inputs = inputs
        self.outputs = [cur_neuron.feedforward(inputs) for cur_neuron in self.list_neurons]
    
    def get_output(self):
        return self.outputs
    
    def inspect(self):
        print(f"Amount of neurons = {len(self.list_neurons)}, inputs = {self.inputs}, outputs = {self.outputs}\n")
        
class NeuronNetwork:
    def __init__(self, size_first_layer, info):
        self.list_hidden_layers = [NeuronLayer(value, size_first_layer) if key == 0 else NeuronLayer(value, info[key - 1]) for key, value in info.items()]
        self.final_layer = NeuronLayer(1, list(info.values())[-1])
        
    def feedforward(self, inputs):
        cur_inputs = inputs
        for i in self.list_hidden_layers:
            i.feedforward(cur_inputs)
            cur_inputs = i.get_output()
        
        self.final_layer.feedforward(cur_inputs)
        self.prediction = self.final_layer.get_output()
        return self.prediction
    
    def get_prediction(self):
        return self.prediction
    
    def inspect(self):
        print(f"Amount of hidden layers = {len(self.list_hidden_layers)}")
    
    def back_propogation(self, output, lr):
        weight = 0
        y_true, y_pred = output, self.prediction[0]
        
        self.final_layer.list_neurons[0].shift = -2 * (y_true - y_pred) * deriv(y_pred)
        
        all_layers = self.list_hidden_layers + [self.final_layer]
         
        for num_layer, layer in enumerate(all_layers[::-1][1:], start = 1):
            for num_neuron, neuron in enumerate(layer.list_neurons): #НОВЫЙ СЛОЙ
                for layer_2 in all_layers[::-1][:num_layer]: #ИДЁМ ПО СТАРЫМ СЛОЯМ
                    for neuron_2 in layer_2.list_neurons:
                        weight += neuron_2.shift * neuron_2.weights[num_neuron]           
                neuron.shift = weight * deriv(layer.outputs[num_neuron])
                weight = 0
        
        for layer in self.list_hidden_layers: 
            for num, neuron in enumerate(layer.list_neurons):
                neuron.weights -= lr * neuron.shift * np.array(layer.inputs)
                neuron.bias -= lr * neuron.shift
        
        self.final_layer.list_neurons[0].weights -= lr * self.final_layer.list_neurons[0].shift * np.array(self.final_layer.inputs)
        self.final_layer.list_neurons[0].bias -= lr * self.final_layer.list_neurons[0].shift
        
    def train(self, inputs, outputs, amnt_epochs):
        self.epochs = []
        self.mse_errors = []
        lr = 0.5
        
        for epoch in range(amnt_epochs):
            for cur_x, cur_y in zip(inputs, outputs):
                self.feedforward(cur_x)
                self.back_propogation(cur_y, lr)
                
            if epoch % 10 == 0:
                self.epochs.append(epoch)
                y_preds = np.apply_along_axis(self.feedforward, 1, inputs)
                self.mse_errors.append(mse_loss(np.ravel(y_preds), outputs))
