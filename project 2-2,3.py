import numpy as np


class DenseLayer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.bias = np.random.randn(n_neurons)

    def forward(self, x):

        output = np.dot(x, self.weights) + self.bias
        output = np.maximum(0, output)
        return output



n_inputs = 3
n_neurons = 2

dense_layer = DenseLayer(n_inputs, n_neurons)
input_data = np.array([[1.0, -1.0, 2.0], [3.0, 0.0, -2.0]])
output_data = dense_layer.forward(input_data)

print(output_data)